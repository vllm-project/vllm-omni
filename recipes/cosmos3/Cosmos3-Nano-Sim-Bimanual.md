# Cosmos3-Nano-Sim-Bimanual

Cosmos3-Nano-Sim-Bimanual is the causal, action-conditioned Cosmos3 world
model. Use an exported artifact whose transformer configuration contains a
`cosmos3_nano_sim_bimanual` manifest. The runtime reads the checkpoint
identity, sampling schedule, history policy, and action contract from that
manifest.

## Actions

Provide 1 normalized unified action row per video-frame transition:

- `action_space="normalized_unified_v1"`
- 59 values per row
- 1 `domain_id`, or 1 domain ID per action row for mixed-domain sequences

The runtime validates the values and domain IDs, pads each action row with 5
zeros to the model's 64D width, and interleaves 4 action tokens with every
latent video frame. Each domain populates the fields it supports in the shared
59D layout and sets unsupported fields to 0. For example, camera-domain rows
use the first 9 values. Input normalization happens before the request reaches
vLLM-Omni.

Full rollouts provide all action rows and normally use `reset=true` and
`close_session=true`. A request with `num_frames=N` requires `N-1` action rows
and `(N-1) % 4 == 0`. Fresh full-video requests decode causal chunks into CPU
memory to bound GPU VAE history; host video memory grows with rollout length.
Typed AgiBot tick requests use `robot_action.v2` and carry 8 normalized unified
rows for each 2-latent-frame chunk.

## Reference images

Pass the original reference image. When height and width are omitted, the
preprocessor selects an aspect-matched canvas from `resolution` (480 by
default), downsizes only when necessary, pads the bottom and right for VAE
encoding, and crops the encoded latent to the aligned content dimensions. For
example, a 640×360 image produces 640×352 video. Explicit height and width use
the requested output geometry.

## Deployment

Start from
[`cosmos3_nano_sim_bimanual.yaml`](../../vllm_omni/deploy/cosmos3_nano_sim_bimanual.yaml).
The distilled checkpoint artifact contains its denoising schedule. Omit
`num_inference_steps` to use the checkpoint-defined schedule.

For dense attention with inference optimizations, merge these settings into
diffusion stage 0:

```yaml
engine_backend: default
enforce_eager: false
diffusion_compile_granularity: regional
diffusion_compile_dynamic: true
vae_fast_path: channels_last
model_config:
  use_cuda_graphs: true
  clean_commit_mode: batched
  overlap_vae_decode: true
  kv_cache_inference_size: 64
  attention_sink_size: 8
```

- `engine_backend: default` selects dense attention. It used less peak memory
  with comparable latency in the Bimanual benchmark and is the path used by
  the supplied deployments.
- Regional `torch.compile` optimizes transformer blocks; `use_cuda_graphs: true`
  enables Inductor CUDA graph replay to reduce launch overhead and requires
  `enforce_eager: false`.
- `clean_commit_mode: batched` refreshes clean-frame KV entries together instead
  of 1 frame at a time.
- `overlap_vae_decode: true` overlaps causal VAE decoding with clean-frame refresh
  for fresh full-video requests.
- `vae_fast_path: channels_last` optimizes VAE decoding with channels-last memory
  layout to reduce decode time.

`kv_cache_inference_size` and `attention_sink_size` configure sliding history
without changing the checkpoint manifest. Units are
latent frames: 64 with 8 sinks retains 8 initial frames and up to 55 recent past
frames, reserving 1 position for the current frame. Other valid cache/sink sizes
are supported. Changing the history window can change generated outputs.

With neither override, the checkpoint history policy applies. The manifest's
`window_frames` and `sink_frames` have the same meaning as
`kv_cache_inference_size` and `attention_sink_size`: `window_frames: 64` with
`sink_frames: 8` is the same 64-frame cache as the overrides above. A checkpoint
with `window_frames: null` and `sink_frames: 0` uses full history and sizes the
cache for the requested rollout.

For typed tick video, the diffusion-stage option
`video_output_transport: {enable_device_postprocess: true}` converts
`output_type="np"` video to uint8 before GPU-to-host transfer. Full-video
rollouts already copy decoded chunks to CPU and do not gain this transfer
optimization. Guardrails, interpolation, and other output formats can retain
the floating-point path; see the
[transport guide](../../docs/user_guide/diffusion/device_side_video_postprocess.md).

Compatible serialized ModelOpt FP8/NVFP4 checkpoints also support
`cosmos3_mixed_precision`. The precision schedule restarts for every denoising
chunk; clean-frame KV refresh uses the final step's precision. For a 4-step
checkpoint, `first_steps=1` and `last_steps=1` select A16/native/native/A16.
This does not enable quantization for BF16 checkpoints. See the
[ModelOpt configuration](../../docs/user_guide/quantization/modelopt.md#cosmos3-mixed-precision-schedule)
for artifact requirements and how to enable the policy.

## Offline inference

`offline_bimanual` runs full rollouts from JSONL records with the default
deployment above. Each record can provide:

- `prompt` (or `ai_caption` or `text`);
- a source `input_video`, `video`, `frames`, or `image`; its first frame is the
  reference image;
- `action` rows, with optional `action_space`, `domain_id`, or
  `domain_name`/`embodiment`;
- `fps` (or `conditioning_fps` or `frame_rate`; 15 by default) and
  `num_frames` (one more than the number of action rows by default);
- `npz_path` or `data_path`, naming an NPZ file that supplies fields the record
  does not set. NPZ object arrays are rejected;
- `action_path`, naming a prepared JSON sidecar with
  `action_space: normalized_unified_v1`, the `action` rows, and either
  `domain_id` or `domain_names`. Domain names are resolved through the
  checkpoint manifest, and `vision_path` supplies the reference image.

Relative paths are resolved against the JSONL directory.

```bash
python -m vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.offline_bimanual \
  --model /checkpoints/cosmos3-nano-sim-bimanual-diffusers \
  --jsonl /data/reference_samples.jsonl \
  --sample-index 0 \
  --seed 42 \
  --output cosmos3_nano_sim_bimanual_sample_0.mp4
```

Each sample is sent as its own session with `reset=true` and
`close_session=true`. Use `--all-samples --output-dir <dir>` to run every
record, and `--output-type latent` to save the pre-VAE latent tensor as a `.pt`
file. Omit `--height` and `--width` to use the reference-image canvas described
above. The runner checks the decoded frame count, FPS, and any explicit
dimensions, writes per-sample status to `sample_outputs.json` in the output
directory, and exits non-zero if any sample fails.
