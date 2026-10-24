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

With neither override, the checkpoint history policy applies. A checkpoint with
`window_frames: null` and `sink_frames: 0` uses full history and sizes the cache
for the requested rollout.

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
