# Cosmos3-Nano-Sim-Bimanual

Cosmos3-Nano-Sim-Bimanual is the causal action-conditioned Cosmos3 world model.
The checkpoint manifest defines the distilled SDE schedule, latent chunk size,
history policy, and action contract. Generation interleaves action and video
tokens per latent frame.

## Artifact contract

Use a Diffusers checkpoint with a schema-v1 `cosmos3_nano_sim_bimanual`
manifest in `transformer/config.json`.

The cookbook camera examples use an action-schema-v4 checkpoint with the
`camera_pose` embodiment and its `cookbook` inference profile. The profile
uses backward pose deltas anchored every 16 pixel frames and `global_asinh`
normalization. Use the matching legacy camera recipe for an action-schema-v3
checkpoint with framewise poses.

## Unified actions

For a schema-v5 checkpoint, provide one prepared 59-value
`normalized_unified_v1` action row per video-frame transition. Supply one
`domain_id` for the rollout or one per action row. Camera rows use the first
nine values and set the other slots to zero. The pipeline pads each row to
64 values. Prepare and normalize these inputs before submitting the request.

In the reference JSONL format, pass a prepared sidecar through `action_path`.
The sidecar contains `action_space: normalized_unified_v1`, the 59-value
`action` rows, and either `domain_id` or `domain_names` with one name for the
rollout or one per row.

An AgiBot tick for a schema-v5 checkpoint uses `robot_action.v2`. A checkpoint
with two latent frames per chunk expects eight action rows per tick. Legacy
AgiBot checkpoints use `robot_action.v1` with raw actions.

## Reference images

With height and width omitted, the preprocessor uses the image aspect ratio
and `resolution` (480 by default) to choose a canvas. It pads the image on the
bottom and right for VAE encoding, then crops the latent to the aligned content
dimensions. A 640×360 input produces 640×352 output.

For an image you have already padded, set
`extra_args.conditioning_image_size=[padded_h, padded_w, content_h, content_w]`.
The reference JSONL runner accepts `conditioning_image_size` directly on the
record. For example, `[480, 832, 360, 640]` preserves the 832×480 input for VAE
encoding, crops the latent to 22×40, and generates 640×352 video. Omit explicit
height and width or pass the output dimensions (352 and 640). The input image
must already match the padded canvas.

## Deployment

Start from [`cosmos3_nano_sim_bimanual.yaml`](../../vllm_omni/deploy/cosmos3_nano_sim_bimanual.yaml).
The default deployment uses eager execution, batch size one, a 720×1280
fallback resolution, and paged KV. Each request resolves an aligned canvas
and selects a matching pool. Startup validates the largest admitted geometry.
Select `engine_backend: default` on the diffusion stage to use dense KV.
Omit `num_inference_steps` to use the checkpoint schedule.

Tick video supports the shared device-side transport optimization. Add
`video_output_transport: {enable_device_postprocess: true}` to the diffusion
stage to convert `output_type="np"` video to uint8 before GPU-to-host transfer.
The optimization is opt-in and preserves typed tick session/chunk/event metadata.
Full video rollouts copy decoded chunks to CPU before shared output processing;
they do not use this optimization to reduce GPU-to-host traffic.
Latent output retains its existing path. Requests with video guardrails enabled
retain the Cosmos3 postprocessor; frame interpolation and non-NumPy presentation
retain floating-point transport. See the
[transport guide](../../docs/user_guide/diffusion/device_side_video_postprocess.md)
for precision and memory fallback behavior.

For compatible serialized ModelOpt FP8/NVFP4 Bimanual artifacts, the inherited
`cosmos3_mixed_precision` policy is evaluated separately for each four-step
denoising chunk. Clean-frame KV commits (including the initial conditioning
frame) use the final step's precision. Generation precision resets after every
chunk or commit, including failures; the reasoner follows its independent policy.
BF16 artifacts do not enable this schedule. The upstream default of three first
and three last A16 steps covers all four Bimanual steps. For example, setting
`first_steps=1` and `last_steps=1` selects A16/native/native/A16 per chunk.
See the [ModelOpt schedule documentation](../../docs/user_guide/quantization/modelopt.md#cosmos3-mixed-precision-schedule)
for checkpoint requirements and runtime configuration.

Checkpoint identity, hash, domain map, and normalizer data are deliberately not
template defaults: they must be read from `transformer/config.json`. Startup
rejects a deploy override that contradicts those embedded artifact fields.

The paged KV startup budget includes the history window, scratch storage,
and text pool. Set `kv_cache_inference_size` and `attention_sink_size` in
`model_config` to override Bimanual history capacity. With a capacity of 64
and eight sinks, the model retains eight initial frames and up to 55 recent
past frames and reserves one position for the current frame.

Schema-v5 checkpoints express `window_frames` as total cache capacity.
Legacy checkpoints express it as recent past frames, with sinks counted
separately. `window_frames: null` with `sink_frames: 0` retains full history,
with the paged pool sized for the requested rollout. Streaming sessions must
declare their total video length up front and keep it fixed across ticks.

Clean K/V commits are batched by default. Set
`model_config.clean_commit_mode: framewise` to commit one frame at a time.

Decode overlap is enabled by default for full-video CUDA requests. Set
`model_config.overlap_vae_decode: false` for serial decode.

### Native blockwise CUDA graphs

Enable regional `torch.compile(mode="reduce-overhead")` in the stage configuration:

```yaml
enforce_eager: false
diffusion_compile_granularity: regional
model_config:
  use_cuda_graphs: true
```

CUDA graphs are disabled by default. Use a deployment without HSDP to enable them.

## Request modes

- Full rollout: provide all actions, `reset=true`, and preferably
  `close_session=true`. The globally last latent frame is not committed because
  it has no downstream reader. Requests require `(num_frames - 1) % 4 == 0`;
  unaligned counts are rejected. Fresh Bimanual video rollouts decode each
  causal chunk into CPU memory, bounding GPU VAE history. The final video still
  occupies host memory proportional to its length. Latent output retains its
  existing path.
- Tick session: use `ARDiffusionSession` with an `ARDiffusionOmniTickConsumer`
  and attach the checkpoint's action control to the typed
  `ARDiffusionTickRequest`. The first tick covers the singleton prefix and one
  checkpoint-sized chunk. Later ticks commit one chunk. Request, event,
  session, and chunk identities are validated against the returned
  `metadata.ar_diffusion` envelope. Reset, close, and disconnect release the
  owning worker through `ARDiffusionWorkerLifecycle`; no inference request is
  submitted solely for cleanup.
- Dense oracle: select the default `DiffusionEngine` with the same checkpoint.
  It maintains dense per-layer history and is the numerical reference for the
  gathered-paged path.

Attention runs one joint softmax over `[real text | committed history |
current]`. The paged path uses the fused `paged_write_attn` operator and carries
the real text K/V in the auxiliary scratch slots; the dense path concatenates the
same three spans.

## Offline inference

### Reference JSONL and NPZ

This runner accepts the reference interactive JSONL plus an optional NPZ
payload. A record may contain `prompt`/`ai_caption`, `input_video`/`video`/`image`,
`action`, `fps`/`conditioning_fps`, `domain_id`, and
`domain_name`/`embodiment`, or point to an NPZ file with `npz_path` or
`data_path`. NPZ object arrays are rejected; store strings as NumPy Unicode
scalars and tensors as numeric arrays. The first source frame is used as the causal prefix;
action rows are validated and normalized using the selected embodiment's
exported raw dimension, layout, and normalizer before being padded to 64
dimensions. Mixed-layout checkpoints select the entry through `domain_name` or
a unique `domain_id`; when neither is supplied, the artifact's declared default
is used. Supply `domain_name` when several normalizers share one domain ID.
For checkpoints containing multiple YAM datasets, select `abc_yam`,
`molmoact2_yam`, or `xdof_yam` by name because all three use domain 16 while
retaining distinct normalizers.

```bash
python -m vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.offline_bimanual \
  --model /checkpoints/cosmos3-nano-sim-bimanual-diffusers \
  --jsonl /data/reference_samples.jsonl \
  --sample-index 0 \
  --num-frames 601 \
  --seed 42 \
  --output cosmos3_nano_sim_bimanual_sample_0.mp4
```

Use `--output-type latent --output sample_0.pt` for the pre-VAE parity gate.
Full rollouts send both `reset=True` and `close_session=True`, preventing the
default session from leaking history into the next sample.

Omit both `--height` and `--width` to infer an aligned, aspect-preserving
canvas from the input media, or to use the deployment default when the record
has no media. Supply both flags to request any policy-valid explicit canvas.

For the preprocessed AgiBot NPZ example:

```bash
python -m vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.offline_bimanual \
  --model /checkpoints/cosmos3-nano-sim-bimanual-diffusers \
  --jsonl agibot_eval_257f_npz/samples.jsonl \
  --sample-index 0 \
  --deploy-config vllm_omni/deploy/cosmos3_nano_sim_bimanual.yaml \
  --num-frames 257 --height 480 --width 640 --fps 30 --seed 42 \
  --output outputs/agibot_257f.mp4
```

### Action-sidecar and camera inputs

For checkpoint-normalized native action sidecars, use `--input-format cookbook`.
The adapter formats the action prompt and selects action-conditioned image preprocessing.

```bash
python -m vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.offline_bimanual \
  --model /checkpoints/cosmos3-nano-sim-bimanual-diffusers \
  --jsonl /data/agibot.jsonl --input-format cookbook --sample-index 0 \
  --deploy-config vllm_omni/deploy/cosmos3_nano_sim_bimanual.yaml \
  --resolution 480 --num-frames 901 --fps 30 --seed 42 \
  --output outputs/agibot_901f.mp4
```

Supply an initial image already at the target canvas (832×480 for 16:9)
to avoid reflection padding in the generated video.

Use `--input-format cookbook` with one JSON object per line. Point `--model`
to a local Diffusers checkpoint that meets the
[artifact contract](#artifact-contract).
To run both examples below with one checkpoint, use one that supports both
`agibotworld` and `camera_pose` and includes the `cookbook` camera profile.

For action conditioning, create `/data/humanoid.jsonl` with a record such as:

```json
{"name":"robot","prompt":"A robot moves objects on a table.","vision_path":"robot.png","action_path":"actions.json","domain_name":"agibotworld","raw_action_dim":29,"view_point":"ego_view","fps":30,"num_frames":901,"action_chunk_size":900,"seed":42}
```

Place the conditioning image and action sidecar beside the JSONL, then run:

```bash
python -m vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.offline_bimanual \
  --model /exports/bimanual-cookbook-diffusers \
  --input-format cookbook \
  --jsonl /data/humanoid.jsonl \
  --all-samples --output-dir outputs/bimanual-actions \
  --resolution 480
```

The sidecar must contain exactly `num_frames - 1` finite rows of the exported
embodiment width: `900 x 29` for the 901-frame example. This format requires
values already normalized into model space. The adapter sets
`action_space="model"`, preserves them, and pads to 64 dimensions without
normalizing them again. Direct/reference callers default to `action_space="raw"`
and apply the exported normalizer once; they may explicitly select `"model"`.
For typed ticks, use the action representation specified in [Unified actions](#unified-actions).

For camera conditioning, create `/data/camera.jsonl` with a record such as:

```json
{"name":"stone_w","prompt":"A stone sculpture in a garden.","vision_path":"stone.png","camera_trajectory":"w-61","num_frames":61,"fps":30,"seed":42}
```

Place the conditioning image beside the JSONL, then run:

```bash
python -m vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.offline_bimanual \
  --model /exports/bimanual-cookbook-diffusers \
  --input-format cookbook \
  --jsonl /data/camera.jsonl \
  --all-samples --output-dir outputs/bimanual-camera \
  --resolution 480 \
  --camera-trajectory w-61 \
  --camera-num-frames 61 \
  --camera-translation-scale 1.0 \
  --camera-action-normalization global_asinh \
  --camera-pose-convention backward_chunk_anchored_16f
```

CLI values override record values; omit `--camera-trajectory` to use different
trajectories per record. Supported commands are translations `w/s/a/d/u/n`,
rotations `up/down/left/right/cw/ccw`, `stay`, `pano`, and `orbit`, separated by commas.
Use `command-steps` syntax, with an optional positive radius for
`orbit`. Alternatively supply an absolute path to a finite `T x 4 x 4` pose JSON
starting at identity. Absolute poses are truncated or padded by holding the
last pose **before** deriving anchored deltas. Malformed commands are rejected.
`camera_num_frames` overrides the generated video's frame count as well as the
camera trajectory length; for example, 61 overrides a record's `num_frames=901`.
Camera actions are normalized once using the artifact's camera contract.

Media and action paths may be local (relative to the JSONL) or HTTP(S) URLs.
Hugging Face `/resolve/` URLs use `huggingface_hub` and its standard cached login
or `HF_TOKEN`; authenticate separately for gated assets. Video inputs use
`decord` to read their first frame. Output uses Diffusers' MP4 writer and
`imageio`/FFmpeg to verify the saved video.

The adapter chooses the 480-resolution aspect bucket by default (832x480 for
the landscape example, or 512x768 with aspect ratio `"2,3"` for a 720x1080
portrait source), honors each record's seed and FPS, and adds action/viewpoint
or camera duration/resolution prompt fields. Set custom camera
`duration_template`/`resolution_template` values directly in the record.

Requests require `F >= 2` and `(F - 1) % 4 == 0`; there is no silent temporal
realignment. Use 61 frames for smoke tests, 901 for the cookbook, and 1801 for
endurance tests, with matching sidecar lengths. The runner uses the checkpoint's
distilled SDE schedule and guidance 1.0.

The deployment retains its configured guardrail behavior. To match the
cookbook's explicit `--no-guardrails` parity configuration, copy the deployment
YAML, set `stages[0].model_config.guardrails: false`, and pass that copy with
`--deploy-config`. Record this choice with the validation results.

Records run sequentially with separate reset/close sessions. With `--output-dir`,
each output filename includes an index and sanitized sample name.
`sample_outputs.json` records success or
failure, errors, effective settings, checkpoint/contract hashes, window and
sampler configuration, elapsed time, and actual saved frame count, FPS, and
dimensions. Resolved camera poses are saved beside the video. A failed record
does not prevent later records from running; any failure makes the final exit
status nonzero. Outputs are marked synthetic in metadata.

For camera runs, `effective_camera_recipe` records the resolved convention,
normalization, translation/rotation scales, camera frame count, and normalizer
method/hash for both v3 and v4 exports. `inference_camera_profile` remains the
artifact's declaration and can be null for v3. `prompt_templates` records the
resolved duration/resolution template values, preserving explicit null or
empty-string disabling. `effective_prompt` is the exact rendered prompt
submitted to the engine. These fields are saved even if inference fails after
input preparation; action-sidecar runs also record their rendered prompt.

## Validation

The CPU suite exercises preprocessing, both contract versions, and the actual
rollout control flow with stub transformer/decoder calls:

```bash
python -m pytest -q -m cpu tests/diffusion/models/cosmos3_nano_sim_bimanual
```

The four `stone_*` camera cases also compare resolved poses, raw actions, and
normalized actions against frozen regression vectors. See the
[fixture descriptions](../../tests/diffusion/models/cosmos3_nano_sim_bimanual/fixtures/README.md).
The vectors are checked in and are never regenerated during tests.

On a compatible CUDA host, run the four 61-frame camera cases, the 901-frame
action rollout, and an 1801-frame endurance run. Check every
`sample_outputs.json` entry. Record frame counts, action and camera adherence,
throughput, and peak GPU and host memory. Evaluate visual quality across the
long rollouts when using a finite KV window.

Decode acceptance must compare vLLM-Omni's one-shot VAE decode with incremental
causal decode using identical saved latents, VAE weights, device, and dtype.
Compare tensors before clamping and video encoding; record maximum and mean
absolute error per frame, with numerical tolerances declared before the run.
Include `[1, 4, 4, ...]` latent chunks, terminal partial chunks of 1, 2, and 3,
cache resets between requests, and each supported VAE execution configuration.
Check chunk boundaries explicitly. The mocked CPU decoders do not establish
pixel equivalence; this GPU/checkpoint acceptance remains outstanding.
