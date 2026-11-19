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

An AgiBot tick for a schema-v5 checkpoint uses `robot_action.v2`. A checkpoint
with two latent frames per chunk expects eight action rows per tick. Legacy
AgiBot checkpoints use `robot_action.v1` with raw actions.

## Reference images

With height and width omitted, the preprocessor uses the image aspect ratio
and `resolution` (480 by default) to choose a canvas. It pads the image on the
bottom and right for VAE encoding, then crops the latent to the aligned content
dimensions. A 640×360 input produces 640×352 output. Callers that supply
their own padded canvas can pass
`conditioning_image_size=[padded_h, padded_w, content_h, content_w]`.

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
with the paged pool sized for the requested rollout.

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
same three spans. Regional compilation and Inductor CUDA graphs can be enabled
for supported deployments with `use_cuda_graphs: true` and `enforce_eager: false`.

See the [offline runner](sim_bimanual/README.md)
for reference JSONL/NPZ input, action-sidecar and camera inputs, and validation
commands. CPU tests do not establish checkpoint quality, GPU memory usage, or
equivalence between one-shot and incremental VAE decoding.

## Cookbook port assessment

The cookbook additions were ported from the uncommitted work based on
`mbala/cosmos_dreams_transfer` (`8f37207b3`) to `mbala/cosmos_dreams`
(`2c974f21e`). The target already has causal generation, action/video packing,
typed robot ticks, session-owned VAE caches, and dense/paged KV history. Its
pipeline has a different request-control structure and newer media transport
and mixed-precision support, so replacing it with the Transfer pipeline would
discard target-branch behavior.

| Change | Applicability on the target branch |
| --- | --- |
| Action-sidecar JSONL, media loading, prompt formatting, batch output metadata | Added through the offline cookbook adapter; the existing reference format remains available. |
| Explicit raw/model action space | Added to request controls and the session fingerprint. Sidecars bypass normalization; raw inputs normalize once. |
| Camera commands and pose JSON | Added with frame alignment, rot6d conversion, 16-frame anchors, and frozen camera regression vectors. |
| Camera inference profile and global-asinh normalizer | Added with action-schema-v4 validation; existing v3 contracts remain supported. |
| Canonical resolution helper and 480p portrait bucket | Added while retaining the target's current framework imports and embodiment aliases. |
| Incremental decoding for long full videos | Adapted to this branch's generation loop and shared output builder. Typed media, guardrail routing, latent output, and denoising precision schedules remain available. |
| Transfer-only pipeline helpers, registration, and documentation | Omitted; the cookbook exercises action/camera conditioning. Unrelated Nano/Transfer wording edits were also omitted. |

The adapter covers both open-loop cookbook inputs, but GPU acceptance is still
required before claiming reproduced model outputs. It accepts aligned frame
counts explicitly instead of silently rounding them, and rejects malformed
camera commands. The supported sampler uses the exported four-step distilled
schedule and guidance 1.0; the framework command's guidance 3.0, shift 5.0,
defaults file, and latency preset are not interchangeable CLI settings here.

With a compatible checkpoint, validate the four 61-frame camera cases,
the 901-frame action rollout, and a 1801-frame endurance run on CUDA. Compare
conditioning and pre-VAE latents first, then causal versus one-shot decoding,
and record actual frame counts, action/camera adherence, memory, and throughput.
The bounded KV window and incremental VAE path need numerical and visual
acceptance independently. Closed-loop policy feedback is outside these examples.
