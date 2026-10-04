# Cosmos3-Nano-Sim-Bimanual

Cosmos3-Nano-Sim-Bimanual is the causal action-conditioned Cosmos3 world model.
The runtime supports distilled four-step SDE checkpoints with `[1, 4, 4, ...]`
latent chunks and per-frame interleaved action/video tokens. The original
reference artifact is `interact_8b_tfdcm_chunk4_agibot` at `iter_000001600`.
Camera-conditioned generation requires a compatible checkpoint export and the
camera inference profile described below.

## Artifact contract

Use the two-stage imaginaire4 exporter with `--cosmos3-nano-sim-bimanual`.
Stage 1 derives checkpoint identity, sampler settings, and the action contract
from the resolved experiment and materialized checkpoint. It validates the
selected EMA/student tensor inventory and publishes
`cosmos3_nano_sim_bimanual_manifest.json`. Stage 2 discovers that manifest
automatically, writes `Cosmos3NanoSimBimanualPipeline` into `model_index.json`,
and embeds the full runtime contract under `cosmos3_nano_sim_bimanual` in
`transformer/config.json`. It emits `cosmos3_nano_sim_bimanual_artifact.json`
with tensor headers and per-shard SHA-256 hashes.

Raw DCP is intentionally not accepted by Stage 2.

Schema-v1 artifacts may also contain the frozen pre-unification YAM datasets
exported by imaginaire4: `abc_yam`, `molmoact2_yam`, and `xdof_yam`. vLLM-Omni
validates their 20D left-arm/right-arm layout, domain 16, and separate legacy
normalizer sources before serving them. The unified `yam_dataset` classes have
different coordinate semantics and are not interchangeable with these legacy
contracts.

From the `imaginaire4/packages/cosmos3` checkout, materialize the complete
checkpoint locally, then run both stages. No hand-written manifest is needed:

```bash
python -m cosmos3.scripts.export_model \
  --checkpoint-path /checkpoints/iter_000001600/model \
  --config-file /runs/interact_8b_tfdcm_chunk4_agibot/config.yaml \
  --experiment interact_8b_tfdcm_chunk4_agibot \
  --cosmos3-nano-sim-bimanual \
  --student-only-checkpoint-metadata \
  -o /exports/cosmos3-nano-sim-bimanual-hf

python -m cosmos3.scripts.convert_model_to_diffusers \
  --checkpoint-path /exports/cosmos3-nano-sim-bimanual-hf \
  -o /exports/cosmos3-nano-sim-bimanual-diffusers
```

The renamed runtime requires newly exported metadata. Regenerate existing
exports through both stages; legacy imports, deployment names, and export
flags are no longer supported. Existing action-schema-v3 exports remain valid;
the cookbook camera profile uses action schema v4. Neither changes model weights.

### Action and camera export

For the private-preview experiment
`causal_8b_sf_dmd_max_4step_cam_chunk4_480p_961f`, supply its actual materialized
checkpoint and saved resolved config. Add these flags to Stage 1 above:

```bash
--cosmos3-nano-sim-bimanual-add-embodiments agibotworld camera_pose \
--cosmos3-nano-sim-bimanual-camera-inference-profile cookbook
```

Then run Stage 2 on that Stage 1 output. The resulting artifact contains both
29D AgiBot (domain 15) and 9D camera (domain 2) contracts. Adding an embodiment
declares a conditioning contract; use a checkpoint trained for that embodiment.
The camera profile records backward pose deltas anchored every 16 pixel frames,
translation scale 1.0, and `global_asinh` normalization using the first nine
channels of the canonical 59D global camera/gripper statistics. Both export
stages bundle and verify those statistics and their hashes.

This is an explicit **inference override**. The saved training dataset
descriptor remains unchanged, including a framewise/scale camera configuration
when present. Do not edit a v3 artifact by hand or pass runtime flags that
contradict its camera contract. Existing framewise/scale exports continue to
work with an explicitly matching legacy camera recipe.

The statistics JSON declares `allowed_pose_conventions=["backward_anchored"]`.
The `cookbook` inference profile explicitly selects chunk-anchored poses. Exports
reconcile this through `normalizer.source.pose_convention_override`, naming
the `bimanual_camera_global_asinh_v1` policy, the original declaration,
the effective `backward_chunk_anchored_16f` convention, and
`inference_camera_profile` as the authority for inference. Both export stages require this
provenance and validate the statistics metadata against the supported profile;
they preserve the original statistics bytes. Existing numerical transforms and
behavioral hashes remain unchanged. Earlier v4 bundles without the override
record need a fresh Stage 1 export before conversion with this exporter.

Stage 1 rejects remote DCP paths for causal exports: download the complete tree
first so `checkpoint_hash` can be verified. It compares the selected
`net_ema.*` namespace against the export target before DCP loading, preventing
missing tensors from surviving as random initialization. Stage 2 then compares
the serialized tensor names, shapes, and dtypes against its strict remap
inventory. Keep `cosmos3_nano_sim_bimanual_artifact.json` with the model directory.

## Deployment

Start from [`cosmos3_nano_sim_bimanual.yaml`](../../vllm_omni/deploy/cosmos3_nano_sim_bimanual.yaml).
The default is eager, batch size one, 720×1280, one resident session, and a
96-latent-frame window. Each request resolves an aligned canvas and selects a
matching paged KV pool; the maximum permitted geometry is validated at load.

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

The KV startup allocation is a hard floor. At TP=1/BF16, one 720p frame across
36 layers is about 133 MiB; the configured window, scratch reservation, and
the 512-token text pool are all counted before model
execution. If the manager reports an insufficient memory budget, use a
checkpoint artifact exported with a smaller `window_frames`; runtime overrides
are rejected. The physical window does not cap the logical session at that
length: evicted positions remain logical placeholders while generation rolls
forward. A sliding window changes long-horizon semantics relative to the
unbounded reference and needs a separate quality evaluation. Engine-level
window/sink/reset overrides would make the paged path diverge from the
manifest-driven dense oracle.

## Request modes

- Full rollout: provide all actions, `reset=true`, and preferably
  `close_session=true`. The globally last latent frame is not committed because
  it has no downstream reader. Requests require `(num_frames - 1) % 4 == 0`;
  unaligned counts are rejected. Fresh Bimanual video rollouts decode each
  causal chunk into CPU memory, bounding GPU VAE history. The final video still
  occupies host memory proportional to its length. Latent output retains its
  existing path.
- Tick session: use `ARDiffusionSession` with an `ARDiffusionOmniTickConsumer`
  and carry AgiBot actions as a `robot_action.v1` control inside the typed
  `ARDiffusionTickRequest`. The first tick covers the singleton prefix and one
  four-frame chunk; later ticks commit one four-frame chunk. Request, event,
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
same three spans. Compile/CUDA graphs, RF/CFG checkpoints, and multi-session
serving remain follow-up work.

See the [offline runner](../../examples/offline_inference/cosmos3_nano_sim_bimanual/README.md)
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

Use the matching two-stage exporter described above to obtain an artifact for
the private-preview checkpoint. Then validate the four 61-frame camera cases,
the 901-frame action rollout, and a 1801-frame endurance run on CUDA. Compare
conditioning and pre-VAE latents first, then causal versus one-shot decoding,
and record actual frame counts, action/camera adherence, memory, and throughput.
The bounded KV window and incremental VAE path need numerical and visual
acceptance independently. Closed-loop policy feedback is outside these examples.
