# Cosmos3-Nano-Sim-Auto

Cosmos3-Nano-Sim-Auto is the causal, chunked, multi-camera Cosmos3 driving world model
(RGB WSM transfer, optionally joint RGB + LiDAR). The reference is imaginaire4's
`cosmos3.scripts.inference` on the Stage 3 student selector `sim_auto_joint_stage3_sfdmd_mads_all`
with rolling KV overrides and regular (non-EMA) weights trained with deduplicated cross-view
attention: per 0.4 s chunk it denoises three RGB latents per camera (and four LiDAR sweeps) with a
four-step SDE schedule at guidance 1, attends a two-slot KV ring (chunk 0 plus the previous chunk)
with per-token view/time visibility (targets: own view unrestricted, other views through a 0.4 s
past-only window; controls: own view only; per-camera captions), adds each camera's physical
rig-view embedding (and the LiDAR row) to its control and target tokens, and commits each chunk
with a clean refresh.

## Artifact contract

Use the two-stage imaginaire4 exporter with `--cosmos3-nano-sim-auto` on the Stage 3 selector plus
the same rolling overrides the reference inference passes. Stage 1 derives the manifest (chunk
clock, ring, replay policy, prompt policy, mRoPE, student schedule, rig-view table, LiDAR V1.2
contract) and hashes the checkpoint; Stage 2 embeds it under
`transformer/config.json["cosmos3_nano_sim_auto"]`, names `Cosmos3NanoSimAutoPipeline` and, for
joint students, packages the LiDAR VAE.

```bash
export COSMOS_INTERNAL=1 COSMOS_TRAINING=1   # COSMOS_TRAINING enables --experiment-overrides
export PYTHONPATH="$PWD/packages/cosmos3:$PWD${PYTHONPATH:+:$PYTHONPATH}"
EXP=sim_auto_joint_stage3_sfdmd_mads_all
# A local copy of the DCP student (the exporter does not read s3:// paths).
CKPT=/checkpoints/YOUR_STAGE3_RUN/checkpoints/iter_XXXXXXXXX/model

python -m cosmos3.scripts.export_model \
  --checkpoint-path "$CKPT" \
  --config-file projects/cosmos3/cosmos3/configs/simulation/config.py \
  --experiment "$EXP" \
  --experiment-overrides \
    model.config.rolling_kv_cache_chunks=2 \
    model.config.rolling_kv_sink_chunks=1 \
    model.config.rolling_prompt_policy=segmented \
  --backbone-type cosmos3_multiview \
  --cosmos3-nano-sim-auto \
  --no-use-ema-weights --no-vit \
  --student-only-checkpoint-metadata \
  -o /exports/cosmos3-nano-sim-auto-hf

python -m cosmos3.scripts.convert_model_to_diffusers \
  --checkpoint-path /exports/cosmos3-nano-sim-auto-hf \
  --skip-vision-encoder --distilled-scheduler on \
  -o /exports/cosmos3-nano-sim-auto
```

Keep `--no-use-ema-weights`: the student uses regular weights. Export the joint selector even
for camera-only work: a joint artifact serves RGB-only requests (exactly as the reference runs a
joint checkpoint on a camera-only input), and the RGB-only selector keeps the LiDAR heads without
a LiDAR tokenizer, which the exporter rejects.

The manifest's `rig_view_embedding` (`num_embeddings`, physical MADS `camera_ids`, final
`lidar_id`) must agree with the converted `rig_view_embed.weight`; the pipeline maps request
cameras to their rows in request order. `prompt.text_cache_max_len` bounds the per-camera caption
tokens of a request (system prompt, rig header and caption of every camera together; default
16384). Requests above it are rejected, so raise it at export with
`--cosmos3-nano-sim-auto-text-cache-max-len` for longer captions.

Verify the training run's actual attention policy before export: the exporter validates the
selected configuration, not the policy used to train the weights. Checkpoints trained with legacy
(non-deduplicated) maskless attention are **unsupported** here: legacy attention counts some
same-view keys twice across its attention partitions, and a matching Boolean visibility mask does
not reproduce that softmax weighting. Export and runtime validation intentionally reject
`deduplicate_cross_view=False`.

Explicit RGB/joint system prompts are included even though the reference recipe sets
`use_system_prompt=False`: the reference tokenizer gives an explicit prompt precedence over
that flag. Do not remove the system message when comparing prompts.

## Deployment

Start from [`cosmos3_nano_sim_auto.yaml`](../../vllm_omni/deploy/cosmos3_nano_sim_auto.yaml): the
AR-Diffusion engine, eager execution, batch size one, 480p. The KV ring lives in the paged pool with
one padded chunk per frame unit (7 cameras: 16,384 tokens RGB-only; joint 23,616 with the current
1x1 LiDAR diffusion patch, 18,208 with 2x2); sink and window come from the artifact and cannot be
overridden. Tensor parallelism is supported (reference
parity target: TP4 on 4x GB200, equivalent to the reference's CP4); sequence, CFG, pipeline and
VAE patch parallelism are rejected.

## Requests

Requests follow the Cosmos3 multiview contract: `multiview.views[*]` with `camera_key`, `prompt`,
`control_path` (WSM) and optional `vision_path` (first frame), top-level `wsm: {}`, `num_frames`
(`4k+1`), `fps`, `resolution`; joint requests add `lidar.control_path` (`.safetensors`,
`frames` `[3,T,128,1800]`), optional `lidar.condition_path` with `num_conditional_sweeps: 1`, and
`return_output: true`. Prepare them from the reference JSONL with
`cosmos3.scripts.prepare_multiview_lidar --rolling --num-views 7 --num-frames 297 --fps 30 --seed 0`;
the first caption span is held for the horizon, controls are cut to its source window so the
server's last-frame padding matches the reference, student-ignored sampling fields are dropped,
and every request gets the same seed (the reference applies `--seed` unchanged to every sample of
an input file).

Guidance must be 1.0 and `num_inference_steps` either unset or the student's step count. Joint
inference requires `fps` equal to the artifact's clock (30). Camera subsets of the exported rig are
accepted in request order.

See the [offline runner](../../examples/offline_inference/cosmos3_nano_sim_auto/README.md).

## Validation status

| Item | Status |
| --- | --- |
| Chunk clock, token geometry, frame unit (297 frames, 7 views, 480p → 26 chunks, 390 RGB tokens per frame) | validated (CPU tests) |
| Dense refresh retains old history until publication; opt-in RGB/LiDAR latent output | CPU regressions pass |
| Replay visibility planner vs token-level oracle and hand-derived reference rules (incl. maskless partitions: controls read own-view history only) | validated (CPU tests) |
| Rig-view embedding: manifest table, request-order camera rows, LiDAR row, applied before the timestep on controls and targets | validated (CPU tests) |
| Fixed-step SDE sampler seed arithmetic | validated (CPU tests) |
| Exporter manifest contract (incl. `rig_view_embedding`) | CPU tests; actual checkpoint export/load pending |
| Transformer / pipeline on GPU, latent parity vs `--save-latents` goldens, TP4 | not tested |
| Tick sessions over the videos API | not implemented (phase 2) |

The Sim-Auto tests do not require generated JSON reference fixtures. Reference comparisons
for replay visibility, prompt text/token IDs, and CPU SDE trajectories are outside this test suite.
