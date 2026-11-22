# Cosmos3-Nano-Sim-Auto

Cosmos3-Nano-Sim-Auto is the causal, chunked, multi-camera Cosmos3 driving world model
(RGB WSM transfer, optionally joint RGB + LiDAR). The reference is imaginaire4's
`sim_auto_{rgb,joint}_distilled_rolling_inference` with regular student weights trained using
deduplicated cross-view attention (the current `sim_auto_joint_stage3_sfdmd_mads_all` defaults): per 0.4 s chunk it
denoises three RGB latents per camera (and four LiDAR sweeps) with a four-step SDE schedule at
guidance 1, attends a two-slot KV ring (chunk 0 plus the previous chunk) with per-token view/time
visibility (own view unrestricted, other views through a 0.4 s past-only window, per-camera
captions), and commits each chunk with a clean refresh.

## Artifact contract

Use the two-stage imaginaire4 exporter with `--cosmos3-nano-sim-auto` on a *rolling inference*
experiment. Stage 1 derives the manifest (chunk clock, ring, replay policy, prompt policy, mRoPE,
student schedule, LiDAR V1.2 contract) and hashes the checkpoint; Stage 2 embeds it under
`transformer/config.json["cosmos3_nano_sim_auto"]`, names `Cosmos3NanoSimAutoPipeline` and, for
joint students, packages the LiDAR VAE.

```bash
export COSMOS_INTERNAL=1 COSMOS_TRAINING=1
export PYTHONPATH="$PWD/packages/cosmos3:$PWD${PYTHONPATH:+:$PYTHONPATH}"
EXP=sim_auto_joint_distilled_rolling_inference   # or sim_auto_rgb_distilled_rolling_inference
CKPT=/checkpoints/YOUR_DEDUPLICATED_STAGE3_RUN/checkpoints/iter_XXXXXXXXX/model

python -m cosmos3.scripts.export_model \
  --checkpoint-path "$CKPT" \
  --config-file projects/cosmos3/interactive/configs/config.py \
  --experiment "$EXP" \
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

Keep `--no-use-ema-weights`: the student uses regular weights. The RGB-only and joint experiments
share weights; only the joint export carries the LiDAR projections and `lidar_vae/`.

The Stage 3 training selector itself has no rolling cache; export with the matching rolling
inference selector above. Verify the training run's actual attention policy before export:
the exporter validates the selected configuration, not the policy used to train the weights.
The older `joint_sfdmd_b12200_s2i1200_seed0/iter_000003600` checkpoint requires the
`*_rolling_inference_legacy` profiles in imaginaire4 and is **unsupported** here. Legacy attention
counts some same-view keys twice across its attention partitions; a matching Boolean visibility
mask does not reproduce that softmax weighting. Export and runtime validation intentionally
reject `deduplicate_cross_view=False`.

Explicit RGB/joint system prompts are included even though the reference recipe sets
`use_system_prompt=False`: the reference tokenizer gives an explicit prompt precedence over
that flag. Do not remove the system message when comparing prompts.

## Deployment

Start from [`cosmos3_nano_sim_auto.yaml`](../../vllm_omni/deploy/cosmos3_nano_sim_auto.yaml): the
AR-Diffusion engine, eager execution, batch size one, 480p. The KV ring lives in the paged pool with
one padded chunk per frame unit (7 cameras: 16,384 tokens RGB-only, 18,208 joint); sink and window
come from the artifact and cannot be overridden. Tensor parallelism is supported (reference
parity target: TP4 on 4x GB200, equivalent to the reference's CP4); sequence, CFG, pipeline and
VAE patch parallelism are rejected.

## Requests

Requests follow the Cosmos3 multiview contract: `multiview.views[*]` with `camera_key`, `prompt`,
`control_path` (WSM) and optional `vision_path` (first frame), top-level `wsm: {}`, `num_frames`
(`4k+1`), `fps`, `resolution`; joint requests add `lidar.control_path` (`.safetensors`,
`frames` `[3,T,128,1800]`), optional `lidar.condition_path` with `num_conditional_sweeps: 1`, and
`return_output: true`. Prepare them from the reference JSONL with
`cosmos3.scripts.prepare_multiview_lidar --rolling --num-views 7 --num-frames 297 --fps 30`; the
first caption span is held for the horizon, controls are cut to its source window so the server's
last-frame padding matches the reference, and student-ignored sampling fields are dropped.

Guidance must be 1.0 and `num_inference_steps` either unset or the student's step count. Joint
inference requires `fps` equal to the artifact's clock (30). Camera subsets of the exported rig are
accepted in request order.

See the [offline runner](../../examples/offline_inference/cosmos3_nano_sim_auto/README.md).

## Validation status

| Item | Status |
| --- | --- |
| Chunk clock, token geometry, frame unit (297 frames, 7 views, 480p → 26 chunks, 390/228 tokens, 18,208-token unit) | validated (CPU tests) |
| Dense refresh retains old history until publication; opt-in RGB/LiDAR latent output | CPU regressions pass |
| Replay visibility planner vs token-level oracle and hand-derived reference rules | validated (CPU tests) |
| Fixed-step SDE sampler seed arithmetic | validated (CPU tests) |
| Exporter manifest contract | CPU tests previously passed in an isolated import harness; actual checkpoint export/load pending |
| Transformer / pipeline on GPU, latent parity vs `--save-latents` goldens, TP4 | not tested |
| Tick sessions over the videos API | not implemented (phase 2) |

The Sim-Auto tests do not require generated JSON reference fixtures. Reference comparisons
for replay visibility, prompt text/token IDs, and CPU SDE trajectories are outside this test suite.
