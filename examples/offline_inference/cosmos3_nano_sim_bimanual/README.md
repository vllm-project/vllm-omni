# Cosmos3-Nano-Sim-Bimanual offline inference

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
For checkpoints containing multiple legacy YAM datasets, select `abc_yam`,
`molmoact2_yam`, or `xdof_yam` by name because all three use domain 16 while
retaining distinct normalizers.

```bash
python examples/offline_inference/cosmos3_nano_sim_bimanual/cosmos3_nano_sim_bimanual.py \
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

## Inference overrides

For `nvidia/Cosmos3-Nano-Sim-Bimanual@1d95a0b5b19d49a24aceebf578cf5e85db310ac0`,
use `--deploy-config vllm_omni/deploy/cosmos3_nano_sim_bimanual_full_history.yaml`.
With a conditioning image, every generated chunk uses two denoising steps:
`[1, 0.8333333333333334]`. Frame 0 is encoded from the image. The four-step
schedule applies only when generating frame 0 without an image. Full history
is retained for up to 901 video frames at 480 resolution.

For the preprocessed AgiBot NPZ example:

```bash
python examples/offline_inference/cosmos3_nano_sim_bimanual/cosmos3_nano_sim_bimanual.py \
  --model /checkpoints/cosmos3-nano-sim-bimanual-diffusers \
  --jsonl agibot_eval_257f_npz/samples.jsonl \
  --sample-index 0 \
  --deploy-config vllm_omni/deploy/cosmos3_nano_sim_bimanual_full_history.yaml \
  --num-frames 257 --height 480 --width 640 --fps 30 --seed 42 \
  --output outputs/agibot_257f.mp4
```

For checkpoint-normalized action sidecars, use `--input-format cookbook` instead of the
raw-action NPZ path. This also formats the action prompt and selects the
action-conditioned image preprocessing:

```bash
python examples/offline_inference/cosmos3_nano_sim_bimanual/cosmos3_nano_sim_bimanual.py \
  --model /checkpoints/cosmos3-nano-sim-bimanual-diffusers \
  --jsonl /data/agibot.jsonl --input-format cookbook --sample-index 0 \
  --deploy-config vllm_omni/deploy/cosmos3_nano_sim_bimanual_full_history.yaml \
  --resolution 480 --num-frames 901 --fps 30 --seed 42 \
  --output outputs/agibot_901f.mp4
```

Supply an initial image already at the target canvas (832×480 for 16:9)
to avoid reflection padding in the generated video.

Overrides live under `stages[0].model_config.inference_overrides`. The last
`frame_sigma_schedules` entry repeats; chunk starts select schedules by absolute
latent-frame index. `history_mode: full` requires `max_num_frames`; longer
rollouts are rejected. Sliding mode accepts `kv_cache_inference_size` and
`attention_sink_size`. Omitted settings retain the artifact defaults.

## Action-sidecar and camera inputs

Use `--input-format cookbook` with one JSON object per line. For a checkpoint
supporting both robot and camera conditioning, export both embodiments and the
`cookbook` camera profile following the
[artifact recipe](../../../recipes/cosmos3/Cosmos3-Nano-Sim-Bimanual.md#action-and-camera-export).
The `--model` argument must point to the local Stage 2 Diffusers directory.

For action conditioning, create `/data/humanoid.jsonl` with a record such as:

```json
{"name":"robot","prompt":"A robot moves objects on a table.","vision_path":"robot.png","action_path":"actions.json","domain_name":"agibotworld","raw_action_dim":29,"view_point":"ego_view","fps":30,"num_frames":901,"action_chunk_size":900,"seed":42}
```

Place the conditioning image and action sidecar beside the JSONL, then run:

```bash
python examples/offline_inference/cosmos3_nano_sim_bimanual/cosmos3_nano_sim_bimanual.py \
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
Typed robot-action ticks continue to accept raw values only.

For camera conditioning, create `/data/camera.jsonl` with a record such as:

```json
{"name":"stone_w","prompt":"A stone sculpture in a garden.","vision_path":"stone.png","camera_trajectory":"w-61","num_frames":61,"fps":30,"seed":42}
```

Place the conditioning image beside the JSONL, then run:

```bash
python examples/offline_inference/cosmos3_nano_sim_bimanual/cosmos3_nano_sim_bimanual.py \
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
endurance tests, with matching sidecar lengths. The default sampler uses the exported
distilled SDE schedule with guidance 1.0; the deployment above explicitly
overrides the frame schedules. Guidance and shift overrides do not change this path.

The deployment retains its configured guardrail behavior. To match the
cookbook's explicit `--no-guardrails` parity configuration, copy the deployment
YAML, set `stages[0].model_config.guardrails: false`, and pass that copy with
`--deploy-config`. Record this choice with the validation results.

Records run sequentially with separate reset/close sessions. Each output has
an index and sanitized sample name. `sample_outputs.json` records success or
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

### Validation

The CPU suite exercises preprocessing, both contract versions, and the actual
rollout control flow with stub transformer/decoder calls:

```bash
python tools/run_bimanual_cpu_tests.py -q
```

The four `stone_*` camera cases also compare resolved poses, raw actions, and
normalized actions against frozen regression vectors. See the
[fixture descriptions](../../../tests/diffusion/models/cosmos3_nano_sim_bimanual/fixtures/README.md).
The vectors are checked in and are never regenerated during tests.

On a supported CUDA host, run the real checkpoint through 61-, 901-, and
1801-frame requests. Check every `sample_outputs.json` entry and record
throughput and peak GPU/host memory. The finite KV window needs visual quality
evaluation over long rollouts.

Decode acceptance must compare vLLM-Omni's one-shot VAE decode with incremental
causal decode using identical saved latents, VAE weights, device, and dtype.
Compare tensors before clamping and video encoding; record maximum and mean
absolute error per frame, with numerical tolerances declared before the run.
Include `[1, 4, 4, ...]` latent chunks, terminal partial chunks of 1, 2, and 3,
cache resets between requests, and each supported VAE execution configuration.
Check chunk boundaries explicitly. The mocked CPU decoders do not establish
pixel equivalence; this GPU/checkpoint acceptance remains outstanding.
