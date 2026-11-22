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

Loading rejects ring shapes and prompt policies the runtime does not implement. The ring must be
the pinned sink chunks plus one rotating chunk (`ring.cache_chunks = ring.sink_chunks + 1`); only
the exported `cache_chunks=2`, `sink_chunks=1` has been compared with the reference. The prompt
must set `use_first_chunk_only: true` (each camera's caption covers the whole horizon) and
`emphasize_control_in_prompt: false` (no control directive is appended to the captions).

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
one padded chunk per frame unit. The pool is sized once at load for the exported rig's worst case
at the artifact's default resolution (every camera, LiDAR on a joint artifact, the aspect ratio
with the most tokens; 7 cameras at 480p: 16,800 tokens RGB-only, joint 24,032 with the current 1x1
LiDAR diffusion patch, 18,624 with 2x2); smaller requests pad to that unit, and a request that
needs a larger one, such as a higher resolution, is rejected on this engine. Sink and window come
from the artifact and cannot be overridden. Tensor parallelism and strict Ulysses context parallelism are supported, alone or
combined (reference parity target: TP4 or Ulysses 4 on 4x GB200, equivalent to the reference's
CP4); ring and all-gather sequence parallelism, CFG, pipeline and VAE patch parallelism are
rejected. Both TP and Ulysses split attention heads, so query and KV head counts must be divisible
by TP x Ulysses. Ulysses runs the GEN layers on a token shard of each chunk and exchanges heads for
the causal maskless attention, whose plan is unchanged; the paged ring and caption cache hold every
token for the rank's head group, so the frame unit is the same as on one GPU. Ulysses keeps a full
weight copy per GPU and replaces TP's per-layer all-reduces with all-to-alls.

The supplied YAML leaves `devices` unset, so the stage inherits `CUDA_VISIBLE_DEVICES`.
Pass `tensor_parallel_size` and `ulysses_degree` to `Omni` and expose at least TP x Ulysses GPUs.
Smaller TP sizes increase memory required per GPU. Memory fit and GPU inference parity remain
unvalidated on the target hardware.

## Requests

Requests follow the Cosmos3 multiview contract: `multiview.views[*]` with `camera_key`, `prompt`,
`control_path` (WSM) and optional `vision_path` (first frame), top-level `wsm: {}`, `num_frames`
(`4k+1`), `fps`, `resolution`; joint requests add `lidar.control_path` (`.safetensors`,
`frames` `[3,T,128,1800]`), optional `lidar.condition_path` with `num_conditional_sweeps: 1`, and
`return_output: true`. The rolling preparation command below holds the first caption span for
the horizon, cuts controls to its source window so last-frame padding matches the reference,
and drops sampling fields the student ignores. Use the same seed for each request unless the
input supplies one.

Guidance must be 1.0 and `num_inference_steps` either unset or the student's step count. Joint
inference requires `fps` equal to the artifact's clock (30). Camera subsets of the exported rig are
accepted in request order.

## Offline rollout and latent parity

In the imaginaire4 checkout, prepare the RGB input with the command below. For joint inputs, use
`wsm_lidar_transfer_i2v.jsonl` and a separate output directory.

```bash
PYTHONPATH="$PWD/packages/cosmos3:$PWD" python -m cosmos3.scripts.prepare_multiview_lidar \
  packages/cosmos3/inputs/omni_multiview/wsm_transfer_i2v.jsonl \
  --output /data/sim_auto_rgb --rolling --num-views 7 --num-frames 297 --fps 30 --seed 0
```

Save this snippet locally as `sim_auto_rollout.py`. It uses the offline `Omni` API and writes a
numbered directory per input row. Use a fresh output directory for each run.

```python
import json
import os
import sys
from pathlib import Path

import torch
from diffusers.utils import export_to_video

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras.cosmos3_lidar import lidar_output_requested, serialize_lidar_output


def main():
    model, manifest_path, output_path = sys.argv[1:]
    manifest = Path(manifest_path).resolve()
    contents = manifest.read_text()
    lines = contents.splitlines() if manifest.suffix.lower() == ".jsonl" else [contents]
    requests = [json.loads(line) for line in lines if line.strip()]
    if not requests:
        raise ValueError(f"No prepared requests in {manifest}")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True)
    save_latents = os.environ.get("SAVE_LATENTS") == "1"
    omni = Omni(
        model=model,
        tensor_parallel_size=int(os.environ.get("TP", "1")),
        ulysses_degree=int(os.environ.get("ULYSSES", "1")),
    )
    try:
        for index, request in enumerate(requests):
            request = {**request.get("extra_params", {}), **request}
            fields = ("multiview", "wsm", "lidar", "resolution", "aspect_ratio")
            extra = {key: request[key] for key in fields if key in request}
            for item in [*extra["multiview"]["views"], extra.get("lidar") or {}]:
                for key in ("control_path", "vision_path", "condition_path", "control", "vision"):
                    path = item.get(key)
                    if isinstance(path, str) and "://" not in path:
                        item[key] = str((manifest.parent / Path(path).expanduser()).resolve())
            extra.update(session_id=f"offline-{index}", reset=True, close_session=True, return_latents=save_latents)
            params = OmniDiffusionSamplingParams(
                seed=int(request.get("seed", 0)),
                num_frames=request.get("num_frames", 1),
                fps=request.get("fps"),
                extra_args=extra,
            )
            [result] = omni.generate({"prompt": request.get("prompt", ""), "modalities": ["video"]}, params)
            [video] = result.images  # [1, cameras * frames_per_view, height, width, 3] in [0, 1]
            payload = result.multimodal_output
            metadata = payload["metadata"]
            views = metadata["multiview"]
            frames_per_view = views["frames_per_view"]
            sample = output_dir / f"{index:04d}"
            sample.mkdir()
            for view, camera in enumerate(views["cameras"]):
                start = view * frames_per_view
                path = sample / f"vision_view{view:02d}_{Path(camera).name}.mp4"
                export_to_video(list(video[0, start : start + frames_per_view]), str(path), fps=views["fps"])
            if lidar_output_requested(extra):
                data, _ = serialize_lidar_output(payload["lidar"], metadata["lidar"])
                (sample / "lidar.safetensors").write_bytes(data)
            if save_latents:
                torch.save([result.latents], sample / "latents.pt")
            summary = {"name": request.get("name"), "seed": params.seed, "metadata": metadata}
            (sample / "sample_outputs.json").write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        omni.close()


if __name__ == "__main__":
    main()
```

Run with explicit GPU visibility. For Ulysses 4, replace `TP=4` with `ULYSSES=4`. For
TP2 × Ulysses 2, set `TP=2 ULYSSES=2`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 TP=4 SAVE_LATENTS=1 python sim_auto_rollout.py \
  /exports/cosmos3-nano-sim-auto /data/sim_auto_rgb/requests.jsonl outputs/sim_auto
```

Camera files follow the request order. When requested, `lidar.safetensors` stores the decoded
`frames` tensor in `[3,T,128,1800]` order (range, intensity, validity). `SAVE_LATENTS=1` also
writes the reference format, a one-element list containing `vision_latent` (`[1,C,V*T,h,w]`,
camera-major) and, for joint requests, `lidar_latent` (`[1,C_l,S,h_l,w_l]`). Both keep their
sampling dtype and observed condition prefix. Compare like-for-like RGB or joint runs with the
same checkpoint, ordered cameras, inputs, horizon, and seed. Load either runner's latent file
with `torch.load(path, map_location="cpu", weights_only=True)[0]`.

## Validation status

| Item | Status |
| --- | --- |
| Chunk clock, token geometry, frame unit (297 frames, 7 views, 480p → 26 chunks, 390 RGB tokens per frame) | validated (CPU tests) |
| Dense refresh retains old history until publication; opt-in RGB/LiDAR latent output | CPU regressions pass |
| Replay visibility planner vs token-level oracle and hand-derived reference rules (incl. maskless partitions: controls read own-view history only) | validated (CPU tests) |
| Rig-view embedding: manifest table, request-order camera rows, LiDAR row, applied before the timestep on controls and targets | validated (CPU tests) |
| Fixed-step SDE sampler seed arithmetic | validated (CPU tests) |
| Exporter manifest contract (incl. `rig_view_embedding`) | CPU tests; actual checkpoint export/load pending |
| Ulysses (2, 4): padded token shards, packed Q/K/V exchange, head-group control/ring/pool K/V vs one rank | validated (CPU Gloo tests) |
| Transformer / pipeline on GPU, latent parity against the reference, TP4, Ulysses | not tested |
| Tick sessions over the videos API | not implemented (phase 2) |

The Sim-Auto tests do not require generated JSON reference fixtures. Reference comparisons
for replay visibility, prompt text/token IDs, and CPU SDE trajectories are outside this test suite.
