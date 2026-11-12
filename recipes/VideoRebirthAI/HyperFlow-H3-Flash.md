# HyperFlow H3 Flash

> Compressed few-step MiniMax-H3 variant for joint video + stereo audio
> (T2VA / FL2VA; Ref2VA phased). Experimental — RFC
> [#10179](https://github.com/vllm-project/vllm-omni/issues/10179).

## Summary

- Vendor: VideoRebirthAI
- Transformer: [`VideoRebirthAI/hyperflow-h3-flash`](https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash)
- Text encoder / VAEs: [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) (base ~31B DiT is **not** downloaded)
- Tasks (P0): T2VA, first/last-frame FL2VA
- Sampling: 4 / 6 (default) / 8 steps, **CFG off**
- Mode: OpenAI-compatible `/v1/videos`
- Status: Experimental

HyperFlow is **not** a FastH3 / FlashGen LoRA on the full H3 base. It ships a
pruned + low-rank + rotated W8A8 INT8 DiT (`fl2va_int8_convrot` /
`ref2va_int8_convrot`) with two-time `(t, r)` conditioning.

### License

MiniMax-H3 Community License territory / redistribution constraints still apply
to the base TE/VAE components. HyperFlow transformer code/weights follow their
upstream Apache-2.0 / model-card terms. Review both before customer deployment.

### Unsupported in v1 (RFC non-goals)

- GGUF / stable-diffusion.cpp / Unsloth consumer paths
- Bitwise SOL-H3 realtime parity
- Fun ControlNet, timeline guides, latent-mask edit, ComfyUI-omni templates
- Replacing FastH3 / FlashGen (those remain few-step adapters on full H3)

## Serve

```bash
hf auth login
export MODEL=VideoRebirthAI/hyperflow-h3-flash
# Optional override if the MiniMax-H3 snapshot is already local:
# export HYPERFLOW_BASE_MODEL=/path/to/MiniMaxAI/MiniMax-H3

vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port 8091 \
  --trust-remote-code \
  --task-type fl2va
```

`ref2va` partition (P1):

```bash
vllm serve "${MODEL}" --omni --task-type ref2va ...
```

Optional extras:

```bash
# Explicit base snapshot / revision
--additional-config '{"extras":{"base_model":"MiniMaxAI/MiniMax-H3","base_revision":"<sha>"}}'
```

## Request contract

Keep the MiniMax-H3 `/v1/videos` schema (`task=t2va|fl2va|ref2va`, duration,
frames, seed). Set `num_inference_steps` to `4`, `6`, or `8` (default `6`).
Do not enable CFG (`guidance_scale` must be omitted or `<= 1`).

For T2VA, pass an explicit `aspect_ratio` (for example `1:1`). To match the
official CLI's minimum length (**124** frames at 24 FPS), use
`duration=5.0` (or an equivalent `num_frames=124`). `duration=5.2` rounds up to
**141** frames under H3's `17n+5` rule.

### Example T2VA request (aligned with CUDA evidence below)

```bash
curl -sS -X POST "http://localhost:8091/v1/videos/sync" \
  -F 'prompt=A clay fox walks through a snowy forest, soft stop-motion style.' \
  -F 'num_inference_steps=6' \
  -F 'seed=42' \
  -F 'width=480' \
  -F 'height=480' \
  -F 'aspect_ratio=1:1' \
  -F 'fps=24' \
  -F 'extra_params={"task":"t2va","duration":5.0,"guidance_scale":1.0,"aspect_ratio":"1:1"}' \
  -o omni_t2va_clayfox.mp4
```

## Experimental results

### Ascend NPU (smoke)

Short-lived 4× Ascend 910B1 (`--num-gpus 4 --usp 4`), `fl2va` first+last,
6 steps, CFG off, ~480×480 / ~4 s: `/v1/videos/sync` HTTP 200, coherent motion
after `(t, r)` wiring (pre-fix mosaic).

### CUDA vs official CLI (L20X)

Same prompt / seed / steps / resolution / frame count:

| Item | Omni (this PR) | Official `hyperflow-h3-flash` CLI |
| --- | --- | --- |
| Prompt | `A clay fox walks through a snowy forest, soft stop-motion style.` | same |
| Seed / steps | `42` / `6` | `42` / `6` |
| Resolution | `480×480`, `aspect_ratio=1:1` | `--height 480 --width 480` |
| Length | `duration=5.0` → **124** frames / 5.175 s | default `MIN_FRAMES=124` / 5.175 s |
| CFG | off (`guidance_scale=1`) | off |
| Hardware | 1× NVIDIA L20X | 1× NVIDIA L20X |
| DiT weights | `/data/models/hyperflow-h3-flash` (`fl2va_int8_convrot`) | same local weights |
| TE / VAE | Omni `MiniMax-H3-FL2VA` layout | Hub `MiniMaxAI/MiniMax-H3` |
| Wall | ~17 s sync request (warm server) | ~12 s denoise; cold CLI longer (Hub fetch) |
| Sample | See inline videos on PR [#10257](https://github.com/vllm-project/vllm-omni/pull/10257) (not committed to the tree) | same |

Notes: Omni P0 **densifies** INT8/low-rank into dense Omni H3 modules, so
outputs are visually close at matched length but not bitwise identical to the
native INT8 official path. Mismatched `duration` (4.0 → 107f, 5.2 → 141f) was
the main cause of earlier large visual drift.

## Implementation notes (current branch)

- Reuses **`MiniMaxH3Pipeline`** (same pattern as FastH3 full checkpoints): Hub
  id resolves to that class; no separate serve pipeline.
- Startup stages a MiniMax-shaped tree: HyperFlow INT8 DiT + base TE/VAE
  symlinks (no 31B transformer download).
- P0 densifies INT8 convrot + low-rank into Omni's fused H3 modules for
  loadability, and wires official pair-unique `(t, r)` (with
  `r = 1 - sigma_{i+1}`) through the MiniMax-H3 denoise loop.
- Native W8A8 kernels (without densify) remain a follow-up.

## Upstream

- Weights: https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash
- Code: https://github.com/Video-Rebirth/hyperflow-h3-flash
- Omni H3 recipe: [MiniMax-H3.md](../MiniMaxAI/MiniMax-H3.md)
