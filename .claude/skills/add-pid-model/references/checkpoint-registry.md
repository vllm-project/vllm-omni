# PiD Checkpoint & Latent-Space Registry (Reference)

> **Important**: this file is a **snapshot** for offline fallback. ALWAYS re-verify
> against the live official sources first (see "Remote Query Procedure" below) —
> the official repo is the single source of truth.
>
> Snapshot verified: 2026-08-25, from `nv-tlabs/PiD` (`docs/checkpoints.md`,
> `pid/_src/inference/checkpoint_registry.py`) and `nvidia/PiD` HF repo.
> vllm-omni registry section updated (2026-08-29).

---

## Remote Query Procedure

```text
GitHub source of truth:   https://github.com/nv-tlabs/PiD/blob/main/pid/_src/inference/checkpoint_registry.py
Checkpoint reference:     https://github.com/nv-tlabs/PiD/blob/main/docs/checkpoints.md
HF weights tree:          https://huggingface.co/nvidia/PiD/tree/main/checkpoints
HF model page:            https://huggingface.co/nvidia/PiD
Paper:                    https://arxiv.org/abs/2605.23902
```

Download all checkpoints locally (for offline use):

```bash
hf download nvidia/PiD --local-dir . --include "checkpoints/*"
```

---

## Official Support Matrix (distilled checkpoints, `model_ema_bf16.pth`)

| Backbone | 2k only | 2k → 4k |
|---|---|---|
| flux | PiD_res2k_sr4x_official_flux_distill_4step | **PiD_v1pt5_res2kto4k_sr4x_official_flux_distill_4step** |
| flux2 | PiD_res2k_sr4x_official_flux2_distill_4step | **PiD_v1pt5_res2kto4k_sr4x_official_flux2_distill_4step** |
| flux2-klein-4b | same as flux2 | same as flux2 |
| flux2-klein-9b | same as flux2 | same as flux2 |
| zimage | same as flux | same as flux |
| zimage-turbo | same as flux | same as flux |
| qwenimage | — | **PiD_v1pt5_res2kto4k_sr4x_official_qwenimage_distill_4step** |
| qwenimage-2512 | — | same as qwenimage |
| sd3 | PiD_res2k_sr4x_official_sd3_distill_4step | **PiD_res2kto4k_sr4x_official_sd3_distill_4step** |
| sdxl | — | **PiD_res2kto4k_sr4x_official_sdxl_distill_4step** |
| dinov2 | PiD_res2k_sr4x_official_dinov2_distill_4step | — |
| siglip | PiD_res2k_sr8x_official_siglip_distill_4step (8x) | — |

**Bold** = the checkpoint currently wired in vllm-omni for that backbone.
All are 4-step distilled. Undistilled variants exist for flux / flux2 / qwenimage
(not used by vllm-omni). Deprecated FLUX/FLUX.2/Qwen-Image `2kto4k` (v1) checkpoints
were moved to `checkpoints_deprecated/` — do not wire them.

`--pid_ckpt_type` guidance: `2kto4k_v1pt5` for FLUX / FLUX.2 / Qwen-Image 4K;
`2kto4k` for SD3 / SDXL; `2k` is 2048px-only.

---

## Latent Space → Backbone → VAE weights

A PiD decoder is tied to a **latent space** (VAE), not a single model. Aliases in
the same row share one checkpoint.

| Latent space | VAE / encoder weights (repo) | Compatible backbones | LDM latent ch / spatial down |
|---|---|---|---|
| Flux1-dev | checkpoints/ae.safetensors | flux, zimage, zimage-turbo | 16 / 8x |
| Flux2-dev | checkpoints/flux2_ae.safetensors | flux2, flux2-klein-4b, flux2-klein-9b | 128 (patchified; 32 raw) / 16x |
| SD3 medium | checkpoints/sd3_vae/ | sd3 | 16 / 8x |
| SDXL | checkpoints/sdxl_vae.safetensors | sdxl | 4 / 8x |
| Qwen-Image | checkpoints/QwenImage_VAE_2d.pth | qwenimage, qwenimage-2512 | 16 / 8x |
| DINOv2-B | checkpoints/rae/ (class-conditional) | dinov2 | n/a (RAE) |
| SigLIP-2 | checkpoints/scale_rae/ (text-conditional) | siglip | n/a (Scale-RAE) |
| Boogu-Image | Flux-style VAE (optional integration) | boogu | 16 / 8x (Flux-like) |

The VAE weight file is what the upstream decoder plugs into; vllm-omni does **not**
download it at runtime — the LDM pipeline's own VAE produces the latent. Use the
row to confirm the latent-space match and derive `lq_latent_channels` /
`latent_spatial_down_factor` (and `lq_latent_unpatchify_factor` for patchified
forms).

---

## Current vllm-omni Registries

### Net configs — `vllm_omni/diffusion/pid/config.py`

Per-backbone net configs (only LQ args differ; shared PixDiT_T2I backbone args are
identical):

| Backbone | lq_latent_channels | latent_spatial_down_factor | lq_latent_unpatchify_factor |
|---|---|---|---|
| qwenimage | 16 | 8 | 1 |
| flux | 16 | 8 | 1 |
| sd3 | 16 | 8 | 1 |
| sdxl | 4 | 8 | 1 |
| flux2 | 128 | 16 | 2 |

`get_pid_net_config(backbone)` validates against exactly these 5 names.
`PID_CHECKPOINT_REGISTRY` maps each to its official `nvidia/PiD` checkpoint
(`_PID_HF_REPO = "nvidia/PiD"`, `_PID_CKPT_ROOT = "checkpoints"`) — qwenimage /
flux / flux2 use the `v1pt5` 2kto4k path, sd3 / sdxl use the `2kto4k` path.

Shared sampling config (`PID_SAMPLING_CONFIG`): 4 SDE steps,
`student_t_list=[0.999, 0.866, 0.634, 0.342, 0.0]`, prediction_type `velocity`,
`fm_timescale=1000.0`.

### LatentForm table — `vllm_omni/diffusion/pid/latent_forms.py`

| Pipeline class | Backbone | `to_x0` | Latent branch returns |
|---|---|---|---|
| `FluxPipeline` | flux | `_unpack_packed_2x2` | `[B, T, 64]` packed tokens |
| `QwenImagePipeline` | qwenimage | `_unpack_packed_2x2` | `[B, T, 64]` packed tokens |
| `ZImagePipeline` | flux | `_identity` | `[B, 16, zH, zW]` grid |
| `Flux2Pipeline` | flux2 | `_patchify_and_normalize` (needs `pipeline`) | `[B, 32, zH, zW]` VAE-ready grid |
| `Flux2KleinPipeline` | flux2 | `_patchify_and_normalize` (needs `pipeline`) | `[B, 32, zH, zW]` VAE-ready grid |
| `StableDiffusion3Pipeline` | sd3 | `_identity` | `[B, 16, zH, zW]` grid |
| `StableDiffusionXLPipeline` | sdxl | `_identity` | `[B, 4, zH, zW]` grid |

`lookup_latent_form(pipeline)` walks `type(pipeline).__mro__`, so unlisted
subclasses of a registered class inherit automatically. Unregistered families
(e.g. `boogu_image`) get a warning at init and a per-request error if PiD is
explicitly requested.

---

## Key Files (vllm-omni)

| File | Role |
|---|---|
| `vllm_omni/diffusion/pid/latent_forms.py` | **Adaptation table**: `LATENT_FORMS`, `LatentForm`, `lookup_latent_form`, `to_x0` pure functions |
| `vllm_omni/diffusion/pid/runner_integration.py` | Runner-layer orchestration (mount / gating / passthrough / stepwise) |
| `vllm_omni/diffusion/pid/config.py` | net configs, `PID_CHECKPOINT_REGISTRY`, getters |
| `vllm_omni/diffusion/pid/decoder.py` | `PidDecodeConfig`, `PidDecoder` (eager load, resident) |
| `vllm_omni/diffusion/pid/checkpoint.py` | `resolve_pid_checkpoint_path` (local/HF-ref/auto-download), `load_pid_checkpoint` |
| `vllm_omni/diffusion/pid/pid_model.py` | `PidInferenceModel` (PidNet + Gemma, sampler, precision) |
| `vllm_omni/diffusion/pid/text_encoder.py` | `GemmaTextEncoder` (gemma-2-2b-it) |
| `vllm_omni/diffusion/worker/diffusion_model_runner.py` | Runner 3-stage PiD wiring (batch passthrough + stepwise) |
| `vllm_omni/diffusion/registry.py` | `init_pid_decoder_on` mount point |
| `docs/design/feature/pid_decode.md` | design doc |
| `vllm_omni/diffusion/pid/PiD_TEST_GUIDE.md` | test-level plan (L1–L3) |
