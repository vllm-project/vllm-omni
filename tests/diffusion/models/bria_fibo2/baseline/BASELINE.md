# Bria Fibo-2.0 Dev — Diffusers baseline reference

This directory documents the reference data the parity test
(`../test_fibo2_transformer_parity.py`) compares against. The reference
tensors themselves live outside the repo (about 85 MiB); `run_baseline.py`
is a snapshot of the script that generated them, reformatted to repo style
and exempted from lint (it targets the external diffusers environment).

## Where the data lives

Set (or rely on the defaults of) two environment variables:

- `FIBO2_HF_HOME` (default `/root/autodl-fs/fibo-baseline/hf-home`) — the HF
  cache holding `briaai/Fibo-2.0-dev`
- `FIBO2_REF_DATA` (default `/root/autodl-fs/fibo-baseline/ref-data`) — the
  captured reference tensors

## How the baseline was produced

Single A800 80GB (driver 590.48.01 / CUDA 13.1), batch 1, 1024x1024, 30
steps, guidance 5.0, seed 42, bf16 weights:

| component | version |
|---|---|
| diffusers | 0.42.0 (PyPI; = PR #14991 merge) |
| transformers | 5.21.0 |
| torch | 2.14.1+cu130 |
| accelerate | 1.15.0 |
| model revision | `72415bd021af33b2cbc63525683a114ff8ac294e` (HF `main`, 2026-10-25) |

The fixed prompt is the diffusers docstring caption (red bicycle, no
`text_render`); the negative prompt is the pipeline's fixed null caption.

Two-phase protocol (both verified bitwise-reproducible across processes):

- **capture** — replays `BriaFibo2Pipeline.__call__`'s T2I path step by step
  (encode -> initial latents from a cuda generator -> 30-step loop -> VAE
  decode), then re-runs the official `pipe.__call__` on the same
  `prompt_embeds`/`latents` and checks the final latents are bitwise equal
  (manual-vs-official: `True`, max diff `0.0`).
- **verify** — an independent process repeats capture and compares every
  tensor bitwise (initial latents, embeds, per-step trajectory, final
  latents, decoded image pixels). Result: `ALL_PASS=True`.

## Reference files (ref-data/run1/)

| file | contents |
|---|---|
| `tensors/initial_latents.safetensors` | seed-42 initial noise latents, fp32 |
| `tensors/prompt_embeds.safetensors` | Qwen3-VL 6-bundle embeds, bf16, `(1, 6, 78, 7680)` |
| `tensors/negative_prompt_embeds.safetensors` | null-caption embeds, bf16, `(1, 6, 46, 7680)` |
| `tensors/first_step.safetensors` | step-0 inputs/outputs: `latent_model_input` (bf16), `noise_pred_cond` / `noise_pred_uncond` / `noise_pred_guided` (fp32) |
| `tensors/selected_steps.safetensors` | same capture at steps 15 and 29 (`step_015_*`, `step_029_*`) |
| `tensors/trajectory.safetensors` | all 30 step latents, fp32 |
| `tensors/final_latents.safetensors` | final latents + the official-pipe rerun |
| `meta.json` | versions, run config, timing, peak memory, sha256 of every artifact |

The transformer input timestep is `timesteps[k] / 1000` (timestep 0 is
1000.0 -> 1.0); scheduler timesteps live in `meta.json` under
`timing_s.timesteps`.

SHA256 of every tensor file is recorded in `meta.json` under
`artifacts_sha256`.

## Parity results (vLLM-Omni port, TP=1, bf16)

Sampled across the whole denoising interval (see the parity test for
tolerances and how to run it):

| point | cosine | max abs diff | rel err |
|---|---|---|---|
| step 0, cond | 0.999932 | 0.047 | 0.91% |
| step 0, uncond | 0.999933 | 0.031 | 0.67% |
| step 15, cond | 0.999933 | 0.078 | 1.31% |
| step 15, uncond | 0.999933 | 0.132 | 2.23% |
| step 29, cond | 0.999929 | 0.078 | 1.34% |
| step 29, uncond | 0.999928 | 0.070 | 1.21% |

The cosine is a flat 0.99993 +- 0.000003 across all six points — one noise
signature — while the max deviation grows with trajectory depth as an
extreme-value statistic of bf16 kernel noise.

Padding-mask semantics are covered by `test_masked_batch_consistency`: a
batched row (78-token prompt padded against a 46-token one) reproduces its
unpadded forward at cosine 0.9999, and a cross-tier batch (78 vs 300 tokens,
gist 256 vs 320, exercising the gist-padding branch with the image-first
layout) likewise reproduces the unpadded row.

A RoPE precision probe compared the shared bf16 `apply_rope_to_qk` against
the diffusers fp32/complex rotation: the isolated rotations differ by up to
0.031 (mean 7e-4), but swapping the fp32 rotation into the full forward
leaves the parity numbers unchanged (rel err 0.00909 vs 0.00909) — the
difference is absorbed by attention/RMSNorm reduction noise, so the shared
bf16 path is kept.

## Regenerating

```bash
python run_baseline.py --mode capture   # writes <out>/run1
python run_baseline.py --mode verify    # independent rerun, expects ALL_PASS
```

`env.sh` in the original workspace pins `HF_HOME`/`HF_ENDPOINT` (hf-mirror)
and the gated-repo token; the script needs `transformers>=5.20.1`,
`diffusers==0.42.0`, and a CUDA GPU.

## Edit Turbo baseline (ref-data/edit-turbo1)

`briaai/Fibo-2.0-edit-turbo` (revision `9a24ee68`; note Bria force-pushed
new weights after the initial `b8dd54e5` download, so pin what you
compare against) is the same
`BriaFibo2Pipeline` architecture, distilled for editing: 4 steps,
guidance 1.0 — no classifier-free guidance, one forward per step, the
negative prompt has no effect. Generated with the same script:

```bash
python run_edit_baseline.py --model briaai/Fibo-2.0-edit-turbo \
  --steps 4 --guidance 1.0 --out ref-data/edit-turbo1
```

Bitwise reproducible (ALL_PASS; manual-vs-official bitwise equal).
vLLM-Omni parity on `9a24ee68`: engine e2e cosine 0.9998 / 39.7 dB
(earlier `b8dd54e5` weights measured 36.1 dB smoke / 43.9 dB e2e).
