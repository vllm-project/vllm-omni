---
name: add-pid-model
description: Add PiD (Pixel Diffusion) super-resolution decode support for a diffusion model in vllm-omni. Invoke when the user names a model to adapt for PiD, asks whether PiD supports a given model, or wants to register a new pipeline family. Flow: check remote PiD repo/HF weights for support -> check vllm-omni support -> latent form analysis -> config registration -> verification.
---

# Adding PiD Super-Resolution Support to a Diffusion Model

## Overview

PiD decode runs at the Runner layer: it temporarily forces
`output_type="latent"` on the batch, takes the pipeline's latent-branch output
as-is, converts it to PiD's `x_0` via the centralized **`LatentForm` table**,
and runs the PiD decoder in place of VAE decode. **Pipelines carry zero PiD
code** — adaptation is pure table/config registration.

Flow (four stages):

1. **Remote analysis** — does official PiD support this model's latent space?
   Verified against `nv-tlabs/PiD` (repo) + `nvidia/PiD` (HF weights). Never
   trust memory.
2. **omni support check** — is the model a vllm-omni pipeline, and is its
   family / latent space already registered?
3. **Latent form** — what does the pipeline's latent branch return, and which
   `to_x0` pure function (existing or new) converts it to `x_0`?
4. **Config registration** — `LATENT_FORMS` entry (+ net config / checkpoint
   registry entries **only** for genuinely new latent spaces), then verify.

Key files:

| File | Role |
|---|---|
| `vllm_omni/diffusion/pid/latent_forms.py` | **The adaptation table.** `LATENT_FORMS: {pipeline class -> LatentForm(backbone, to_x0)}`; `lookup_latent_form()` walks `__mro__` so subclasses inherit. |
| `vllm_omni/diffusion/pid/runner_integration.py` | Runner orchestration (mount, gating, force/restore/decode). Generic — never edit per model. |
| `vllm_omni/diffusion/pid/config.py` | Per-backbone net configs + `PID_CHECKPOINT_REGISTRY`. |
| `vllm_omni/diffusion/models/<family>/pipeline_*.py` | **Never edited** for PiD. |

Registered families today: **Qwen-Image / Flux / Z-Image (reuses flux) /
Flux2 / Flux2-Klein / SD3 / SDXL**.

---

## Step 1: Remote Analysis (MANDATORY — source of truth is upstream)

Confirm support against the live official sources:

```text
GitHub source of truth:   https://github.com/nv-tlabs/PiD/blob/main/pid/_src/inference/checkpoint_registry.py
Checkpoint reference:     https://github.com/nv-tlabs/PiD/blob/main/docs/checkpoints.md
                          (raw: https://raw.githubusercontent.com/nv-tlabs/PiD/main/docs/checkpoints.md)
HF weights tree:          https://huggingface.co/nvidia/PiD/tree/main/checkpoints
HF model page:            https://huggingface.co/nvidia/PiD
```

Use `WebFetch` on the checkpoint reference + HF tree; fall back to `WebSearch`
(`PiD checkpoint <model> nvidia/PiD`) if fetches fail.

A PiD decoder is tied to a **latent space** (VAE), not to a single model.
Record from upstream: the backbone name, the distilled checkpoint dir
(`model_ema_bf16.pth` inside), and the VAE the decoder was trained against.

| Observation | Verdict |
|---|---|
| Backbone listed with a distilled 2k/2kto4k checkpoint | **Supported** → Step 2 |
| Listed but only undistilled / deprecated / in `checkpoints_deprecated/` | **Not recommended** — report, do NOT adapt |
| Absent from both `docs/checkpoints.md` and `checkpoint_registry.py` | **Not supported** — report clearly, stop |
| Latent space ≈ a registered one (same ch / down-factor, e.g. Z-Image ≈ Flux) | **Supported via reuse** → Step 2 alias path |

> **Offline fallback**: if remote access is unavailable, use the snapshot in
> `references/checkpoint-registry.md` and warn the user that support was
> judged from a cached snapshot.

---

## Step 2: omni Support Check

Two questions, in order:

**2a. Is the model a vllm-omni pipeline?** Check
`vllm_omni/diffusion/models/` and the model registry for the pipeline class
(e.g. user says "flux2-klein-4b" → `Flux2KleinPipeline`). If vllm-omni does
not support the model at all, PiD adaptation is moot — report and stop
(model support must land first).

**2b. Is the family already registered?** Match `type(pipeline).__name__`
against `LATENT_FORMS` (MRO inheritance covers subclasses). Then classify:

| Case | Meaning | Work in Step 3/4 |
|---|---|---|
| Pipeline class (or a superclass) already in `LATENT_FORMS` | Family registered | **None** — verify with a smoke test only |
| New pipeline class, latent space already registered (shares a registered VAE) | Sibling / alias (e.g. Flux2Klein → flux2, Z-Image → flux) | One `LATENT_FORMS` entry |
| New latent space, upstream checkpoint exists | New backbone | Net config + checkpoint registry + `LATENT_FORMS` + new `to_x0` |
| New latent space, no upstream checkpoint | Unsupported | **Stop.** Report; do not invent configs |

Derive the latent space from the pipeline itself: VAE latent channels (e.g.
`vae.config.z_dim`), `vae_scale_factor`, and whether the DiT loop keeps
latents packed/patchified.

---

## Step 3: Latent Form Analysis

**3a. Read the pipeline's decode site.** Grep the pipeline file for
`output_type == "latent"`. The branch `image = latents` returns whatever the
pipeline natively holds there — **that exact value is your `to_x0` input, and
the pipeline must not be changed** (PiD must not alter non-PiD behavior,
including latent-branch return values). Note which transforms run *before*
the branch (unpack, BN denormalization, unpatchify, `/ scaling + shift`,
dtype casts): whatever they produce is what you must convert from.

Current family forms (verify against the file before reusing):

| Family | Latent branch returns | `to_x0` | `x_0` handed to PiD |
|---|---|---|---|
| `FluxPipeline`, `QwenImagePipeline` | `[B, T, 4C]` 2x2-packed tokens (row-major canonical order) | `_unpack_packed_2x2` | `[B, C, zH, zW]` grid |
| `ZImagePipeline`, `StableDiffusion3Pipeline`, `StableDiffusionXLPipeline` | `[B, C, zH, zW]` native grid | `_identity` | unchanged |
| `Flux2Pipeline`, `Flux2KleinPipeline` | `[B, 32, zH, zW]` VAE-ready grid (unpack + BN denorm + unpatchify all run *before* the branch — original behavior, preserved) | `_patchify_and_normalize` (needs `pipeline` for `vae.bn` stats) | `[B, 128, zH/2, zW/2]` BN-normalized 2x2-patchified grid |

**3b. New family → write a new pure function** in `latent_forms.py`
(see `references/adaptation-recipe.md` for the contract and templates):

```python
def _your_form(latent, height, width, vae_scale_factor, *, pipeline=None) -> (x0, pid_h, pid_w):
```

Pure tensor ops only (unit-testable on CPU); the optional `pipeline` kwarg is
passed by the Runner and may be used **only to read model-side constants**
(e.g. flux2's `vae.bn` running stats) — never mutable session state. Raise
`ValueError` on non-canonical shapes or a missing required `pipeline`
context (Runner is fail-loud). Contract: `x0` is `[B, C, zH, zW]` with
`C == lq_latent_channels` of the backbone and
`zH * latent_spatial_down_factor == pid_h` (`latent_spatial_down_factor`
describes x0's full spatial compression; `lq_latent_unpatchify_factor` is a
channel-level op inside `LQProjection2D`, not a spatial term).

**3c. Do NOT edit the pipeline.** If the latent branch returns *processed*
latents (denormalized / unpacked / cast to VAE dtype), that is the pipeline's
own contract — keep it and invert the processing in the `to_x0` pure function
instead. This is exactly what flux2 does: its latent branch returns the
VAE-ready 32ch grid, and `_patchify_and_normalize` inverts the pipeline's
BN denorm (re-normalize with `vae.bn` stats) and unpatchify (2x2 patchify)
at the Runner layer. Editing the pipeline's branch order changes
`output_type="latent"` return values for non-PiD users, which is a hard no.

---

## Step 4: Config Registration

**4a. `LATENT_FORMS` entry** (`vllm_omni/diffusion/pid/latent_forms.py`) —
required for every new pipeline class:

```python
LATENT_FORMS["YourPipeline"] = LatentForm("<backbone>", _your_or_existing_to_x0)
```

Alias families reuse the parent backbone and its `to_x0` (e.g.
`Flux2KleinPipeline` → `LatentForm("flux2", _patchify_and_normalize)`).

**4b. Net config + checkpoint registry** (`vllm_omni/diffusion/pid/config.py`)
— only for a genuinely new latent space:

```python
YOUR_BACKBONE_PID_NET_CONFIG = _make_net_config(
    lq_latent_channels=<VAE latent ch>,          # e.g. 4/16/128
    latent_spatial_down_factor=<VAE down>,       # e.g. 8 or 16
    lq_latent_unpatchify_factor=<1 or 2>,        # 2 if x_0 is 2x2-patchified
)
# + mapping entry in get_pid_net_config()
# + PID_CHECKPOINT_REGISTRY["<backbone>"] = (experiment_dir, path, scale)
# + export from pid/__init__.py
```

> **flux2 lesson (do not repeat it)**: the checkpoint's
> `lq_proj.latent_proj.0` is `Conv2d(32, 1024, 3)` while the patchified x_0
> has 128 channels. Set `lq_latent_unpatchify_factor=2` so
> `LQProjection2D` unpatchifies 128→32 internally; the raw VAE latent is
> 32ch @ 8x, hence `latent_spatial_down_factor=16` with the 2x patch grid.
> Symptom if you get it wrong: `size mismatch for lq_proj.latent_proj.0.weight`
> at load, or `expected input ... to have 32 channels, but got 128` at runtime.

**4c. Things NOT to touch**

- **No pipeline edits at all.** No `PID_BACKBONE`, no mixin, no hooks, no
  `_pid_caption`/`_pid_override` attrs — the Runner reads request intent from
  `req.sampling_params.pid_decode` / `sp.output_type`.
- Runner code (`runner_integration.py`, `diffusion_model_runner.py` wiring)
  is generic — no per-model changes.
- Do NOT add `--pid-scale/--pid-num-steps/--pid-seed` CLI flags — they live in
  `PidDecodeConfig` defaults, overridden per request via `pid_decode`.
- CLI surface is fixed: `--pid-enable / --pid-checkpoint / --pid-gemma`.

---

## Step 5: Verification

### 5a. L1 unit tests (CPU, no weights) — `tests/diffusion/pid/`

- `test_pid_latent_forms.py`: new `to_x0` shape math (token count ↔ grid,
  channels, `pid_h/w`), error branches for non-canonical latents;
  `lookup_latent_form` MRO inheritance.
- `test_pid_runner_integration.py`: gating table, `PidPassthrough`
  force/restore/decode.
- `test_pid_config.py` / `test_pid_checkpoint.py`: registry keys, net config
  values, checkpoint resolution.

Markers: `pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]`.
Mock with `mocker`/`monkeypatch`; never instantiate PidNet/Gemma on CPU.

### 5b. L2/L3 e2e — `tests/e2e/online_serving/test_<model>_pid.py`

L2: server with `--pid-enable` → output size == LDM size × `scale`.
L3: per-request `pid_decode` override; `enabled:false` falls back to VAE;
requesting PiD without `--pid-enable` errors.

### 5c. Manual serve smoke test

```bash
vllm serve <HF-id-or-local-path> --omni --port 8091 \
  --pid-enable \
  --pid-checkpoint <local model_ema_bf16.pth or omit for auto-download> \
  --pid-gemma /path/to/gemma-2-2b-it
```

Send a request with `"pid_decode": {"enabled": true, "scale": 4}`; confirm
the returned image is `scale`× the LDM resolution. Also confirm
`"pid_decode": {"enabled": false}` matches a non-PiD server (VAE path
unchanged).

---

## References

- `references/checkpoint-registry.md` — official support matrix (snapshot),
  remote query procedure, latent-space ↔ VAE characteristics, current
  vllm-omni registries.
- `references/adaptation-recipe.md` — `to_x0` contract, per-family templates,
  registration snippets, runtime checklist.
- Design doc: `docs/design/feature/pid_decode.md`
- Test guide: `vllm_omni/diffusion/pid/PiD_TEST_GUIDE.md`
- Upstream: `https://github.com/nv-tlabs/PiD` · `https://huggingface.co/nvidia/PiD` ·
  paper `https://arxiv.org/abs/2605.23902`
