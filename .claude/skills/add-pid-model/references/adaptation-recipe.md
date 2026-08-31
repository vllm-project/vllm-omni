# PiD Adaptation Recipe (zero-intrusion templates)

Code templates for registering a new model family for PiD. All adaptation
lands in `vllm_omni/diffusion/pid/latent_forms.py` (+ `config.py` for new
latent spaces). **Pipelines are never edited for PiD** — the Runner's
`PidPassthrough` already handles force/restore/decode generically
(`vllm_omni/diffusion/pid/runner_integration.py`).

---

## 0. The `to_x0` contract (read this first)

`LatentForm.to_x0` converts the pipeline's latent-branch output (**as-is —
whatever `output_type == "latent"` returns; the pipeline is never edited**)
into PiD's `x_0`:

```python
def to_x0(
    latent: torch.Tensor,          # pipeline latent-branch output, as-is
    height: int | None,            # request target pixel height (may be None)
    width: int | None,             # request target pixel width (may be None)
    vae_scale_factor: int,         # pipeline.vae_scale_factor
    *, pipeline: object | None = None,   # Runner-passed context, see below
) -> tuple[torch.Tensor, int, int]:
    ...
    return x0, pid_h, pid_w
```

- `x0` must be 4D `[B, C, zH, zW]` with `C == lq_latent_channels` (net
  config) and `zH * latent_spatial_down_factor == pid_h` (same for W;
  `latent_spatial_down_factor` describes x0's full spatial compression —
  `lq_latent_unpatchify_factor` is a channel-level op inside
  `LQProjection2D`, not a spatial term). `pid_h/pid_w` are the **LDM output
  pixel size** (before super-resolution).
- **Pure tensor ops only** — no session/global state — so it is
  unit-testable on CPU with dummy tensors. The optional `pipeline` kwarg
  (passed by the Runner) may be used **only to read model-side constants**
  (e.g. flux2's `vae.bn` running stats); fail-loud if a required context is
  missing.
- **Fail-loud**: raise `ValueError` on non-canonical shapes (wrong dims,
  token count mismatch, `None` size where required). The Runner surfaces
  these; never silently best-effort.
- A trailing singleton dim (QwenImage-style 5D `[B, T, 1, zH, zW]`) is
  tolerated — squeeze it.

Family cheat-sheet:

| Family | Latent branch returns | `to_x0` | Notes |
|---|---|---|---|
| `FluxPipeline`, `QwenImagePipeline` | `[B, T, 4C]` 2x2-packed tokens, row-major canonical grid order | `_unpack_packed_2x2` | 4C = 64 for 16ch VAE; requires non-None height/width |
| `ZImagePipeline`, `StableDiffusion3Pipeline`, `StableDiffusionXLPipeline` | `[B, C, zH, zW]` native grid | `_identity` | loop latent IS the VAE-ready grid |
| `Flux2Pipeline`, `Flux2KleinPipeline` | `[B, 32, zH, zW]` VAE-ready grid (unpack + BN denorm + unpatchify run *before* the branch — original behavior, preserved) | `_patchify_and_normalize` (uses `pipeline` for `vae.bn` stats) | x_0 is re-patchified and BN re-normalized (128ch); `LQProjection2D` unpatchifies internally |

---

## 1. Path A — family already registered (no code)

If `type(pipeline)` or any class in its `__mro__` is a `LATENT_FORMS` key,
**nothing to do** — `lookup_latent_form` resolves it. Just smoke-test:

```bash
vllm serve <model> --omni --pid-enable --pid-gemma <gemma-2-2b-it>
```

---

## 2. Path B — new pipeline class, registered latent space (one entry)

Example: wiring `Flux2KleinPipeline` (shares the Flux2 latent space):

```python
# vllm_omni/diffusion/pid/latent_forms.py
LATENT_FORMS: dict[str, LatentForm] = {
    ...
    "Flux2KleinPipeline": LatentForm("flux2", _patchify_and_normalize),  # reuse
}
```

- Reuse the parent backbone's `to_x0` unless the sibling's latent-branch
  output form genuinely differs (then write a new pure function — Path C style).
- No `config.py` change (backbone key unchanged → same net config, same
  checkpoint).
- If the sibling's latent branch returns *processed* latents, invert the
  processing in the `to_x0` pure function (Section 4) — do not edit the
  pipeline.

---

## 3. Path C — new latent space (net config + registries + to_x0)

### 3a. New `to_x0` pure function

```python
# vllm_omni/diffusion/pid/latent_forms.py
def _your_form(
    latent: torch.Tensor,
    height: int | None,
    width: int | None,
    vae_scale_factor: int,
    *, pipeline: object | None = None,
) -> tuple[torch.Tensor, int, int]:
    """<Family>: <token/grid form> -> x_0 grid.

    Input is the loop output <describe exact shape and layout>. Output
    <[B, C, zH, zW]> grid, equivalent to <pipeline's own unpack helper>.
    """
    if latent.dim() != 3:                      # or 4, per the family
        raise ValueError(f"your-form latent must be 3D, got {tuple(latent.shape)}")
    if height is None or width is None:
        raise ValueError("your-form latent requires non-None height/width")
    b, t, c = latent.shape
    h = int(height) // vae_scale_factor        # grid math for YOUR family
    w = int(width) // vae_scale_factor
    if t != h * w:
        raise ValueError(
            f"your-form token count {t} != grid {h * w}; non-canonical token "
            "grids (edit/img2img latents) are not supported by PiD yet"
        )
    x0 = latent.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    return x0, h * vae_scale_factor, w * vae_scale_factor
```

Copy the guard style from `_unpack_packed_2x2` / `_patchify_and_normalize`
(dims check → size fallback → token-count check → reshape). The reshape must
be **mathematically identical** to (or the exact inverse of) the pipeline's
own transform — derive it by reading that helper, not by guessing.

### 3b. Register the family

```python
# vllm_omni/diffusion/pid/latent_forms.py
LATENT_FORMS["YourPipeline"] = LatentForm("your_backbone", _your_form)
```

### 3c. Net config + checkpoint registry

```python
# vllm_omni/diffusion/pid/config.py
YOUR_BACKBONE_PID_NET_CONFIG = _make_net_config(
    lq_latent_channels=<VAE latent ch>,        # channels of x_0 AFTER to_x0
    latent_spatial_down_factor=<VAE spatial down>,
    lq_latent_unpatchify_factor=<2 if x_0 is 2x2-patchified else 1>,
)

def get_pid_net_config(backbone: str) -> dict:
    mapping = {
        ...,
        "your_backbone": YOUR_BACKBONE_PID_NET_CONFIG,
    }
    ...

PID_CHECKPOINT_REGISTRY["your_backbone"] = (
    "PiD_..._official_yourbackbone_distill_4step",   # from docs/checkpoints.md
    f"{_PID_CKPT_ROOT}/PiD_..._official_yourbackbone_distill_4step/model_ema_bf16.pth",
    4,                                                # pid_scale
)
```

Also export `YOUR_BACKBONE_PID_NET_CONFIG` from `pid/__init__.py`, and add
the backbone name to `PidNetConfig.backbone`'s `Literal` if you want typed
validation.

> **VAE characteristics** come from the target LDM: `vae.config` (latent
> channels) + `vae_scale_factor` + packing. `(lq_latent_channels,
> latent_spatial_down_factor, lq_latent_unpatchify_factor)` must describe
> the x_0 **at the to_x0 output**, not the raw VAE latent.
>
> **flux2 reference values**: patchified x_0 = 128ch on a 16x-compressed
> grid → `(128, 16, 2)`; the raw VAE latent is 32ch @ 8x. The checkpoint's
> `lq_proj.latent_proj.0` is `Conv2d(32, 1024, 3)` — `LQProjection2D`
> unpatchifies 128→32 via the factor. Getting channels/factor wrong yields:
> `size mismatch for lq_proj.latent_proj.0.weight` at load, or
> `expected input[...] to have 32 channels, but got 128` at runtime.

---

## 4. Inverting processed latents — never edit the pipeline

If the pipeline's latent branch returns *processed* latents (VAE-denormed /
unpacked / dtype-cast), that is the pipeline's own contract — **keep it
byte-for-byte** and invert the processing inside your `to_x0` pure function.
Editing the branch order was attempted for flux2 (moving the unpack chain
into the VAE `else` branch) and **reverted**: it changes
`output_type="latent"` return values for non-PiD users, which violates the
hard rule that PiD must not alter existing behavior.

The flux2 case study — the branch runs `unpack → BN denorm → unpatchify`
*before* the split (original behavior, untouched):

```python
# pipeline (original behavior, preserved):
latents = self._unpack_latents_with_ids(latents, latent_ids)
latents = latents * latents_bn_std + latents_bn_mean    # BN denorm
latents = self._unpatchify_latents(latents)             # 128ch -> 32ch
if output_type == "latent":
    image = latents        # VAE-ready grid — the pipeline's own contract
else:
    ...
```

The inversion lives entirely in `latent_forms.py`:

```python
# to_x0: VAE-ready 32ch grid -> BN-normalized 2x2-patchified 128ch grid
# 1) 2x2 patchify: [B, 32, zH, zW] -> [B, 128, zH/2, zW/2]   (exact inverse
#    of _unpatchify_latents; channel order c*4 + ph*2 + pw, matching
#    _patchify_latents)
# 2) BN re-normalize: (x - mean) / sqrt(var + eps) using pipeline.vae.bn
#    running stats (BatchNorm2d(128)) and vae.config.batch_norm_eps,
#    computed in fp32 then cast back to the latent dtype
def _patchify_and_normalize(latent, height, width, vae_scale_factor, *, pipeline=None):
    bn = getattr(getattr(pipeline, "vae", None), "bn", None)
    if bn is None or not hasattr(bn, "running_mean"):
        raise ValueError(
            "flux2 latent form requires the pipeline (for vae.bn running stats) "
            "to re-normalize the VAE-ready latent into PiD's BN-normalized space"
        )
    ...
```

Why this inversion is safe: the transform is **deterministic math** —
patchify is an exact
reshape/permute, and BN stats are fixed model constants loaded with the VAE.
Any drift breaks the shape validation or the round-trip unit test loudly;
there is no silent-wrong-image path. Round-trip test:
`denorm(unpatchify(to_x0(grid)))` must recover the pipeline's latent
(`torch.allclose`, atol 1e-5, fp32).

---

## 5. What the Runner already does (do NOT reimplement)

`vllm_omni/diffusion/pid/runner_integration.py` + `diffusion_model_runner.py`:

- **Mount**: `init_pid_decoder_on` resolves the backbone via
  `lookup_latent_form(model)` and eager-loads weights, resident, aligned with
  `enforce_eager`.
- **Gating** (`maybe_pid_passthrough`): global `--pid-enable` off → per-request
  request raises (weights are not lazily loaded); family unregistered →
  warning (explicit request → error); `output_type == "latent"` → skip;
  initial latent present (img2img/edit, `strength < 1`, `latents` /
  `image_latent` non-None) → whole batch falls back to VAE with warning;
  mixed enabled/disabled batch → whole batch falls back to VAE.
- **Passthrough**: `force_latent_output` (batch `output_type` → `"latent"`,
  originals saved) → pipeline forward → `restore_output_type` →
  `decode_outputs` (per request: `to_x0` → `_validate_x0` → `decode_with_pid`
  → replace `DiffusionOutput.output`).
- **Stepwise/streaming**: `stepwise_pid_active` + `decode_stepwise_output`.
- **Per-request overrides**: `sp.pid_decode` dict (`scale` / `num_steps` /
  `seed` / `degrade_sigma` / `enabled`), applied via frozen `PidDecodeConfig`
  replacement. Caption comes from the request prompt
  (`_prompt_text`: str or `{"prompt": ...}`).

---

## 6. Runtime verification checklist

- [ ] `lookup_latent_form(<pipeline instance>)` returns the new form (and
      subclasses via MRO).
- [ ] `get_pid_net_config(<backbone>)` returns the expected
      `lq_latent_channels` / `latent_spatial_down_factor` /
      `lq_latent_unpatchify_factor`.
- [ ] `PID_CHECKPOINT_REGISTRY[<backbone>]` resolves (auto-download works, or
      `--pid-checkpoint <local .pth>`).
- [ ] `to_x0` unit tests: token-count ↔ grid math, `pid_h/w` size, error
      branches for non-canonical latents (CPU, no weights).
- [ ] Gating unit tests pass (`test_pid_runner_integration.py` decision table).
- [ ] With `--pid-enable`, checkpoint loads with no fatal
      `missing/unexpected` keys beyond expected LQ keys.
- [ ] Output size == LDM size × `scale` (e.g. 1024 → 4096 with `scale=4`).
- [ ] `"pid_decode": {"enabled": false}` returns the VAE path (identical to a
      non-PiD server).
- [ ] Per-request `{"scale": N, "num_steps": N, "seed": N, "degrade_sigma": X}`
      applies (frozen `PidDecodeConfig` replaced, never mutated).
- [ ] Without SP, PiD runs only on global rank 0 under TP; with SP, all SP
      ranks participate.
- [ ] `--enforce-eager` parity: PiD compiles iff the main model compiles.
