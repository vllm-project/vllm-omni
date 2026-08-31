# PiD (Pixel Diffusion) Super-Resolution Decode

!!! info "Feature Status"
    Enabled per pipeline. Currently wired for **Qwen-Image**; the core
    decoder is backbone-agnostic (Flux / SD3 / SDXL / Flux2 net configs are
    already registered). PiD is an optional **post-denoise** decoder — when
    disabled, the standard VAE path is unchanged.

This document describes how PiD (Pixel Diffusion) is integrated as an
optional super-resolution decoder for LDM pipelines. It captures the
**parameter surface** added by the feature and the **adaptation recipe**
for wiring a new model to PiD (one table entry + config registration — no
pipeline edits).

---

## Table of Contents

- [References](#references)
- [Overview](#overview)
- [Parameter Surface](#parameter-surface)
- [Architecture](#architecture)
- [Adaptation Recipe](#adaptation-recipe)
- [Usage Examples](#usage-examples)
- [Related Files](#related-files)

---

## References

- **Paper**: [PiD: Pixel Diffusion for Fast Super-Resolution](https://arxiv.org/abs/2605.23902)
- **Project**: [PiD — NVIDIA Research (SIL)](https://research.nvidia.com/labs/sil/projects/pid/)
- **Official repository**: [nv-tlabs/PiD](https://github.com/nv-tlabs/PiD)
- **Weights / checkpoints (Hugging Face)**: [nvidia/PiD](https://huggingface.co/nvidia/PiD)
- **Qwen-Image backbone checkpoint** (used by this feature):
  `checkpoints/PiD_v1pt5_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth`
  — the v1pt5 **4-step distilled** checkpoint in `.pth` format.

---

## Overview

PiD replaces the VAE decode step with a distilled pixel-diffusion model that
takes the LDM `x_0` latent as a low-quality (LQ) condition and synthesises a
higher-resolution RGB image. The distilled checkpoint runs **4 SDE steps** by
default and uses a Gemma-2-2b-it text encoder for caption conditioning.

**Core behaviour:**

- **Pipeline Non-invasive**：pipelines contain no PiD code. The Runner
  (`diffusion_model_runner.py`) temporarily forces the pipeline's existing
  `output_type == "latent"` branch to hand back the **raw loop latents**,
  converts them to `x_0` via a pure function, and runs PiD super-resolution
  in place of the VAE decode.
- **Centralized adaptation table**：`LATENT_FORMS` (pipeline class name →
  `{backbone, to_x0 pure function}`) maps every pipeline family to its PiD
  backbone and latent transformation. Subclasses inherit via `__mro__`
  fallback. A new model sharing a registered latent form is **zero-change**.
- Weights are **eager-loaded** at model initialisation
  (`registry.initialize_model` → `init_pid_decoder_on`) and stay resident on
  GPU; PiD is declared as a *resident module* so the CPU offloader never
  swaps it out.
- Compilation follows the main model: `PidDecoder` inherits the pipeline's
  `enforce_eager` flag, so PiD is compiled when the main model compiles and
  stays eager when the main model is eager.
- Checkpoint resolution supports a **local `.pth` path**, an **HF reference**
  (`<repo>/<subfolder>/<file>`), or — when `--pid-checkpoint` is omitted —
  an **automatic download** of the matching official checkpoint from the
  `nvidia/PiD` repo, keyed by backbone.
- Per-request overrides (enable/disable, scale, num_steps, seed,
  `degrade_sigma`) flow through `OmniDiffusionSamplingParams.pid_decode`.
- Gating is fail-loud: shape mismatches between the latent form and the
  backbone net config raise errors (never silently produce a wrong image);
  unsupported cases (img2img/edit latents, video, chunked streaming batches
  mixing intent) fall back to the VAE with a warning.
- Under tensor parallelism without SP, PiD decode runs **only on rank 0**;
  with sequence parallelism the PiD net is sharded and every rank in the SP
  group participates.

---

## Parameter Surface

The feature adds parameters at three layers. Each lower layer is the typed
form of the layer above; CLI flags are packed into a `dict` and re-injected
as `pid_decode` so they flow through the normal stage-config plumbing.

### 1. CLI Flags (`vllm_omni/entrypoints/cli/serve.py`)

All flags live in the `omni_config_group` and are prefixed with `--pid-`.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--pid-enable` | flag (bool) | `False` | Master switch. When set, the matching `--pid-*` keys are packed into a `pid_decode` dict. |
| `--pid-checkpoint` | str | `None` | PiD decoder checkpoint: a local `.pth` path, an HF reference `<repo>/<subfolder>/<file>`, or `None` for auto-download. |
| `--pid-gemma` | str | `Efficient-Large-Model/gemma-2-2b-it` | Gemma text encoder used by PiD (HF id or a local directory). |

> `scale` / `num_steps` / `seed` / `degrade_sigma` are **not** exposed as CLI
> flags. They default in `PidDecodeConfig` and can be overridden **per
> request** through `pid_decode` (see §4).

**Packing logic** (`AsyncOmniEngine.__init__`): when `--pid-enable` is set,
the engine pops the two keys above from `kwargs` and re-injects a single
`pid_decode` dict using the **config field names**:

| CLI key | Config key |
|---|---|
| `pid_checkpoint` | `checkpoint_path` |
| `pid_gemma` | `gemma_model` |

The dict is then forwarded into `OmniDiffusionConfig.pid_decode` via the
stage-config plumbing (`async_omni_engine.py::_init_diffusion_engine` adds
`"pid_decode": kwargs.get("pid_decode")` to the engine kwargs).

### 2. Stage Config (`vllm_omni/diffusion/data.py`)

`OmniDiffusionConfig` gains one field:

```python
pid_decode: dict[str, Any] | None = None
```

`None` keeps the standard VAE path. A `dict` is accepted for ergonomic CLI
plumbing; it is normalised to `PidDecodeConfig` by
`runner_integration._resolve_pid_config` at mount/gate time.

### 3. Typed Config (`vllm_omni/diffusion/pid/decoder.py`)

```python
@dataclass(frozen=True)
class PidDecodeConfig:
    enabled: bool = False
    checkpoint_path: str = ""      # empty -> auto-download from nvidia/PiD by backbone
    gemma_model: str = ""          # Gemma text encoder (HF id or local dir)
    scale: int = 4
    num_steps: int = 4
    seed: int = 0
    degrade_sigma: float = 0.0     # noise injected into the LQ latent
    precision: str = "bfloat16"    # "bfloat16" | "float16" | "float32"
```

### 4. Per-Request Override (`vllm_omni/inputs/data.py`)

```python
@dataclass
class OmniDiffusionSamplingParams:
    ...
    pid_decode: dict[str, Any] | None = None
```

Accepted keys (all optional): `enabled`, `scale`, `num_steps`, `seed`,
`degrade_sigma`. The override is applied with `dataclasses.replace` on the
frozen `PidDecodeConfig`, so the pipeline-level config is never mutated.

| Override `enabled` | Startup `--pid-enable` | Behaviour |
|---|---|---|
| `False` | any | Skip PiD, use VAE (a mixed batch falls back entirely — the batch `output_type` is uniform). |
| `True` | `False` | **Error** — PiD weights are not lazily loaded; restart with `--pid-enable`. |
| `True` | `True` | Run PiD with per-request overrides. |
| `None` / absent | `True` | Run PiD with pipeline-level config. |

### 5. HTTP API (`vllm_omni/entrypoints/openai/protocol/images.py`)

`ImageGenerationRequest` gains one optional field:

```python
pid_decode: dict[str, Any] | None = Field(
    default=None,
    description="Per-request PiD decode configuration. "
                "Keys: enabled, scale, num_steps, seed, degrade_sigma.",
)
```

The field is wired through two paths in `api_server.py::generate_images`:

- `extra_body["pid_decode"]` — forwarded to the chat handler for the
  chat-completion-style image path.
- `_update_if_not_none(gen_params, "pid_decode", ...)` — forwarded to the
  standalone image-generation path.

`serving_chat.py::OmniOpenAIServingChat.generate_diffusion_images` reads
`extra_body.get("pid_decode")` and passes it into
`OmniDiffusionSamplingParams`.

---

## Architecture

### Data Flow (Runner-layer passthrough)

```
scheduler → runner._execute_request_list
  │
  ├─ pid_passthrough = maybe_pid_passthrough(pipeline, reqs, od_config)
  │    ├─ disabled / family unregistered / user wants latents / img2img → None
  │    └─ hit → PidPassthrough(pipeline, form, decoder, config)
  │
  ├─ pid_passthrough.force_latent_output(reqs)        # output_type := "latent"
  ├─ raw_outputs = pipeline.forward(batch)            # ← pipeline unchanged
  ├─ (finally) pid_passthrough.restore_output_type(reqs)
  │
  └─ outputs = pid_passthrough.decode_outputs(outputs, reqs)
       ├─ per output: form.to_x0(latent, h, w, vae_scale_factor) → (B, C, zH, zW)
       ├─ shape hard-validation against the backbone net config (fail-loud)
       ├─ caption = req.prompt; override = req.sampling_params.pid_decode
       ├─ decode_with_pid(...)  # override merge + rank/SP gating
       └─ DiffusionOutput.output ← (B, 3, pid_h·scale, pid_w·scale) in [-1, 1]
```

The stepwise (streaming) path mirrors this:
`stepwise_pid_active(pipeline, req)` gates, the Runner calls
`pipeline.post_decode(req, output_type="latent")`, then
`decode_stepwise_output(pipeline, req, result)` super-resolves.

### Mount Point (`registry.initialize_model`)

```python
model = model_class(od_config=od_config)
...
init_pid_decoder_on(model, od_config)   # eager load + resident declaration
...
_apply_sequence_parallel_if_enabled(model, od_config)
```

The mount runs **before** sequence parallelism so the PiD net
(`model._pid_decoder._model.net`) receives the same SP hooks as the main
DiT.

### Key Components

1. **`LatentForm` table** (`pid/latent_forms.py`): the single source of
   truth for "what latent form does this pipeline family emit, and how to
   convert it to PiD `x_0`". Pure functions (the optional `pipeline` kwarg is
   used only to read model-side constants like flux2's `vae.bn` stats),
   directly unit-testable. `lookup_latent_form(pipeline)` resolves by class
   name with `__mro__` fallback (subclasses inherit the family entry).

   | Pipeline class | Backbone | `to_x0` | Input form (latent branch returns) |
   |---|---|---|---|
   | `FluxPipeline` | `flux` | `_unpack_packed_2x2` | `(B, T, 64)` 2x2-packed tokens, 16ch/8x |
   | `QwenImagePipeline` | `qwenimage` | `_unpack_packed_2x2` | `(B, T, 64)` 2x2-packed tokens, 16ch/8x |
   | `ZImagePipeline` | `flux` | `_identity` | native `(B, 16, zH, zW)`/8x |
   | `StableDiffusion3Pipeline` | `sd3` | `_identity` | native `(B, 16, zH, zW)`/8x |
   | `StableDiffusionXLPipeline` | `sdxl` | `_identity` | native `(B, 4, zH, zW)`/8x |
   | `Flux2Pipeline`, `Flux2KleinPipeline` | `flux2` | `_patchify_and_normalize` | `(B, 32, zH, zW)` VAE-ready grid (pipeline's original latent-branch output, untouched) |

   Flux2's latent branch has always emitted the VAE-ready 32ch/8x grid
   (unpack + BN denorm + unpatchify run before the branch) — that behavior is
   preserved byte-for-byte; `_patchify_and_normalize` inverts it at the
   Runner layer (2x2 patchify back to 128ch + BN re-normalization with
   `vae.bn` running stats). The PiD flux2 checkpoint's LQ projection then
   performs the 128→32 channel unpatchify internally
   (`lq_latent_unpatchify_factor=2`, see §Adaptation Recipe).

2. **`PidPassthrough`** (`pid/runner_integration.py`): batch-level
   orchestrator owned by the Runner for one forward pass:
   `force_latent_output(reqs)` → `restore_output_type(reqs)` →
   `decode_outputs(outputs, reqs)`. Supports both single-batched outputs and
   per-request slices.

3. **`init_pid_decoder_on`** (`pid/runner_integration.py`): mount entry
   called from `registry.initialize_model`. Resolves the backbone from the
   `LatentForm` table, constructs `PidDecoder`, eagerly loads weights,
   registers it as an `nn.Module` submodule and appends `"_pid_decoder"` to
   `model._resident_modules`. No-op when PiD is disabled or the family is
   unregistered (warning).

4. **`decode_with_pid`** (`pid/runner_integration.py`): shared decode core —
   merges the per-request override, gates ranks (rank-0 only without SP;
   all ranks participate under SP), falls back to an empty caption with a
   warning, and calls `PidDecoder.decode`.

5. **`PidDecoder`** (`pid/decoder.py`): `nn.Module` wrapping
   `PidInferenceModel`. Loads weights eagerly in `load_weights()` and
   exposes a `decode(...)` entry point. Its `enforce_eager` flag mirrors
   the pipeline's so PiD compiles iff the main model compiles.

6. **`PidInferenceModel`** (`pid/pid_model.py`): holds `PidNet` +
   `GemmaTextEncoder`, runs the 4-step distilled SDE sampler. Precision is
   resolved once at construction (autocast dtype vs. pure fp32).

7. **`PidDecodeConfig`** (`pid/decoder.py`): frozen dataclass normalising
   both CLI dict and programmatic construction.

8. **Config registry** (`pid/config.py`): per-backbone `*_PID_NET_CONFIG`
   (Qwen-Image / Flux / SD3 / SDXL / Flux2) + shared `PID_SAMPLING_CONFIG`
   + `PID_CHECKPOINT_REGISTRY` mapping each backbone to its official
   `nvidia/PiD` checkpoint.

9. **Checkpoint loader** (`pid/checkpoint.py`): `resolve_pid_checkpoint_path`
   handles local / HF-ref / auto-download; `load_pid_checkpoint` strips the
   `net.` prefix, drops `net_ema.*` and training-only aux heads
   (`lq_proj.lq_aux_rgb_head`), and tolerates zero-init LQ-projection keys.

### Gating Rules (`maybe_pid_passthrough`)

| Condition | Result |
|---|---|
| PiD disabled globally, no per-request `enabled=True` | `None` (zero overhead) |
| PiD disabled globally, request sets `enabled=True` | `RuntimeError` (weights are not lazily loaded) |
| Family unregistered in `LATENT_FORMS` | warning + `None`; `RuntimeError` if explicitly requested |
| Request `output_type == "latent"` | `None` (user semantics win) |
| Request carries initial latents (img2img/edit: `strength < 1`, `latents`/`image_latent` set) | warning + `None` (non-canonical token grids unsupported) |
| Batch mixes `enabled=False` requests | whole batch falls back to VAE (batch `output_type` is uniform) |
| All checks pass | `PidPassthrough` |

### Safety of the temporary `output_type` overwrite

- The overwrite happens **after** batch formation (the scheduler's
  `output_type`-based request grouping is unaffected).
- `finally` restores the original values; request objects are not polluted.
- The module-level `od_config.output_type == "latent"` fast paths (e.g.
  `pipeline_flux.py` skips VAE loading) read the **global** config, not the
  per-request overwrite — the VAE stays loaded as the fallback path.

---

## Adaptation Recipe

Adapting a new model to PiD is **config-only** — no pipeline edits. The
`.claude/skills/add-pid-model` skill automates this flow.

### Step 0: Check support in the remote PiD repo

Does an official PiD checkpoint exist whose LQ conditioning latent matches
the model's latent space (channels × compression)? See
`PID_CHECKPOINT_REGISTRY` for the known set. No official checkpoint → stop
(Plan B does not train checkpoints).

### Step 1: Same latent space as a registered backbone → one table entry

If the model's loop latents have the same form as a registered family, add
one entry to `LATENT_FORMS` in `vllm_omni/diffusion/pid/latent_forms.py`:

```python
"YourPipeline": LatentForm("flux", _unpack_packed_2x2),   # pick backbone + to_x0
```

Subclasses of a registered class need **nothing** — `lookup_latent_form`
resolves via `__mro__`.

### Step 2: New latent form → one pure function

If the loop latents need a transformation, write a pure function with the
module contract:

```python
def _your_transform(latent, height, width, vae_scale_factor):
    """(loop latents) -> (x0 [B, C, zH, zW], pid_h, pid_w)"""
```

- Pure tensor ops only (no model/session/global state) — directly testable.
- Raise `ValueError` on non-canonical shapes; the Runner side is fail-loud.
- Add a round-trip test in `tests/diffusion/pid/test_pid_latent_forms.py`
  against the pipeline's private implementation.

### Step 3: New latent space → new net config + checkpoint entry

If no registered backbone shares the latent space, register it in
`vllm_omni/diffusion/pid/config.py`:

```python
YOUR_BACKBONE_PID_NET_CONFIG = _make_net_config(
    lq_latent_channels=<C of x_0>,
    latent_spatial_down_factor=<zH * down == pid_h>,
    lq_latent_unpatchify_factor=<1 unless x_0 stays patchified, cf. flux2>,
)
```

then register it in `get_pid_net_config` and — if an official checkpoint
exists — in `PID_CHECKPOINT_REGISTRY` for auto-download.

**Checkpoint weight compatibility** (the flux2 lesson): the net config must
describe the latent **exactly as the checkpoint's LQ projection expects
it**. If the checkpoint's `lq_proj.latent_proj.0` is `Conv2d(32, …)` but the
`x_0` carries 128 patchified channels, set
`lq_latent_channels=128` **and** `lq_latent_unpatchify_factor=2` —
`LQProjection2D` then performs the channel unpatchify internally
(`128 / 2² == 32`). A mismatch surfaces as
`size mismatch for lq_proj.latent_proj.0.weight` at load, or as a conv2d
channel error at inference.

### Step 4: Verify

1. `pytest -s -v tests/diffusion/pid/ -m "core_model and cpu"` — table +
   gating + shape validation.
2. Start with `--pid-enable` and send a request; the output size must be
   `LDM size × scale` (see `PiD_TEST_GUIDE.md` L2/L3).

---

## Usage Examples

### Enable PiD at startup (CLI)

```bash
vllm serve Qwen/Qwen-Image --omni \
  --port 8091 \
  --pid-enable \
  --pid-checkpoint /path/to/PiD_v1pt5_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth \
  --pid-gemma /path/to/gemma-2-2b-it
```

Every request on this server will use PiD decode with these defaults.
`--pid-checkpoint` and `--pid-gemma` may be omitted: the former then
auto-downloads the official checkpoint from `nvidia/PiD` (by backbone), and
the latter falls back to the `Efficient-Large-Model/gemma-2-2b-it` HF id.
There is **no** `--pid-scale` / `--pid-num-steps` / `--pid-seed` CLI flag;
tune those per request via `pid_decode`.

### Per-request override via HTTP API

Once the server is started with `--pid-enable`, individual requests can
tune or disable PiD. Send `pid_decode` inside the request body:

```json
{
  "model": "Qwen/Qwen-Image",
  "prompt": "a cat sleeping on a windowsill",
  "size": "1024x1024",
  "pid_decode": {
    "enabled": true,
    "scale": 4,
    "num_steps": 4,
    "seed": 42
  }
}
```

To disable PiD for one request on a PiD-enabled server:

```json
{
  "model": "Qwen/Qwen-Image",
  "prompt": "...",
  "pid_decode": {"enabled": false}
}
```

### Programmatic construction

When building `OmniDiffusionConfig` directly (e.g. in tests or benchmarks):

```python
od_config = OmniDiffusionConfig(
    ...,
    pid_decode=PidDecodeConfig(
        enabled=True,
        checkpoint_path="/path/to/model_ema_bf16.pth",   # or "" for auto-download
        gemma_model="/path/to/gemma-2-2b-it",
        scale=4,
        num_steps=4,
        seed=0,
        precision="bfloat16",
    ),
)
```

A `dict` with the same keys is also accepted (`pid_decode={...}`) for CLI
ergonomics.

---

## Related Files

- `vllm_omni/diffusion/pid/__init__.py` — public API exports
- `vllm_omni/diffusion/pid/latent_forms.py`
  `LatentForm`, `LATENT_FORMS`, `lookup_latent_form`
- `vllm_omni/diffusion/pid/runner_integration.py`
  `init_pid_decoder_on`, `maybe_pid_passthrough`, `PidPassthrough`,
  `decode_with_pid`, `stepwise_pid_active`, `decode_stepwise_output`
- `vllm_omni/diffusion/pid/config.py` — per-backbone `*_PID_NET_CONFIG`,
  `PID_SAMPLING_CONFIG`, `PID_CHECKPOINT_REGISTRY`, typed wrappers
- `vllm_omni/diffusion/pid/decoder.py` — `PidDecodeConfig`, `PidDecoder`
- `vllm_omni/diffusion/pid/pid_model.py` — `PidInferenceModel`
  (PidNet + Gemma, precision handling, sampler)
- `vllm_omni/diffusion/pid/pid_net.py` / `pixeldit.py` / `lq_projection_2d.py`
  — PidNet backbone / PixDiT-T2I modules / LQ projection
  (channel unpatchify via `lq_latent_unpatchify_factor`)
- `vllm_omni/diffusion/pid/text_encoder.py` — `GemmaTextEncoder`
  (chi-prompt prefix, 300-token layout, `from_pretrained_with_prefetch`)
- `vllm_omni/diffusion/pid/checkpoint.py` — checkpoint resolution
  (local / HF-ref / auto-download) and loader (`net.` prefix stripping)
- `vllm_omni/diffusion/pid/context_parallel.py` — context-parallel helpers
  adapted to standard `torch.distributed`
- `vllm_omni/diffusion/registry.py` — `initialize_model()` PiD mount (before SP)
- `vllm_omni/diffusion/worker/diffusion_model_runner.py` — Runner three-stage
  wiring (`_execute_request_list`) + stepwise path
- `vllm_omni/diffusion/data.py` — `OmniDiffusionConfig.pid_decode` field
- `vllm_omni/inputs/data.py` — `OmniDiffusionSamplingParams.pid_decode`
- `vllm_omni/engine/async_omni_engine.py` — CLI flag packing
  (`--pid-*` → `pid_decode` dict)
- `vllm_omni/entrypoints/cli/serve.py` — `--pid-*` CLI flags
- `vllm_omni/entrypoints/openai/protocol/images.py` —
  `ImageGenerationRequest.pid_decode`
- `vllm_omni/diffusion/models/flux2/pipeline_flux2.py`,
  `vllm_omni/diffusion/models/flux2_klein/pipeline_flux2_klein.py` —
  latent-branch semantic alignment (raw loop latents; see plan B §6)
- `vllm_omni/diffusion/pid/PiD_TEST_GUIDE.md` — test levels and recipes
- `tests/diffusion/pid/test_pid_latent_forms.py`,
  `tests/diffusion/pid/test_pid_runner_integration.py`,
  `tests/diffusion/pid/test_pid_pipeline.py`
