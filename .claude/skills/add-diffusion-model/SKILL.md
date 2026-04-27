---
name: add-diffusion-model
description: Add a new diffusion model (text-to-image, text-to-video, image-to-video, text-to-audio, image editing) to vLLM-Omni, including Cache-DiT acceleration and parallelism support (TP, SP/USP, CFG-Parallel, HSDP). Use when integrating a new diffusion model, porting a diffusers pipeline or a custom model repo to vllm-omni, creating a new DiT transformer adapter, adding diffusion model support, or enabling multi-GPU parallelism and cache acceleration for an existing model.
---

# Adding a Diffusion Model to vLLM-Omni

## Overview

This skill guides you through adding a new diffusion model to vLLM-Omni. The model may come from HuggingFace Diffusers (structured pipeline) or from a private/custom repo. The workflow differs significantly depending on the source.

## Prerequisites

Before starting, determine:

1. **Model category**: Text-to-Image, Text-to-Video, Image-to-Video, Image Editing, Text-to-Audio, or Omni
2. **Reference source**: Diffusers pipeline, custom repo, or a combination
3. **Model HuggingFace ID** or local checkpoint path
4. **Architecture**: Scheduler, text encoder, VAE, transformer/backbone

## Step 0: Classify the Migration Path

Check the model's HF repo for `model_index.json`. This determines your path:

| Scenario | How to identify | Migration path |
|----------|----------------|----------------|
| **Already supported** | `_class_name` in `model_index.json` matches a key in `_DIFFUSION_MODELS` in `registry.py` | Skip to Step 5 (test) and Step 7 (docs) |
| **Diffusers-based** | Has standard `model_index.json` with `_diffusers_version`, subfolders for `transformer/`, `vae/`, etc. | Follow **Path A** below |
| **Custom/private repo** | No diffusers `model_index.json`, weights in non-standard format, custom model code in a separate git repo | Follow **Path B** below |
| **Hybrid** | Has some diffusers components (VAE) but custom transformer/fusion | Mix of Path A and Path B |

## Path A: Diffusers-Based Model

For models with a standard diffusers layout. See [references/transformer-adaptation.md](references/transformer-adaptation.md) for detailed code patterns.

### A1. Analyze `model_index.json`

Identify components: `transformer`, `scheduler`, `vae`, `text_encoder`, `tokenizer`.

### A2. Create model directory

```
vllm_omni/diffusion/models/your_model_name/
├── __init__.py
├── pipeline_your_model.py
└── your_model_transformer.py
```

### A3. Adapt transformer

1. Copy from diffusers source. Remove mixins (`ModelMixin`, `ConfigMixin`, `AttentionModuleMixin`).
2. Replace attention with `vllm_omni.diffusion.attention.layer.Attention` (QKV shape: `[B, seq, heads, head_dim]`).
3. Add `od_config: OmniDiffusionConfig | None = None` to `__init__`.
4. Add `load_weights()` method mapping diffusers weight names to vllm-omni names.
5. Add class attributes: `_repeated_blocks`, `_layerwise_offload_blocks_attr`.

### A4. Adapt pipeline

Inherit from `nn.Module`. The key contract:

```python
class YourPipeline(nn.Module):
    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        # Load VAE, text encoder, tokenizer via from_pretrained()
        # Instantiate transformer (weights loaded later via weights_sources)
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model, subfolder="transformer",
                prefix="transformer.", fall_back_to_pt=True)]

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        # Encode prompt → prepare latents → denoise loop → VAE decode
        return DiffusionOutput(output=output)

    def load_weights(self, weights):
        return AutoWeightsLoader(self).load_weights(weights)
```

Add post/pre-process functions in the same pipeline file. Register them in `registry.py`.

### A5. Register, test, docs → continue at Step 4 below.

---

## Path B: Custom/Private Repo Model

For models without a diffusers pipeline — weights in custom format, model code in a private repo. Real examples: DreamID-Omni, BAGEL, HunyuanImage3.

### B1. Understand the reference repo

Study the original model's code to identify:
- **Model architecture files** (transformers, fusion modules, embeddings)
- **Weight format** (safetensors, `.pth`, custom checkpoint structure)
- **Weight loading helpers** (custom init functions, checkpoint loaders)
- **Pre/post-processing** (image/audio transforms, tokenization, VAE encode/decode)
- **External dependencies** (packages not on PyPI)
- **Config format** (JSON config files, hardcoded dicts)

### B2. Decide what lives WHERE

This is the key design decision for custom models. Follow these placement rules:

| Code type | Where to place | Example |
|-----------|---------------|---------|
| **Pipeline orchestration** (init, forward, denoise loop) | `vllm_omni/diffusion/models/<name>/pipeline_<name>.py` | Always required |
| **Custom transformer/backbone** (ported and adapted to vllm-omni) | `vllm_omni/diffusion/models/<name>/<name>_transformer.py` or similar | `wan2_2.py`, `fusion.py`, `bagel_transformer.py` |
| **Custom sub-models** (VAE, fusion, autoencoder) | `vllm_omni/diffusion/models/<name>/` as separate files | `autoencoder.py`, `fusion.py` |
| **External dependency code** (original repo utilities) | **External repo**, installed via download script or pip | `dreamid_omni` package via git clone |
| **Hardcoded model configs** | Module-level dicts in pipeline file | `VIDEO_CONFIG`, `AUDIO_CONFIG` dicts |
| **Download/setup script** | `examples/offline_inference/<name>/download_<name>.py` | `download_dreamid_omni.py` |
| **Custom `model_index.json`** | Generated by download script, placed at model root | Minimal: `{"_class_name": "YourPipeline", ...}` |

### B3. Handle external dependencies

If the model's code lives in a separate git repo:

**Option 1: Import with graceful fallback** (recommended for models with external utils)

```python
try:
    from external_model.utils import init_vae, load_checkpoint
except ImportError:
    raise ImportError(
        "Failed to import from dependency 'external_model'. "
        "Please run the download script first."
    )
```

**Option 2: Port the code directly** (preferred when feasible)

Copy the essential model files into `vllm_omni/diffusion/models/<name>/` and adapt them. This avoids external dependencies. BAGEL does this — `autoencoder.py` and `bagel_transformer.py` are ported directly.

**Decision criteria**: Port if the code is self-contained and won't diverge. Use external deps if the model repo is actively maintained and the code is complex.

### B4. Handle custom weight loading

Custom models have two patterns for weight loading:

**Pattern 1: Bypass standard loader** (DreamID-Omni style)

When the original model has complex custom init functions that load weights in `__init__`:

```python
class CustomPipeline(nn.Module):
    def __init__(self, *, od_config, prefix=""):
        super().__init__()
        model = od_config.model
        # Load everything eagerly in __init__ using custom helpers
        self.vae = custom_init_vae(model, device=self.device)
        self.text_encoder = custom_init_text_encoder(model, device=self.device)
        self.transformer = CustomFusionModel(CONFIG)
        load_custom_checkpoint(self.transformer,
            checkpoint_path=os.path.join(model, "model.safetensors"))
        # NO weights_sources defined — bypasses standard loader

    def load_weights(self, weights):
        pass  # No-op — all weights loaded in __init__
```

**Pattern 2: Use standard loader with custom `load_weights`** (BAGEL style)

When weights are in safetensors format but need name remapping:

```python
class CustomPipeline(nn.Module):
    def __init__(self, *, od_config, prefix=""):
        super().__init__()
        # Instantiate model architecture without weights
        self.bagel = BagelModel(config)
        self.vae = AutoEncoder(ae_params)

        # Point loader at the safetensors in the model root
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder=None,  # weights at root, not in subfolder
                prefix="",
                fall_back_to_pt=False,
            )
        ]

    def load_weights(self, weights):
        # Custom name remapping for non-diffusers weight names
        params = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            # Remap original weight names to vllm-omni module names
            name = self._remap_weight_name(name)
            if name in params:
                default_weight_loader(params[name], tensor)
                loaded.add(name)
        return loaded
```

### B5. Create the `model_index.json`

Custom models need a `model_index.json` at the model root for vllm-omni to discover them. For custom models, this is minimal:

```json
{
    "_class_name": "YourModelPipeline",
    "custom_key": "path/to/custom_weights.safetensors"
}
```

The `_class_name` must match a key in `_DIFFUSION_MODELS` in `registry.py`. Additional keys are model-specific (accessed via `od_config.model_config`).

If the model's weights come from multiple HF repos, write a **download script** that:
1. Downloads from each repo
2. Assembles into a single directory
3. Generates `model_index.json`
4. Installs any external dependencies (git clone + `.pth` file)

Place at: `examples/offline_inference/<name>/download_<name>.py`

### B6. Handle multi-modal inputs

If the model accepts images, audio, or other multi-modal inputs, implement the protocol classes from `vllm_omni/diffusion/models/interface.py`:

```python
from vllm_omni.diffusion.models.interface import SupportImageInput, SupportAudioInput

class MyPipeline(nn.Module, SupportImageInput, SupportAudioInput):
    # Protocol markers — the engine uses these to enable proper input routing
    pass
```

Preprocessing for custom models is typically done **inside `forward()`** rather than via registered pre-process functions, since the logic is often tightly coupled to the model.

### B7. Continue at Step 4 below.

---

## Common Steps (Both Paths)

### Step 4: Register Model in registry.py

Edit `vllm_omni/diffusion/registry.py`:

```python
_DIFFUSION_MODELS = {
    "YourModelPipeline": ("your_model_name", "pipeline_your_model", "YourModelPipeline"),
}
_DIFFUSION_POST_PROCESS_FUNCS = {
    "YourModelPipeline": "get_your_model_post_process_func",  # if applicable
}
_DIFFUSION_PRE_PROCESS_FUNCS = {
    "YourModelPipeline": "get_your_model_pre_process_func",  # if applicable
}
```

The registry key is the `_class_name` from `model_index.json`. The tuple is `(folder_name, module_file, class_name)`.

Create `__init__.py` exporting the pipeline class and any factory functions.

### Step 5: Run, Test, Debug

Use the appropriate existing example script:

| Category | Script |
|----------|--------|
| Text-to-Image | `examples/offline_inference/text_to_image/text_to_image.py` |
| Text-to-Video | `examples/offline_inference/text_to_video/text_to_video.py` |
| Image-to-Video | `examples/offline_inference/image_to_video/image_to_video.py` |
| Image-to-Image | `examples/offline_inference/image_to_image/image_edit.py` |
| Text-to-Audio | `examples/offline_inference/text_to_audio/text_to_audio.py` |

For custom/Omni models that don't fit these categories, create a dedicated example script.

**Validation**: No errors, output is meaningful, quality matches reference implementation.

See [references/troubleshooting.md](references/troubleshooting.md) for common errors.

### Step 6: Add Example Scripts

For Omni or custom models, create:
- `examples/offline_inference/your_model_name/` — offline script + README
- `examples/online_serving/your_model_name/` — server script + client
- Download script if weights require assembly from multiple sources

### Step 7: Update Documentation

Required updates:
1. `docs/user_guide/diffusion/parallelism_acceleration.md` — parallelism support table
2. `docs/user_guide/diffusion/cpu_offload_diffusion.md` — if CPU offload supported (add to supported models table)
3. `docs/user_guide/diffusion/teacache.md` — if TeaCache supported
4. `docs/user_guide/diffusion/cache_dit_acceleration.md` — if Cache-DiT supported
5. `examples/offline_inference/xxx/README.md` — offline example docs
6. `examples/online_serve/xxx/README.md` — online serve docs

### Step 8: Add E2E Tests (Recommended)

Create `tests/e2e/online_serving/test_your_model_expansion.py`.

### Step 9: Add Cache-DiT Acceleration

Cache-DiT accelerates inference by caching intermediate computation results across denoising steps. After your model is working correctly on a single GPU, add cache-dit support.

See [references/cache-dit-patterns.md](references/cache-dit-patterns.md) for detailed code patterns.

#### 9a. Determine your model type

| Model Type | Description | Action |
|------------|-------------|--------|
| **Standard single-transformer** | One transformer with one `ModuleList` of blocks | No code needed — `CacheDiTBackend` auto-detects via `enable_cache_for_dit()` |
| **Multi-block-list** | One transformer with multiple block lists (e.g., `transformer_blocks` + `single_transformer_blocks`) | Write custom enabler with `BlockAdapter` |
| **Dual-transformer** | Two transformers (e.g., high-noise + low-noise) | Write custom enabler with `BlockAdapter` wrapping both |

#### 9b. Standard models — verify automatic support

For standard single-transformer models, test directly:

```python
omni = Omni(
    model="your-model-name",
    cache_backend="cache_dit",
    cache_config={
        "Fn_compute_blocks": 1,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 4,
    }
)
```

Check logs for "Cache-dit enabled successfully on xxx". If it works, skip to Step 9e.

#### 9c. Custom architectures — write a custom enabler

For multi-block-list or dual-transformer models, write a custom enabler function:

```python
from cache_dit import BlockAdapter, ForwardPattern, ParamsModifier, DBCacheConfig

def enable_cache_for_your_model(pipeline, cache_config):
    db_cache_config = DBCacheConfig(
        num_inference_steps=None,
        Fn_compute_blocks=cache_config.Fn_compute_blocks,
        Bn_compute_blocks=cache_config.Bn_compute_blocks,
        max_warmup_steps=cache_config.max_warmup_steps,
        max_cached_steps=cache_config.max_cached_steps,
        max_continuous_cached_steps=cache_config.max_continuous_cached_steps,
        residual_diff_threshold=cache_config.residual_diff_threshold,
    )

    cache_dit.enable_cache(
        BlockAdapter(
            transformer=pipeline.transformer,
            blocks=[
                pipeline.transformer.transformer_blocks,
                pipeline.transformer.single_transformer_blocks,
            ],
            forward_pattern=[ForwardPattern.Pattern_1, ForwardPattern.Pattern_1],
            params_modifiers=[ParamsModifier(...)],
        ),
        cache_config=db_cache_config,
    )

    def refresh_cache_context(pipeline, num_inference_steps, verbose=True):
        cache_dit.refresh_context(
            pipeline.transformer, num_inference_steps=num_inference_steps, verbose=verbose
        )
    return refresh_cache_context
```

#### 9d. Register the custom enabler

Add your enabler to `CUSTOM_DIT_ENABLERS` in `vllm_omni/diffusion/cache/cache_dit_backend.py`:

```python
CUSTOM_DIT_ENABLERS = {
    "Wan22Pipeline": enable_cache_for_wan22,
    "LongCatImagePipeline": enable_cache_for_longcat_image,
    "YourModelPipeline": enable_cache_for_your_model,  # Add here
}
```

#### 9e. Test Cache-DiT

```python
omni = Omni(
    model="your-model-name",
    cache_backend="cache_dit",
    cache_config={
        "Fn_compute_blocks": 1, "Bn_compute_blocks": 0,
        "max_warmup_steps": 4, "residual_diff_threshold": 0.24,
    }
)
images = omni.generate("a beautiful landscape",
    OmniDiffusionSamplingParams(num_inference_steps=50))
```

**Verify**: 1) logs show cache enabled, 2) 1.5-2x speedup, 3) output quality acceptable vs baseline.

If quality degrades, lower `residual_diff_threshold` (try 0.12-0.18) or increase `max_warmup_steps` (try 6-8).

---

### Step 10: Add Parallelism Support

After the model works on a single GPU, add multi-GPU parallelism. Add each type incrementally, testing after each addition.

See [references/parallelism-patterns.md](references/parallelism-patterns.md) for detailed code patterns and API reference.

**Recommended order**: TP → SP/USP → CFG Parallel → HSDP

#### 10a. Tensor Parallelism (TP)

Shards DiT linear layers across GPUs. Requires code changes in the transformer.

**What to change in the transformer**:
1. Replace `nn.Linear` with `ColumnParallelLinear` / `RowParallelLinear` / `QKVParallelLinear`
2. Update `load_weights()` to handle QKV fusion with `stacked_params_mapping`
3. Use `self.to_qkv.num_heads` (local heads) instead of total heads for split sizes

```python
from vllm.model_executor.layers.linear import (
    QKVParallelLinear, RowParallelLinear, ColumnParallelLinear,
)

# Attention: QKV → RowParallel output
self.to_qkv = QKVParallelLinear(dim, head_dim, num_heads, num_kv_heads)
self.to_out = RowParallelLinear(dim, dim, input_is_parallel=True)

# FFN: ColumnParallel → RowParallel
self.w1 = ColumnParallelLinear(dim, ffn_dim)
self.w2 = RowParallelLinear(ffn_dim, dim, input_is_parallel=True)
```

**Constraints**: `num_heads % tp_size == 0` and `num_kv_heads % tp_size == 0`.

**Test**: `--tensor-parallel-size 2`

#### 10b. Sequence Parallelism (SP / USP)

Splits sequence tokens across GPUs. Non-intrusive via `_sp_plan` on the transformer class — no changes to `forward()`.

**What to change in the transformer**:

Add `_sp_plan` class attribute:

```python
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput, SequenceParallelOutput,
)

class YourTransformer(nn.Module):
    _sp_plan = {
        "blocks.0": {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
        },
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }
```

If inline tensor ops (e.g., `torch.cat`) exist between shard/gather points, extract them into `nn.Module` submodules so hooks can intercept them.

For RoPE that needs splitting, add an entry for the RoPE module with `split_output=True`.

**Test**: `--ulysses-degree 2` (offline) or `--usp 2` (online serving)

#### 10c. CFG Parallel

Distributes positive/negative CFG branches across 2 GPUs. Requires the pipeline to inherit `CFGParallelMixin`.

**What to change in the pipeline**:

```python
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin

class YourPipeline(nn.Module, CFGParallelMixin):
    def diffuse(self, ...) -> torch.Tensor:
        for i, t in enumerate(timesteps):
            positive_kwargs = {...}
            negative_kwargs = {...} if do_true_cfg else None
            noise_pred = self.predict_noise_maybe_with_cfg(
                do_true_cfg=do_true_cfg, true_cfg_scale=cfg_scale,
                positive_kwargs=positive_kwargs, negative_kwargs=negative_kwargs,
            )
            latents = self.scheduler_step_maybe_with_cfg(
                noise_pred, t, latents, do_true_cfg
            )
        return latents
```

Override `predict_noise()` if your transformer call is non-standard. Override `combine_cfg_noise()` for multi-output models (e.g., video + audio).

**Constraint**: Exactly 2 GPUs. Only for models using classifier-free guidance.

**Test**: `--cfg-parallel-size 2`

#### 10d. HSDP (Hybrid Sharded Data Parallel)

Shards transformer weights via PyTorch FSDP2 to reduce per-GPU VRAM. No code changes to the forward pass — just add a class attribute.

**What to change in the transformer**:

```python
class YourTransformer(nn.Module):
    @staticmethod
    def _is_transformer_block(name: str, module) -> bool:
        return "blocks" in name and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_transformer_block]
```

**Constraint**: Cannot combine with TP. For standalone HSDP, set `hsdp_shard_size` explicitly.

**Test**: `--use-hsdp` or `DiffusionParallelConfig(use_hsdp=True)`

#### 10e. Update parallelism documentation

After adding parallelism support, update:
1. `docs/user_guide/diffusion/parallelism_acceleration.md` — add your model to the support table
2. Record which parallelism methods are supported (USP, Ring, CFG, TP, HSDP, VAE-Patch)

### Step 11: Add CPU Offload Support

Implement `SupportsModuleOffload` on your pipeline class to enable
`--enable-cpu-offload` and `--enable-layerwise-offload`. The protocol
declares which submodules the offloader should manage:

```python
from typing import ClassVar
from vllm_omni.diffusion.models.interface import SupportsModuleOffload

class YourPipeline(nn.Module, SupportsModuleOffload):
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _resident_modules: ClassVar[list[str]] = []  # optional
```

- `_dit_modules`: denoising submodules (kept on GPU during diffusion loop)
- `_encoder_modules`: encoder/vision submodules (offloaded to CPU during diffusion loop)
- `_vae_modules`: VAE(s) (handled by both sequential and layerwise backends)
- `_resident_modules`: additional modules to pin on GPU during layerwise
  offloading (e.g. embedders, connectors). Only used by the layerwise
  backend. Optional — defaults to `[]`.

All attribute names support dotted paths for nested submodules
(e.g. `"pipe.transformer"`, `"bagel.time_embedder"`).

Pipelines without `SupportsModuleOffload` fall back to scanning
well-known attribute names (`transformer`, `text_encoder`, `vae`,
etc.), which fails for non-standard names.

---

## Iterative Development Tips

1. **Start minimal**: Basic generation first, no parallelism/caching
2. **Use `--enforce-eager`**: Disable torch.compile during debugging
3. **Use small models**: Test with smaller variants first
4. **Check tensor shapes**: Most errors are reshape mismatches in attention
5. **Add features incrementally**: Single GPU → TP → SP → CFG → HSDP → Cache-DiT
6. **For custom models**: Get the model running with the original code first, then progressively replace components with vllm-omni equivalents
7. **Cache-DiT before parallelism tuning**: Cache-DiT is lossy — verify quality at baseline before combining with parallelism
8. **Combine lossless + lossy**: e.g., TP + SP + Cache-DiT for maximum throughput

## Reference Files

- [Transformer Adaptation](references/transformer-adaptation.md) — porting transformers from diffusers
- [Custom Model Patterns](references/custom-model-patterns.md) — patterns for non-diffusers models
- [Parallelism Patterns](references/parallelism-patterns.md) — TP, SP/USP, CFG parallel, HSDP implementation details
- [Cache-DiT Patterns](references/cache-dit-patterns.md) — cache-dit acceleration for standard and custom architectures
- [Troubleshooting](references/troubleshooting.md) — common errors and fixes
