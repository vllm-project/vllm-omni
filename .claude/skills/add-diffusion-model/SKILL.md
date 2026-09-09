---
name: add-diffusion-model
description: Add a new diffusion model (text-to-image, text-to-video, image-to-video, text-to-audio, image editing) to vLLM-Omni, including native non-Diffusers ports, reference-parity validation, Cache-DiT, offload, and parallelism support (TP, SP/USP, CFG-Parallel, HSDP). Use when integrating or reviewing a new diffusion model, porting a Diffusers pipeline or custom model repository, creating a DiT adapter, reusing shared examples, or qualifying multi-GPU and memory optimizations.
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
| **Already supported** | `_class_name` in `model_index.json` matches a key in `_DIFFUSION_MODELS` in `registry.py` | Skip implementation, then validate model-specific examples, tests, and docs as needed |
| **Diffusers-based** | Has standard `model_index.json` with `_diffusers_version`, subfolders for `transformer/`, `vae/`, etc. | Follow **Path A** below |
| **Native non-Diffusers model** | No Diffusers index, non-standard checkpoint hierarchy, or custom architecture in a separate repo | Follow **Path B** below; port the runtime natively unless an external adapter was explicitly requested |
| **Hybrid** | Has some diffusers components (VAE) but custom transformer/fusion | Mix of Path A and Path B |

Before coding, write a short integration contract covering runtime ownership,
checkpoint discovery, reference revision, I/O geometry, attention semantics,
CFG behavior, auxiliary components, target hardware, and the default deployment.
For Path B, hybrid models, or any optimization work, read
[references/native-model-integration-checklist.md](references/native-model-integration-checklist.md)
and use its phase gates.

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
5. Add class attributes for acceleration features such as `_repeated_blocks` and `_layerwise_offload_blocks_attrs` (see [references/transformer-adaptation.md](references/transformer-adaptation.md) for examples).

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

### A4.1 Add progress bar support (recommended)

For pipelines with a standard denoising loop, prefer the existing progress bar pattern instead of hand-rolled logging.

```python
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin

class YourPipeline(nn.Module, ProgressBarMixin):
    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        # ... prepare timesteps / latents ...
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                # predict noise / scheduler step
                latents = ...
                progress_bar.update()

        return DiffusionOutput(output=output)
```

For custom loop structures, follow `vllm_omni/diffusion/models/progress_bar.py` and existing pipelines using `ProgressBarMixin`.

### A5. Register, test, docs → continue at Step 4 below.

---

## Path B: Native / Non-Diffusers Model

For models without a Diffusers pipeline—weights in custom formats and model
code in another public or private repository. Treat that repository as a pinned
correctness oracle. A request for native support means the vLLM-Omni runtime
must not import the reference implementation; an external adapter is appropriate
only when the requested scope explicitly permits that dependency. See
[references/custom-model-patterns.md](references/custom-model-patterns.md) for
concrete integration patterns.

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
| **Reference-only code** | Keep outside the runtime; use a pinned revision for golden outputs and architecture analysis | Reference inference script |
| **Explicit external adapter dependency** | External package, only when the requested scope and maintainer direction allow it | Compatibility adapter, not native support |
| **Hardcoded model configs** | Module-level dicts in pipeline file | `VIDEO_CONFIG`, `AUDIO_CONFIG` dicts |
| **Download/setup script** | `examples/offline_inference/<name>/download_<name>.py` | `download_<name>.py` |
| **Custom `model_index.json`** | Generated by download script, placed at model root | Minimal: `{"_class_name": "YourPipeline", ...}` |

### B3. Handle external dependencies

If the model's code lives in a separate git repo, first decide whether the
requested deliverable is native support or an external adapter. Do not silently
choose the adapter path.

**Option 1: Port the code directly** (default for native support)

Copy the essential model files into `vllm_omni/diffusion/models/<name>/` and
adapt them to shared vLLM-Omni contracts. Keep checkpoint loading strict and
use the pinned reference only to generate parity evidence.

**Option 2: Import with graceful fallback** (adapter scope only)

```python
try:
    from external_model.utils import init_vae, load_checkpoint
except ImportError:
    raise ImportError(
        "Failed to import from dependency 'external_model'. "
        "Please run the download script first."
    )
```

Use an external runtime dependency only when the user explicitly requested an
adapter or a maintainer approved the exception. Document the dependency,
pinned revision, installation path, and unsupported native features.

### B4. Handle custom weight loading

Custom models have two common patterns for weight loading:

**Pattern 1: Bypass standard loader** (eager custom init)

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
        load_custom_checkpoint(
            self.transformer,
            checkpoint_path=os.path.join(model, "model.safetensors"),
        )
        # NO weights_sources defined — bypasses standard loader

    def load_weights(self, weights):
        pass  # No-op — all weights loaded in __init__
```

**Pattern 2: Use standard loader with custom `load_weights`** (BAGEL style)

When weights are in safetensors format but need name remapping:

```python
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

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
                param = params[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, tensor)
                loaded.add(name)
        return loaded
```

### B5. Create the `model_index.json`

Prefer a `model_index.json` at the model root when vLLM-Omni owns an assembled
checkpoint directory. For custom models, this is minimal:

```json
{
    "_class_name": "YourModelPipeline",
    "custom_key": "path/to/custom_weights.safetensors"
}
```

The `_class_name` must match a key in `_DIFFUSION_MODELS` in `registry.py`.
Additional keys are model-specific (accessed via `od_config.model_config`).

If the released repository is immutable and has neither a root `config.json`
nor a Diffusers index, add it to a generic native-checkpoint signature resolver:
match the exact Hub ID or a distinctive set of local files and return the
pipeline class. Do not use model-name substrings or add parallel one-off
predicates in CLI, config, and serving consumers.

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

Reuse these shared scripts even for custom models when their request and output
contracts fit. Create a dedicated model script only when the shared category
cannot represent the protocol, and document that gap in the PR.

**Validation**: No errors, output is meaningful, quality matches reference implementation.

See [references/troubleshooting.md](references/troubleshooting.md) for common errors.

### Step 6: Add Example Scripts

Only when the shared category scripts cannot represent the model, create:
- `examples/offline_inference/your_model_name/` — offline script + README
- `examples/online_serving/your_model_name/` — server script + client
- Download script if weights require assembly from multiple sources

### Step 7: Update Documentation

Follow the [`add-recipe` skill](../add-recipe/SKILL.md) to add or update the
model-family recipe and its `recipes/README.md` row with verified specifications,
hardware, commands, feature links, and qualification evidence.

Required updates:
1. `docs/user_guide/diffusion/parallelism/overview.md` — parallelism support overview/table
2. `docs/user_guide/diffusion/cpu_offload.md` — if CPU offload supported (add to supported models table)
3. `docs/user_guide/diffusion/cache_acceleration/teacache.md` — if TeaCache supported
4. `docs/user_guide/diffusion/cache_acceleration/cache_dit.md` — if Cache-DiT supported
5. Offline example docs under `examples/offline_inference/<name>/` (`README.md` or category-specific `.md`)
6. `examples/online_serving/<name>/README.md` — online serving docs

### Step 8: Add E2E Tests

**Follow the [vllm-omni-test skill](../vllm-omni-test/SKILL.md)** for markers, file naming, Buildkite wiring, and run commands. Also read [l4_functionality_tests.inc.md](https://github.com/vllm-project/vllm-omni/blob/main/docs/contributing/ci/test_examples/l4_functionality_tests.inc.md), [test_system_overview.md](https://github.com/vllm-project/vllm-omni/blob/main/docs/contributing/ci/test_system_overview.md), and [test_writing_guide.md](https://github.com/vllm-project/vllm-omni/blob/main/docs/contributing/ci/test_writing_guide.md).

Classify the model's **CI priority** first:

| Priority | Required test levels | Files & markers |
|----------|---------------------|-----------------|
| **High** (listed in [#1832](https://github.com/vllm-project/vllm-omni/issues/1832) or on the diffusion hot path) | **L1** · **L2** online · **L3** online + offline · **L4** feature + performance | See table below |
| **Medium** (*normal priority* in L4 docs) | **L3** online + offline · **L4** feature only | Fewer L4 parametrized rows |
| **Low** | **L4** feature only | One or two `*_expansion.py` cases |

**Per-level deliverables (diffusion / `pytest.mark.diffusion`):**

| Level | Location | Marker | CI pipeline | Notes |
|-------|----------|--------|-------------|-------|
| **L1** | `tests/diffusion/models/{slug}/`, `tests/diffusion/cache/`, transformer unit tests | `core_model` + `cpu` | `test-ready.yml` | Weight remap, `_sp_plan`, cache enabler registration, shape contracts |
| **L2** | `tests/e2e/online_serving/test_{slug}.py` (and offline if the category is offline-first) | **`core_model` + `advanced_model`** (both on baseline smoke) + `diffusion` + `@hardware_test` / `hardware_marks` | `test-ready.yml` | Default deploy smoke — minimal `num_inference_steps`, single prompt |
| **L3** | `tests/e2e/online_serving/test_{slug}.py` **and** `tests/e2e/offline_inference/test_{slug}.py` when offline matters | Baseline smoke: **`core_model` + `advanced_model`**; heavier cases: `advanced_model` only (+ `diffusion`) | `test-merge.yml` **or** merged into nightly diffusion function job | Real weights, streaming/API paths, LoRA/offload smoke |
| **L4** | `tests/e2e/online_serving/test_{slug}_expansion.py` (+ offline expansion if needed) | `full_model` + `diffusion` | `test-nightly.yml` (X2I / X2V / X2A function groups) | Feature combos per [#1832](https://github.com/vllm-project/vllm-omni/issues/1832); perf → `tests/dfx/perf/tests/test_{model}_vllm_omni.json` with per-case **`mark`** (`hardware_marks` + `full_model` + `diffusion`) |

**L2 & L3 online — same file, dual marks on the baseline smoke:** The **first / simplest** case in `test_{slug}.py` (default deploy, minimal steps, single prompt) should carry **both** `@pytest.mark.core_model` and `@pytest.mark.advanced_model` on the **same** function so L2 (`test-ready.yml`) and L3 (`test-merge.yml`) share one smoke test. Heavier deploy variants or API paths in the same file use **`advanced_model` only**. When L3 moves to nightly, migrate those heavier cases into `test_{slug}_expansion.py` with `full_model` and remove the dedicated `test-merge.yml` job (see `test_longcat_image_expansion.py`, `test_qwen_image_expansion.py`).

**L4 design (high priority):** Combine multiple supported features (Cache-DiT, TP, USP, CFG, HSDP, CPU offload, quantization) into **few parametrized** `OmniServerParams` rows so each feature appears in at least one case without exploding GPU jobs. Shard single-GPU vs multi-GPU cases across the nightly X2I/X2V function steps (`cards_1` vs `not cards_1`).

**L4 design (medium / low):** One or two parametrized rows covering the best quality/perf trade-off; skip perf JSON unless the model is high priority.

**Reference implementations:** `tests/e2e/online_serving/test_qwen_image_edit_expansion.py`, `tests/e2e/online_serving/test_longcat_image_expansion.py`, `tests/e2e/online_serving/test_hunyuan_video_15_expansion.py`.

**Keep model-specific code inside test modules — not `tests/helpers/{slug}.py`:** deploy constants, prompts, sampling dicts, and inline `request_config` / `form_data` belong in each `test_{slug}.py` and `test_{slug}_expansion.py`. Do not add per-model files under `tests/helpers/`; reuse only repo-wide harness (`mark`, `media`, `runtime`, `stage_config`, `assertions`). **L2+ online/offline e2e:** reuse or add `send_*_request` in `tests/helpers/runtime.py` — tests call the handler, not raw `omni.generate` / HTTP. See [vllm-omni-test skill](../vllm-omni-test/SKILL.md) § **Runtime send helpers**.

Keep the model suite proportional using the six distinct failure owners in the
native integration checklist. Combine supported features into a few
parametrized E2E rows instead of creating one test per optimization.

### Step 9: Add Cache-DiT Acceleration

Add caching only after uncached single-device correctness. Read
[references/cache-dit-patterns.md](references/cache-dit-patterns.md), use the
automatic single-block-list path when possible, and add a registered
`BlockAdapter` only for genuinely custom block topology.

Verify a real cache hit and compare quality with the uncached baseline. Make a
speed claim only at a realistic step count where warmup permits hits; an
all-warmup smoke proves integration, not acceleration.

---

### Step 10: Add Parallelism Support

After the model works on a single GPU, add multi-GPU parallelism. Add each type incrementally, testing after each addition.

See [references/parallelism-patterns.md](references/parallelism-patterns.md) for detailed code patterns and API reference.

**Recommended order**: TP → SP/USP → CFG Parallel → HSDP

#### 10a. Tensor Parallelism (TP)

Replace compatible projections with vLLM parallel linears, preserve checkpoint
fusion/loading, and use local head counts. Require query/KV head divisibility
and compare a multi-rank forward with the one-rank oracle.

#### 10b. Sequence Parallelism (SP / USP)

Prefer the declarative `_sp_plan`. For packed variable-length attention or
learned-sink LSE correction, keep model math explicit and reuse shared exchange
utilities. Validate uneven sequence splits, RoPE coordinates, and outputs
against the one-rank oracle.

#### 10c. CFG Parallel

Confirm the model uses CFG. Then reuse `CFGParallelMixin`, overriding prediction
or recombination only for non-standard or multi-output pipelines. Distinguish a
packed positive/negative implementation from two independent branches and
validate the two-rank result against the packed one-rank oracle.

#### 10d. HSDP (Hybrid Sharded Data Parallel)

Declare layer shard conditions and ignored rank-local modules; preserve mixed
checkpoint dtypes. HSDP cannot combine with TP. Measure parameter loading,
FSDP materialization, warm HBM, and host PSS rather than assuming sharding saves
peak memory. Keep the resident layout as default if HSDP is worse.

#### 10e. Update parallelism documentation

After adding parallelism support, update:
1. `docs/user_guide/diffusion/parallelism/overview.md` — add your model to the support overview/table
2. Record which parallelism methods are supported (USP, Ring, CFG, TP, HSDP, VAE-Patch)

### Step 11: Add CPU Offload Support

Implement `SupportsComponentDiscovery` on your pipeline class to enable
`--enable-cpu-offload` and `--enable-layerwise-offload`. The protocol
declares which submodules the offloader should manage:

```python
from typing import ClassVar
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery

class YourPipeline(nn.Module, SupportsComponentDiscovery):
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

Pipelines without `SupportsComponentDiscovery` fall back to scanning
well-known attribute names (`transformer`, `text_encoder`, `vae`,
etc.), which fails for non-standard names.

Keep model-specific checkpoint paths, nested block aliases, layout transforms,
and component lifecycles in the model package. Change a shared offloader only
for a general contract, demonstrate another consumer or a framework-level bug,
and add one focused shared regression. Avoid `if ModelName` branches in shared
backends.

### Step 12: Performance Profiling

After verifying correctness and implementing parallelism/caching, profile the model's performance to identify bottlenecks and ensure optimal execution.

See the [Profiling Single-Stage Diffusion](../../../docs/contributing/profiling.md#3-profiling-single-stage-diffusion) guide for detailed instructions on:
1. Using the PyTorch profiler (`profiler: "torch"`) to capture detailed CPU/CUDA traces.
2. Using Nsight Systems (`nsys`) with `profiler: "cuda"` for low-overhead CUDA traces.
3. Controlling profiling via `omni.start_profile()` and `omni.stop_profile()`.

Report cold E2E, warm user latency, steady-state wave time, peak allocated and
reserved HBM, host PSS for the full process tree, and stage boundaries. State
whether prompt encoding and VAE/audio decoding are included. For multi-device
layouts, plot user latency against throughput per device and keep different
denoising-step counts on separate Pareto frontiers.

---

## Pre-commit conventions

New library files must pass the local gates in
[docs/contributing/README.md](../../../docs/contributing/README.md#linting).
That page is the full hook list (SPDX, forbidden imports including Hugging Face
Hub / Triton / pickle, `torch.cuda`, mypy, test marks, markdownlint, Buildkite,
shellcheck). In particular:

- SPDX copyright is `vLLM-Omni project` (stale `vLLM project` is rewritten).
- Use `import regex as re` and `pybase64` in `vllm_omni/`; do not import stdlib
  `re` or `base64`. Hugging Face Hub downloads go through
  `vllm.transformers_utils.repo_utils`.
- Do not add `torch.cuda.*` call sites; use `current_omni_platform`.
- New `tests/**/test_*.py` files need a CI level mark and a hardware mark.
- Do not expand `CHECK_IMPORTS[*].allowed_files` or `ALLOWED_FILES` without review.
- GitHub Actions skips SPDX/shellcheck/mypy-3.10/test-marks/markdownlint; run
  `pre-commit` locally.

## Iterative Development Tips

1. **Start minimal**: Basic generation first, no parallelism/caching
2. **Use `--enforce-eager`**: Disable torch.compile during debugging
3. **Use small models**: Test with smaller variants first
4. **Check tensor shapes**: Most errors are reshape mismatches in attention
5. **Add features incrementally**: Single GPU → TP → SP → CFG → HSDP → Cache-DiT
6. **For custom models**: Run the pinned reference separately, then port the runtime natively; do not ship temporary imports from the reference implementation
7. **Cache-DiT before parallelism tuning**: Cache-DiT is lossy — verify quality at baseline before combining with parallelism
8. **Combine lossless + lossy**: e.g., TP + SP + Cache-DiT for maximum throughput

## Reference Files

- [vllm-omni-test skill](../vllm-omni-test/SKILL.md) — L1–L4 markers, naming, Buildkite wiring, run commands
- [Transformer Adaptation](references/transformer-adaptation.md) — porting transformers from diffusers
- [Custom Model Patterns](references/custom-model-patterns.md) — patterns for non-diffusers models
- [Native Model Integration Checklist](references/native-model-integration-checklist.md) — ownership boundaries, phase gates, qualification matrix, and review evidence
- [Parallelism Patterns](references/parallelism-patterns.md) — TP, SP/USP, CFG parallel, HSDP implementation details
- [Cache-DiT Patterns](references/cache-dit-patterns.md) — cache-dit acceleration for standard and custom architectures
- [Troubleshooting](references/troubleshooting.md) — common errors and fixes
