# vLLM-Omni Compatibility Test Framework

A batch-processing-based feature compatibility testing and performance evaluation framework covering all 13 diffusion acceleration/optimization features.


## 🚀 Quick Start

### 1. Minimal test (3 prompts)

```bash
cd compatibility

python run_compat_test.py \
    --baseline-feature cfg_parallel \
    --addons teacache \
    --num-prompts 3 \
    --steps 10
```

### 2. Analyze results

```bash
python analyze_compat_results.py \
    --results-dir ./compat_results/cfg_parallel \
    --charts
```

### 3. Diagnose image differences

```bash
# Diagnose a single configuration
python diagnose_diff.py \
    --results-dir ./compat_results/cfg_parallel \
    --config cfg_parallel+teacache

# Diagnose all configurations at once and save a JSON report
python diagnose_diff.py \
    --results-dir ./compat_results/cfg_parallel \
    --all --save-json
```

That's it! 🎉

## 🎯 Supported Features (13 total)

### Acceleration: Cache Methods (Lossy)

| Feature ID | Description | GPU Requirement | Lossy? | Typical Speedup |
|------------|-------------|-----------------|--------|-----------------|
| `teacache` | TeaCache adaptive caching | ×1 | ✅ | ~1.5× |
| `cache_dit` | Cache-DiT (DBCache + TaylorSeer + SCM) | ×1 | ✅ | ~1.7× |

> `teacache` and `cache_dit` are **incompatible** and cannot be used together.

### Acceleration: Parallelism Methods (Lossless)

| Feature ID | Description | GPU Requirement | Lossy? | Typical Speedup |
|------------|-------------|-----------------|--------|-----------------|
| `cfg_parallel` | CFG positive/negative branches dispatched to 2 GPUs | ×2 | ❌ | ~1.8× |
| `ulysses` | Ulysses sequence parallelism (all-to-all) | ×2 | ❌ | ~1.6× |
| `ring` | Ring sequence parallelism (ring communication) | ×2 | ❌ | ~1.5× |
| `tp` | Tensor parallelism (weight sharding) | ×2 | ❌ | ~1.4× |
| `hsdp` | HSDP (FSDP2 weight sharding with runtime reassembly) | ×2 | ❌ | ~1.3× |

> `tp` and `hsdp` are **incompatible** and cannot be used together.

### Memory Optimization

| Feature ID | Description | GPU Requirement | Lossy? | Notes |
|------------|-------------|-----------------|--------|-------|
| `cpu_offload` | Module-level CPU offloading (DiT + text encoder) | ×1 | ❌ | Single-GPU only |
| `layerwise_offload` | Layer-wise CPU offloading (keeps only 1 block on GPU at a time) | ×1 | ❌ | Single-GPU only |
| `vae_patch_parallel` ⚠️ | VAE patch parallel decoding | — (reuses GPUs from parallel baseline) | ❌ | **addon-only**, see note below |
| `fp8` | FP8 quantization (Ada/Hopper W8A8) | ×1 | ✅ (slight) | Incompatible with gguf |
| `gguf` | GGUF quantization (Q4/Q8, etc.) | ×1 | ✅ (slight) | Incompatible with fp8 |

> `layerwise_offload` and `vae_patch_parallel` are **incompatible**.  
> `fp8` and `gguf` are **incompatible**.

#### Special Notes on vae_patch_parallel

`vae_patch_parallel` is an **addon-only** feature and **cannot** be used as a standalone `--baseline-feature`.  
`--vae-patch-parallel-size` must equal the product of the parallel sizes of the baseline parallel method:

```bash
# ✅ Correct: vae_patch_parallel as addon stacked on top of tp (×2)
python run_compat_test.py \
    --baseline-feature tp \
    --addons vae_patch_parallel \
    --model Qwen/Qwen-Image

# ✅ Correct: stacked on top of cfg_parallel (×2)
python run_compat_test.py \
    --baseline-feature cfg_parallel \
    --addons vae_patch_parallel \
    --model Qwen/Qwen-Image

# ❌ Wrong: cannot be used as a standalone baseline
python run_compat_test.py --baseline-feature vae_patch_parallel  # raises error
```

### Extended Features

| Feature ID | Description | GPU Requirement | Lossy? | Notes |
|------------|-------------|-----------------|--------|-------|
| `lora` | LoRA inference adapter | ×1 | ❌ | Requires `--lora-path` |

## ⛔ Conflict Rules

The following feature combinations are **incompatible**. When a conflicting combination appears in the test matrix, the test case is automatically marked as `SKIP (conflict)` and skipped — **it does not count as a failure**.

| Feature A | Feature B | Reason |
|-----------|-----------|--------|
| `tp` | `hsdp` | Tensor Parallel and HSDP are not compatible |
| `teacache` | `cache_dit` | Two cache methods cannot be enabled simultaneously |
| `layerwise_offload` | `cpu_offload` | Two CPU offloading methods are incompatible |
| `fp8` | `gguf` | Two quantization methods are incompatible |
| `layerwise_offload` | any multi-GPU feature | Layer-wise offloading currently supports single-GPU only (all features with `gpu_multiplier > 1` conflict) |

Multi-GPU features (`gpu_multiplier > 1`) include: `cfg_parallel`, `ulysses`, `ring`, `tp`, `hsdp`.

### Conflict Detection Example

```
# Runtime terminal output:
[WARN]  SKIP 'tp+hsdp'             — Tensor Parallel and HSDP are not compatible
[WARN]  SKIP 'teacache+cache_dit'  — TeaCache and Cache-DiT are not compatible
[WARN]  SKIP 'fp8+gguf'            — FP8 quantization and GGUF quantization are not compatible
[WARN]  SKIP 'layerwise_offload+tp'— 'layerwise_offload' supports single-card only and cannot
                                      be combined with multi-GPU feature(s): ['tp']

# Final summary distinguishes conflict skips vs. insufficient-GPU skips:
  SKIP (conflict)  : 2  (incompatible feature pairs)
  SKIP (GPU)       : 1  (insufficient GPUs)
  SKIP total       : 3  (configs, not prompts)
```

### Extending Conflict Rules

Add entries to the `CONFLICT_RULES` list in `run_compat_test.py`:

```python
CONFLICT_RULES: list[tuple[str, str, str]] = [
    ("tp",               "hsdp",          "Tensor Parallel and HSDP are not compatible"),
    ("teacache",         "cache_dit",     "TeaCache and Cache-DiT are not compatible"),
    ("layerwise_offload","cpu_offload",   "Layerwise and module-level CPU offloading are not compatible"),
    ("fp8",              "gguf",          "FP8 and GGUF quantization are not compatible"),
    # Add new entry:
    ("my_feature_a",     "my_feature_b",  "Describe the reason"),
]
```

If a feature only supports single-GPU, add it to `SINGLE_CARD_ONLY`:

```python
SINGLE_CARD_ONLY: frozenset[str] = frozenset({"layerwise_offload", "my_single_card_feature"})
```

## 📖 Usage Scenarios

### Scenario 1: New Feature Development

After developing a new feature, quickly verify compatibility with existing features:

```bash
python run_compat_test.py \
    --baseline-feature <your_new_feature> \
    --addons cfg_parallel teacache ulysses \
    --num-prompts 20 \
    --steps 30
```

### Scenario 2: Memory Optimization Verification

```bash
# Verify FP8 quantization does not affect image quality
python run_compat_test.py \
    --baseline-feature fp8 \
    --addons cfg_parallel \
    --model Qwen/Qwen-Image-2512 \
    --num-prompts 10 --steps 20

# Verify layer-wise CPU offloading
python run_compat_test.py \
    --baseline-feature layerwise_offload \
    --num-prompts 5 --steps 10
```

### Scenario 3: LoRA Inference Verification

```bash
python run_compat_test.py \
    --baseline-feature lora \
    --lora-path /path/to/my/lora_adapter \
    --model Tongyi-MAI/Z-Image-Turbo \
    --num-prompts 10 --steps 20
```

### Scenario 4: Performance Optimization Comparison

```bash
# Before optimization
python run_compat_test.py --baseline-feature cfg_parallel \
    --output-dir ./before_optimization

# After optimization
python run_compat_test.py --baseline-feature cfg_parallel \
    --output-dir ./after_optimization

# Compare
python compare_results.py \
    ./before_optimization/cfg_parallel/report.json \
    ./after_optimization/cfg_parallel/report.json \
    --best
```

### Scenario 5: Verify Conflict Skipping

When an incompatible feature combination appears, the test case is automatically skipped (not counted as a failure):

```bash
# teacache and cache_dit are incompatible → cfg+teacache+cache_dit is automatically skipped
python run_compat_test.py \
    --baseline-feature teacache \
    --addons cache_dit \
    --model Qwen/Qwen-Image-2512

# Example terminal output:
# [WARN]  SKIP 'teacache+cache_dit' — TeaCache and Cache-DiT are not compatible
```

### Scenario 7: CI/CD Integration

```bash
python run_compat_test.py \
    --baseline-feature cfg_parallel \
    --addons teacache cache_dit fp8 \
    --num-prompts 10 \
    --steps 20 \
    --output-dir ./ci_test

python analyze_compat_results.py --results-dir ./ci_test/cfg_parallel
```

## 🔧 Tools

### Core Tools

| Tool | Function | Input | Output |
|------|----------|-------|--------|
| `batch_text_to_image.py` | Batch image generation | Prompt file | Images + timing stats |
| `run_compat_test.py` | Compatibility test execution | Feature configuration | Test results directory |
| `analyze_compat_results.py` | Results analysis | Test results directory | JSON report + charts |
| `diagnose_diff.py` | Image difference diagnosis | Test results directory | Diff report (terminal + JSON) |
| `compare_results.py` | Multi-result comparison | Multiple JSON reports | Comparative analysis |

### diagnose_diff.py Arguments

| Argument | Description |
|----------|-------------|
| `--results-dir PATH` | Results directory (containing `baseline/` and config subdirectories) |
| `--config NAME...` | One or more configuration names to diagnose |
| `--all` | Auto-discover and diagnose all non-reference configs in the directory |
| `--reference NAME` | Reference configuration name (default: `baseline`) |
| `--top N` | Show at most N worst images per configuration (default: 10) |
| `--save-json` | Save a JSON report for each configuration |

> SSIM metric requires `pip install scikit-image`; falls back to MeanDiff/MaxDiff if not installed.

## 📊 Output Structure

### Test Results Directory Layout

```
compat_results/
└── cfg_parallel/                    # Baseline feature directory
    ├── manifest.json                # Test metadata
    ├── report.json                  # Analysis report (generated after running analyze)
    ├── chart_quality.png            # Quality comparison chart
    ├── chart_speedgain.png          # Performance comparison chart
    ├── baseline/                    # Pure baseline configuration
    │   ├── batch_generation.log     # Batch generation log
    │   ├── batch_generation.exitcode
    │   ├── prompt_00.png
    │   ├── prompt_00.exitcode
    │   └── ...
    ├── cfg_parallel/                # Baseline feature run alone
    └── cfg_parallel+teacache/       # Combined feature run
```

### Key Metrics

- **Speedup**: speedup relative to the pure baseline
- **MeanDiff**: mean pixel difference (0–1, lower is better)
- **MaxDiff**: maximum pixel difference (0–1)
- **SSIM**: structural similarity (0–1, higher is better; requires scikit-image)
- **Status**: OK ✅ / WARN ⚠️ / LARGE ❌

## 🔍 FAQ

### Q: Why is a configuration being skipped?

```
SKIP 'cfg_parallel+ulysses' — requires 4 GPUs, only 2 available
```

**A**: Insufficient GPUs. Reduce the number of feature combinations or use more GPUs.

### Q: How do I run LoRA tests?

**A**: Provide the `--lora-path` argument:

```bash
python run_compat_test.py \
    --baseline-feature lora \
    --lora-path /path/to/adapter \
    --model Tongyi-MAI/Z-Image-Turbo
```

### Q: What should I do about HSDP and TP incompatibility?

**A**: HSDP cannot be used together with `--tensor-parallel-size > 1` or data parallelism. Test them separately:

```bash
python run_compat_test.py \
    --baseline-feature hsdp \
    --model black-forest-labs/FLUX.1-dev \
    --num-prompts 5 --steps 10
```

### Q: How can I speed up the tests?

**A**: Use the following arguments:
- `--num-prompts 3` — reduce number of prompts
- `--steps 10` — reduce number of inference steps
- `--height 512 --width 512` — reduce image resolution

### Q: Should I be concerned about WARN status?

**A**: WARN typically appears for lossy features (TeaCache, Cache-DiT, FP8, GGUF); the quality loss is within an acceptable range.

### Q: How do I add a new feature?

**A**: Add an entry to `FEATURE_REGISTRY` in `run_compat_test.py`:

```python
"my_feature": {
    "args": ["--my-arg", "value"],
    "gpu_multiplier": 1,    # GPU multiplier
    "lossy": False,         # whether lossy
    "label": "My Feature",  # display name
    "category": "parallelism",  # category
    "note": "Brief description.",
},
```

## 📈 Performance Benchmark Reference

Based on 20 prompts, 30 inference steps, 1024×1024, Qwen/Qwen-Image-2512:

| Configuration | Avg Time | Speedup | Quality Loss (MeanDiff) |
|---------------|----------|---------|-------------------------|
| Pure baseline | 10.2s | 1.0× | — |
| CFG Parallel | 5.6s | 1.82× | 0.0000 |
| CFG + TeaCache | 2.9s | 3.52× | 0.0823 |
| CFG + Cache-DiT | 2.7s | 3.78× | 0.1124 |
| FP8 | 7.1s | 1.44× | ~0.01 |

*Actual performance varies by hardware and configuration*

## 🤝 Contributing

### Adding a New Feature

1. Register in `FEATURE_REGISTRY` in `run_compat_test.py`
2. If new CLI arguments are needed, add them in `batch_text_to_image.py`
3. Run tests to verify and submit a PR

### Improving Documentation

1. Update this README and related Markdown files
2. Ensure example code is runnable
3. Submit a PR
