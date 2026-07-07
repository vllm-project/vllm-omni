# Quantization

vLLM-Omni exposes quantization through the unified `quantization_config`
path. The same configuration entrypoint is used across diffusion-only models,
multi-stage omni/TTS models, and multi-stage diffusion models, but each model
type has a different quantization scope.

## Quantization Modes

| Mode | Guide | Description | Methods |
|------|-------|-------------|---------|
| Online quantization | [Online Quantization](online.md) | vLLM-Omni computes quantized weights and scales while loading the model. | FP8 W8A8, Int8 W8A8, MXFP8 W8A8, MXFP4 W4A4 |
| Runtime attention quantization | [Quantized KV Cache](quantized_kvcache.md) | vLLM-Omni dynamically quantizes eligible diffusion Flash Attention tensors during inference. | FP8 FA |
| Pre-quantized checkpoints | Method-specific guides | The checkpoint or an offline quantizer provides quantized weights and scales before serving. | ModelOpt, GGUF, AutoRound, msModelSlim, serialized Int8, offline MXFP8, offline MXFP4 DualScale |

## Hardware Support

| Device | FP8 W8A8 | Int8 W8A8 | ModelOpt | MXFP8 W8A8 | MXFP4 W4A4 | GGUF | AutoRound | msModelSlim |
|--------|----------|-----------|----------|------------|------------|------|-----------|-------------|
| NVIDIA Blackwell GPU (SM 100+) | ✅ | ✅ | ✅ | ⭕ | ⭕ | ✅ | ✅ | ❌ |
| NVIDIA Ada/Hopper GPU (SM 89+) | ✅ | ✅ | ✅ | ⭕ | ⭕ | ✅ | ✅ | ❌ |
| NVIDIA Ampere GPU (SM 80+) | ✅ | ✅ | ⭕ | ⭕ | ⭕ | ✅ | ✅ | ❌ |
| AMD ROCm | ⭕ | ⭕ | ⭕ | ⭕ | ✅ | ⭕ | ⭕ | ❌ |
| Intel XPU | ⭕ | ⭕ | ⭕ | ⭕ | ⭕ | ⭕ | ✅ | ❌ |
| Ascend NPU | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ |

Legend: `✅` supported, `❌` unsupported, `⭕` not verified in this
guide. FP8 on Ampere may use a weight-only path where available.

## Model Type Support

### Diffusion Model (Qwen-Image, Wan2.2)

These models run a diffusion transformer as the primary inference module. The
default quantization target is the transformer; tokenizer, scheduler, text
encoder, and VAE stay on the base checkpoint unless a method guide says
otherwise.

| Method | Guide | Mode | Example models | Status |
|--------|-------|------|----------------|--------|
| FP8 W8A8 | [FP8](fp8.md) | Online W8A8 or checkpoint FP8 | Qwen-Image; Wan2.2 is not validated | Validated for Qwen-Image family and other DiT models |
| Int8 W8A8 | [Int8](int8.md) | Online or serialized W8A8 | Qwen-Image; Wan2.2 is not validated | Validated for Qwen-Image and Z-Image |
| ModelOpt | [ModelOpt](modelopt.md) | Pre-quantized FP8 checkpoints | Qwen-Image, Z-Image, FLUX.2, HunyuanImage-3.0 | Validated for ModelOpt FP8 diffusion checkpoints |
| MXFP8 W8A8 | [MXFP8](mxfp8.md) | Online W8A8 or offline pre-quantized | Wan2.2-T2V-A14B, I2V-A14B, TI2V-5B | Ascend NPU only; validated for Wan2.2 |
| MXFP4 W4A4 | [MXFP4](mxfp4.md) | `mxfp4`: online single-scale only; `mxfp4_dualscale`: online or offline dual-scale (offline recommended) | Wan2.2-T2V-A14B, I2V-A14B | Ascend NPU only; validated for Wan2.2 A14B cascade models; TI2V-5B not supported; offline `mxfp4_dualscale` uses calibrated `mul_scale` for best accuracy |
| GGUF | [GGUF](gguf.md) | Pre-quantized transformer weights | Qwen-Image | Validated where a model-specific GGUF adapter exists |
| AutoRound | [AutoRound](autoround.md) | Pre-quantized W4A16 checkpoints | FLUX.1-dev; Qwen-Image/Wan2.2 not validated | Checkpoint-driven |
| msModelSlim | [msModelSlim](msmodelslim.md) | Pre-quantized Ascend checkpoints | Wan2.2 recipe; HunyuanImage-3.0 inference target | Ascend/NPU path |

### Multi-Stage Omni/TTS Model (Qwen3-Omni, Qwen3-TTS)

These models combine an AR language model with audio, vision, talker, or TTS
stages. Quantization is scoped to the AR language-model stage when the
checkpoint contains a supported `quantization_config`; the non-AR stages stay
in BF16 unless the model guide explicitly adds support.

| Method | Guide | Scope | Example models | Status |
|--------|-------|-------|----------------|--------|
| ModelOpt | [ModelOpt](modelopt.md) | Thinker or language-model checkpoint config | Qwen3-Omni thinker | ModelOpt checkpoint path |
| Int8 | [Int8](int8.md) | Not currently validated for omni/TTS stages | Qwen3-Omni, Qwen3-TTS | Not validated |
| MXFP8 | [MXFP8](mxfp8.md) | Not currently validated for omni/TTS stages | Qwen3-Omni, Qwen3-TTS | Not validated |
| MXFP4 | [MXFP4](mxfp4.md) | Not currently validated for omni/TTS stages | Qwen3-Omni, Qwen3-TTS | Not validated |
| GGUF | [GGUF](gguf.md) | Not currently validated for omni/TTS stages | Qwen3-Omni, Qwen3-TTS | Not validated |
| AutoRound | [AutoRound](autoround.md) | Thinker or language-model checkpoint config | Qwen2.5-Omni, Qwen3-Omni | Supported through AutoRound checkpoints |
| msModelSlim | [msModelSlim](msmodelslim.md) | Not currently validated for omni/TTS stages | Qwen3-Omni, Qwen3-TTS | Not validated |

### Multi-Stage Diffusion Model (BAGEL, GLM-Image)

These models split generation across multiple stages. Quantization must be
attached to the intended stage rather than applied globally.

| Method | Guide | Scope | Example models | Status |
|--------|-------|-------|----------------|--------|
| FP8 | [FP8](fp8.md) | Stage-specific DiT or transformer module | BAGEL, GLM-Image | Requires model-specific validation |
| Int8 | [Int8](int8.md) | Stage-specific DiT or transformer module | BAGEL, GLM-Image | Requires model-specific validation |
| ModelOpt | [ModelOpt](modelopt.md) | Checkpoint-defined diffusion stage | BAGEL, GLM-Image | Requires model-specific validation |
| MXFP8 | [MXFP8](mxfp8.md) | Stage-specific DiT or transformer module | BAGEL, GLM-Image | Not validated |
| MXFP4 | [MXFP4](mxfp4.md) | Stage-specific DiT or transformer module | BAGEL, GLM-Image | Not validated |
| GGUF | [GGUF](gguf.md) | Stage-specific transformer weights | BAGEL, GLM-Image | No validated adapter listed |
| AutoRound | [AutoRound](autoround.md) | Checkpoint-defined stage | BAGEL, GLM-Image | No validated checkpoint listed |
| msModelSlim | [msModelSlim](msmodelslim.md) | Ascend-generated stage weights | GLM-Image | Requires model-specific adaptation |

!!! note
    "Online quantization" means vLLM-Omni computes the quantization data while
    loading the model. "Pre-quantized" means the checkpoint or external
    quantizer provides the required quantized weights and scales.

## Quantization Scope

### Diffusion Model (Qwen-Image, Wan2.2)

The default target is the diffusion transformer. Component routing is available
through `build_quant_config()`:

```python
from vllm_omni.quantization import build_quant_config

config = build_quant_config({
    "transformer": {"method": "fp8"},
    "vae": None,
})
```

| Component | Default quantized? | Notes |
|-----------|--------------------|-------|
| Diffusion transformer | Yes | Primary target for FP8, Int8, ModelOpt, MXFP8, MXFP4, GGUF, AutoRound, and msModelSlim |
| Text encoder | No | Keep BF16 unless a method-specific guide documents support |
| VAE | No | Keep BF16; storage-only paths are method-specific |
| Scheduler/tokenizer | No | Loaded from the base model repository |

### Multi-Stage Omni/TTS Model (Qwen3-Omni, Qwen3-TTS)

| Component | Default quantized? | Notes |
|-----------|--------------------|-------|
| Thinker or AR language model | Yes, when checkpoint config is supported | ModelOpt FP8/NVFP4 or AutoRound checkpoint config |
| Audio encoder | No | BF16 |
| Vision encoder | No | BF16 |
| Talker or TTS stage | No | BF16 unless model-specific support is documented |
| Code2Wav | No | BF16 |

### Multi-Stage Diffusion Model (BAGEL, GLM-Image)

| Component | Default quantized? | Notes |
|-----------|--------------------|-------|
| Selected diffusion or transformer stage | Method-specific | Must be routed to the intended stage |
| Other generation stages | No | Keep BF16 unless separately validated |
| VAE, tokenizer, scheduler | No | Loaded from the base checkpoint |

## Python API

`build_quant_config()` accepts strings, dictionaries, per-component
dictionaries, existing `QuantizationConfig` objects, or `None`.

```python
from vllm_omni.quantization import build_quant_config

build_quant_config("fp8")
build_quant_config({"method": "fp8", "activation_scheme": "static"})
build_quant_config("auto-round", bits=4, group_size=128)
build_quant_config({"method": "gguf", "gguf_model": "/path/to/model.gguf"})
build_quant_config({"transformer": {"method": "fp8"}, "vae": None})
build_quant_config(None)
```

## Output Similarity Comparison Tool

Use `vllm_omni.quantization.tools.compare_diffusion_trajectory_similarity`
to compare a reference diffusion run with a quantized candidate run using the
same prompt, seed, resolution, scheduler settings, and inference steps. The
tool compares final decoded images or video frames, and also reports generation
latency and worker-reported peak memory when available.

This is useful when validating whether online quantization, an offline
pre-quantized checkpoint, or a new `ignored_layers` choice keeps generation
quality close to the BF16 reference.

### Online Quantization Example

```bash
python -m vllm_omni.quantization.tools.compare_diffusion_trajectory_similarity \
  --task t2i \
  --model Qwen/Qwen-Image \
  --candidate-quantization fp8 \
  --ignored-layers img_mlp \
  --prompt "a cup of coffee on the table" \
  --height 512 --width 512 \
  --num-inference-steps 20 \
  --seed 142 \
  --output-json /tmp/qwen_image_fp8_similarity/result.json \
  --save-output-dir /tmp/qwen_image_fp8_similarity/images \
  --enforce-eager
```

### Offline Checkpoint Example

Use `--candidate-model` when the candidate is already quantized or lives at a
different model path:

```bash
python -m vllm_omni.quantization.tools.compare_diffusion_trajectory_similarity \
  --task t2i \
  --reference-model Qwen/Qwen-Image \
  --candidate-model /path/to/qwen-image-fp8-checkpoint \
  --prompt "a cup of coffee on the table" \
  --height 512 --width 512 \
  --num-inference-steps 20 \
  --seed 142 \
  --output-json /tmp/qwen_image_fp8_checkpoint_similarity/result.json
```

If the checkpoint does not include a loadable quantization config, pass one
explicitly:

```bash
--candidate-quantization-config-json '{"method":"fp8"}'
```

### Output Metrics

The output JSON includes `output_metrics`, `reference_generation`, and
`candidate_generation`.

| Metric | Direction | Meaning |
|--------|-----------|---------|
| `cosine_similarity` | Higher is better | Vector direction similarity between output pixels or frames. Useful as a broad sanity check. |
| `mae` | Lower is better | Mean absolute pixel or frame error. For decoded outputs, values are in uint8 pixel units. |
| `mse` / `rmse` | Lower is better | Squared error and its square root. These penalize localized large differences more than `mae`. |
| `max_abs` | Lower is better | Worst single-element absolute error. Treat it as an outlier/debug signal, not as a release gate. |
| `l2` / `relative_l2` | Lower is better | Absolute and reference-normalized L2 distance. `relative_l2` is easier to compare across resolutions. |
| `psnr_db` | Higher is better | Pixel-space signal-to-noise ratio in dB for uint8 images or frames. |
| `avg_generation_time_s` | Lower is better | Average wall-clock generation time across measured runs. |
| `max_peak_memory_mb` | Lower is better | Maximum worker-reported peak device memory across measured runs, when the worker reports it. |

Recommended starting thresholds for same-seed diffusion comparisons:

| Metric | Smoke threshold | Stricter target | Notes |
|--------|-----------------|-----------------|-------|
| `psnr_db` | `>= 20.0` | `>= 25.0` | Good for quick image or frame regression checks. |
| `mae` | `<= 12.0` | `<= 6.0` | Interpreted in decoded uint8 pixel units. |
| `cosine_similarity` | `>= 0.98` | `>= 0.995` | Less sensitive to global scale than L2-style metrics. |
| `relative_l2` | `<= 0.20` | `<= 0.08` | Useful when comparing across prompts or resolutions. |

These thresholds are heuristics. Tune them by model family, task, resolution,
quantization method, and deployment tolerance. For release gating, pair the
numeric report with visual inspection of saved reference and candidate outputs.

The tool intentionally reports separate quality, latency, and memory metrics
instead of a single consolidated similarity score. A single score can hide
important tradeoffs, for example a candidate with good PSNR but a meaningful
memory regression, or a candidate with low average error but localized visual
artifacts. If you need a project-specific pass/fail gate, define it as an
explicit policy over the individual metrics.

Pixel-level metrics do not measure semantic consistency. For higher-cost
evaluation, you can complement this report with a vision-language judge that
describes the reference and candidate outputs and compares those descriptions.
Keep that semantic check separate from this lightweight tool so users can
choose whether the additional model cost and latency are appropriate.
