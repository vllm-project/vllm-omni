# Quantization

vLLM-Omni exposes quantization through the unified `quantization_config`
path. The same configuration entrypoint is used across diffusion-only models,
multi-stage omni/TTS models, and multi-stage diffusion models, but each model
type has a different quantization scope.

For the internal architecture and backend extension points, see the
[quantization design document](../../design/feature/quantization.md).

## Quantization Modes

| Mode | Guide | Description | Methods |
|------|-------|-------------|---------|
| Online quantization | [Online Quantization](online.md) | vLLM-Omni computes quantized weights and scales while loading the model. | FP8 W8A8, Int8 W8A8, BitsAndBytes W4, MXFP8 W8A8, MXFP4 W4A4 |
| Runtime attention quantization | [Quantized KV Cache](quantized_kvcache.md) | vLLM-Omni dynamically quantizes eligible diffusion Flash Attention tensors during inference. | FP8 FA |
| Pre-quantized checkpoints | Method-specific guides | The checkpoint or an offline quantizer provides quantized weights and scales before serving. | ModelOpt, AutoRound, msModelSlim, serialized Int8, offline MXFP8, offline MXFP4 DualScale |

Online quantization starts from a normal BF16/FP16 checkpoint and repeats the
conversion on each fresh model load. It is the simplest path for experiments.
Pre-quantized checkpoints store the packed weights and scales produced by a
separate tool; they are easier to reuse across deployments and can carry
calibrated or mixed-precision policies that load-time conversion cannot
reproduce.

## Support Levels

Quantization support is specific to a model, component, checkpoint format, and
hardware backend. The tables in this guide use the following levels:

| Level | Meaning |
|-------|---------|
| **CI-backed** | A named model and quantized checkpoint are exercised by scheduled full-model hardware CI. |
| **Validated** | A named model and checkpoint have end-to-end or quality-validation evidence, but the check is not necessarily scheduled in CI. |
| **Integrated** | The config, loader, quantized layers, or model wiring exists, but this guide has no completed end-to-end validation for the named model and checkpoint. |
| **Not validated** | There is no current support recommendation for this model/component combination. |
| **Unsupported** | The combination is intentionally rejected or known not to work. |

A method being available on a device does not make every model on that device
supported. Likewise, successfully detecting a checkpoint's
`quantization_config` establishes loader compatibility, not output quality.

## Hardware × Quantization Method

This matrix uses concrete inference paths as columns. Producer toolkits are not
separate methods: NVIDIA ModelOpt produces the ModelOpt FP8/NVFP4 formats,
Intel AutoRound produces AutoRound checkpoints, and Ascend msModelSlim produces
checkpoints consumed by the native MXFP or compatible Ascend paths.

| Hardware | [Online FP8](fp8.md) | [Int8](int8.md) | [BitsAndBytes W4](bitsandbytes.md) | [ModelOpt FP8](modelopt.md) | [ModelOpt NVFP4](modelopt.md#supported-modelopt-checkpoint-formats) | [AutoRound W4A16](autoround.md) | [OCP MXFP8](mxfp8.md) | [OCP MXFP4](mxfp4.md) |
|----------|-----------------------|------------------|------------------------------------------|--------------------------------|-------------------------------------------------------------------------|--------------------------------------|--------------------------|--------------------------|
| NVIDIA Blackwell (SM 100+) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| NVIDIA Hopper (SM 90) | ✅ | ✅ | ✅ | ✅ | ✅ (Marlin fallback) | ✅ | ❌ | ❌ |
| NVIDIA Ada (SM 89) | ✅ | ✅ | ✅ | ✅ | ⭕ | ✅ | ❌ | ❌ |
| NVIDIA Ampere (SM 80+) | ✅ | ✅ | ✅ | ⭕ | ⭕ | ✅ | ❌ | ❌ |
| AMD ROCm (gfx950 for MXFP4) | ⭕ | ⭕ | ❌ | ⭕ | ⭕ | ⭕ | ❌ | ✅ (online only) |
| Intel XPU | ⭕ | ⭕ | ❌ | ⭕ | ⭕ | ✅ | ✅ | ❌ |
| Ascend NPU | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |

Legend: `✅` backend available, `❌` rejected or no implementation, `⭕` not
verified in this guide. This matrix does not establish model-level support; use
the model-and-stage tables below for that.

The Int8 column covers online and serialized Int8 paths. OCP MXFP8 is online on
Intel XPU and online/offline on Ascend; Intel's separate AutoRound MXFP8
checkpoint path is described in the [MXFP8 guide](mxfp8.md). OCP MXFP4 on AMD
means online single-scale `mxfp4` on gfx950, while MXFP4 DualScale is
Ascend-only. NVIDIA's NVFP4 and OCP MXFP4 have different checkpoint metadata,
scale layouts, and kernel contracts. Mixed ModelOpt FP8/NVFP4 checkpoints also
require a model-specific mixed-precision policy and validation. FP8 on Ampere
may use a weight-only path where available.

## Current Model and Stage Support

The current offline checkpoint guidance is transformer-first. A listed
checkpoint quantizes the stage named in the **Scope** column; it does not imply
that the model's encoders, VAE, decoder, or other stages are also quantized.

### Vendor and Runtime Ecosystems

The producer and runtime backend are related, but they are not the same support
claim. A producer can emit several checkpoint formats, and each format still
needs a compatible vLLM-Omni loader, model-stage wiring, and validation. A
runtime may also provide online quantization without an offline producer path.

| Ecosystem | Current paths | Current vLLM-Omni scope | Model evidence |
|-----------|---------------|-------------------------|----------------|
| NVIDIA | ModelOpt FP8, NVFP4, and mixed FP8/NVFP4 | Diffusion transformer or Qwen-Omni thinker | Qwen-Image and other named diffusion checkpoints are validated; Qwen3-Omni NVFP4 has H100 CI for the Marlin fallback |
| Intel | AutoRound W4A16 and AutoRound MXFP8 | Diffusion/world-model transformer or Qwen-Omni thinker | Wan A14B and GLM-Image W4A16 are CI-backed; Qwen-Image, Qwen-Omni, and Cosmos3-Super have validation outside scheduled CI |
| AMD ROCm | Online OCP MXFP4 through AITER on gfx950; no pre-quantized MXFP4 or AMD Quark workflow is integrated | Diffusion transformer linear layers | Integrated backend with dispatch and weight-allocation unit coverage; no named-model quality validation or scheduled gfx950 full-model CI |
| Ascend | msModelSlim output through native MXFP8/MXFP4 merge workflows or a compatible generic Ascend format | Wan diffusion transformers | Native Wan MXFP8/MXFP4 paths are validated outside scheduled full-model NPU CI; other model families remain integrated or unvalidated |

### Pre-Quantized Model Matrix

| Model family | Method | Scope | Level | Notes |
|--------------|--------|-------|-------|-------|
| Qwen-Image | [ModelOpt](modelopt.md) FP8 or mixed FP8/NVFP4 | Diffusion transformer | Validated | Named Qwen-Image 2512 checkpoints and recipes are available. Text encoder and VAE stay BF16. |
| Qwen-Image | [AutoRound](autoround.md) W4A16 | Diffusion transformer | Validated | Uses `INC4AI/Qwen-Image-AutoRound-W4A16`; text encoder and VAE stay BF16. |
| Wan2.2 I2V/T2V A14B | [AutoRound](autoround.md) W4A16 | Both diffusion transformers | CI-backed | The high-noise and low-noise DiTs are quantized; text encoder and VAE stay BF16. |
| Wan2.2 TI2V 5B | [AutoRound](autoround.md) W4A16 | Diffusion transformer | Validated | Named checkpoint available; not covered by the scheduled Wan AutoRound job. |
| Wan2.2 | [MXFP8](mxfp8.md) | One or both diffusion transformers | Validated | Native Ascend and AutoRound checkpoint formats are integrated; no scheduled full-model offline NPU job. |
| Wan2.2 I2V/T2V A14B | [MXFP4 DualScale](mxfp4.md) | Both diffusion transformers | Validated | Offline calibrated checkpoints are recommended; no scheduled full-model offline NPU job. |
| Qwen2.5-Omni | [AutoRound](autoround.md) W4A16 | Thinker language model | Validated | Encoders and output/audio stages stay BF16. |
| Qwen3-Omni | [ModelOpt](modelopt.md) FP8 | Thinker language model | Validated | Tested outside scheduled CI. |
| Qwen3-Omni | [ModelOpt](modelopt.md) NVFP4 | Thinker language model | CI-backed | H100 CI exercises the Marlin fallback; the native Blackwell FP4 path has separate recipe validation. |
| Qwen3-Omni | [AutoRound](autoround.md) W4A16 | Thinker language model | Validated | Audio encoder, vision encoder, talker, and Code2Wav stay BF16. |
| GLM-Image | [AutoRound](autoround.md) W4A16 | Diffusion transformer stage | CI-backed | Text-to-image and image-to-image full-model smoke coverage. |
| Cosmos3-Super | [AutoRound](autoround.md) W4A16 | World-model transformer | Validated | Manually validated serving path; VAE and other components stay BF16. |
| Cosmos3-Nano | [AutoRound](autoround.md) W4A16 | World-model transformer | Integrated | Named checkpoint is listed, but no model-specific end-to-end test is maintained in-tree. |
| MiniMax H3 | None | None | Not validated | No offline vendor checkpoint path is documented; online FP8 targets the DiT only. |

The method pages contain the complete checkpoint lists, including FLUX,
Z-Image, and HunyuanImage-3.0 variants.

### Online Quantization

| Model family | Methods | Scope | Level |
|--------------|---------|-------|-------|
| Qwen-Image | FP8, Int8 | Diffusion transformer | Validated |
| Z-Image | Int8, BitsAndBytes W4 | Diffusion transformer | Validated |
| Wan2.2 | MXFP8 | One or both diffusion transformers | Validated on the documented XPU/NPU paths |
| Wan2.2 A14B | MXFP4 | Both diffusion transformers | Validated on Ascend; the ROCm gfx950 online backend is integrated but not model-validated |
| MiniMax H3 | FP8 | DiT and optional reference DiT | Validated |
| Qwen3-Omni | Generic online methods, such as FP8 | Thinker language model only | Integrated; not model-validated as an online recommendation |

See [Online Quantization](online.md) for configuration and hardware details.

### Unlisted and Long-Tail Models

For models not listed above, generic quantized linear layers or compatible
checkpoint metadata are only an integration starting point. The default support
target is the model's primary transformer, DiT, or AR language-model stage.
Text/audio/vision encoders, VAEs, vocoders, decoders, and output stages remain
BF16 until a model-specific integration supplies quant-aware layers, checkpoint
mapping, end-to-end output validation, and backend coverage.

!!! note
    "Online quantization" means vLLM-Omni computes the quantization data while
    loading the model. "Pre-quantized" means the checkpoint or external
    quantizer provides the required quantized weights and scales.

## Quantization Scope

### Diffusion Model (Qwen-Image, Wan2.2)

The default target is the diffusion transformer. `build_quant_config()` can
construct a component router:

```python
from vllm_omni.quantization import build_quant_config

config = build_quant_config({
    "transformer": {"method": "fp8"},
    "vae": None,
})
```

!!! warning "Routing syntax is not a support claim"
    A component dictionary only selects a quantization config by module prefix.
    It does not make an encoder or VAE quantizable. The model must construct
    that component from quantization-aware layers, and an offline checkpoint
    must contain matching packed weights and scales. Current user guidance
    keeps diffusion text encoders and VAEs in BF16.

| Component | Default quantized? | Notes |
|-----------|--------------------|-------|
| Diffusion transformer | Yes, for listed model/method pairs | Primary target for FP8, Int8, BitsAndBytes, ModelOpt, MXFP8, MXFP4, AutoRound, and msModelSlim |
| Second diffusion transformer | Yes, for listed Wan2.2 A14B methods | `transformer` and `transformer_2` read their own offline checkpoint metadata; arbitrary mixed methods are not currently validated |
| Text encoder | No | Keep BF16 unless a method-specific guide documents support |
| VAE | No | Keep BF16 |
| Scheduler/tokenizer | No | Loaded from the base model repository |

### Multi-Stage Omni/TTS Model (Qwen3-Omni, Qwen3-TTS)

| Component | Default quantized? | Notes |
|-----------|--------------------|-------|
| Thinker or AR language model | Yes, when checkpoint config is supported | ModelOpt FP8/NVFP4 or AutoRound checkpoint config |
| Audio encoder | No | BF16 |
| Vision encoder | No | BF16 |
| Talker or TTS stage | No | BF16 unless model-specific support is documented |
| Code2Wav | No | BF16 |

An explicit `ComponentQuantizationConfig` can be used by model developers to
route different configs to `audio_tower`, `visual`, and `language_model`, but
the supported Qwen-Omni checkpoints currently quantize only the thinker
language model.

### World Model (Cosmos3)

| Component | Default quantized? | Notes |
|-----------|--------------------|-------|
| World-model transformer | AutoRound checkpoints only | Cosmos3-Super is validated; Cosmos3-Nano is integrated but not end-to-end tested in-tree |
| VAE and guardrail components | No | BF16 |

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
build_quant_config("bitsandbytes")
build_quant_config("auto-round", bits=4, group_size=128)
# Component routing syntax; model-specific support is still required.
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
