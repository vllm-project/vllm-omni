# T2I/I2I Accuracy Benchmarks

This directory contains evaluation tools for measuring the accuracy and quality of Text-to-Image (T2I) and Image-to-Image (I2I) generation in vLLM-Omni.

## Overview

When applying inference optimizations (quantization, flash-attention, prefix caching) to diffusion/generation models, there is a risk of silent quality degradation. This benchmark suite provides automated evaluation to ensure optimizations maintain visual fidelity and prompt alignment.

## Metrics

### T2I (Text-to-Image)

| Metric | Purpose | Description |
|--------|---------|-------------|
| **VQAScore** | Prompt faithfulness | Uses VLM to answer questions about generated images |
| **GenEval** | Compositional correctness | Evaluates attribute binding, spatial relationships, numeracy, action/state |

### I2I (Image-to-Image)

| Metric | Purpose | Description |
|--------|---------|-------------|
| **VLM-Judge** | Edit success | Evaluates if the edit achieved the instruction |
| **LPIPS** | Background preservation | Measures perceptual similarity of unchanged regions |

## Installation

The accuracy benchmarks require additional dependencies:

```bash
pip install -e ".[eval]"
```

Or install manually:

```bash
pip install lpips torchmetrics transformers torch pillow numpy
```

## Usage

### Command Line

Run T2I benchmark:

```bash
python benchmarks/accuracy/run_omni_accuracy.py \
    --mode t2i \
    --prompts prompts.txt \
    --images ./generated/ \
    --output t2i_results.json
```

Run I2I benchmark:

```bash
python benchmarks/accuracy/run_omni_accuracy.py \
    --mode i2i \
    --original ./original/ \
    --edited ./edited/ \
    --instructions instructions.txt \
    --output i2i_results.json
```

Run both:

```bash
python benchmarks/accuracy/run_omni_accuracy.py \
    --mode both \
    --config benchmark_config.json \
    --output results.json
```

### Python API

```python
from benchmarks.accuracy import T2IEvaluator, I2IEvaluator

# T2I evaluation
t2i_eval = T2IEvaluator(
    use_vqascore=True,
    use_geneval=True,
    vlm_model="Qwen2.5-VL-7B"
)
results = t2i_eval.evaluate(prompts, images)

# I2I evaluation
i2i_eval = I2IEvaluator(
    use_lpips=True,
    use_vlm_judge=True,
    lpips_net="alex"
)
results = i2i_eval.evaluate(
    original_images,
    edited_images,
    instructions
)
```

## Configuration

Example `benchmark_config.json`:

```json
{
  "vlm_model": "Qwen2.5-VL-7B",
  "lpips_net": "alex",
  "device": "cuda",
  "use_vqascore": true,
  "use_geneval": true,
  "use_lpips": true,
  "use_vlm_judge": true
}
```

## Datasets

The benchmarks support the following evaluation datasets:

- **GEBench**: Fine-grained prompt adherence evaluation
- **GEdit-Bench**: Image editing evaluation
- **GenEval**: Object-focused generation evaluation

Datasets are not automatically downloaded. Please obtain the datasets (for example, from their HuggingFace repositories) and prepare them according to the benchmark configuration before running these evaluations.

## CI Integration

These benchmarks are designed to run as L4/L5 tests in the CI pipeline:

- **L4**: Quick sanity check (small dataset subset)
- **L5**: Full accuracy evaluation (complete dataset)

Example GitHub Actions workflow:

```yaml
- name: Run Accuracy Benchmarks
  run: |
    python benchmarks/accuracy/run_omni_accuracy.py \
      --mode both \
      --config configs/l4_benchmark.json \
      --output l4_results.json
```

## References

1. Lin, T., et al. (2024). VQAScore: Evaluating Text-to-Image Generation with Visual Question Answering.
2. Ghosh, S., et al. (2024). GenEval: An Object-Focused Framework for Evaluating Text-to-Image Generation.
3. Zhang, R., et al. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. CVPR.
4. GEBench: https://github.com/stepfun-ai/GEBench
5. GEdit-Bench: https://github.com/stepfun-ai/Step1X-Edit

## Contributing

To add new metrics or improve existing ones:

1. Implement the metric in `t2i.py` or `i2i.py`
2. Add tests in `tests/`
3. Update this README
4. Submit a PR
