# DiNa-LRM — Offline Inference Example

[DiNa-LRM](https://github.com/HKUST-C4G/diffusion-rm) (**Di**ffusion-**Na**tive **L**atent **R**eward **M**odel) is a discriminative reward model that scores text-image alignment in the SD3.5-M latent space.
It does **not** generate images — it returns a scalar reward for each (prompt, image) pair.

## Model

| Architecture | HF Checkpoint |
|---|---|
| `DiNaLRMPipeline` | [`liuhuohuo/DiNa-LRM-SD35M-12layers`](https://huggingface.co/liuhuohuo/DiNa-LRM-SD35M-12layers) |

The checkpoint encodes a truncated SD3.5-M backbone (first 12 transformer layers, LoRA fine-tuned)
and a cross-attention reward head.

## Requirements

```bash
pip install vllm-omni diffusers transformers peft omegaconf pyyaml \
            huggingface_hub torchvision pillow
```

## Quick Start

```bash
cd examples/offline_inference/dina_lrm
```

**Score a single prompt against an image:**

```bash
python end2end.py \
    --model liuhuohuo/DiNa-LRM-SD35M-12layers \
    --prompt "A cat sitting on a wooden floor" \
    --image-path /path/to/image.png
```

**Use a higher noise level for pipeline-generated latents:**

```bash
python end2end.py \
    --model liuhuohuo/DiNa-LRM-SD35M-12layers \
    --prompt "A red sports car" \
    --image-path /path/to/image.png \
    --noise-level 0.4
```

**Use a local checkpoint:**

```bash
python end2end.py \
    --model /path/to/DiNa-LRM-SD35M-12layers \
    --prompt "A beautiful sunset over the ocean" \
    --image-path /path/to/image.png
```

## Output

The script prints raw and normalised reward scores for each prompt:

```
============================================================
Prompt                                          Raw score   Norm score
------------------------------------------------------------
A cat sitting on a wooden floor                    0.8312       1.0831
============================================================

Normalised score = (raw + 10) / 10  — typically in [0, 2], centred ~1.
```

| Field | Description |
|---|---|
| **Raw score** | Direct output of the reward head (unbounded float) |
| **Norm score** | `(raw + 10) / 10` — human-readable value, typically in `[0, 2]` |

## Parameters

| Argument | Default | Description |
|---|---|---|
| `--model` | `liuhuohuo/DiNa-LRM-SD35M-12layers` | HF repo ID or local path |
| `--prompt` | — | Text prompt to score against the image |
| `--image-path` | — | Local image file (JPEG / PNG) |
| `--noise-level` | `0.1` | Noise sigma σ. Use `0.1` for clean images, `0.4` for generated latents |
| `--dtype` | `bfloat16` | Compute dtype: `bfloat16` / `float16` / `float32` |