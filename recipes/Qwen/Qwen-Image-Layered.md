# Qwen-Image-Layered

> Layered image decomposition on a single NVIDIA L40S using layer-wise CPU offloading.

## Summary

- Vendor: Qwen
- Model: `Qwen/Qwen-Image-Layered`
- Task: Layered image generation / image decomposition
- Mode: Offline inference
- Maintainer: Community

## When to use this recipe

Use this recipe as a known-good starting point for running
`Qwen/Qwen-Image-Layered` on a single NVIDIA L40S 48 GB GPU.

The configuration uses layer-wise CPU offloading for the image transformer and
was validated end-to-end with four RGBA output layers at 640 resolution.

## References

- Related example:
  [`examples/offline_inference/image_to_image/image_edit.py`](../../examples/offline_inference/image_to_image/image_edit.py)

## Hardware Support

## GPU

### 1x NVIDIA L40S 48GB

#### Environment

- OS: Linux
- Python: 3.12.14
- GPU: NVIDIA L40S, 46068 MiB available VRAM
- Host RAM: 124 GiB
- NVIDIA driver: 580.173.02
- CUDA: 13.0
- PyTorch: `2.13.0+cu130`
- vLLM: `0.27.1`
- vLLM-Omni: `0.27.0rc2.dev78+g2fdbf2234`
- vLLM-Omni commit: `2fdbf2234aeb76715618a5b236f0016f115e3e64`

#### Input

The validation used the public Qwen bear image:

```bash
wget -O qwen-bear.png \
  https://vllm-public-assets.s3.us-west-2.amazonaws.com/omni-assets/qwen-bear.png
```

#### Command

Run from the repository root:

```bash
python examples/offline_inference/image_to_image/image_edit.py \
  --model "Qwen/Qwen-Image-Layered" \
  --image qwen-bear.png \
  --prompt "" \
  --output "layered_50" \
  --num-inference-steps 50 \
  --cfg-scale 4.0 \
  --layers 4 \
  --resolution 640 \
  --color-format RGBA \
  --enable-layerwise-offload
```
