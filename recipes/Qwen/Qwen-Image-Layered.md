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

#### Verification

Confirm that four non-empty RGBA layers were generated:

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image

for index in range(4):
    path = Path(f"layered_50_{index}.png")
    assert path.is_file() and path.stat().st_size > 0, f"Missing or empty: {path}"

    with Image.open(path) as image:
        assert image.mode == "RGBA", f"{path}: expected RGBA, got {image.mode}"
        print(f"{path}: size={image.size}, mode={image.mode}")
PY
```

Expected output:

```text
layered_50_0.png: size=(...), mode=RGBA
layered_50_1.png: size=(...), mode=RGBA
layered_50_2.png: size=(...), mode=RGBA
layered_50_3.png: size=(...), mode=RGBA
```

All four generated layers should also be manually inspected to confirm that they contain meaningful decomposed image content.

#### Observed Results

| Metric                         |                  Result |
| ------------------------------ | ----------------------: |
| Inference steps                |                   50/50 |
| Output layers                  |                       4 |
| Resolution                     |                     640 |
| Color format                   |                    RGBA |
| Layer-wise offload             |                 Enabled |
| Total generation time          |                155.97 s |
| Denoising loop                 |                  ~150 s |
| Average denoising step         |                 ~3.00 s |
| GPU memory after model loading |              ~17.17 GiB |
| Peak observed GPU memory       | 26,863 MiB (~26.23 GiB) |

During the main denoising loop, GPU utilization remained approximately 99–100%, while observed GPU memory stayed stable at approximately 26,863 MiB.

#### Notes

* `--enable-layerwise-offload` moves transformer blocks between host memory and GPU memory to reduce GPU-memory requirements.
* The runtime reported layer-wise offloading across 60 `QwenImageTransformerBlock` instances.
* The validated machine had 124 GiB of host RAM.
* Validation used tensor parallel size 1 with no distributed execution.
* Although `--cfg-scale 4.0` was supplied, true classifier-free guidance was not enabled because no negative prompt was provided.
* The reported latency and memory measurements are observations from this specific L40S environment and should not be treated as performance guarantees.

