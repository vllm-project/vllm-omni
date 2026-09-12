# HiDream-O1-Image

> HiDream-O1 text-to-image generation through the shared offline example

## Summary

- Vendor: HiDream.ai
- Model: `HiDream-ai/HiDream-O1-Image`
- Task: Text-to-image
- Mode: Offline inference
- Maintainer: Community

## When to use this recipe

Use this recipe to generate native-resolution images with HiDream-O1 on one
or two CUDA GPUs. The two-GPU command shards the Qwen3-VL attention and MLP
weights, as well as the timestep and pixel projections.

## References

- Upstream model:
  [`HiDream-ai/HiDream-O1-Image`](https://huggingface.co/HiDream-ai/HiDream-O1-Image)
- Related offline example:
  [`examples/offline_inference/text_to_image/text_to_image.py`](../../examples/offline_inference/text_to_image/text_to_image.py)
- Related issue:
  [`vllm-project/vllm-omni#3479`](https://github.com/vllm-project/vllm-omni/issues/3479)

## Hardware Support

## GPU

### 1x H100 80GB

#### Environment

- OS: Linux
- Python: Match the repository requirements for your checkout
- Driver / runtime: NVIDIA CUDA
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Command

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model HiDream-ai/HiDream-O1-Image \
  --prompt "A cat is sitting next to a sign that says 'HiDream-O1 vLLM-Omni'" \
  --height 2048 \
  --width 2048 \
  --num-inference-steps 50 \
  --guidance-scale 5.0 \
  --seed 42 \
  --output hidream_o1_output.png
```

#### Verification

```bash
python -c "from PIL import Image; im = Image.open('hidream_o1_output.png'); print(im.mode, im.size)"
```

Expected output:

```text
RGB (2048, 2048)
```

#### Notes

- The pipeline supports BF16 text-to-image generation.
- Requested sizes are matched to the closest resolution supported by the checkpoint.

### 2x H100 80GB

#### Environment

- OS: Linux
- Python: Match the repository requirements for your checkout
- Driver / runtime: NVIDIA CUDA
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Command

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model HiDream-ai/HiDream-O1-Image \
  --tensor-parallel-size 2 \
  --prompt "A cat is sitting next to a sign that says 'HiDream-O1 vLLM-Omni'" \
  --height 2048 \
  --width 2048 \
  --num-inference-steps 50 \
  --guidance-scale 5.0 \
  --seed 42 \
  --output hidream_o1_tp2_output.png
```

#### Verification

```bash
python -c "from PIL import Image; im = Image.open('hidream_o1_tp2_output.png'); print(im.mode, im.size)"
```

Expected output:

```text
RGB (2048, 2048)
```

#### Notes

- Tensor parallelism shards the attention, MLP, timestep, and pixel projection weights across both GPUs.
- Image editing and image-conditioned generation are not supported.

## TeaCache acceleration

TeaCache for HiDream-O1 uses coefficients calibrated from HiDream-O1 full-compute trajectories. The model-specific default is `rel_l1_thresh=0.10`; the example below specifies the same value explicitly so that its speed/quality policy is visible and reproducible.

The validated configuration used 2048 x 2048 output, 50 inference steps, guidance scale 5.0, BF16, and tensor parallel size 2. Other resolutions, schedulers, step counts, and guidance scales should be validated separately.

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

engine = Omni(
    model="HiDream-ai/HiDream-O1-Image",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.10},
    tensor_parallel_size=2,
    enforce_eager=True,
)

outputs = engine.generate(
    {
        "prompt": "A cat is sitting next to a sign that says 'HiDream-O1 vLLM-Omni'",
        "modalities": ["image"],
    },
    OmniDiffusionSamplingParams(
        height=2048,
        width=2048,
        num_inference_steps=50,
        guidance_scale=5.0,
        seed=42,
    ),
)
outputs[0].images[0].save("hidream_o1_teacache.png")
engine.close()
```
