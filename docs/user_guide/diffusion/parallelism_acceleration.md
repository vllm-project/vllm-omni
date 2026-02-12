# Parallelism Acceleration Guide

This guide includes how to use parallelism methods in vLLM-Omni to speed up diffusion model inference as well as reduce the memory requirement on each device.

## Overview

The following parallelism methods are currently supported in vLLM-Omni:

1. DeepSpeed Ulysses Sequence Parallel (DeepSpeed Ulysses-SP) ([arxiv paper](https://arxiv.org/pdf/2309.14509)): Ulysses-SP splits the input along the sequence dimension and uses all-to-all communication to allow each device to compute only a subset of attention heads.

2. [Ring-Attention](#ring-attention) - splits the input along the sequence dimension and uses ring-based P2P communication to accumulate attention results, keeping the sequence dimension sharded

3. Classifier-Free-Guidance Parallel (CFG-Parallel): CFG-Parallel runs the positive/negative prompts of classifier-free guidance (CFG) on different devices, then merges on a single device to perform the scheduler step.

4. [Tensor Parallelism](#tensor-parallelism): Tensor parallelism shards model weights across devices. This can reduce per-GPU memory usage. Note that for diffusion models we currently shard the majority of layers within the DiT.

5. [VAE Patch Parallelism](#vae-patch-parallelism): VAE patch parallelism shards VAE decode spatially across ranks. This can reduce the peak memory of VAE decode and (depending on resolution and communication overhead) speed up VAE decode.

The following table shows which models are currently supported by parallelism method:

### ImageGen

| Model                    | Model Identifier                     | Ulysses-SP | Ring-SP | CFG-Parallel | Tensor-Parallel | VAE-Patch-Parallel |
|--------------------------|--------------------------------------|:----------:|:-------:|:------------:|:---------------:|:------------------:|
| **LongCat-Image**        | `meituan-longcat/LongCat-Image`      |     ✅      |    ✅    |      ❌       |        ✅        |         ❌          |
| **LongCat-Image-Edit**   | `meituan-longcat/LongCat-Image-Edit` |     ✅      |    ✅    |      ❌       |        ✅        |         ❌          |
| **Ovis-Image**           | `OvisAI/Ovis-Image`                  |     ❌      |    ❌    |      ❌       |        ❌        |         ❌          |
| **Qwen-Image**           | `Qwen/Qwen-Image`                    |     ✅      |    ✅    |      ✅       |        ✅        |         ❌          |
| **Qwen-Image-Edit**      | `Qwen/Qwen-Image-Edit`               |     ✅      |    ✅    |      ✅       |        ✅        |         ❌          |
| **Qwen-Image-Edit-2509** | `Qwen/Qwen-Image-Edit-2509`          |     ✅      |    ✅    |      ✅       |        ✅        |         ❌          |
| **Qwen-Image-Layered**   | `Qwen/Qwen-Image-Layered`            |     ✅      |    ✅    |      ✅       |        ✅        |         ❌          |
| **Z-Image**              | `Tongyi-MAI/Z-Image-Turbo`           |     ✅      |    ✅    |      ❌       |  ✅ (TP=2 only)  |         ✅          |
| **Stable-Diffusion3.5**  | `stabilityai/stable-diffusion-3.5`   |     ❌      |    ❌    |      ❌       |        ❌        |         ❌          |
| **FLUX.2-klein**         | `black-forest-labs/FLUX.2-klein-4B`  |     ❌      |    ❌    |      ❌       |        ✅        |         ❌          |
| **FLUX.1-dev**           | `black-forest-labs/FLUX.1-dev`       |     ❌      |    ❌    |      ✅       |        ✅        |         ❌          |

!!! note "TP Limitations for Diffusion Models"
    We currently implement Tensor Parallelism (TP) only for the DiT (Diffusion Transformer) blocks. This is because the `text_encoder` component in vLLM-Omni uses the original Transformers implementation, which does not yet support TP.

    - Good news: The text_encoder typically has minimal impact on overall inference performance.
    - Bad news: When TP is enabled, every TP process retains a full copy of the text_encoder weights, leading to significant GPU memory waste.

    We are actively refactoring this design to address this. For details and progress, please refer to [Issue #771](https://github.com/vllm-project/vllm-omni/issues/771).


!!! note "Why Z-Image is TP=2 only"
    Z-Image Turbo is currently limited to `tensor_parallel_size` of **1 or 2** due to model shape divisibility constraints.
    For example, the model has `n_heads=30` and a final projection out dimension of `64`, so valid TP sizes must divide both 30 and 64; the only common divisors are **1 and 2**.

### VideoGen

| Model | Model Identifier | Ulysses-SP | Ring-Attention | Tensor-Parallel |
|-------|------------------|------------|---------|--------------------------|
| **Wan2.2** | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | ✅ | ✅ | ✅ |

### Tensor Parallelism

Tensor parallelism splits model parameters across GPUs. In vLLM-Omni, tensor parallelism is configured via `DiffusionParallelConfig.tensor_parallel_size`.

#### Offline Inference

```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig

omni = Omni(
    model="Tongyi-MAI/Z-Image-Turbo",
    parallel_config=DiffusionParallelConfig(tensor_parallel_size=2),
)

outputs = omni.generate(
    "a cat reading a book",
    OmniDiffusionSamplingParams(
        num_inference_steps=9,
        width=512,
        height=512,
    ),
)
```

### VAE Patch Parallelism

VAE patch parallelism distributes the VAE decode workload across multiple ranks by splitting the latent spatially. It is configured via `DiffusionParallelConfig.vae_patch_parallel_size` and can be combined with other parallelism methods (e.g., TP).

!!! note "Enablement and feature gate"
    - VAE patch parallelism is currently **enabled only for validated pipelines** (currently: `Tongyi-MAI/Z-Image-Turbo`).
    - If `vae_patch_parallel_size > 1` is set for a validated pipeline, vLLM-Omni will automatically enable `vae_use_tiling` as a safety gate. (We use `vae_use_tiling` because it indicates the VAE supports diffusers tiling parameters like `tile_latent_min_size` and `tile_overlap_factor`.)

#### Offline Inference

```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig

omni = Omni(
    model="Tongyi-MAI/Z-Image-Turbo",
    parallel_config=DiffusionParallelConfig(
        tensor_parallel_size=2,
        vae_patch_parallel_size=2,
    ),
    vae_use_tiling=True,
)

outputs = omni.generate(
    prompt="a cat reading a book",
    num_inference_steps=9,
    width=1024,
    height=1024,
)
```

#### How it works (method selection)

VAE patch parallelism automatically selects between two internal decode methods based on whether diffusers tiling would kick in:

- `_distributed_tiled_decode`: Used when the latent spatial size exceeds `vae.tile_latent_min_size` (i.e., diffusers tiled decode). Each rank decodes a subset of tiles; rank0 gathers and runs the same overlap+blend+stitch logic as diffusers. This matches the single-rank diffusers tiled output.

- `_distributed_patch_decode`: Used when diffusers tiling would not kick in. Each rank decodes a grid patch expanded with a latent-space halo; then rank0 gathers the cropped core patches and stitches them into the full image. This path has no blending and can introduce small numerical differences compared to the non-parallel decode.

### Sequence Parallelism

#### Ulysses-SP

##### Offline Inference

An example of offline inference script using [Ulysses-SP](https://arxiv.org/pdf/2309.14509) is shown below:
```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.diffusion.data import DiffusionParallelConfig
ulysses_degree = 2

omni = Omni(
    model="Qwen/Qwen-Image",
    parallel_config=DiffusionParallelConfig(ulysses_degree=2)
)

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=50, width=2048, height=2048),
)
```

See `examples/offline_inference/text_to_image/text_to_image.py` for a complete working example.

##### Online Serving

You can enable Ulysses-SP in online serving for diffusion models via `--usp`:

```bash
# Text-to-image (requires >= 2 GPUs)
vllm serve Qwen/Qwen-Image --omni --port 8091 --usp 2
```

##### Benchmarks
!!! note "Benchmark Disclaimer"
    These benchmarks are provided for **general reference only**. The configurations shown use default or common parameter settings and have not been exhaustively optimized for maximum performance. Actual performance may vary based on:

    - Specific model and use case
    - Hardware configuration
    - Careful parameter tuning
    - Different inference settings (e.g., number of steps, image resolution)


To measure the parallelism methods, we run benchmarks with **Qwen/Qwen-Image** model generating images (**2048x2048** as long sequence input) with 50 inference steps. The hardware devices are NVIDIA H800 GPUs. `sdpa` is the attention backends.

| Configuration | Ulysses degree |Generation Time | Speedup |
|---------------|----------------|---------|---------|
| **Baseline (diffusers)** | - | 112.5s | 1.0x |
| Ulysses-SP  |  2  |  65.2s | 1.73x |
| Ulysses-SP  |  4  | 39.6s | 2.84x |
| Ulysses-SP  |  8  | 30.8s | 3.65x |

#### Ring-Attention

Ring-Attention ([arxiv paper](https://arxiv.org/abs/2310.01889)) splits the input along the sequence dimension and uses ring-based P2P communication to accumulate attention results. Unlike Ulysses-SP which uses all-to-all communication, Ring-Attention keeps the sequence dimension sharded throughout the computation and circulates Key/Value blocks through a ring topology.

##### Offline Inference

An example of offline inference script using Ring-Attention is shown below:
```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.diffusion.data import DiffusionParallelConfig
ring_degree = 2

omni = Omni(
    model="Qwen/Qwen-Image",
    parallel_config=DiffusionParallelConfig(ring_degree=2)
)

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=50, width=2048, height=2048),
)
```

See `examples/offline_inference/text_to_image/text_to_image.py` for a complete working example.


##### Online Serving

You can enable Ring-Attention in online serving for diffusion models via `--ring`:

```bash
# Text-to-image (requires >= 2 GPUs)
vllm serve Qwen/Qwen-Image --omni --port 8091 --ring 2
```

##### Benchmarks
!!! note "Benchmark Disclaimer"
    These benchmarks are provided for **general reference only**. The configurations shown use default or common parameter settings and have not been exhaustively optimized for maximum performance. Actual performance may vary based on:

    - Specific model and use case
    - Hardware configuration
    - Careful parameter tuning
    - Different inference settings (e.g., number of steps, image resolution)


To measure the parallelism methods, we run benchmarks with **Qwen/Qwen-Image** model generating images (**1024x1024** as long sequence input) with 50 inference steps. The hardware devices are NVIDIA A100 GPUs. `flash_attn` is the attention backends.

| Configuration | Ring degree |Generation Time | Speedup |
|---------------|----------------|---------|---------|
| **Baseline (diffusers)** | - | 45.2s | 1.0x |
| Ring-Attention  |  2  |  29.9s | 1.51x |
| Ring-Attention  |  4  | 23.3s | 1.94x |


#### Hybrid Ulysses + Ring

You can combine both Ulysses-SP and Ring-Attention for larger scale parallelism. The total sequence parallel size equals `ulysses_degree × ring_degree`.

##### Offline Inference

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.diffusion.data import DiffusionParallelConfig

# Hybrid: 2 Ulysses × 2 Ring = 4 GPUs total
omni = Omni(
    model="Qwen/Qwen-Image",
    parallel_config=DiffusionParallelConfig(ulysses_degree=2, ring_degree=2)
)

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=50, width=2048, height=2048),
)
```

##### Online Serving

```bash
# Text-to-image (requires >= 4 GPUs)
vllm serve Qwen/Qwen-Image --omni --port 8091 --usp 2 --ring 2
```

##### Benchmarks
!!! note "Benchmark Disclaimer"
    These benchmarks are provided for **general reference only**. The configurations shown use default or common parameter settings and have not been exhaustively optimized for maximum performance. Actual performance may vary based on:

    - Specific model and use case
    - Hardware configuration
    - Careful parameter tuning
    - Different inference settings (e.g., number of steps, image resolution)


To measure the parallelism methods, we run benchmarks with **Qwen/Qwen-Image** model generating images (**1024x1024** as long sequence input) with 50 inference steps. The hardware devices are NVIDIA A100 GPUs. `flash_attn` is the attention backends.

| Configuration | Ulysses degree | Ring degree | Generation Time | Speedup |
|---------------|----------------|-------------|-----------------|---------|
| **Baseline (diffusers)** | - | - | 45.2s | 1.0x |
| Hybrid Ulysses + Ring  |  2  |  2  |  24.3s | 1.87x |


### CFG-Parallel

#### Offline Inference

CFG-Parallel is enabled through `DiffusionParallelConfig(cfg_parallel_size=2)`, which runs one rank for the positive branch and one rank for the negative branch.

An example of offline inference using CFG-Parallel (image-to-image) is shown below:

```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig

image_path = "path_to_image.png"
omni = Omni(
    model="Qwen/Qwen-Image-Edit",
    parallel_config=DiffusionParallelConfig(cfg_parallel_size=2),
)
input_image = Image.open(image_path).convert("RGB")

outputs = omni.generate(
    {
        "prompt": "turn this cat to a dog",
        "negative_prompt": "low quality, blurry",
        "multi_modal_data": {"image": input_image},
    },
    OmniDiffusionSamplingParams(
        true_cfg_scale=4.0,
        num_inference_steps=50,
    ),
)
```

Notes:

- CFG-Parallel is only effective when a `negative_prompt` is provided AND a guidance scale (or `cfg_scale`) is greater than 1.

See `examples/offline_inference/image_to_image/image_edit.py` for a complete working example.
```bash
cd examples/offline_inference/image_to_image/
python image_edit.py \
  --model "Qwen/Qwen-Image-Edit" \
  --image "qwen_image_output.png" \
  --prompt "turn this cat to a dog" \
  --negative-prompt "low quality, blurry" \
  --cfg-scale 4.0 \
  --output "edited_image.png" \
  --cfg-parallel-size 2
```

#### Online Serving

You can enable CFG-Parallel in online serving for diffusion models via `--cfg-parallel-size`:

```bash
vllm serve Qwen/Qwen-Image-Edit --omni --port 8091 --cfg-parallel-size 2
```
