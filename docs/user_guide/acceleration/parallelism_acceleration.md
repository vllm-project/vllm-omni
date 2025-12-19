# Parallelism Acceleration Guide

This guide includes how to use parallelism methods in vLLM-Omni to speed up diffusion model inference as well as reduce the memory requirement on each device.

## Overview

The following parallelism methods are currently supported in vLLM-Omni:

1. DeepSpeed Ulysses Sequence Parallel (Ulysses-SP) ([paper](https://arxiv.org/pdf/2309.14509)): Ulysses-SP splits the input along the sequence dimension and uses all-to-all communication to allow each device to compute only a subset of attention heads.


The following table shows which models are currently supported by parallelism method:


| Model | Model Identifier |  Ulysses-SP |
|-------|-----------------|-----------|
| **Qwen-Image** | `Qwen/Qwen-Image` |  ✅ |
| **Z-Image** | `Tongyi-MAI/Z-Image-Turbo` | ❌ |
| **Qwen-Image-Edit** | `Qwen/Qwen-Image-Edit` | ✅ |

### Sequence Parallelism

#### Ulysses-SP

##### Quick Start

An example of using Ulysses-SP is shown below:
```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
ulysses_degree = 2

omni = Omni(
    model="Qwen/Qwen-Image",
    parallel_config=DiffusionParallelConfig(ulysses_degree=2)
)

outputs = omni.generate(prompt="A cat sitting on a windowsill", num_inference_steps=50, width=2048, height=2048)
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

##### How to parallelize a new model

If a diffusion model has been deployed in vLLM-Omni and supports single-card inference, you can refer to the following instruction on how to parallelize this model with Ulysses-SP.

First, please edit the `TransformerModel`'s `forward` function in the `xxx_model_transformer.py` to make the inputs (image hidden states, positional embeddings, etc.) as chunks separated at the sequence dimension. Taking `qwen_image_transformer.py` as an example:

```diff
class QwenImageTransformer2DModel(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        ...
    ):
+   if self.parallel_config.sequence_parallel_size > 1:
+       hidden_states = torch.chunk(hidden_states, get_sequence_parallel_world_size(), dim=-2)[
+           get_sequence_parallel_rank()
+      ]

    hidden_states = self.img_in(hidden_states)

    ...
    image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

+   def get_rotary_emb_chunk(freqs):
+       freqs = torch.chunk(freqs, get_sequence_parallel_world_size(), dim=0)[get_sequence_parallel_rank()]
+       return freqs

+   if self.parallel_config.sequence_parallel_size > 1:
+       img_freqs, txt_freqs = image_rotary_emb
+       img_freqs = get_rotary_emb_chunk(img_freqs)
+       image_rotary_emb = (img_freqs, txt_freqs)
```

Next, at the end of the `forward` function, please call `get_sp_group().all_gather` to gather the chunked outputs across devices, and concatenate them at the sequence dimension.


```diff
class QwenImageTransformer2DModel(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        ...
    ):
    # Use only the image part (hidden_states) from the dual-stream blocks
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

+   if self.parallel_config.sequence_parallel_size > 1:
+       output = get_sp_group().all_gather(output, dim=-2)
    return Transformer2DModelOutput(sample=output)
```

Finally, you can set the parallel configuration and pass it to `Omni` and start parallel inference with:
```diff
from vllm_omni import Omni
+from vllm_omni.diffusion.data import DiffusionParallelConfig
ulysses_degree = 2

omni = Omni(
    model="Qwen/Qwen-Image",
+    parallel_config=DiffusionParallelConfig(ulysses_degree=2)
)

outputs = omni.generate(prompt="A cat sitting on a windowsill", num_inference_steps=50)
```
