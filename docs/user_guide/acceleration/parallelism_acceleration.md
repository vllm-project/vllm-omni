# Parallelism Acceleration Guide

This guide includes how to use parallelism methods in vLLM-Omni to speed up diffusion model inference as well as reduce the memory requirement on each device.

## Overview

The following parallelism methods are currently supported in vLLM-Omni:

1. DeepSpeed Ulysses Sequence Parallel (DeepSpeed Ulysses-SP) ([arxiv paper](https://arxiv.org/pdf/2309.14509)): Ulysses-SP splits the input along the sequence dimension and uses all-to-all communication to allow each device to compute only a subset of attention heads.

2. [Ring-Attention](#ring-attention) - splits the input along the sequence dimension and uses ring-based P2P communication to accumulate attention results, keeping the sequence dimension sharded


The following table shows which models are currently supported by parallelism method:

### ImageGen

| Model | Model Identifier | Ulysses-SP | Ring-SP |
|-------|------------------|-----------|---------|
| **LongCat-Image** | `meituan-longcat/LongCat-Image` | ❌ | ❌ |
| **LongCat-Image-Edit** | `meituan-longcat/LongCat-Image-Edit` | ❌ | ❌ |
| **Ovis-Image** | `OvisAI/Ovis-Image` | ❌ | ❌ |
| **Qwen-Image** | `Qwen/Qwen-Image` | ✅ | ✅ |
| **Qwen-Image-Edit** | `Qwen/Qwen-Image-Edit` | ✅ | ✅ |
| **Qwen-Image-Edit-2509** | `Qwen/Qwen-Image-Edit-2509` | ✅ | ✅ |
| **Qwen-Image-Layered** | `Qwen/Qwen-Image-Layered` | ✅ | ✅ |
| **Z-Image** | `Tongyi-MAI/Z-Image-Turbo` | ❌ | ❌ |

### VideoGen

| Model | Model Identifier | Ulysses-SP | Ring-SP |
|-------|------------------|-----------|---------|
| **Wan2.2** | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | ❌ | ❌ |

### Sequence Parallelism

#### Ulysses-SP

##### Offline Inference

An example of offline inference script using [Ulysses-SP](https://arxiv.org/pdf/2309.14509) is shown below:
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

#### Ring-Attention

Ring-Attention ([arxiv paper](https://arxiv.org/abs/2310.01889)) splits the input along the sequence dimension and uses ring-based P2P communication to accumulate attention results. Unlike Ulysses-SP which uses all-to-all communication, Ring-Attention keeps the sequence dimension sharded throughout the computation and circulates Key/Value blocks through a ring topology.

##### Offline Inference

An example of offline inference script using Ring-Attention is shown below:
```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
ring_degree = 2

omni = Omni(
    model="Qwen/Qwen-Image",
    parallel_config=DiffusionParallelConfig(ring_degree=2)
)

outputs = omni.generate(prompt="A cat sitting on a windowsill", num_inference_steps=50, width=2048, height=2048)
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
from vllm_omni.diffusion.data import DiffusionParallelConfig

# Hybrid: 2 Ulysses × 2 Ring = 4 GPUs total
omni = Omni(
    model="Qwen/Qwen-Image",
    parallel_config=DiffusionParallelConfig(ulysses_degree=2, ring_degree=2)
)

outputs = omni.generate(prompt="A cat sitting on a windowsill", num_inference_steps=50, width=2048, height=2048)
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


##### How to parallelize a new model

If a diffusion model has been deployed in vLLM-Omni and supports single-card inference, you can refer to the following instructions to parallelize it with [Ulysses-SP](https://arxiv.org/pdf/2309.14509).

This section uses **Qwen-Image** (`QwenImageTransformer2DModel`) as the reference implementation. Qwen-Image is a **dual-stream** transformer (text + image) that performs **joint attention** across the concatenated sequences. Because of that, when enabling sequence parallel you typically:

- Chunk **image tokens** (`hidden_states`) across SP ranks along the **sequence dimension**.
- Keep **text embeddings** (`encoder_hidden_states`) **replicated** on all SP ranks for correctness (unless you implement full joint-SP semantics and explicitly split text too).
- Chunk **image RoPE freqs** to match the chunked image tokens.
- `all_gather` the final output back along the sequence dimension.

First, add the sequence-parallel helpers and (for Qwen-Image) the forward-context flag. Then, in the transformer's `forward()`:

- Chunk `hidden_states` by SP world size.
- Set `get_forward_context().split_text_embed_in_sp = False` because Qwen-Image uses joint attention and we would use a full text embedding in each rank.
- After `pos_embed`, chunk `img_freqs` on `dim=0` (token axis) to match the chunked image tokens; only chunk `txt_freqs` if you explicitly split text embeddings in SP.

Taking `vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py` as an example:

```diff
from vllm_omni.diffusion.distributed.parallel_state import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from vllm_omni.diffusion.forward_context import get_forward_context

class QwenImageTransformer2DModel(...):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: list[tuple[int, int, int]] | None = None,
        txt_seq_lens: list[int] | None = None,
        ...
    ):
+       if self.parallel_config.sequence_parallel_size > 1:
+           # Chunk image tokens along sequence dimension.
+           hidden_states = torch.chunk(
+               hidden_states,
+               get_sequence_parallel_world_size(),
+               dim=-2,
+           )[get_sequence_parallel_rank()]
+
+           # Qwen-Image uses *dual-stream* (text + image) and runs *joint attention*.
+           # Text embeddings should be replicated across SP ranks for correctness.
+           get_forward_context().split_text_embed_in_sp = False

        hidden_states = self.img_in(hidden_states)

        ...
        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

+       def get_rotary_emb_chunk(freqs):
+           return torch.chunk(freqs, get_sequence_parallel_world_size(), dim=0)[
+               get_sequence_parallel_rank()
+           ]
+
+       if self.parallel_config.sequence_parallel_size > 1:
+           img_freqs, txt_freqs = image_rotary_emb
+           img_freqs = get_rotary_emb_chunk(img_freqs)
+           if get_forward_context().split_text_embed_in_sp:
+               txt_freqs = get_rotary_emb_chunk(txt_freqs)
+           image_rotary_emb = (img_freqs, txt_freqs)
```

Next, at the end of the `forward()` function, call `get_sp_group().all_gather` to gather the chunked outputs across devices and concatenate them along the sequence dimension (matching the earlier `dim=-2` chunking):

```diff
class QwenImageTransformer2DModel(...):
    def forward(...):
        ...
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

+       if self.parallel_config.sequence_parallel_size > 1:
+           output = get_sp_group().all_gather(output, dim=-2)
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
