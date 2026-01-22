# Parallelism Acceleration Guide

This guide includes how to use parallelism methods in vLLM-Omni to speed up diffusion model inference as well as reduce the memory requirement on each device.

## Overview

The following parallelism methods are currently supported in vLLM-Omni:

1. DeepSpeed Ulysses Sequence Parallel (DeepSpeed Ulysses-SP) ([arxiv paper](https://arxiv.org/pdf/2309.14509)): Ulysses-SP splits the input along the sequence dimension and uses all-to-all communication to allow each device to compute only a subset of attention heads.

2. [Ring-Attention](#ring-attention) - splits the input along the sequence dimension and uses ring-based P2P communication to accumulate attention results, keeping the sequence dimension sharded

3. Classifier-Free-Guidance Parallel (CFG-Parallel): CFG-Parallel runs the positive/negative prompts of classifier-free guidance (CFG) on different devices, then merges on a single device to perform the scheduler step.

4. [Tensor Parallelism](#tensor-parallelism): Tensor parallelism shards model weights across devices. This can reduce per-GPU memory usage. Note that for diffusion models we currently shard the majority of layers within the DiT.

The following table shows which models are currently supported by parallelism method:

### ImageGen

| Model | Model Identifier | Ulysses-SP | Ring-SP | CFG-Parallel | Tensor-Parallel |
|-------|------------------|------------|---------|--------------|--------------------------|
| **LongCat-Image** | `meituan-longcat/LongCat-Image` | ✅ | ✅ | ❌ | ❌ |
| **LongCat-Image-Edit** | `meituan-longcat/LongCat-Image-Edit` | ✅ | ✅ | ❌ | ❌ |
| **Ovis-Image** | `OvisAI/Ovis-Image` | ❌ | ❌ | ❌ | ❌ |
| **Qwen-Image** | `Qwen/Qwen-Image` | ✅ | ✅ | ✅ | ✅ |
| **Qwen-Image-Edit** | `Qwen/Qwen-Image-Edit` | ✅ | ✅ | ✅ | ❌ |
| **Qwen-Image-Edit-2509** | `Qwen/Qwen-Image-Edit-2509` | ✅ | ✅ | ✅ | ❌ |
| **Qwen-Image-Layered** | `Qwen/Qwen-Image-Layered` | ✅ | ✅ | ✅ | ❌ |
| **Z-Image** | `Tongyi-MAI/Z-Image-Turbo` | ✅ | ✅ | ❌ | ✅ (TP=2 only) |
| **Stable-Diffusion3.5** | `stabilityai/stable-diffusion-3.5` | ❌ | ❌ | ❌ | ❌ |


!!! note "TP Limitations for Diffusion Models"
    We currently implement Tensor Parallelism (TP) only for the DiT (Diffusion Transformer) blocks. This is because the `text_encoder` component in vLLM-Omni uses the original Transformers implementation, which does not yet support TP.

    - Good news: The text_encoder typically has minimal impact on overall inference performance.
    - Bad news: When TP is enabled, every TP process retains a full copy of the text_encoder weights, leading to significant GPU memory waste.

    We are actively refactoring this design to address this. For details and progress, please refer to [Issue #771](https://github.com/vllm-project/vllm-omni/issues/771).


!!! note "Why Z-Image is TP=2 only"
    Z-Image Turbo is currently limited to `tensor_parallel_size` of **1 or 2** due to model shape divisibility constraints.
    For example, the model has `n_heads=30` and a final projection out dimension of `64`, so valid TP sizes must divide both 30 and 64; the only common divisors are **1 and 2**.

### VideoGen

| Model | Model Identifier | Ulysses-SP | Ring-SP | Tensor-Parallel |
|-------|------------------|------------|---------|--------------------------|
| **Wan2.2** | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | ❌ | ❌ | ❌ |

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
    prompt="a cat reading a book",
    num_inference_steps=9,
    width=512,
    height=512,
)
```

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

NOTE: "Terminology: SP vs CP"
    Our "Sequence Parallelism" (SP) corresponds to "Context Parallelism" (CP) in the [diffusers library](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/_modeling_parallel.py).
    We use "Sequence Parallelism" to align with vLLM-Omni's terminology.

---

###### Non-intrusive `_sp_plan` (Recommended)

The `_sp_plan` mechanism allows SP without modifying `forward()` logic. The framework automatically registers hooks to shard inputs and gather outputs at module boundaries.

**Requirements for `forward()` function:**

- Tensor operations that need sharding/gathering must happen at **`nn.Module` boundaries** (not inline Python operations)
- If your `forward()` contains inline tensor operations (e.g., `torch.cat`, `pad_sequence`) that need sharding, **extract them into a submodule**

**When to create a submodule:**

```python
# ❌ BAD: Inline operations - hooks cannot intercept
def forward(self, x, cap_feats):
    unified = torch.cat([x, cap_feats], dim=1)  # Cannot be sharded via _sp_plan
    ...

# ✅ GOOD: Extract into a submodule
class UnifiedPrepare(nn.Module):
    def forward(self, x, cap_feats):
        return torch.cat([x, cap_feats], dim=1)  # Now can be sharded via _sp_plan

class MyModel(nn.Module):
    def __init__(self):
        self.unified_prepare = UnifiedPrepare()  # Submodule

    def forward(self, x, cap_feats):
        unified = self.unified_prepare(x, cap_feats)  # Hook can intercept here
```

---

###### Defining `_sp_plan`

**Type definitions** (see [diffusers `_modeling_parallel.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/_modeling_parallel.py) for reference):

```python
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,   # Corresponds to diffusers' ContextParallelInput
    SequenceParallelOutput,  # Corresponds to diffusers' ContextParallelOutput
)
```

| Parameter | Description |
|-----------|-------------|
| `split_dim` | Dimension to split/gather (usually `1` for sequence) |
| `expected_dims` | Expected tensor rank for validation (optional) |
| `split_output` | `False`: shard **input** parameters; `True`: shard **output** tensors |
| `auto_pad` | Auto-pad if sequence not divisible by world_size (Ulysses only) |

**Key naming convention:**

| Key | Meaning | Python equivalent |
|-----|---------|-------------------|
| `""` | Root model | `model` |
| `"blocks.0"` | First element of ModuleList | `model.blocks[0]` |
| `"blocks.*"` | All elements of ModuleList | `for b in model.blocks` |
| `"outputs.main"` | ModuleDict entry | `model.outputs["main"]` |

**Dictionary key types:**

| Key type | `split_output` | Description |
|----------|----------------|-------------|
| `"param_name"` (str) | `False` | Shard **input parameter** by name |
| `0`, `1` (int) | `True` | Shard **output tuple** by index |

**Example** (similar to [diffusers `transformer_wan.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_wan.py)):

```python
class MyTransformer(nn.Module):
    _sp_plan = {
        # Shard rope module OUTPUTS (returns tuple)
        "rope": {
            0: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),  # cos
            1: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),  # sin
        },
        # Shard transformer block INPUT parameter
        "blocks.0": {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
        },
        # Gather at final projection
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }
```

---

###### Hook flow

```
Input → [SequenceParallelSplitHook: pre_forward] → Module.forward() → [post_forward] → ...
                                                                              ↓
... → [SequenceParallelGatherHook: post_forward] → Output
```

1. **SplitHook** shards tensors before/after the target module
2. **Attention layers** handle Ulysses/Ring communication internally
3. **GatherHook** collects sharded outputs

The framework automatically applies these hooks when `sequence_parallel_size > 1`.

---

###### Method 2: Intrusive modification (For complex cases)

For models with dynamic sharding logic that cannot be expressed via `_sp_plan`:

```python
from vllm_omni.diffusion.distributed.sp_sharding import sp_shard, sp_gather

def forward(self, hidden_states, ...):
    if self.parallel_config.sequence_parallel_size > 1:
        hidden_states = sp_shard(hidden_states, dim=1)
        # ... computation ...
        output = sp_gather(output, dim=1)
    return output
```

---

###### Choosing the right approach

| Scenario | Approach |
|----------|----------|
| Standard transformer | `_sp_plan` |
| Inline tensor ops need sharding | Extract to submodule + `_sp_plan` |
| Dynamic/conditional sharding | Intrusive modification |


### CFG-Parallel

##### Offline Inference

CFG-Parallel is enabled through `DiffusionParallelConfig(cfg_parallel_size=...)`. The recommended configuration is `cfg_parallel_size=2` (one rank for the positive branch and one rank for the negative branch).

An example of offline inference using CFG-Parallel (image-to-image) is shown below:

```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig

omni = Omni(
    model="Qwen/Qwen-Image-Edit",
    parallel_config=DiffusionParallelConfig(cfg_parallel_size=2),
)

outputs = omni.generate(
    prompt="turn this cat to a dog",
    negative_prompt="low quality, blurry",
    true_cfg_scale=4.0,
    pil_image=input_image,
    num_inference_steps=50,
)
```

Notes:

- CFG-Parallel is only effective when **true CFG** is enabled (i.e., `true_cfg_scale > 1` and a `negative_prompt` is provided).

#### How to parallelize a pipeline

This section describes how to add CFG-Parallel to a diffusion **pipeline**. We use the Qwen-Image pipeline (`vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py`) as the reference implementation.

In `QwenImagePipeline`, each diffusion step runs two denoiser forward passes sequentially:

- positive (prompt-conditioned)
- negative (negative-prompt-conditioned)

CFG-Parallel assigns these two branches to different ranks in the **CFG group** and synchronizes the results.

Below is an example of CFG-Parallel implementation:

```python
def diffuse(
        self,
        ...
        ):
    # Enable CFG-parallel: rank0 computes positive, rank1 computes negative.
    cfg_parallel_ready = do_true_cfg and get_classifier_free_guidance_world_size() > 1

    self.transformer.do_true_cfg = do_true_cfg

    if cfg_parallel_ready:
        cfg_group = get_cfg_group()
        cfg_rank = get_classifier_free_guidance_rank()

        if cfg_rank == 0:
            local_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states_mask=prompt_embeds_mask,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                attention_kwargs=self.attention_kwargs,
                return_dict=False,
            )[0]
        else:
            local_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states_mask=negative_prompt_embeds_mask,
                encoder_hidden_states=negative_prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=negative_txt_seq_lens,
                attention_kwargs=self.attention_kwargs,
                return_dict=False,
            )[0]

        gathered = cfg_group.all_gather(local_pred, separate_tensors=True)
        if cfg_rank == 0:
            noise_pred = gathered[0]
            neg_noise_pred = gathered[1]
            comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            noise_pred = comb_pred * (cond_norm / noise_norm)
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        cfg_group.broadcast(latents, src=0)
    else:
        # fallback: run positive then negative sequentially on one rank
        ...
```
