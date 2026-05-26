# Parallelism Patterns Reference

## Overview

vLLM-Omni supports multiple parallelism strategies for diffusion models. Each targets a different bottleneck:

| Strategy | Splits | Best For | Constraint |
|----------|--------|----------|------------|
| Tensor Parallel (TP) | Model layers across GPUs | Latency reduction, large models | Requires fast GPU interconnect, `num_heads % tp == 0` |
| Sequence Parallel (SP/USP) | Sequence tokens across GPUs | Long sequences (video, high-res) | Near-linear scaling |
| CFG Parallel | Positive/negative CFG branches | Models using classifier-free guidance | Exactly 2 GPUs |
| HSDP | Weight shards via FSDP2 | VRAM reduction | Cannot combine with TP |
| VAE Patch Parallel | VAE decode spatial tiles | Large VAE outputs | Auto-enables tiling |

**Recommended integration order**: TP → SP → CFG Parallel → HSDP

**Official design docs**:
- TP: https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/tensor_parallel
- SP: https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/sequence_parallel
- CFG: https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/cfg_parallel
- HSDP: https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/hsdp

---

## Tensor Parallelism (TP)

Replace standard `nn.Linear` with vLLM's parallel linear layers. This is the most invasive change but provides direct VRAM savings and compute speedup.

### Layer replacement rules

| Pattern | vLLM Layer | When to Use |
|---------|-----------|-------------|
| Fan-out (first in FFN) | `ColumnParallelLinear` | Projection that splits output across ranks |
| Fan-in (second in FFN) | `RowParallelLinear` | Projection that gathers across ranks |
| QKV projection | `QKVParallelLinear` | Fused Q/K/V for self-attention |
| Single Q or K or V | `ColumnParallelLinear` | Separate projections (cross-attention) |
| Attention output | `RowParallelLinear` | Output projection after attention |
| Must not shard | `ReplicatedLinear` | Layers that must stay replicated |

### MLP Block (Up-Down Pattern)

```python
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear, RowParallelLinear,
)

class TPFeedForward(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.fc1 = ColumnParallelLinear(dim, ffn_dim, bias=False, return_bias=False)
        self.fc2 = RowParallelLinear(
            ffn_dim, dim, bias=False,
            input_is_parallel=True,  # Input already sharded from fc1
            return_bias=False,
        )

    def forward(self, x):
        x, _ = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x, _ = self.fc2(x)
        return x
```

### Attention Block (QKV-Out Pattern)

```python
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm_omni.diffusion.attention.layer import Attention

class TPSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads=None):
        super().__init__()
        num_kv_heads = num_kv_heads or num_heads
        self.head_dim = dim // num_heads

        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=False,
            return_bias=False,
        )
        self.to_out = RowParallelLinear(
            dim, dim, bias=False,
            input_is_parallel=True,
            return_bias=False,
        )
        self.attn = Attention(
            num_heads=self.to_qkv.num_heads,      # Local heads per GPU
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim ** 0.5),
            causal=False,
            num_kv_heads=self.to_qkv.num_kv_heads,  # Local KV heads per GPU
            role="self",
        )

    def forward(self, x):
        qkv, _ = self.to_qkv(x)
        q, k, v = qkv.split(
            [self.to_qkv.num_heads * self.head_dim,
             self.to_qkv.num_kv_heads * self.head_dim,
             self.to_qkv.num_kv_heads * self.head_dim],
            dim=-1,
        )
        B, S, _ = x.shape
        q = q.view(B, S, self.to_qkv.num_heads, self.head_dim)
        k = k.view(B, S, self.to_qkv.num_kv_heads, self.head_dim)
        v = v.view(B, S, self.to_qkv.num_kv_heads, self.head_dim)
        out = self.attn(q, k, v)
        out = out.reshape(B, S, -1)
        out, _ = self.to_out(out)
        return out
```

### QKV Fusion in load_weights

When you fuse separate Q/K/V into `QKVParallelLinear`, map diffusers' separate weight names:

```python
stacked_params_mapping = [
    ("to_qkv", "to_q", "q"),
    ("to_qkv", "to_k", "k"),
    ("to_qkv", "to_v", "v"),
]

def load_weights(self, weights):
    params = dict(self.named_parameters())
    loaded = set()
    for name, tensor in weights:
        for fused_name, orig_name, shard_id in stacked_params_mapping:
            if orig_name in name:
                name = name.replace(orig_name, fused_name)
                param = params[name]
                param.weight_loader(param, tensor, shard_id)
                loaded.add(name)
                break
        else:
            if name in params:
                param = params[name]
                if hasattr(param, "weight_loader"):
                    param.weight_loader(param, tensor)
                else:
                    default_weight_loader(param, tensor)
                loaded.add(name)
    return loaded
```

### RMSNorm with TP

When RMSNorm sits between TP-sharded dimensions, use `DistributedRMSNorm` — it computes global RMS via all-reduce across TP ranks. See the Wan2.2 implementation for the pattern.

### TP Constraints

- `num_heads % tp_size == 0`
- `num_kv_heads % tp_size == 0`
- Use `self.to_qkv.num_heads` (local per-GPU count), not total heads, for split sizes

### Testing TP

```bash
python text_to_image.py --model Your-org/your-model \
  --tensor-parallel-size 2 --output "tp_test.png"
```

**Verify**: speedup, memory reduction proportional to TP size, quality matches single-GPU.

### Reference implementations

| Model | Path |
|-------|------|
| Z-Image | `vllm_omni/diffusion/models/z_image/z_image_transformer.py` |
| FLUX | `vllm_omni/diffusion/models/flux/flux_transformer.py` |
| Qwen-Image | `vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py` |

---

## Sequence Parallelism (SP / USP)

SP splits sequence tokens across GPUs using Ulysses (all-to-all) or Ring (P2P) communication. It is applied non-intrusively via the `_sp_plan` dict — no changes to `forward()` logic.

### Approach 1: Non-Intrusive `_sp_plan` (Recommended)

The framework automatically registers hooks to shard inputs and gather outputs at `nn.Module` boundaries.

#### Step 1: Identify module boundaries

Find where tensors need sharding/gathering:

```python
class MyTransformer(nn.Module):
    def __init__(self):
        self.patch_embed = PatchEmbed()    # Before blocks
        self.pos_embed = RoPE()            # RoPE may need splitting
        self.blocks = nn.ModuleList([...]) # Blocks process sharded x
        self.norm_out = LayerNorm()
        self.proj_out = Linear()           # Gather after this

    def forward(self, x):
        x = self.patch_embed(x)
        pos = self.pos_embed(x)
        for block in self.blocks:
            x = block(x, pos)
        x = self.norm_out(x)
        return self.proj_out(x)
```

#### Step 2: Handle inline operations

`_sp_plan` hooks only work at `nn.Module` boundaries. Inline ops like `torch.cat()` must be extracted into submodules:

```python
# BAD: Inline — hooks can't intercept
unified = torch.cat([x, cap_feats], dim=1)

# GOOD: Extract into submodule
class UnifiedPrepare(nn.Module):
    def forward(self, x, cap_feats):
        return torch.cat([x, cap_feats], dim=1)

self.unified_prepare = UnifiedPrepare()
unified = self.unified_prepare(x, cap_feats)
```

Common cases: `torch.cat()`, `pad_sequence()`, `tensor.reshape()`, complex preprocessing.

#### Step 3: Write `_sp_plan`

**Pattern 1: Shard at first block, gather at output** (most common)

```python
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput, SequenceParallelOutput,
)

class StandardTransformer(nn.Module):
    _sp_plan = {
        "blocks.0": {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
        },
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }
```

**Pattern 2: Shard RoPE outputs separately**

```python
class TransformerWithRoPE(nn.Module):
    _sp_plan = {
        "rope": {
            0: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),
            1: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),
        },
        "blocks.0": {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
        },
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }
```

**Pattern 3: Dual-stream (shard image, replicate text)**

```python
class DualStreamTransformer(nn.Module):
    _sp_plan = {
        "rope_preparer": {
            2: SequenceParallelInput(split_dim=0, expected_dims=2, split_output=True),
            3: SequenceParallelInput(split_dim=0, expected_dims=2, split_output=True),
        },
        "transformer_blocks.0": {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
        },
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }
```

### API Reference

**SequenceParallelInput**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `split_dim` | int | Dimension to split (usually 1 for sequence) |
| `expected_dims` | int/None | Expected tensor rank for validation |
| `split_output` | bool | `False`: shard input params; `True`: shard output tensors |
| `auto_pad` | bool | Auto-pad if sequence not divisible by world_size |

**SequenceParallelOutput**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `gather_dim` | int | Dimension to gather (usually 1 for sequence) |
| `expected_dims` | int/None | Expected tensor rank for validation |

**Module naming**:

| Key | Meaning |
|-----|---------|
| `"blocks.0"` | First element of ModuleList |
| `"blocks.*"` | All elements of ModuleList |
| `"rope"` | Named submodule |

**Dictionary value types**:

| Key type | split_output | Description |
|----------|-------------|-------------|
| `"param_name"` (str) | False | Shard input parameter by name |
| `0, 1, ...` (int) | True | Shard output tuple by index |

### Approach 2: Intrusive Modification (Complex Cases)

For dynamic sharding logic that can't be expressed via `_sp_plan`:

```python
from vllm_omni.diffusion.distributed.sp_sharding import sp_shard, sp_gather

def forward(self, hidden_states, ...):
    if self.parallel_config.sequence_parallel_size > 1:
        hidden_states = sp_shard(hidden_states, dim=1)
    for block in self.blocks:
        hidden_states = block(hidden_states)
    if self.parallel_config.sequence_parallel_size > 1:
        hidden_states = sp_gather(hidden_states, dim=1)
    return hidden_states
```

Use intrusive modification as a last resort — `_sp_plan` is preferred for maintainability.

### UAA Mode (Experimental)

`ulysses_mode="advanced_uaa"` handles arbitrary sequence lengths and head counts that aren't divisible by `ulysses_degree`. Uses variable all-to-all split sizes and temporary head padding.

### Combining SP methods

Ulysses and Ring can be combined: `ulysses_degree × ring_degree = total SP GPUs`.

```python
DiffusionParallelConfig(ulysses_degree=2, ring_degree=2)  # 4 GPUs total
```

### Testing SP

```bash
# Offline
python text_to_image.py --model Your-model --ulysses-degree 2

# Online serving
vllm serve Your-model --omni --usp 2
```

### Reference implementations

| Model | Path |
|-------|------|
| Qwen-Image | `vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py` |
| Wan2.2 | `vllm_omni/diffusion/models/wan2_2/wan2_2_transformer.py` |
| Z-Image | `vllm_omni/diffusion/models/z_image/z_image_transformer.py` |

---

## CFG Parallelism

Distributes positive/negative Classifier-Free Guidance branches across 2 GPUs.

### Implementation

Inherit `CFGParallelMixin` and implement `diffuse()`:

```python
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin

class YourPipeline(nn.Module, CFGParallelMixin):
    def diffuse(self, latents, timesteps, prompt_embeds, negative_embeds,
                do_true_cfg, true_cfg_scale, **kwargs):
        for i, t in enumerate(timesteps):
            positive_kwargs = {
                "hidden_states": latents,
                "encoder_hidden_states": prompt_embeds,
                "timestep": t,
            }
            negative_kwargs = {
                "hidden_states": latents,
                "encoder_hidden_states": negative_embeds,
                "timestep": t,
            } if do_true_cfg else None

            noise_pred = self.predict_noise_maybe_with_cfg(
                do_true_cfg=do_true_cfg,
                true_cfg_scale=true_cfg_scale,
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
            )
            latents = self.scheduler_step_maybe_with_cfg(
                noise_pred, t, latents, do_true_cfg
            )
        return latents
```

### Customization hooks

| Method | Override when |
|--------|-------------|
| `predict_noise()` | Non-standard transformer call (e.g., dual-transformer like Wan2.2) |
| `cfg_normalize_function()` | Custom normalization (e.g., LongCat with clamping) |
| `combine_cfg_noise()` | Multi-output models (e.g., video + audio: CFG on video, positive-only on audio) |

**Custom predict_noise** (Wan2.2 — selects active transformer):

```python
def predict_noise(self, current_model=None, **kwargs):
    if current_model is None:
        current_model = self.transformer
    return current_model(**kwargs)[0]
```

**Custom combine_cfg_noise** (multi-output):

```python
def combine_cfg_noise(self, positive_pred, negative_pred, scale, normalize):
    video_pos, audio_pos = positive_pred
    video_neg, audio_neg = negative_pred
    video_combined = super().combine_cfg_noise(video_pos, video_neg, scale, normalize)
    return (video_combined, audio_pos)
```

### Composite scheduler for multi-output

When each output has its own schedule:

```python
class VideoAudioScheduler:
    def __init__(self, video_scheduler, audio_scheduler):
        self.video_scheduler = video_scheduler
        self.audio_scheduler = audio_scheduler

    def step(self, noise_pred, t, latents, return_dict=False, generator=None):
        video_out = self.video_scheduler.step(
            noise_pred[0], t[0], latents[0], return_dict=False, generator=generator
        )[0]
        audio_out = self.audio_scheduler.step(
            noise_pred[1], t[1], latents[1], return_dict=False, generator=generator
        )[0]
        return ((video_out, audio_out),)
```

### Testing CFG Parallel

```bash
python text_to_image.py --model Your-model \
  --cfg-parallel-size 2 --cfg-scale 4.0 \
  --negative-prompt "ugly, unclear"
```

**Constraint**: `guidance_scale > 1.0` and negative prompt must be provided.

### Reference implementations

| Model | Path |
|-------|------|
| Qwen-Image | `vllm_omni/diffusion/models/qwen_image/cfg_parallel.py` |
| Wan2.2 | `vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py` |
| Mixin base | `vllm_omni/diffusion/distributed/cfg_parallel.py` |

---

## HSDP (Hybrid Sharded Data Parallel)

Shards model weights across GPUs using PyTorch FSDP2. Reduces per-GPU VRAM without changing computation.

### Implementation

Add `_hsdp_shard_conditions` to the transformer class:

```python
class YourTransformer(nn.Module):
    @staticmethod
    def _is_transformer_block(name: str, module) -> bool:
        return "blocks" in name and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_transformer_block]
```

For MoE models, add additional conditions:

```python
class MoETransformer(nn.Module):
    @staticmethod
    def _is_transformer_block(name, module):
        return "blocks" in name and name.split(".")[-1].isdigit()

    @staticmethod
    def _is_moe_expert(name, module):
        return "experts" in name and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_transformer_block, _is_moe_expert]
```

A module is sharded if **any** condition returns `True`.

### Constraints

- Cannot combine with Tensor Parallelism
- For standalone HSDP (no other parallelism), `hsdp_shard_size` must be specified explicitly
- Can combine with SP: HSDP reduces memory while SP distributes sequence

### Testing HSDP

```python
from vllm_omni.diffusion.data import DiffusionParallelConfig

parallel_config = DiffusionParallelConfig(use_hsdp=True, hsdp_shard_size=8)
omni = Omni(model="your-model", parallel_config=parallel_config)
```

Or CLI:

```bash
vllm serve Your-model --omni --use-hsdp
```

**Verify**: logs show "HSDP Inference: replicate_size=..., shard_size=..." and "Sharded N modules + root". Check VRAM reduction.

### Reference implementations

| Model | Path |
|-------|------|
| Wan2.2 | `vllm_omni/diffusion/models/wan2_2/wan2_2_transformer.py` |
| HSDP Core | `vllm_omni/diffusion/distributed/hsdp.py` |

---

## VAE Patch Parallelism

Shards VAE decode spatially across ranks using tiling:

```bash
python text_to_image.py --model Your-model --vae-patch-parallel-size 4
```

Auto-enables `--vae-use-tiling`. Uses `DistributedAutoencoderKLWan` or similar distributed VAE. Set `vae_patch_parallel_size` in `DiffusionParallelConfig`.

---

## Combining Parallelism Methods

Common multi-GPU recipes:

```bash
# 4 GPUs: CFG (2) × Ulysses (2)
python text_to_image.py --model Qwen/Qwen-Image \
  --cfg-parallel-size 2 --ulysses-degree 2

# 8 GPUs: Ulysses (4) × Ring (2) + VAE patch (8)
python text_to_video.py --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --ulysses-degree 4 --ring-degree 2 --vae-patch-parallel-size 8

# 2 GPUs: HSDP + Ulysses (cannot combine HSDP with TP)
vllm serve Your-model --omni --use-hsdp --usp 2
```

## Discovering Parallelism Support

Check which parallelism methods a model supports:

| Check | How |
|-------|-----|
| **Ulysses / Ring SP** | Transformer defines `_sp_plan`. Search: `grep -r '_sp_plan' vllm_omni/diffusion/models/` |
| **CFG Parallel** | Pipeline inherits `CFGParallelMixin`. Search: `grep -r 'CFGParallelMixin' vllm_omni/diffusion/models/` |
| **TP** | Uses `ColumnParallelLinear` / `QKVParallelLinear`. Search: `grep -r 'ParallelLinear\|QKVParallel' vllm_omni/diffusion/models/<model>/` |
| **HSDP** | Transformer defines `_hsdp_shard_conditions`. Search: `grep -r '_hsdp_shard_conditions' vllm_omni/diffusion/models/` |

The canonical per-model support table is in `docs/user_guide/diffusion/parallelism_acceleration.md`.
