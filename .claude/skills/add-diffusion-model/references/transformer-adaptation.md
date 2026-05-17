# Transformer Adaptation Reference

## Adapting a Diffusers Transformer to vLLM-Omni

### Step-by-step Checklist

1. Copy the transformer class from diffusers source
2. Remove all mixin classes — inherit only from `nn.Module`
3. Replace attention dispatch with `vllm_omni.diffusion.attention.layer.Attention`
4. Replace logger with `vllm.logger.init_logger`
5. Add `od_config: OmniDiffusionConfig | None = None` to `__init__`
6. Remove training-only code (gradient checkpointing, dropout)
7. Add `load_weights()` method for weight loading from safetensors
8. Add class-level attributes for acceleration features

### Mixin Removal

Remove these diffusers mixins (and their imports):

```python
# Remove all of these:
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import AttentionModuleMixin
from diffusers.loaders import PeftAdapterMixin, FromOriginalModelMixin

# Replace:
class MyTransformer(ModelMixin, ConfigMixin, AttentionModuleMixin):
# With:
class MyTransformer(nn.Module):
```

Also remove `@register_to_config` decorators from `__init__`.

### Attention Replacement

The vLLM-Omni `Attention` layer wraps backend selection (FlashAttention, SDPA, SageAttn, etc.) and supports sequence parallelism hooks.

**QKV tensor shape must be `[batch, seq_len, num_heads, head_dim]`.**

#### Self-Attention Pattern

```python
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)

        self.attn = Attention(
            num_heads=num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim ** 0.5),
            causal=False,
            num_kv_heads=num_heads,
            role="self",
        )

    def forward(self, x, attn_mask=None):
        B, S, _ = x.shape
        q = self.to_q(x).view(B, S, self.num_heads, self.head_dim)
        k = self.to_k(x).view(B, S, self.num_heads, self.head_dim)
        v = self.to_v(x).view(B, S, self.num_heads, self.head_dim)

        attn_metadata = AttentionMetadata(attn_mask=attn_mask)
        out = self.attn(q, k, v, attn_metadata=attn_metadata)
        out = out.reshape(B, S, -1)
        return self.to_out(out)
```

**Declaring the attention role.** Always pass a `role` string to `Attention(...)`. It is the lookup key users target with `--diffusion-attention-config.per_role.<role>.backend` to swap kernels per site without touching model code.

| Convention | When to use |
|---|---|
| `role="self"` | Q/K/V come from the same hidden state |
| `role="cross"` | K/V come from a separate `encoder_hidden_states`; pair with `skip_sequence_parallel=True` if K/V is replicated across SP ranks |

For multi-modal sites that don't fit `self` / `cross`, use a dot-namespaced role and provide `role_category` so user config can fall back to the generic category:

```python
self.audio_to_video_attn = Attention(
    num_heads=num_heads, head_size=self.head_dim,
    softmax_scale=1.0 / (self.head_dim ** 0.5), causal=False,
    role="mymodel.audio_to_video",
    role_category="cross",
)
```

#### Fused QKV with TP (Advanced)

For tensor parallelism, use vLLM's parallel linear layers:

```python
from vllm.model_executor.layers.linear import (
    QKVParallelLinear, RowParallelLinear
)

class TPSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
        )
        self.to_out = RowParallelLinear(dim, dim)

        self.attn = Attention(
            num_heads=num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim ** 0.5),
            causal=False,
            num_kv_heads=num_heads,
            role="self",
        )
```

### Logger Replacement

```python
# Replace:
from diffusers.utils import logging
logger = logging.get_logger(__name__)

# With:
from vllm.logger import init_logger
logger = init_logger(__name__)
```

### Custom Layers from vLLM-Omni

Available utility layers:

```python
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm_omni.diffusion.layers.rope import RotaryEmbedding
from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNorm
```

### Config Support

```python
from vllm_omni.diffusion.data import OmniDiffusionConfig

class MyTransformer(nn.Module):
    def __init__(self, *, od_config=None, num_layers=28, hidden_size=3072, **kwargs):
        super().__init__()
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config if od_config else None
        # ... build layers
```

The transformer config values come from `model_index.json` → `config.json` in the transformer subfolder. The pipeline uses `get_transformer_config_kwargs(od_config.tf_model_config, TransformerClass)` to filter config keys to match the `__init__` signature.

### Weight Loading

The `load_weights` method receives an iterable of `(name, tensor)` from safetensors files, with the prefix (e.g., `"transformer."`) already stripped by the loader.

```python
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

class MyTransformer(nn.Module):
    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            # Optional: remap names from diffusers to vllm-omni naming
            # e.g., "ff.net.0.proj" -> "ff.net_0.proj"

            if name in params:
                param = params[name]
                if hasattr(param, "weight_loader"):
                    param.weight_loader(param, tensor)
                else:
                    default_weight_loader(param, tensor)
                loaded.add(name)
        return loaded
```

#### QKV Fusion in load_weights

If you fused separate Q/K/V into a `QKVParallelLinear`, you need to map diffusers' separate weight names:

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
            # Normal loading
            ...
    return loaded
```

### Class-Level Attributes for Features

```python
class MyTransformer(nn.Module):
    # torch.compile: list block class names that repeat and can be compiled
    _repeated_blocks = ["MyTransformerBlock"]

    # CPU offload: attribute name of the nn.ModuleList containing blocks
    _layerwise_offload_blocks_attr = "blocks"

    # LoRA: mapping of fused param names to original param names
    packed_modules_mapping = {"to_qkv": ["to_q", "to_k", "to_v"]}

    # Sequence parallelism plan (advanced — add after basic impl works)
    _sp_plan = {
        "blocks.0": SequenceParallelInput(split_dim=1),
        "proj_out": SequenceParallelOutput(gather_dim=1),
    }
```
