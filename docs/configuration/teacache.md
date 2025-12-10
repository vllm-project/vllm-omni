# TeaCache Configuration Guide

TeaCache (Timestep Embedding Aware Cache) is an adaptive caching technique that speeds up diffusion model inference by reusing transformer block computations when consecutive timestep embeddings are similar. This guide explains how to enable and configure TeaCache for DiT (Diffusion Transformer) models in vLLM-Omni.

## Overview

TeaCache accelerates diffusion inference by:

1. **Extracting modulated inputs** from the first transformer block at each timestep
2. **Computing similarity** between consecutive timestep embeddings using relative L1 distance
3. **Applying polynomial rescaling** with model-specific coefficients to convert distance to a caching signal
4. **Making caching decisions**: When similarity is high (below threshold), reuse cached residuals instead of computing full transformer blocks
5. **Always computing** the first and last timesteps to ensure maximum quality

This typically provides **1.5x-2.0x speedup** with minimal quality loss, depending on the threshold configuration.

## Quick Start

### Basic Configuration

Enable TeaCache by setting `cache_adapter` to `"tea_cache"` in your `OmniDiffusionConfig`:

```python
from vllm_omni.diffusion.data import OmniDiffusionConfig

config = OmniDiffusionConfig(
    model="Qwen/Qwen-Image",
    model_class_name="QwenImagePipeline",
    cache_adapter="tea_cache",
    cache_config={
        "rel_l1_thresh": 0.2,
        "model_type": "QwenImagePipeline"
    }
)
```

### Using Environment Variable

You can also enable TeaCache via environment variable:

```bash
export DIFFUSION_CACHE_ADAPTER=tea_cache
```

Then create your config:

```python
config = OmniDiffusionConfig.from_kwargs(
    model="Qwen/Qwen-Image",
    model_class_name="QwenImagePipeline",
    cache_config={
        "rel_l1_thresh": 0.2,
        "model_type": "QwenImagePipeline"
    }
)
```

> **Note**: If `model_type` is not provided in `cache_config`, it will be automatically injected from `model_class_name` if available.

## Configuration Parameters

### Required Parameters

#### `model_type` (str)
Pipeline class name that matches your model. Must exactly match `OmniDiffusionConfig.model_class_name`.

**Supported values:**
- `"QwenImagePipeline"` - For Qwen-Image models
- `"FluxPipeline"` - For FLUX models (future support)

**Example:**
```python
cache_config={
    "model_type": "QwenImagePipeline"  # Must match model_class_name
}
```

### Optional Parameters

#### `rel_l1_thresh` (float, default: `0.2`)
Threshold for accumulated relative L1 distance. When the accumulated distance is below this threshold, cached residuals are reused instead of computing full transformer blocks.

**Recommended values** (based on TeaCache paper):
- `0.2` - **~1.5x speedup** with minimal quality loss (recommended)
- `0.4` - **~1.8x speedup** with slight quality loss
- `0.6` - **~2.0x speedup** with noticeable quality loss
- `0.8` - **~2.25x speedup** with significant quality loss

Higher thresholds lead to more aggressive caching and faster inference, but may reduce output quality.

**Example:**
```python
cache_config={
    "rel_l1_thresh": 0.2,  # Balanced speed/quality
    "model_type": "QwenImagePipeline"
}
```

#### `coefficients` (list[float], optional)
Polynomial coefficients for rescaling L1 distances. If not provided, model-specific defaults are used based on `model_type`.

**Default coefficients:**
- **QwenImagePipeline**: `[-4.5e+02, 2.8e+02, -4.5e+01, 3.2e+00, -2.0e-02]`
- **FluxPipeline**: `[4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01]`

The polynomial is evaluated as: `c[0]*x^4 + c[1]*x^3 + c[2]*x^2 + c[3]*x + c[4]` where `x` is the relative L1 distance.

**Example:**
```python
cache_config={
    "rel_l1_thresh": 0.2,
    "model_type": "QwenImagePipeline",
    "coefficients": [-4.5e+02, 2.8e+02, -4.5e+01, 3.2e+00, -2.0e-02]  # Optional
}
```

## Complete Example

### Python API Usage

```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import OmniDiffusionConfig

# Configure TeaCache
config = OmniDiffusionConfig(
    model="Qwen/Qwen-Image",
    model_class_name="QwenImagePipeline",
    cache_adapter="tea_cache",
    cache_config={
        "rel_l1_thresh": 0.2,  # 1.5x speedup with minimal quality loss
        "model_type": "QwenImagePipeline"
    }
)

# Initialize Omni with TeaCache enabled
omni = Omni(config)

# Generate image (cache state is automatically managed)
prompt = "A cat sitting on a windowsill"
outputs = omni.generate(prompt=prompt, num_inference_steps=50)

# Cache state is automatically reset between generations
```

### Stage Config YAML

For YAML-based stage configurations:

```yaml
model: "Qwen/Qwen-Image"
model_class_name: "QwenImagePipeline"
cache_adapter: "tea_cache"
cache_config:
  rel_l1_thresh: 0.2
  model_type: "QwenImagePipeline"
```

### Command-Line Usage

When using CLI tools, TeaCache can be enabled via environment variable:

```bash
export DIFFUSION_CACHE_ADAPTER=tea_cache

python your_script.py \
  --model Qwen/Qwen-Image \
  --model-class-name QwenImagePipeline \
  --cache-config '{"rel_l1_thresh": 0.2, "model_type": "QwenImagePipeline"}'
```

## How It Works

### Algorithm Overview

1. **First timestep**: Always compute full transformer (never cached)
2. **Intermediate timesteps**:
   - Extract modulated input from first transformer block
   - Compute relative L1 distance: `|current - previous| / |previous|`
   - Apply polynomial rescaling with model-specific coefficients
   - Accumulate rescaled distances
   - If accumulated distance < threshold: **reuse cached residual** (fast path)
   - If accumulated distance >= threshold: **compute full transformer** and cache new residual (slow path)
3. **Last timestep**: Always compute full transformer (never cached)

### State Management

TeaCache maintains separate state for each CFG (Classifier-Free Guidance) branch:
- Positive branch (conditioned)
- Negative branch (unconditioned)

State is automatically reset between inference runs. The cache adapter handles state management transparently.

## Performance Tuning

### Finding Optimal Threshold

Start with the default `rel_l1_thresh=0.2` and adjust based on your quality/speed requirements:

1. **For maximum quality**: Use `0.1-0.2`
2. **For balanced performance**: Use `0.2-0.4`
3. **For maximum speed**: Use `0.6-0.8` (may have noticeable quality loss)

### Benchmarking

To measure TeaCache performance:

```python
import time

# Without TeaCache
config_no_cache = OmniDiffusionConfig(
    model="Qwen/Qwen-Image",
    cache_adapter="none"
)

# With TeaCache
config_with_cache = OmniDiffusionConfig(
    model="Qwen/Qwen-Image",
    cache_adapter="tea_cache",
    cache_config={"rel_l1_thresh": 0.2, "model_type": "QwenImagePipeline"}
)

# Time inference
start = time.time()
outputs = omni.generate(prompt="test", num_inference_steps=50)
elapsed = time.time() - start
print(f"Inference time: {elapsed:.2f}s")
```

## Troubleshooting

### Error: "model_type must be provided in cache_config"

**Solution**: Ensure `model_type` is set in `cache_config` or matches `model_class_name`:

```python
config = OmniDiffusionConfig(
    model="Qwen/Qwen-Image",
    model_class_name="QwenImagePipeline",  # This will auto-inject into cache_config
    cache_adapter="tea_cache",
    cache_config={"rel_l1_thresh": 0.2}  # model_type auto-injected
)
```

### Error: "Unknown model type"

**Solution**: Verify `model_type` exactly matches a supported pipeline class name. Check available types:

```python
from vllm_omni.diffusion.cache.teacache.extractors import EXTRACTOR_REGISTRY
print("Supported model types:", list(EXTRACTOR_REGISTRY.keys()))
```

### Quality Degradation

**Solution**: Lower the `rel_l1_thresh` value:

```python
cache_config={
    "rel_l1_thresh": 0.1,  # More conservative caching
    "model_type": "QwenImagePipeline"
}
```

### State Not Resetting

**Solution**: Ensure you're creating a new inference run. The cache adapter automatically resets state, but if you're reusing the same transformer module, manually reset:

```python
# Cache adapter handles this automatically, but if needed:
if adapter is not None:
    adapter.reset(transformer)
```

## Supported Models

Currently supported models:

- **Qwen-Image** (`QwenImagePipeline`) - Fully supported
- **FLUX** (`FluxPipeline`) - Extractor available, full support coming soon

## Adding Support for New Models

To add TeaCache support for a new DiT model, you need to create an **extractor function** that knows how to extract the modulated input from your model's first transformer block. This is the only model-specific code required.

### Understanding Extractors

An extractor function encapsulates all model-specific logic needed for TeaCache:
1. **Preprocessing** - Prepare hidden states, timestep embeddings, encoder states
2. **Modulated input extraction** - Extract the tensor used for cache decision from the first transformer block
3. **Transformer execution** - Define how to run all transformer blocks
4. **Postprocessing** - Apply final normalization and projection

The extractor returns a `CacheContext` object containing all this information, allowing the generic TeaCache hook to work with any model.

### Step-by-Step Guide

#### Step 1: Understand Your Model's Architecture

Before writing an extractor, identify:

- **Transformer block attribute name**: `transformer_blocks`, `layers`, `blocks`, etc.
- **Single-stream vs dual-stream**: Does your model process image and text separately or together?
- **Modulation pattern**: How does the first block apply timestep conditioning?
- **Input/output format**: What are the forward pass arguments and return types?

#### Step 2: Create the Extractor Function

Create a function with this signature:

```python
from typing import Any, Union
import torch
import torch.nn as nn
from vllm_omni.diffusion.cache.teacache.extractors import CacheContext

def extract_yourmodel_context(
    module: nn.Module,
    hidden_states: torch.Tensor,
    # ... other model-specific arguments ...
    **kwargs: Any,
) -> CacheContext:
    """
    Extract cache context for YourModelTransformer.
    
    Args:
        module: Your transformer model instance
        hidden_states: Input hidden states tensor
        # ... document your model's specific arguments ...
        
    Returns:
        CacheContext with all information needed for generic caching
    """
    # Implementation below
```

#### Step 3: Implement Preprocessing

Apply model-specific preprocessing (input embedding, normalization, timestep embedding):

```python
# Example: Single-stream model (like FLUX)
hidden_states = module.x_embedder(hidden_states)
timestep = timestep.to(hidden_states.dtype) * 1000
if guidance is not None:
    guidance = guidance.to(hidden_states.dtype) * 1000
temb = module.time_text_embed(timestep, guidance, pooled_projections)
encoder_hidden_states = module.context_embedder(encoder_hidden_states)

# Example: Dual-stream model (like Qwen)
hidden_states = module.img_in(hidden_states)
encoder_hidden_states = module.txt_norm(encoder_hidden_states)
encoder_hidden_states = module.txt_in(encoder_hidden_states)
temb = module.time_text_embed(timestep, hidden_states)
```

#### Step 4: Extract Modulated Input

Extract the modulated input from the **first transformer block**. This is the tensor used for cache decision:

```python
# Get first transformer block
block = module.transformer_blocks[0]  # or module.layers[0], etc.

# Extract modulated input (pattern varies by model)
# Example 1: Using norm1 with timestep embedding
modulated_input, _, _, _, _ = block.norm1(hidden_states.clone(), emb=temb.clone())

# Example 2: Using modulation parameters (Qwen-style)
img_mod_params = block.img_mod(temb)
img_mod1, _ = img_mod_params.chunk(2, dim=-1)
img_normed = block.img_norm1(hidden_states)
modulated_input, _ = block._modulate(img_normed, img_mod1)

# Example 3: Using timestep embedding directly (CogVideoX-style)
modulated_input = temb.clone()
```

> **Important**: The modulated input should be extracted **before** running transformer blocks. It represents the state after normalization and modulation but before attention/MLP.

#### Step 5: Define Transformer Execution

Create a callable that executes all transformer blocks:

```python
# For single-stream models
def run_transformer_blocks():
    """Execute all transformer blocks."""
    h = hidden_states
    for block in module.transformer_blocks:
        h = block(h, temb=temb, encoder_hidden_states=encoder_hidden_states)
    return (h,)  # Return tuple with single element

# For dual-stream models
def run_transformer_blocks():
    """Execute all transformer blocks."""
    h = hidden_states
    e = encoder_hidden_states
    for block in module.transformer_blocks:
        e, h = block(
            hidden_states=h,
            encoder_hidden_states=e,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
        )
    return (h, e)  # Return tuple with both streams
```

#### Step 6: Define Postprocessing

Create a callable that applies final transformations:

```python
from diffusers.models.modeling_outputs import Transformer2DModelOutput

return_dict = kwargs.get("return_dict", True)

def postprocess(h):
    """Apply model-specific output postprocessing."""
    h = module.norm_out(h, temb)
    output = module.proj_out(h)
    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)
```

#### Step 7: Return CacheContext

Construct and return the `CacheContext`:

```python
return CacheContext(
    modulated_input=modulated_input,  # Tensor for cache decision
    hidden_states=hidden_states,  # Preprocessed hidden states
    encoder_hidden_states=encoder_hidden_states if dual_stream else None,
    temb=temb,  # Timestep embedding
    run_transformer_blocks=run_transformer_blocks,  # Execution callable
    postprocess=postprocess,  # Postprocessing callable
)
```

#### Step 8: Register the Extractor

Register your extractor function:

```python
from vllm_omni.diffusion.cache.teacache.extractors import register_extractor

# Register with pipeline class name (must match OmniDiffusionConfig.model_class_name)
register_extractor("YourModelPipeline", extract_yourmodel_context)
```

Or add directly to the registry:

```python
from vllm_omni.diffusion.cache.teacache.extractors import EXTRACTOR_REGISTRY

EXTRACTOR_REGISTRY["YourModelPipeline"] = extract_yourmodel_context
```

#### Step 9: Add Model Coefficients

Add polynomial coefficients for your model to `config.py`:

```python
# In vllm_omni/diffusion/cache/teacache/config.py
_MODEL_COEFFICIENTS = {
    # ... existing coefficients ...
    "YourModelPipeline": [
        1.0,  # c[0] * x^4
        -2.0,  # c[1] * x^3
        3.0,  # c[2] * x^2
        -4.0,  # c[3] * x
        5.0,  # c[4]
    ],
}
```

> **Note**: Coefficients are typically obtained from the TeaCache paper or through empirical tuning. If unavailable, you can use FLUX coefficients as a starting point and tune empirically.

### Complete Example: Single-Stream Model

Here's a complete example for a hypothetical single-stream model:

```python
from typing import Any
import torch
import torch.nn as nn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from vllm_omni.diffusion.cache.teacache.extractors import CacheContext, register_extractor

def extract_simplemodel_context(
    module: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    pooled_projections: torch.Tensor = None,
    guidance: torch.Tensor = None,
    return_dict: bool = True,
    **kwargs: Any,
) -> CacheContext:
    """
    Extract cache context for SimpleModelTransformer2DModel.
    
    This is a single-stream model similar to FLUX.
    """
    # Preprocessing
    hidden_states = module.x_embedder(hidden_states)
    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
        temb = module.time_text_embed(timestep, guidance, pooled_projections)
    else:
        temb = module.time_text_embed(timestep, pooled_projections)
    encoder_hidden_states = module.context_embedder(encoder_hidden_states)
    
    # Extract modulated input from first block
    block = module.transformer_blocks[0]
    modulated_input, _, _, _, _ = block.norm1(hidden_states.clone(), emb=temb.clone())
    
    # Define transformer execution
    def run_transformer_blocks():
        h = hidden_states
        e = encoder_hidden_states
        for block in module.transformer_blocks:
            e, h = block(h, e, temb=temb)
        return (h,)
    
    # Define postprocessing
    def postprocess(h):
        h = module.norm_out(h, temb)
        output = module.proj_out(h)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
    
    # Return context
    return CacheContext(
        modulated_input=modulated_input,
        hidden_states=hidden_states,
        encoder_hidden_states=None,  # Single-stream, no separate encoder stream
        temb=temb,
        run_transformer_blocks=run_transformer_blocks,
        postprocess=postprocess,
    )

# Register the extractor
register_extractor("SimpleModelPipeline", extract_simplemodel_context)
```

### Complete Example: Dual-Stream Model

Here's a complete example for a dual-stream model (similar to Qwen):

```python
from typing import Any, Union
import torch
import torch.nn as nn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from vllm_omni.diffusion.cache.teacache.extractors import CacheContext, register_extractor

def extract_dualstream_context(
    module: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_mask: torch.Tensor,
    timestep: Union[torch.Tensor, float, int],
    img_shapes: torch.Tensor,
    txt_seq_lens: torch.Tensor,
    guidance: torch.Tensor | None = None,
    attention_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> CacheContext:
    """
    Extract cache context for DualStreamTransformer2DModel.
    
    This is a dual-stream model that processes image and text separately.
    """
    # Preprocessing
    hidden_states = module.img_in(hidden_states)
    timestep = timestep.to(device=hidden_states.device, dtype=hidden_states.dtype)
    encoder_hidden_states = module.txt_norm(encoder_hidden_states)
    encoder_hidden_states = module.txt_in(encoder_hidden_states)
    
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    
    temb = (
        module.time_text_embed(timestep, hidden_states)
        if guidance is None
        else module.time_text_embed(timestep, guidance, hidden_states)
    )
    
    image_rotary_emb = module.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)
    
    # Extract modulated input from first block
    block = module.transformer_blocks[0]
    img_mod_params = block.img_mod(temb)
    img_mod1, _ = img_mod_params.chunk(2, dim=-1)
    img_normed = block.img_norm1(hidden_states)
    modulated_input, _ = block._modulate(img_normed, img_mod1)
    
    # Define transformer execution
    def run_transformer_blocks():
        h = hidden_states
        e = encoder_hidden_states
        for block in module.transformer_blocks:
            e, h = block(
                hidden_states=h,
                encoder_hidden_states=e,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=attention_kwargs,
            )
        return (h, e)
    
    # Define postprocessing
    return_dict = kwargs.get("return_dict", True)
    
    def postprocess(h):
        h = module.norm_out(h, temb)
        output = module.proj_out(h)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
    
    # Return context
    return CacheContext(
        modulated_input=modulated_input,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        run_transformer_blocks=run_transformer_blocks,
        postprocess=postprocess,
    )

# Register the extractor
register_extractor("DualStreamPipeline", extract_dualstream_context)
```

### Testing Your Extractor

After creating your extractor, test it:

```python
from vllm_omni.diffusion.cache.teacache.extractors import get_extractor

# Get your extractor
extractor = get_extractor("YourModelPipeline")

# Test with your model
ctx = extractor(
    module=your_transformer,
    hidden_states=test_hidden_states,
    # ... other arguments ...
)

# Validate the context
ctx.validate()

# Test transformer execution
outputs = ctx.run_transformer_blocks()
assert len(outputs) >= 1

# Test postprocessing
final_output = ctx.postprocess(outputs[0])
```

### Common Patterns

#### Pattern 1: Standard DiT with AdaLN

```python
# Modulation via AdaLN (Adaptive Layer Normalization)
block = module.transformer_blocks[0]
modulated_input, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(
    hidden_states.clone(), emb=temb.clone()
)
```

#### Pattern 2: Separate Modulation Parameters

```python
# Separate modulation (Qwen-style)
mod_params = block.img_mod(temb)
mod1, mod2 = mod_params.chunk(2, dim=-1)
normed = block.norm1(hidden_states)
modulated_input, _ = block._modulate(normed, mod1)
```

#### Pattern 3: Timestep Embedding Directly

```python
# Use timestep embedding directly (CogVideoX-style)
modulated_input = temb.clone()
```

### Tips and Best Practices

1. **Use `.clone()`** when extracting modulated input to avoid modifying the original tensor
2. **Match forward signature** - Your extractor should accept the same arguments as your model's forward method
3. **Handle optional arguments** - Use `**kwargs` to accept additional arguments that may be passed
4. **Validate early** - Call `ctx.validate()` during development to catch errors early
5. **Test with real data** - Test your extractor with actual model inputs, not just dummy tensors
6. **Check device/dtype** - Ensure all tensors are on the correct device and dtype
7. **Document your extractor** - Add clear docstrings explaining model-specific behavior

### Getting Help

If you encounter issues:

1. **Check existing extractors** - Look at `extract_qwen_context` in `extractors.py` as a reference
2. **Review CacheContext docs** - See the `CacheContext` class documentation for field requirements
3. **Test incrementally** - Test preprocessing, then extraction, then execution separately
4. **Compare with original** - If your model has TeaCache support elsewhere, compare implementations

### Contributing

Once you've successfully added support for a new model:

1. **Add to registry** - Submit a PR adding your extractor to `EXTRACTOR_REGISTRY`
2. **Add coefficients** - Include empirically-tuned polynomial coefficients
3. **Add tests** - Include unit tests for your extractor
4. **Update docs** - Add your model to the "Supported Models" section

## References

- [TeaCache Paper](https://liewfeng.github.io/TeaCache/)
- [ComfyUI-TeaCache Implementation](https://github.com/liewfeng/ComfyUI-TeaCache)
- [Original TeaCache Repository](https://github.com/liewfeng/TeaCache)

## API Reference

For detailed API documentation, see:
- [`TeaCacheConfig`](../../api/vllm_omni.diffusion.cache.teacache.config/#vllm_omni.diffusion.cache.teacache.config.TeaCacheConfig)
- [`TeaCacheAdapter`](../../api/vllm_omni.diffusion.cache.teacache.adapter/#vllm_omni.diffusion.cache.teacache.adapter.TeaCacheAdapter)
- [`setup_cache`](../../api/vllm_omni.diffusion.cache.apply/#vllm_omni.diffusion.cache.apply.setup_cache)

