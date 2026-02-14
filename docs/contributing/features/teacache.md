# Support TeaCache

This section describes how to add TeaCache to a diffusion transformer model. We use the Qwen-Image transformer as the reference implementation.

---

## Table of Contents

- [Overview](#overview)
- [Step-by-Step Implementation](#step-by-step-implementation)
- [Customization](#customization)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Reference Implementations](#reference-implementations)
- [Summary](#summary)

---

## Overview

### What is TeaCache?

TeaCache speeds up diffusion inference by caching transformer block computations when consecutive timesteps are similar. It provides **1.5x-2.0x speedup** with minimal quality loss.

The core insight is that the modulated input (after normalization and timestep conditioning) changes gradually across timesteps. By measuring the L1 distance between consecutive modulated inputs and comparing it to a threshold, TeaCache decides whether to execute the full transformer blocks or reuse the cached residual from the previous step.

vLLM-omni provides a **hook-based** TeaCache system that requires **zero changes to model code**. The hook completely intercepts the transformer's forward pass and implements adaptive caching transparently. This design allows easy integration with any transformer model by simply writing an extractor function.

### Architecture

The TeaCache system consists of three main components:

| Component | Purpose | Location |
|-----------|---------|----------|
| [`CacheContext`](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/cache/#vllm_omni.diffusion.cache.CacheContext) | Dataclass containing model-specific information for caching | `vllm_omni/diffusion/cache/teacache/context.py` |
| [`EXTRACTOR_REGISTRY`](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/cache/teacache/extractors/#vllm_omni.diffusion.cache.teacache.extractors.EXTRACTOR_REGISTRY) | Maps transformer class names to extractor functions | `vllm_omni/diffusion/cache/teacache/extractors.py` |
| [`TeaCacheConfig`](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/cache/#vllm_omni.diffusion.cache.TeaCacheConfig) | Configuration including thresholds and polynomial coefficients | `vllm_omni/diffusion/cache/teacache/config.py` |

The hook handles all caching logic automatically, including:

- CFG-aware state management (separate states for positive/negative branches)
- CFG-parallel compatibility
- L1 distance computation with polynomial rescaling
- Residual caching and reuse


---

## Step-by-Step Implementation

To add TeaCache support for a new model, you need to:

1. Write an **extractor function** that returns a `CacheContext` object
2. Register the extractor in the `EXTRACTOR_REGISTRY`
3. Add model-specific polynomial coefficients to `TeaCacheConfig`

### Step 1: Model-Specific Preprocessing

Extract and process model inputs. This typically involves:
- Embedding image/latent inputs
- Processing text encoder outputs (if dual-stream)
- Creating timestep embeddings
- Computing positional embeddings

**Example (Qwen-Image):**

```python
def extract_qwen_context(
    module: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_mask: torch.Tensor,
    timestep: torch.Tensor,
    img_shapes: torch.Tensor,
    txt_seq_lens: torch.Tensor,
    guidance: torch.Tensor | None = None,
    **kwargs: Any,
) -> CacheContext:
    # Validate model structure
    if not hasattr(module, "transformer_blocks") or len(module.transformer_blocks) == 0:
        raise ValueError("Module must have transformer_blocks")

    # Preprocessing: embed inputs
    hidden_states = module.img_in(hidden_states)
    timestep = timestep.to(device=hidden_states.device, dtype=hidden_states.dtype)
    encoder_hidden_states = module.txt_norm(encoder_hidden_states)
    encoder_hidden_states = module.txt_in(encoder_hidden_states)

    # Create timestep embedding
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    temb = (
        module.time_text_embed(timestep, hidden_states)
        if guidance is None
        else module.time_text_embed(timestep, guidance, hidden_states)
    )

    # Compute position embeddings
    image_rotary_emb = module.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)
```

### Step 2: Extract Modulated Input

The modulated input is used for cache decisions. Extract it from the **first transformer block** after normalization and modulation.

**Example (Qwen-Image):**

```python
    # Extract modulated input from first transformer block
    block = module.transformer_blocks[0]
    img_mod_params = block.img_mod(temb)
    img_mod1, _ = img_mod_params.chunk(2, dim=-1)
    img_modulated, _ = block.img_norm1(hidden_states, img_mod1)
```

**Key Points:**

- Use the **first block** to extract modulated input early
- Apply the same normalization and modulation as the actual forward pass
- The tensor should represent the processed features that will change across timesteps

### Step 3: Define Transformer Execution

Create a callable that executes all transformer blocks. This encapsulates the main computation loop.

**Example (Qwen-Image dual-stream):**

```python
    def run_transformer_blocks():
        """Execute all Qwen transformer blocks."""
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
        return (h, e)  # Return both image and text hidden states
```

**Example (Single-stream model like Flux):**

```python
    def run_transformer_blocks():
        """Execute all Flux transformer blocks."""
        h = hidden_states

        for block in module.transformer_blocks:
            h = block(h, temb=temb)
        return (h,)  # Return only image hidden states
```

**Key Points:**

- Return format:
- For single-stream models: return `(hidden_states,)`
- For dual-stream models: return `(hidden_states, encoder_hidden_states)`

### Step 4: Define Postprocessing

Create a callable that applies final transformations to produce the model output.

**Example (Qwen-Image):**

```python
    return_dict = kwargs.get("return_dict", True)

    def postprocess(h):
        """Apply Qwen-specific output postprocessing."""
        h = module.norm_out(h, temb)
        output = module.proj_out(h)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
```

### Step 5: Return CacheContext

Package all information into a `CacheContext` object.

```python
    return CacheContext(
        modulated_input=img_modulated,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,  # or None for single-stream
        temb=temb,
        run_transformer_blocks=run_transformer_blocks,
        postprocess=postprocess,
    )
```

**CacheContext Fields:**

| Field | Type | Purpose |
|-------|------|---------|
| `modulated_input` | `torch.Tensor` | Tensor used for cache decision (similarity comparison) |
| `hidden_states` | `torch.Tensor` | Current hidden states (will be modified by caching) |
| `encoder_hidden_states` | `torch.Tensor | None` | Encoder states for dual-stream models, `None` for single-stream |
| `temb` | `torch.Tensor` | Timestep embedding tensor |
| `run_transformer_blocks` | `Callable[[], tuple]` | Executes transformer blocks, returns `(hidden_states, [encoder_hidden_states])` |
| `postprocess` | `Callable[[torch.Tensor], Any]` | Applies final transformations to produce model output |
| `extra_states` | `dict | None` | Optional dict for additional model-specific state |

### Step 6: Register the Extractor

Add your extractor to the `EXTRACTOR_REGISTRY` in `vllm_omni/diffusion/cache/teacache/extractors.py`:

```python
EXTRACTOR_REGISTRY: dict[str, Callable] = {
    "QwenImageTransformer2DModel": extract_qwen_context,
    "Bagel": extract_bagel_context,
    "ZImageTransformer2DModel": extract_zimage_context,
    "YourModelTransformer2DModel": extract_your_model_context,  # Add here
}
```

**Key:** Use the transformer class name (`module.__class__.__name__`)

### Step 7: Add Model Coefficients

Add polynomial rescaling coefficients to `vllm_omni/diffusion/cache/teacache/config.py`:

```python
_MODEL_COEFFICIENTS = {
    "QwenImageTransformer2DModel": [
        -4.50000000e02,
        2.80000000e02,
        -4.50000000e01,
        3.20000000e00,
        -2.00000000e-02,
    ],
    "YourModelTransformer2DModel": [  # Add your model's coefficients
        # 5 polynomial coefficients (can reuse similar model's coefficients initially)
    ],
}
```


**Initial approach:** Start with coefficients from a similar model architecture, then tune empirically following [Customization](#customization) section.

---

## Customization

### Coefficient Estimation

While you can start with coefficients from a similar model architecture, estimating custom coefficients for your specific model typically improves TeaCache performance.

**Why Estimate Coefficients?**

The polynomial coefficients rescale L1 distances between consecutive modulated inputs to better predict when cached residuals can be reused. Model-specific coefficients account for:

- Architecture differences (layer count, hidden size, attention patterns)
- Training data characteristics
- Noise prediction behavior across timesteps

| Approach | Performance | Effort |
|----------|-------------|--------|
| Using defaults from similar model | Within 5-10% of optimal | Low |
| Estimating custom coefficients | Best performance | Medium |

#### Implement Data Collection Adapter

Add an adapter in `vllm_omni/diffusion/cache/teacache/coefficient_estimator.py`:

```python
class YourModelAdapter:
    """Adapter for coefficient estimation on your model."""

    @staticmethod
    def load_pipeline(model_path: str, device: str, dtype: torch.dtype) -> Any:
        """Load your diffusion pipeline."""
        from your_model_package import YourModelPipeline

        pipeline = YourModelPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
        )
        pipeline = pipeline.to(device)
        return pipeline

    @staticmethod
    def get_transformer(pipeline: Any) -> tuple[Any, str]:
        """Extract transformer from pipeline."""
        return pipeline.transformer, "YourTransformer2DModel"

    @staticmethod
    def install_hook(transformer: Any, hook: DataCollectionHook) -> None:
        """Install data collection hook on transformer."""
        from vllm_omni.diffusion.hooks import HookRegistry

        registry = HookRegistry.get_or_create(transformer)
        registry.register_hook(hook._HOOK_NAME, hook)


# Register your adapter
_MODEL_ADAPTERS["YourModel"] = YourModelAdapter
```

#### Collect Data and Estimate

```python
from vllm_omni.diffusion.cache.teacache.coefficient_estimator import (
    TeaCacheCoefficientEstimator,
)
from datasets import load_dataset
from tqdm import tqdm

# Initialize estimator
estimator = TeaCacheCoefficientEstimator(
    model_path="/path/to/your/model",
    model_type="YourModel",
)

# Load diverse prompts (paper recommends ~70 prompts)
dataset = load_dataset("nateraw/parti-prompts", split="train")
prompts = dataset["Prompt"][:70]

# Collect data
for prompt in tqdm(prompts, desc="Collecting data"):
    estimator.collect_from_prompt(prompt=prompt, num_inference_steps=50)

# Estimate coefficients
coeffs = estimator.estimate(poly_order=4)
print(f"Estimated coefficients: {coeffs.tolist()}")
```

**Data Statistics Guide:**

| Metric | Good Range | Warning Signs |
|--------|------------|---------------|
| **Count** | 2000-5000+ | < 1000: too few prompts |
| **Input Diffs (x)** | 0.01-0.10 | Very small (<0.001): model may not modulate properly |
| **Output Diffs (y)** | Should correlate with x | No correlation: check extractor |
| **Coefficient magnitude** | -1e6 to 1e6 | > 1e8: numerical instability |

---

## Testing

After adding TeaCache support, test with:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="your-model-name",
    cache_backend="tea_cache",
    cache_config={
        "rel_l1_thresh": 0.2,
        "coefficients": [1.33e6, -1.69e5, 7.95e3, -1.64e2, 1.26],  # Your coefficients
    }
)

images = omni.generate(
    "a beautiful landscape",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

**Verify:**

1. **Check logs** - Look for TeaCache initialization messages
2. **Compare performance** - Measure speedup vs baseline (expect 1.5x-2.0x)
3. **Verify output quality** - Visually compare cached vs uncached outputs (should be nearly identical)

See more detailed examples in [user guide for teacache](../../user_guide/diffusion/teacache.md).

---

## Troubleshooting

### Issue: "Unknown model type"

**Symptoms:** Error message indicating the model type is not recognized when enabling TeaCache.

**Causes & Solutions:**

- **Extractor not registered:**

**Problem:** The transformer class name doesn't exist in `EXTRACTOR_REGISTRY`.

**Solution:** Check the class name and add to registry:
```python
# Check transformer class name
print(pipeline.transformer.__class__.__name__)

# Add to EXTRACTOR_REGISTRY
EXTRACTOR_REGISTRY["YourTransformer2DModel"] = extract_your_context
```

- **Transformer class name mismatch:**

**Solution:** Ensure the registry key matches exactly with `module.__class__.__name__`.

### Issue: "Cannot find coefficients"

**Symptoms:** Error when initializing TeaCache about missing model coefficients.

**Causes & Solutions:**

- **Missing coefficients in config:**

**Solution:** Add coefficients to `_MODEL_COEFFICIENTS` in `config.py`, or pass custom coefficients:
```python
omni = Omni(
    model="your-model",
    cache_backend="tea_cache",
    cache_config={"coefficients": [1.0, -0.5, 0.1, -0.01, 0.001]}
)
```

### Issue: Quality Degradation

**Symptoms:** Output images look noticeably different or have artifacts compared to baseline.

**Causes & Solutions:**

- **Threshold too high:**

**Problem:** `rel_l1_thresh` is too aggressive, causing cache reuse when outputs differ significantly.

**Solution:** Lower the threshold:
```python
cache_config={"rel_l1_thresh": 0.1}  # Try 0.1-0.2
```

- **Coefficients not tuned:**

**Solution:** Estimate model-specific coefficients using the coefficient estimation process described above.

---

## Reference Implementations

Complete examples in the codebase:

| Model | Path | Pattern | Notes |
|-------|------|---------|-------|
| **Qwen-Image** | `vllm_omni/diffusion/cache/teacache/extractors.py` | Dual-stream | `extract_qwen_context` |
| **Bagel** | `vllm_omni/diffusion/cache/teacache/extractors.py` | Omni model | `extract_bagel_context` |
| **TeaCache Core** | `vllm_omni/diffusion/cache/teacache/` | Base implementation | Hook and config |
| **Coefficient Estimator** | `vllm_omni/diffusion/cache/teacache/coefficient_estimator.py` | Estimation tool | Adapter pattern |

---

## Summary

Adding TeaCache support:

1. ✅ **Write extractor** - Create function returning `CacheContext` with model-specific preprocessing
2. ✅ **Register extractor** - Add to `EXTRACTOR_REGISTRY` with transformer class name
3. ✅ **Add coefficients** - Add polynomial coefficients to `_MODEL_COEFFICIENTS`
4. ✅ **Test** - Verify with `cache_backend="tea_cache"`
