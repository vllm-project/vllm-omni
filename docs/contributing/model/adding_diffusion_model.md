# Adding a Diffusion Model to vLLM-Omni

This guide walks you through adding a new diffusion model to vLLM-Omni. We use **Qwen-Image** as the primary example, with references to other models (LongCat, Flux, Wan2.2) to illustrate different patterns.


---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Basic Implementation](#basic-implementation)
4. [Advanced Features](#advanced-features)
5. [Troubleshooting](#troubleshooting)
6. [Pull Request Checklist](#pull-request-checklist)
7. [Reference Implementations](#reference-implementations)
8. [Summary](#summary)

---

## Overview

vLLM-Omni's diffusion inference follows this architecture:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/vllm-omni-diffusion-flow.png">
    <img alt="Diffusion Flow" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/vllm-omni-diffusion-flow.png" width=55%>
  </picture>
</p>

**Key Components:**

1. **Request Handling:** User prompts → `OmniDiffusionRequest`
2. **Diffusion Engine:**  Request →  Preprocessing (Optional) → Pipeline execution -> Post-processing
3. **Pipeline Execution:** Request → Encode prompt → Diffusion steps → Vae decode


## Directory Structure

Organize your model files following this structure:

```
vllm_omni/
└── diffusion/
    ├── registry.py                          # ← Register your model here
    ├── request.py                           # Request data structures
    └── models/
        └── your_model_name/                 # ← Create this directory
            ├── __init__.py                  # Export pipeline and transformer
            ├── pipeline_xxx.py              # Pipeline implementation
            └── xxx_transformer.py           # Transformer implementation
```

**Naming Conventions:**

- **Model directory:** `your_model_name` (lowercase, underscores),  e.g., `qwen_image`, `flux`, `longcat_image`, `wan2_2`
- **Pipeline file:** `pipeline_xxx.py` where `xxx` describes the task, e.g., `pipeline_qwen_image.py`, `pipeline_qwen_image_edit.py`
- **Transformer file:** `xxx_transformer.py` matching transformer class name, e.g.,  `qwen_image_transformer.py`, `flux_transformer.py`

---

## Basic Implementation

This section covers the minimal steps to get a model working in vLLM-Omni with basic features (online/offline serving, batch requests).

### Step 1: Adapt Transformer Model

The transformer is the core denoising network. Start by copying the transformer implementation from Diffusers and making these adaptations.


#### 1.1: Remove Diffusers Mixins

Diffusers' `Mixin` classes are not needed in vLLM-Omni. Remove them:

```diff
# Before (Diffusers)
- from diffusers.models.modeling_utils import ModelMixin
- from diffusers.models.attention_processor import AttentionModuleMixin

- class YourModelTransformer2DModel(ModelMixin, AttentionModuleMixin):
+ class YourModelTransformer2DModel(nn.Module):
    """Your transformer model."""
```

**Example mixins to remove:**

- `ModelMixin` - Weight loading utilities (vLLM-Omni has its own weight loader)
- `AttentionModuleMixin` - Attention processors (using vLLM-Omni's Attention layer instead)
- `ConfigMixin` - Config management (not needed)
- `PeftAdapterMixin` - Parameter efficient finetune utilities (not needed)

#### 1.2: Replace Attention Implementation

**The most important adaptation:** Replace Diffusers' attention with vLLM-Omni's optimized `Attention` layer.

**Before (Diffusers):**
```python
from diffusers.models.attention_processor import dispatch_attention_fn

class YourAttentionBlock(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states=None, ...):
        ...
        hidden_states = dispatch_attention_fn(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )
```

**After (vLLM-Omni):**
```python
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

class YourAttentionBlock(nn.Module):
    def __init__(self, ...):
        super().__init__()

        # Initialize vLLM-Omni's Attention layer
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim ** 0.5),
            causal=False,  # Diffusion models typically use bidirectional attention
            num_kv_heads=self.num_kv_heads,
        )

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, ...):
        ...
        # Create attention metadata
        attn_metadata = AttentionMetadata(attn_mask=attention_mask)
        hidden_states = self.attn(query, key, value, attn_metadata=attn_metadata)

```

**Key Points:**

- **Attention layer initialization:** Done in `__init__`, not per-forward
- **Tensor shapes:** vLLM-Omni `Attention` expects QKV to have `[B, seq, num_heads, head_dim]` shape
- **AttentionMetadata:** Wraps attention mask and other metadata

**Attention backends:** vLLM-Omni automatically selects the attention backend given the environmental variable `DIFFUSION_ATTENTION_BACKEND`. The default attention backend is `FLASH_ATTN` for diffusion models.

#### 1.3: Replace Imports and Utilities

**Logger:**
```diff
- from diffusers.utils import logging
- logger = logging.get_logger(__name__)

+ from vllm.logger import init_logger
+ logger = init_logger(__name__)
```

**Custom layers from vLLM and vLLM-Omni (if needed):**

```python
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm_omni.diffusion.layers.rope import RotaryEmbedding
from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNorm
```

#### 1.4: Remove Training-Only Code

Remove code that's only needed for training:

```diff
# Remove gradient checkpointing
- if torch.is_grad_enabled() and self.gradient_checkpointing:
-     hidden_states = torch.utils.checkpoint.checkpoint(
-         self._forward_block, hidden_states, ...
-     )
- else:
-     hidden_states = self._forward_block(hidden_states, ...)
+ hidden_states = self._forward_block(hidden_states, ...)

# Remove training-specific attributes
- self.gradient_checkpointing = False

# Remove dropout (set to 0 or remove)
- self.dropout = nn.Dropout(dropout_prob)
+ # Removed dropout for inference
```

#### 1.5: Add Configuration Support

Add support for vLLM-Omni's `OmniDiffusionConfig`:

```python
from vllm_omni.diffusion.data import OmniDiffusionConfig

class YourModelTransformer2DModel(nn.Module):
    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig | None = None,  # ← Add vLLM-Omni config
        # ... other model-specific parameters
        num_layers: int = 28,
        hidden_size: int = 3072,
        num_heads: int = 24,
        **kwargs,
    ):
        super().__init__()

        # Store config
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config if od_config else None

        # Model architecture
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # ... initialize layers
```

### Step 2: Adapt Pipeline

The pipeline orchestrates the full generation process (text encoding, denoising loop, VAE decoding). Adapt it from Diffusers format to vLLM-Omni's interface.

#### 2.1: Remove Diffusers Inheritance

**Remove Diffusers base classes:**
```diff
- from diffusers import DiffusionPipeline
- from diffusers.loaders import LoraLoaderMixin

- class YourModelPipeline(DiffusionPipeline, LoraLoaderMixin):
+ class YourModelPipeline(nn.Module):
    """Your model pipeline for vLLM-Omni."""
```

#### 2.2: Adapt `__init__` Method

**Before (Diffusers):**
```python
class YourModelPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        transformer: YourTransformer,
        scheduler: FlowMatchScheduler,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
```

**After (vLLM-Omni):**
```python
import os
from diffusers import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.diffusion.models.your_model_name.your_model_transformer import (
    YourModelTransformer2DModel,
)


class YourModelPipeline(nn.Module):
    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        self.device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)

        # Load components from checkpoint
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only)
        self.text_encoder = CLIPTextModel.from_pretrained(
            model, subfolder="text_encoder", local_files_only=local_files_only).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model, subfolder="tokenizer", local_files_only=local_files_only)
        self.vae = AutoencoderKL.from_pretrained(
            model, subfolder="vae", local_files_only=local_files_only).to(self.device)

        # Initialize transformer with vLLM-Omni config
        transformer_kwargs = get_transformer_config_kwargs(
            od_config.tf_model_config, YourModelTransformer2DModel)
        self.transformer = YourModelTransformer2DModel(
            od_config=od_config, **transformer_kwargs)

        # Store VAE scale factor for latent space conversions
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.default_sample_size = 128  # Default latent size
```

**Key Changes:**

1. **`od_config` parameter:** All configuration through `OmniDiffusionConfig`
2. **Manual component loading:** No `register_modules()`, load each component explicitly
3. **Local files support:** Check `os.path.exists(model)` for local checkpoints
4. **Transformer with config:** Pass `od_config` to transformer constructor

#### 2.3: Adapt `__call__` → `forward` Method

**Change signature:**

```diff
- @torch.no_grad()
- def __call__(
+ def forward(
    self,
+   req: OmniDiffusionRequest,  # ← Add request parameter here
- ):
+ ) -> DiffusionOutput:  # ← Add return type
```

[`OmniDiffusionRequest`](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/request/#vllm_omni.diffusion.request.OmniDiffusionRequest) is a dataclass that contains the **prompts** and **sampling parameters** [`OmniDiffusionSamplingParams`](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/inputs/data/#vllm_omni.inputs.data.OmniDiffusionSamplingParams) for the diffusion pipeline execution. It also contains a request_id for other components to trace this request and its outputs.

See some parameters in `OmniDiffusionSamplingParams` as follows:

| parameters | type |value | function |
|:---:|:---:|:---:|:---:|
| `num_inference_steps` | `int` | 50 |  The number of diffusion steps during inference|
| `guidance_scale` |  `float` | 0.0 |  The classifier free guidance scale |
| `width` and `height` | `int` | None | The width and height of the generated image |

**Extract parameters from request:**

```python
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.data import DiffusionOutput

def forward(
    self,
    req: OmniDiffusionRequest,
) -> DiffusionOutput:
    # Extract prompts from request
    if req.prompts is not None:
        prompt = [
            p if isinstance(p, str) else (p.get("prompt") or "")
            for p in req.prompts
        ]

    # Extract sampling parameters
    sampling_params = req.sampling_params
    num_inference_steps = sampling_params.num_inference_steps or 50
    guidance_scale = sampling_params.guidance_scale or 7.5
    height = sampling_params.height or (self.default_sample_size * self.vae_scale_factor)
    width = sampling_params.width or (self.default_sample_size * self.vae_scale_factor)

    # For image editing pipelines, extract images from multi_modal_data
    if hasattr(req, 'multi_modal_data') and req.multi_modal_data:
        input_images = req.multi_modal_data.get('image', [])

    # ... rest of generation logic
```

For an image editing model, an example `OmniDiffusionRequest` is like:
```python
{
    "prompt": "turn this cat to a dog",
    "multi_modal_data": {"image": input_image}
},
```

**Wrap output:**

```diff
    # Generate images
    images = self.vae.decode(latents)[0]

-   return {"images": images}
+   return DiffusionOutput(output=images)
```

#### 2.4: Extract Pre/Post-Processing Functions

vLLM-Omni separates image processing from the main pipeline for better modularity.

**Post-processing function (required):**
```python
def get_your_model_post_process_func(
    od_config: OmniDiffusionConfig,
):
    """
    Create post-processing function for your model.

    Returns a function that converts latents to images.
    """
    from diffusers.image_processor import VaeImageProcessor
    import json

    # Load VAE config to get scale factor
    model_path = od_config.model
    if not os.path.exists(model_path):
        from vllm_omni.diffusion.model_loader.utils import download_weights_from_hf_specific
        model_path = download_weights_from_hf_specific(model_path, None, ["*"])

    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** (len(vae_config["block_out_channels"]) - 1)

    # Create image processor
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    def post_process_func(images: torch.Tensor):
        return image_processor.postprocess(images, output_type="pil")

    return post_process_func
```

**Pre-processing function (for image editing pipelines):**

```python
def get_your_model_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    """
    Create pre-processing function for image editing.

    Returns a function that prepares input images.
    """
    from PIL import Image
    from diffusers.image_processor import VaeImageProcessor

    # Load VAE config
    # ... (similar to post_process_func)

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    def pre_process_func(
        request: OmniDiffusionRequest,
        ):
        for i, prompt in enumerate(request.prompts):
            multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else None
            raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
            # image pre-processing
            # after pre-processing, update the request attributes
            ...
        return request

    return pre_process_func
```

#### 2.5: Add Weight Loading Support

Add methods for automatic weight downloading and loading:

```python
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm.model_executor.models.utils import AutoWeightsLoader

class YourModelPipeline(nn.Module):
    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        # ... initialization code

        # Define weight sources for automatic loading
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Customize the weight loading behavior, such as filter weights name.

        Args:
            weights: Iterable of (param_name, param_tensor) tuples

        Returns:
            Set of loaded parameter names
        """
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
```

### Step 3: Register Model

Register your model in `vllm_omni/diffusion/registry.py` so vLLM-Omni can discover and load it.

#### 3.1: Register Pipeline Class

```python
# vllm_omni/diffusion/registry.py

_DIFFUSION_MODELS = {
    # Format: "PipelineClassName": (module_folder, module_file, class_name)

    # Existing models
    "QwenImagePipeline": ("qwen_image", "pipeline_qwen_image", "QwenImagePipeline"),
    "FluxPipeline": ("flux", "pipeline_flux", "FluxPipeline"),

    # Add your model
    "YourModelPipeline": (
        "your_model_name",           # Module folder name
        "pipeline_your_model",       # Python file name (without .py)
        "YourModelPipeline",         # Pipeline class name
    ),
}
```

#### 3.2: Register Pre/Post-Processing Function

```python
# vllm_omni/diffusion/registry.py
_DIFFUSION_PRE_PROCESS_FUNCS = {
    # arch: pre_process_func
    # `pre_process_func` function must be placed in {mod_folder}/{mod_relname}.py,
    # where mod_folder and mod_relname are  defined and mapped using `_DIFFUSION_MODELS` via the `arch` key
    "GlmImagePipeline": "get_glm_image_pre_process_func",
    "QwenImageEditPipeline": "get_qwen_image_edit_pre_process_func",

    # Add your model
    "YourModelPipeline": "get_your_model_pre_process_func", # Optional
}
_DIFFUSION_POST_PROCESS_FUNCS = {
    # Format: "PipelineClassName": "function_name"

    # Existing models
    "QwenImagePipeline": "get_qwen_image_post_process_func",
    "FluxPipeline": "get_flux_post_process_func",

    # Add your model
    "YourModelPipeline": "get_your_model_post_process_func",
}
```


#### 3.3: Export from Module

Create/update `__init__.py` to export your classes:

```python
# vllm_omni/diffusion/models/your_model_name/__init__.py

from .pipeline_your_model import (
    YourModelPipeline,
    get_your_model_post_process_func,
)
from .your_model_transformer import YourModelTransformer2DModel

__all__ = [
    "YourModelPipeline",
    "YourModelTransformer2DModel",
    "get_your_model_post_process_func",
]
```

---

### Step 4: Add Example Script


If your model is one of Text-to-Image, Text-to-Audio, Text-to-Video, Image-to-Image, Image-to-Video models, you can simply try one of the following offline inference scripts to run your model:

| Model Category | Offline Inference Script |
|---|---|
| Image-to-Image | `examples/offline_inference/image_to_image/image_edit.py` |
| Image-to-Video | `examples/offline_inference/image_to_video/image_to_video.py` |
| Text-to-Image | `examples/offline_inference/text_to_image/text_to_image.py` |
| Text-to-Audio | `examples/offline_inference/text_to_audio/text_to_audio.py` |
| Text-to-Video | `examples/offline_inference/text_to_video/text_to_video.py` |


If new CLI arguments need to be added, please edit the offline inference script corresponding to your model category from the table above, and update the example inference script in its corresponding document file (e.g., `examples/offline_inference/text_to_video/text_to_video.md`).

For online inference, all the supported tasks are listed in `docs/user_guide/examples/online_serving/`. If your model falls into these categories, please check the corresponding documentation in this folder and the example at `examples/online_serving/TASK_NAME`. Update them accordingly if needed.

---

If your model is an Omni (understanding and generation) model, please follow the steps below.

#### 4.1: Create Example File

Taking **BAGEL** model as examples for both offline and online:

- Offline: `examples/offline_inference/bagel/`
- Online: `examples/online_serving/bagel/`

Add **two example folders** for your model:

```bash
mkdir -p examples/offline_inference/your_model_name
mkdir -p examples/online_serving/your_model_name
```

**Offline (recommended minimum):** create `examples/offline_inference/your_model_name/end2end.py` and a README.

- Script: `examples/offline_inference/your_model_name/end2end.py`
  - Parse args like BAGEL (`--model`, `--modality`, optional `--image-path`, `--steps`, etc.)
  - Use `from vllm_omni.entrypoints.omni import Omni` (or `OmniDiffusion` if your model is diffusion-only)
  - Save outputs (images/audio/video/text) with deterministic filenames (e.g., `output_0_0.png`)
- Doc: `examples/offline_inference/your_model_name/README.md`
  - Include at least one runnable command, e.g.:

```bash
cd examples/offline_inference/your_model_name
python end2end.py --model your-org/your-model-name --modality text2img --prompts "A cute cat"
```

#### 4.2: Add Online Serving Example (OpenAI-Compatible)

Mirror BAGEL’s online serving setup:

- Server launcher: `examples/online_serving/your_model_name/run_server.sh`
  - Wrap `vllm serve ... --omni --port ...` (and `--stage-configs-path ...` if needed)
- Client: `examples/online_serving/your_model_name/openai_chat_client.py`
  - Send requests to `POST /v1/chat/completions`
  - Support multimodal inputs (e.g., base64 image) if your model needs it
- Doc: `examples/online_serving/your_model_name/README.md`
  - Include both “launch server” and “send request”:

```bash
# Terminal 1: launch server
cd examples/online_serving/your_model_name
bash run_server.sh

# Terminal 2: send request
python openai_chat_client.py --prompt "A cute cat" --modality text2img
```


### Step 5: Test Your Implementation

Before submitting, thoroughly test your implementation.

#### 5.1: Performance/Speed Check

Manually compare **latency/throughput** and **output quality** against a Diffusers baseline.

For a fair comparison, keep the same **prompt**, **seed**, **resolution**, **num_inference_steps**, and **guidance settings**, and run multiple trials to reduce randomness. Record the results (and your hardware / driver / CUDA versions) in your PR description.


#### 5.2 Functionality Check in CI

To ensure project maintainability and sustainable development, we encourage contributors to submit test code (unit tests, system tests, or end-to-end tests) alongside their code changes.
For comprehensive testing guidelines, please refer to the [Test File Structure and Style Guide](../ci/tests_style.md).

---

## Advanced Features

Once basic implementation works, add advanced features for better performance.

### torch.compile Support

Enable automatic compilation for repeated blocks:

```python
# In your_model_transformer.py

class YourModelTransformer2DModel(nn.Module):
    # Specify which blocks can be compiled
    _repeated_blocks = ["YourTransformerBlock"]  # List of block class names

    def __init__(self, ...):
        super().__init__()
        # ... initialization
```

vLLM-Omni automatically compiles blocks in `_repeated_blocks` when `torch.compile` is available.

### Tensor Parallelism

See detailed guide: [How to add Tensor Parallel support](../features/tensor_parallel.md)

**Quick setup:**

1. Replace Linear layers by various parallel linear layers (e.g., `ColumnParallelLinear`) in vLLM
2. Check `tp_size` validity: `hidden_dim`, `num_heads`, and `num_kv_heads` must be divisible by `tp_size`

**Usage:** Set `tensor_parallel_size` when initializing:
```python
omni = Omni(model="your-model", tensor_parallel_size=2)
```

### CFG Parallelism

See detailed guide: [How to add CFG-Parallel support](../features/cfg_parallel.md)

**Quick setup:**

1. Implement `diffuse()` method
2. Inherit `CFGParallelMixin` in your pipeline class

**Usage:** Set `cfg_parallel_size` when initializing:
```python
omni = Omni(model="your-model", cfg_parallel_size=2)
```

### Sequence Parallelism

See detailed guide: [How to add Sequence Parallel support](../features/sequence_parallel.md)

**Quick setup:**

1. Add `_sp_plan` class attribute to transformer
2. Specify where to shard/gather tensors

**Usage:** Set `ulysses_degree` and `ring_degree` when initializing:
```python
omni = Omni(model="your-model", ulysses_degree=2, ring_degree=2)
```

### Cache Acceleration

#### TeaCache

See detailed guide: [How to add TeaCache support](../features/teacache.md)

**Quick setup:**

1. Write extractor function
2. Register in `EXTRACTOR_REGISTRY`
3. Add polynomial coefficients

**Usage:** Set `cache_backend` and `cache_config` when initializing:
```python
omni = Omni(model="your-model",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.2}
)

```


#### Cache-DiT

See detailed guide: [How to add Cache-DiT support](../features/cache_dit.md)

**Quick setup:**

- For standard models: Works automatically
- For complex architectures: Write custom cache config

**Usage:** Set `cache_backend` and `cache_config` when initializing:
```python
omni = Omni(model="your-model",  
    cache_backend="cache_dit",
    cache_config={
        "Fn_compute_blocks": 1,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 4,
    }
)
```

### CPU Offload

See detailed guide: [CPU Offloading for Diffusion Models](../../user_guide/diffusion/cpu_offload_diffusion.md)

vLLM-Omni provides two offloading strategies to reduce GPU memory usage:

1. **Model-level offload**: Mutual exclusion between DiT and encoders (only one on GPU at a time)
2. **Layerwise (Blockwise) offload**: Keeps only a single transformer block on GPU at a time with compute-memory overlap

**Usage:** Enable offload when initializing:
```python
# Model-level offload
omni = Omni(model="your-model", enable_cpu_offload=True)

# Layerwise offload
omni = Omni(model="your-model", enable_layerwise_offload=True)
```

**To support layerwise offloading:** Define the blocks attribute name in your transformer:

```python
class WanTransformer3DModel(nn.Module):
    _layerwise_offload_blocks_attr = "blocks"  # Attribute name containing transformer blocks

    def __init__(self):
        self.blocks = nn.ModuleList([...])  # Transformer blocks
```

**Note:** Layerwise offloading is primarily recommended for large **video generation models** where the compute cost per block is high enough to effectively overlap with memory prefetch operations.


---

## Troubleshooting


**Issue: ImportError when loading model**

**Symptoms:** `ModuleNotFoundError` or `ImportError` when calling `Omni(model="your-model")`

**Causes:**

1. Model not registered in `registry.py`
2. Wrong class name in registry
3. Missing `__init__.py` exports


**Issue: Shape mismatch in attention**

**Symptoms:** `RuntimeError: shape mismatch` in attention forward

**Cause:** Incorrect tensor reshaping for vLLM-Omni's attention interface

**Solution:** Ensure correct shapes:

```python
# vLLM-Omni expects: [batch, seq_len, num_heads, head_dim]
query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
key = key.view(batch_size, kv_seq_len, self.num_kv_heads, self.head_dim)
value = value.view(batch_size, kv_seq_len, self.num_kv_heads, self.head_dim)

hidden_states = self.attn(query, key, value, attn_metadata=attn_metadata)

# Reshape back: [batch, seq_len, num_heads, head_dim] → [batch, seq_len, hidden_size]
hidden_states = hidden_states.reshape(batch_size, seq_len, -1)
```

**Issue: Different outputs compared to Diffusers**

**Symptoms:** Generated images look different from Diffusers

**Causes:**

1. Attention backend differences (FlashAttention vs PyTorch SDPA)
2. Missing normalization or scaling

**4. Issue: Out of memory (OOM)**

**Symptoms:** CUDA out of memory errors

**Solutions:**

1. **Reduce batch size:**
   ```python
   omni.generate(prompts=[...], max_batch_size=2)
   ```

2. **Use smaller image size:**
   ```python
   sampling_params = OmniDiffusionSamplingParams(height=512, width=512)
   ```

3. **Enable model offloading:**
   ```python
   omni = Omni(model="...", enable_cpu_offload=True)
   ```

4. **Apply vae tiling and slicing**
   ```python
   omni = Omni(model="...", vae_use_slicing=True, vae_use_tiling=True,)
   ```

---

## Pull Request Checklist

When submitting a PR to add your model, include:

**1. Implementation Files**

- ✅ Transformer model (`xxx_transformer.py`)
- ✅ Pipeline (`pipeline_xxx.py`)
- ✅ Registry entries in `registry.py`
- ✅ `__init__.py` with proper exports

**2. Example and Tests**

- ✅ Example script in `examples/`
- ✅ Test file in `tests/e2e/`
- ✅ Documentation (`docs/`) creation or updates

_Note: End-to-end test files in `tests/e2e/` are optional but strongly recommended. README updates are required for all new models._

**3. Documentation Updates**

- ✅ Add model to supported models table in `docs/models/supported_models.md`
- ✅ If supporting acceleration features (e.g., sequence parallelism, CFG parallel), update acceleration feature tables in:
  - `docs/user_guide/diffusion_acceleration.md`
  - `docs/user_guide/diffusion/parallelism_acceleration.md`

---

## Model Recipe

After implementing and testing your model, please add a model recipe to the [vllm-project/recipes](https://github.com/vllm-project/recipes) repository. This helps other users understand how to use your model with vLLM-Omni.

**What to Include**

Your recipe should include:

1. **Model Overview**: Brief description of the model and its capabilities
2. **Installation Instructions**: Step-by-step setup instructions including:
   - Installing vllm-omni and dependencies
   - Installing any additional required packages (e.g., xformers, diffusers)
   - Any version requirements
3. **Usage Examples**: Command-line examples demonstrating how to run the model
4. **Configuration Details**: Important configuration parameters and their meanings

**Example**

For reference, see the [LongCat recipe example](https://github.com/vllm-project/recipes/pull/179) which demonstrates the expected format and structure.

**Recipe Location**

Create your recipe file in the appropriate directory structure:
- For organization-specific models: `OrganizationName/ModelName.md`
- For general models: `ModelName.md`

The recipe should be a Markdown file that provides clear, reproducible instructions for users to get started with your model.

---

## Reference Implementations

Study these complete examples:

| Model | Architecture | Key Features | Files |
|-------|--------------|--------------|-------|
| **Qwen-Image** | Dual-stream transformer | CFG-Parallel, SP, TP, Cache | `vllm_omni/diffusion/models/qwen_image/` |
| **Wan2.2** | Video transformer | Dual transformers, SP, CFG-Parallel | `vllm_omni/diffusion/models/wan2_2/` |

---

## Summary

Adding a diffusion model to vLLM-Omni involves:

1. ✅ **Adapt transformer** - Replace attention, remove mixins, add config support
2. ✅ **Adapt pipeline** - Change interface, add request handling, extract processing
3. ✅ **Register model** - Add entries to `registry.py`
4. ✅ **Add examples** - Provide runnable scripts
5. ✅ **Test thoroughly** - Verify correctness and performance
6. ✅ **Add advanced features** - Enable parallelism and acceleration (optional)
7. ✅ **Submit PR** - Include verification results and documentation

**Need help?** Check reference implementations or ask in [slack.vllm.ai](https://slack.vllm.ai) or vLLM user forum at [discuss.vllm.ai](https://discuss.vllm.ai).
