# Adding a Diffusion Model
This guide walks through the process of adding a new Diffusion model to vLLM-Omni, using Qwen/Qwen-Image-Edit as a comprehensive example.

# Table of Contents
1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Step-by-Step Implementation](#step-by-step-implementation)
4. [Testing](#testing)
5. [Adding a Model Recipe](#adding-a-model-recipe)


# Overview
When add a new diffusion model into vLLM-Omni, additional adaptation work is required due to the following reasons:

+ New model must follow the framework’s parameter passing mechanisms and inference flow.

+ Replacing the model’s default implementations with optimized modules, which is necessary to achieve the better performance.

The diffusion execution flow as follow:
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/vllm-omni-diffusion-flow.png">
    <img alt="Diffusion Flow" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/vllm-omni-diffusion-flow.png" width=55%>
  </picture>
</p>


# Directory Structure
File Structure for Adding a New Diffusion Model

```
vllm_omni/
└── examples/
    └──offline_inference
        └── example script                # reuse existing if possible (e.g., image_edit.py)
    └──online_serving
        └── example script
└── diffusion/
    └── registry.py                       # Registry work
    ├── request.py                        # Request Info
    └── models/your_model_name/           # Model directory (e.g., qwen_image)
        └── pipeline_xxx.py               # Model implementation (e.g., pipeline_qwen_image_edit.py)
```

# Step-by-step-implementation
## Step 1: Model Implementation
The diffusion pipeline’s implementation follows **HuggingFace Diffusers**.
### 1.1 Define the Pipeline Class
Define the pipeline class, e.g., `QwenImageEditPipeline`, and initialize all required submodules, either from HuggingFace `diffusers` or custom implementations. In `QwenImageEditPipeline`, only `QwenImageTransformer2DModel` is re-implemented to support optimizations such as Ulysses-SP. When adding new models in the future, you can either reuse this re-implemented `QwenImageTransformer2DModel` or extend it as needed.

### 1.2 Pre-Processing and Post-Processing Extraction
Extract the pre-processing and post-processing logic from the pipeline class to follow vLLM-Omni’s execution flow. For Qwen-Image-Edit:
```python
def get_qwen_image_edit_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    """
    Define a pre-processing function that resizes input images and
    pre-process for subsequent inference.
    """
```

```python
def get_qwen_image_edit_post_process_func(
    od_config: OmniDiffusionConfig,
):
    """
    Defines a post-processing function that post-process images.
    """
```

### 1.3 Define the forward function
The forward function of `QwenImageEditPipeline` follows the HuggingFace `diffusers` design for the most part. The key differences are:
+ As described in the overview, arguments are passed through `OnniDiffusionRequest`, so we need to get user parameters from it accordingly.
```python
prompt = req.prompt
```
+ pre/post-processing are handled by the framework elsewhere, so skip them.

### 1.4 Replace some ops or layers in DiT component

vLLM-Omni provides a set of optimized operators with better performance and built-in support for parallelism, including attention, rotary embeddings (RoPE), and linear layers.

Below is an example showing how to replace standard Transformer attention and FFN layers with vLLM-Omni implementations:

```python
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.layers.rope import RotaryEmbedding

class MyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention()
        self.to_qkv = QKVParallelLinear()
        self.to_out = RowParallelLinear()
        self.rope = RotaryEmbedding(is_neox_style=False)

    def forward(self, hidden_states):
        qkv, _ = self.to_qkv(hidden_states)
        q, k, v = qkv.split(...)
        q, k = self.rope(...)
        attn_output = self.attn(q, k, v)
        output = self.to_out(attn_output)

class MyFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = ColumnParallelLinear()
        self.fc2 = RowParallelLinear()
        self.act = F.gelu

    def forward(self, hidden_states):
        hidden, _ = self.fc1(hidden_states)
        hidden = self.act(hidden)
        output = self.fc2(hidden)
        return output
```

In this example:

+ Attention uses vLLM-Omni’s optimized attention kernel together with parallel QKV projection and RoPE.

+ Linear layers are replaced with column- and row-parallel variants to enable tensor parallelism.

+ The FFN follows a standard two-layer structure and can be further optimized (e.g., using fused or merged projections) if needed.


### 1.5 Provide a `_repeated_blocks` in DiT model
`_repeated_blocks` is the small and frequently-repeated block(s) of a model -- typically a transformer layer.

It's used for torch compile optimizations.
```python
_repeated_blocks = ["QwenImageTransformerBlock"]
```


### 1.6 (Optional) implement sequence parallelism
vLLM-Omni has a non-intrusive `_sp_plan` that enable sequence parallel without modifying `forward()` logic.
You can refer to [How to parallelize a new model](../../user_guide/diffusion/parallelism_acceleration.md)


### 1.7 (Optional) integrate with Cache-Dit
vLLM-Omni supports acceleration via [Cache-Dit](../../user_guide/diffusion/cache_dit_acceleration.md). Most models compatible with Diffusers can use Cache-Dit seamlessly. For new models, you can extend support by modifying`cache_dit_backend.py`

## Step 2: Extend OmniDiffusionRequest Fields
User-provided inputs are ultimately passed to the model’s forward method through OmniDiffusionRequest, so we add the required fields here to support the new model.
```python
prompt: str | list[str] | None = None
negative_prompt: str | list[str] | None = None
...
```

## Step 3: Registry
+ registry diffusion model in registry.py
```python
_DIFFUSION_MODELS = {
    # arch:(mod_folder, mod_relname, cls_name)
    ...
    "QwenImageEditPipeline": (
        "qwen_image",
        "pipeline_qwen_image_edit",
        "QwenImageEditPipeline",
    ),
    ...
}
```
+ registry pre-process get function
```python
_DIFFUSION_PRE_PROCESS_FUNCS = {
    # arch: pre_process_func
    ...
    "QwenImageEditPipeline": "get_qwen_image_edit_pre_process_func",
    ...
}
```

+ registry post-process get function
```python
_DIFFUSION_POST_PROCESS_FUNCS = {
    # arch: post_process_func
    ...
    "QwenImageEditPipeline": "get_qwen_image_edit_post_process_func",
    ...
}
```

## Step 4: Add an Example Script
For each newly integrated model, we need to provide examples script under the examples/ to demonstrate how to initialize the pipeline with Omni, pass in user inputs, and generate outputs.
Key point for writing the example:

+ Use the Omni entrypoint to load the model and construct the pipeline.

+ Show how to format user inputs and pass them via omni.generate(...).

+ Demonstrate the common runtime arguments, such as:

    + model path or model name

    + input image(s) or prompt text

    + key diffusion parameters (e.g., inference steps, guidance scale)

    + optional acceleration backends (e.g., Cache-DiT, TeaCache)

+ Save or display the generated results so users can validate the integration.

## Step 5: Open a Pull Request

When submitting a pull request to add support for a new model, please include the following information in the PR description:

+ Output verification: provide generation outputs to verify correctness and model behavior.

+ Inference speed: provide a comparison with the corresponding implementation in Diffusers.

+ Parallelism support: specify the supported parallel sizes and any relevant limitations.

+ Cache acceleration: check whether the model can be accelerated using Cache-Dit or not.


Providing these details helps reviewers evaluate correctness, performance improvements, and parallel scalability of the new model integration.

# Testing
For comprehensive testing guidelines, please refer to the [Test File Structure and Style Guide](../ci/tests_style.md).


## Adding a Model Recipe
After implementing and testing your model, please add a model recipe to the [vllm-project/recipes](https://github.com/vllm-project/recipes) repository. This helps other users understand how to use your model with vLLM-Omni.
