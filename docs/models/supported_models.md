# Supported Models

vLLM-Omni supports unified multimodal comprehension and generation models across various tasks.

## Model Implementation

If vLLM-Omni natively supports a model, its implementation can be found in <gh-file:vllm_omni/model_executor/models> and <gh-file:vllm_omni/diffusion/models>.

## List of Supported Models for Nvidia GPU / AMD GPU

<style>
th {
  white-space: nowrap;
  min-width: 0 !important;
}
</style>

| Architecture | Models | Example HF Models |
|--------------|--------|-------------------|
| `Qwen3OmniMoeForConditionalGeneration` | Qwen3-Omni | `Qwen/Qwen3-Omni-30B-A3B-Instruct` |
| `Qwen2_5OmniForConditionalGeneration` | Qwen2.5-Omni | `Qwen/Qwen2.5-Omni-7B`, `Qwen/Qwen2.5-Omni-3B` |
| `QwenImagePipeline` | Qwen-Image | `Qwen/Qwen-Image` |
| `QwenImageEditPipeline` | Qwen-Image-Edit | `Qwen/Qwen-Image-Edit` |
| `QwenImageEditPlusPipeline` | Qwen-Image-Edit-2509 | `Qwen/Qwen-Image-Edit-2509` |
| `QwenImageLayeredPipeline` | Qwen-Image-Layered | `Qwen/Qwen-Image-Layered` |
|`ZImagePipeline` | Z-Image | `Tongyi-MAI/Z-Image-Turbo` |
| `WanPipeline` | Wan2.2 | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` |
| `OvisImagePipeline` | Ovis-Image | `OvisAI/Ovis-Image` |
|`LongcatImagePipeline` | LongCat-Image | `meituan-longcat/LongCat-Image` |
|`LongCatImageEditPipeline` | LongCat-Image-Edit | `meituan-longcat/LongCat-Image-Edit` |


## List of Supported Models for NPU

<style>
th {
  white-space: nowrap;
  min-width: 0 !important;
}
</style>

| Architecture | Models | Example HF Models |
|--------------|--------|-------------------|
| `Qwen2_5OmniForConditionalGeneration` | Qwen2.5-Omni | `Qwen/Qwen2.5-Omni-7B`, `Qwen/Qwen2.5-Omni-3B`|
| `QwenImagePipeline` | Qwen-Image | `Qwen/Qwen-Image` |
