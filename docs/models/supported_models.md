# Supported Models

vLLM-Omni supports unified multimodal comprehension and generation models across various tasks.

## Model Implementation

If vLLM-Omni natively supports a model, its implementation can be found in <gh-file:vllm-omni/model_executor/models> and <gh-file:vllm_omni/diffusion/models>.

## List of Supported Models for Nvidia GPU

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
|`ZImagePipeline` | Z-Image | `Tongyi-MAI/Z-Image-Turbo` |


## List of Supported Models for Ascend NPU

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
