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
| `BagelForConditionalGeneration` | BAGEL (DiT-only) | `ByteDance-Seed/BAGEL-7B-MoT` |
| `QwenImagePipeline` | Qwen-Image | `Qwen/Qwen-Image` |
| `QwenImagePipeline` | Qwen-Image-2512 | `Qwen/Qwen-Image-2512` |
| `QwenImageEditPipeline` | Qwen-Image-Edit | `Qwen/Qwen-Image-Edit` |
| `QwenImageEditPlusPipeline` | Qwen-Image-Edit-2509 | `Qwen/Qwen-Image-Edit-2509` |
| `QwenImageLayeredPipeline` | Qwen-Image-Layered | `Qwen/Qwen-Image-Layered` |
|`ZImagePipeline` | Z-Image | `Tongyi-MAI/Z-Image-Turbo` |
| `WanPipeline` | Wan2.2-T2V, Wan2.2-TI2V | `Wan-AI/Wan2.2-T2V-A14B-Diffusers`, `Wan-AI/Wan2.2-TI2V-5B-Diffusers` |
| `WanImageToVideoPipeline` | Wan2.2-I2V | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` |
| `OvisImagePipeline` | Ovis-Image | `OvisAI/Ovis-Image` |
|`LongcatImagePipeline` | LongCat-Image | `meituan-longcat/LongCat-Image` |
|`LongCatImageEditPipeline` | LongCat-Image-Edit | `meituan-longcat/LongCat-Image-Edit` |
|`StableDiffusion3Pipeline` | Stable-Diffusion-3 | `stabilityai/stable-diffusion-3.5-medium` |
|`Flux2KleinPipeline` | FLUX.2-klein | `black-forest-labs/FLUX.2-klein-4B`, `black-forest-labs/FLUX.2-klein-9B` |
|`FluxPipeline` | FLUX.1-dev | `black-forest-labs/FLUX.1-dev` |
|`StableAudioPipeline` | Stable-Audio-Open | `stabilityai/stable-audio-open-1.0` |
|`Qwen3TTSForConditionalGeneration` | Qwen3-TTS-12Hz-1.7B-CustomVoice | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
|`Qwen3TTSForConditionalGeneration` | Qwen3-TTS-12Hz-1.7B-VoiceDesign | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` |
|`Qwen3TTSForConditionalGeneration` | Qwen3-TTS-12Hz-1.7B-Base | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` |


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
| `QwenImagePipeline` | Qwen-Image-2512 | `Qwen/Qwen-Image-2512` |
| `QwenImageEditPipeline` | Qwen-Image-Edit | `Qwen/Qwen-Image-Edit` |
| `QwenImageEditPlusPipeline` | Qwen-Image-Edit-2509 | `Qwen/Qwen-Image-Edit-2509` |
| `QwenImageLayeredPipeline` | Qwen-Image-Layered | `Qwen/Qwen-Image-Layered` |
| `QwenImageEditPlusPipeline` | Qwen-Image-Edit-2511 | `Qwen/Qwen-Image-Edit-2511` |
|`ZImagePipeline` | Z-Image | `Tongyi-MAI/Z-Image-Turbo` |
