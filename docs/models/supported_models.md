# Supported Models

vLLM-Omni supports unified multimodal comprehension and generation models across various tasks.

## Model Implementation

If vLLM-Omni natively supports a model, its implementation can be found in <gh-file:vllm_omni/model_executor/models> and <gh-file:vllm_omni/diffusion/models>.

## List of Supported Models

<style>
th {
  white-space: nowrap;
  min-width: 0 !important;
}
</style>

| Architecture | Models | Example HF Models | NVIDIA GPU | AMD GPU | Ascend NPU | Intel GPU |
|--------------|--------|-------------------|------------|---------|-----|-----------|
| `Qwen3OmniMoeForConditionalGeneration` | Qwen3-Omni | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `Qwen2_5OmniForConditionalGeneration` | Qwen2.5-Omni | `Qwen/Qwen2.5-Omni-7B`, `Qwen/Qwen2.5-Omni-3B` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `BagelForConditionalGeneration` | BAGEL (DiT-only) | `ByteDance-Seed/BAGEL-7B-MoT` | ✅︎ | ✅︎ | | ✅︎ |
| `HunyuanImage3ForCausalMM` | HunyuanImage3.0 (DiT-only) | `tencent/HunyuanImage-3.0`, `tencent/HunyuanImage-3.0-Instruct` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `QwenImagePipeline` | Qwen-Image | `Qwen/Qwen-Image` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `QwenImagePipeline` | Qwen-Image-2512 | `Qwen/Qwen-Image-2512` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `QwenImageEditPipeline` | Qwen-Image-Edit | `Qwen/Qwen-Image-Edit` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `QwenImageEditPlusPipeline` | Qwen-Image-Edit-2509 | `Qwen/Qwen-Image-Edit-2509` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `QwenImageLayeredPipeline` | Qwen-Image-Layered | `Qwen/Qwen-Image-Layered` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `QwenImageEditPlusPipeline` | Qwen-Image-Edit-2511 | `Qwen/Qwen-Image-Edit-2511` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `GlmImagePipeline` | GLM-Image | `zai-org/GLM-Image` | ✅︎ | ✅︎ | | |
| `ZImagePipeline` | Z-Image | `Tongyi-MAI/Z-Image-Turbo` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `WanPipeline` | Wan2.1-T2V, Wan2.2-T2V, Wan2.2-TI2V | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, `Wan-AI/Wan2.1-T2V-14B-Diffusers`, `Wan-AI/Wan2.2-T2V-A14B-Diffusers`, `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `WanImageToVideoPipeline` | Wan2.2-I2V | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `Wan22VACEPipeline` | Wan2.1-VACE | `Wan-AI/Wan2.1-VACE-1.3B-diffusers`, `Wan-AI/Wan2.1-VACE-14B-diffusers` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `LTX2Pipeline` | LTX-2-T2V | `Lightricks/LTX-2` | ✅︎ | ✅︎ | | |
| `LTX2ImageToVideoPipeline` | LTX-2-I2V | `Lightricks/LTX-2` | ✅︎ | ✅︎ | | |
| `LTX2TwoStagesPipeline` | LTX-2-T2V | `rootonchair/LTX-2-19b-distilled` | ✅︎ | ✅︎ | | |
| `LTX2ImageToVideoTwoStagesPipeline` | LTX-2-I2V | `rootonchair/LTX-2-19b-distilled` | ✅︎ | ✅︎ | | |
| `HeliosPipeline`, `HeliosPyramidPipeline` | Helios | `BestWishYsh/Helios-Base`, `BestWishYsh/Helios-Mid`, `BestWishYsh/Helios-Distilled` | ✅︎ | ✅︎ | ✅︎ | |
| `MagiHumanPipeline` | MagiHuman | `princepride/daVinci-MagiHuman` | ✅︎ | ✅︎ | | |
| `OvisImagePipeline` | Ovis-Image | `OvisAI/Ovis-Image` | ✅︎ | ✅︎ | | ✅︎ |
| `LongcatImagePipeline` | LongCat-Image | `meituan-longcat/LongCat-Image` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `LongCatImageEditPipeline` | LongCat-Image-Edit | `meituan-longcat/LongCat-Image-Edit` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `StableDiffusion3Pipeline` | Stable-Diffusion-3 | `stabilityai/stable-diffusion-3.5-medium` | ✅︎ | ✅︎ | | ✅︎ |
| `CosyVoice3Model` | CosyVoice3 | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | ✅︎ | ✅︎ | | |
| `MammothModa2ForConditionalGeneration` | MammothModa2-Preview | `bytedance-research/MammothModa2-Preview` | ✅︎ | ✅︎ | | |
| `Flux2KleinPipeline` | FLUX.2-klein | `black-forest-labs/FLUX.2-klein-4B`, `black-forest-labs/FLUX.2-klein-9B` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `FluxKontextPipeline` | FLUX.1-Kontext-dev | `black-forest-labs/FLUX.1-Kontext-dev` | ✅︎ | ✅︎ | | |
| `FluxPipeline` | FLUX.1-dev | `black-forest-labs/FLUX.1-dev` | ✅︎ | ✅︎ | | ✅︎ |
| `OmniGen2Pipeline` | OmniGen2 | `OmniGen2/OmniGen2` | ✅︎ | ✅︎ | | ✅︎ |
| `StableAudioPipeline` | Stable-Audio-Open | `stabilityai/stable-audio-open-1.0` | ✅︎ | ✅︎ | | ✅︎ |
| `Qwen3TTSForConditionalGeneration` | Qwen3-TTS-12Hz-1.7B-CustomVoice | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `Qwen3TTSForConditionalGeneration` | Qwen3-TTS-12Hz-1.7B-VoiceDesign | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `Qwen3TTSForConditionalGeneration` | Qwen3-TTS-12Hz-1.7B-Base | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `NextStep11Pipeline` | NextStep-1.1 | `stepfun-ai/NextStep-1.1` | ✅︎ | ✅︎ | | ✅︎ |
| `MiMoAudioForConditionalGeneration` | MiMo-Audio-7B-Instruct | `XiaomiMiMo/MiMo-Audio-7B-Instruct` | ✅︎ | ✅︎ | | |
| `Flux2Pipeline` | FLUX.2-dev | `black-forest-labs/FLUX.2-dev` | ✅︎ | ✅︎ | | |
| `FishSpeechSlowARForConditionalGeneration` | Fish Speech S2 Pro | `fishaudio/s2-pro` | ✅︎ | ✅︎ | | |
| `DreamIDOmniPipeline` | DreamID-Omni | `XuGuo699/DreamID-Omni` | ✅︎ | ✅︎ | | |
| `HunyuanVideo15Pipeline` | HunyuanVideo-1.5-T2V | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v`, `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v` | ✅︎ | ✅︎ | | |
| `HunyuanVideo15ImageToVideoPipeline` | HunyuanVideo-1.5-I2V | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v`, `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v` | ✅︎ | ✅︎ | | |
| `VoxtralTTSForConditionalGeneration` | Voxtral TTS | `mistralai/Voxtral-4B-TTS-2603` | ✅︎ | ✅︎ | | |

✅︎ indicates the model is supported on that backend. Empty cells mean not listed as supported on that backend.
