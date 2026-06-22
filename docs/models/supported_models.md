# Supported Models

vLLM-Omni supports unified multimodal comprehension and generation models across various tasks.

## Model Implementation

If vLLM-Omni natively supports a model, its implementation can be found in <gh-file:vllm_omni/model_executor/models> and <gh-file:vllm_omni/diffusion/models>.

For deployment recipes (TTS and multimodal), see [recipes.vllm.ai](https://recipes.vllm.ai) and the in-repo [`recipes/`](https://github.com/vllm-project/vllm-omni/tree/main/recipes) directory.

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
| `MingFlashOmniForConditionalGeneration` + `MingImagePipeline` | Ming-flash-omni-2.0 (omni-speech + imagegen<sup>1</sup>) | `Jonathan1909/Ming-flash-omni-2.0` | ✅︎ |   |   |   |
| `BagelForConditionalGeneration` | BAGEL (DiT-only) | `ByteDance-Seed/BAGEL-7B-MoT` | ✅︎ | ✅︎ | | ✅︎ |
| `InternVLAA1Pipeline` | InternVLA-A1 | `InternRobotics/InternVLA-A1-3B` | ✅︎ | ✅︎ | | |
| `Gr00tN1d7Pipeline` | GR00T N1.7 | `nvidia/GR00T-N1.7-3B` | ✅︎ | | | |
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
| `Cosmos3OmniDiffusersPipeline` | Cosmos3 T2I, T2V, I2V, V2V, T2V with sound, action policy | `nvidia/Cosmos3-Nano` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `WanSpeechToVideoPipeline` | Wan2.2-S2V | `Wan-AI/Wan2.2-S2V-14B` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `Wan22VACEPipeline` | Wan2.1-VACE | `Wan-AI/Wan2.1-VACE-1.3B-diffusers`, `Wan-AI/Wan2.1-VACE-14B-diffusers` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `LTX2Pipeline` | LTX-2-T2V | `Lightricks/LTX-2` | ✅︎ | ✅︎ | | |
| `LTX2ImageToVideoPipeline` | LTX-2-I2V | `Lightricks/LTX-2` | ✅︎ | ✅︎ | | |
| `LTX2TwoStagesPipeline` | LTX-2-T2V | `rootonchair/LTX-2-19b-distilled` | ✅︎ | ✅︎ | | |
| `LTX2ImageToVideoTwoStagesPipeline` | LTX-2-I2V | `rootonchair/LTX-2-19b-distilled` | ✅︎ | ✅︎ | | |
| `LTX23Pipeline` | LTX-2.3-T2V | `dg845/LTX-2.3-Diffusers` | ✅︎ | ✅︎ | | |
| `LTX23ImageToVideoPipeline` | LTX-2.3-I2V | `dg845/LTX-2.3-Diffusers` | ✅︎ | ✅︎ | | |
| `DreamZeroPipeline` | DreamZero-DROID | `GEAR-Dreams/DreamZero-DROID` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `HeliosPipeline`, `HeliosPyramidPipeline` | Helios | `BestWishYsh/Helios-Base`, `BestWishYsh/Helios-Mid`, `BestWishYsh/Helios-Distilled` | ✅︎ | ✅︎ | ✅︎ | |
| `MagiHumanPipeline` | MagiHuman | `SII-GAIR/daVinci-MagiHuman-Base-1080p` | ✅︎ | ✅︎ | | |
| `OvisImagePipeline` | Ovis-Image | `OvisAI/Ovis-Image` | ✅︎ | ✅︎ | | ✅︎ |
| `LongcatImagePipeline` | LongCat-Image | `meituan-longcat/LongCat-Image` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `LongCatImageEditPipeline` | LongCat-Image-Edit | `meituan-longcat/LongCat-Image-Edit` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `StableDiffusionXLPipeline` | Stable-Diffusion-XL | `stabilityai/stable-diffusion-xl-base-1.0` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `StableDiffusion3Pipeline` | Stable-Diffusion-3 | `stabilityai/stable-diffusion-3.5-medium` | ✅︎ | ✅︎ | | ✅︎ |
| `CosyVoice3Model` | CosyVoice3 | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | ✅︎ | ✅︎ | | ✅︎ |
| `OmniVoiceModel` | OmniVoice | `k2-fsa/OmniVoice` | ✅︎ | | | |
| `VoxCPM2TalkerForConditionalGeneration` | VoxCPM2 | `openbmb/VoxCPM2` | ✅︎ | | | |
| `MammothModa2ForConditionalGeneration` | MammothModa2-Preview | `bytedance-research/MammothModa2-Preview` | ✅︎ | ✅︎ | | |
| `Flux2KleinPipeline` | FLUX.2-klein | `black-forest-labs/FLUX.2-klein-4B`, `black-forest-labs/FLUX.2-klein-9B` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `FluxKontextPipeline` | FLUX.1-Kontext-dev | `black-forest-labs/FLUX.1-Kontext-dev` | ✅︎ | ✅︎ | | |
| `FluxPipeline` | FLUX.1-dev | `black-forest-labs/FLUX.1-dev` | ✅︎ | ✅︎ | | ✅︎ |
| `FluxPipeline` | FLUX.1-schnell | `black-forest-labs/FLUX.1-schnell` | ✅︎ | ✅︎ | | ✅︎ |
| `OmniGen2Pipeline` | OmniGen2 | `OmniGen2/OmniGen2` | ✅︎ | ✅︎ | | ✅︎ |
| `StableAudioPipeline` | Stable-Audio-Open | `stabilityai/stable-audio-open-1.0` | ✅︎ | ✅︎ | | ✅︎ |
| `AudioXPipeline` | AudioX | `zhangj1an/AudioX` | ✅︎ | ✅︎ | | |
| `Qwen3TTSForConditionalGeneration` | Qwen3-TTS-12Hz-1.7B-CustomVoice | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `Qwen3TTSForConditionalGeneration` | Qwen3-TTS-12Hz-1.7B-VoiceDesign | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `Qwen3TTSForConditionalGeneration` | Qwen3-TTS-12Hz-1.7B-Base | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
| `MingTTSForConditionalGeneration` | Ming-omni-tts dense 0.5B | `inclusionAI/Ming-omni-tts-0.5B` | ✅︎ | | | |
| `GLMTTSForConditionalGeneration` | GLM-TTS | `zai-org/GLM-TTS` | ✅︎ | | | |
| `MossTTSNanoForCausalLM` | MOSS-TTS-Nano | `OpenMOSS-Team/MOSS-TTS-Nano` | ✅︎ | | | |
| `MossTTSDelayModel` | MOSS-TTS, MOSS-TTSD, MOSS-SoundEffect, MOSS-VoiceGenerator | `OpenMOSS-Team/MOSS-VoiceGenerator` | ✅︎ | | | |
| `MossTTSRealtime` | MOSS-TTS-Realtime | `OpenMOSS-Team/MOSS-TTS-Realtime` | ✅︎ | | | |
| `HiggsAudioV2ForConditionalGeneration` | Higgs-Audio v2 | `bosonai/higgs-audio-v2-generation-3B-base` | ✅︎ | | | |
| `HiggsMultimodalQwen3ForConditionalGeneration` | Higgs-Audio v3 (TTS) | `bosonai/higgs-audio-v3-tts-4b` | ✅︎ | | | |
| `NextStep11Pipeline` | NextStep-1.1 | `stepfun-ai/NextStep-1.1` | ✅︎ | ✅︎ | | ✅︎ |
| `MiMoAudioModel` | MiMo-Audio-7B-Instruct | `XiaomiMiMo/MiMo-Audio-7B-Instruct` | ✅︎ | ✅︎ | | |
| `MiMoV2ASRForCausalLM` | MiMo-V2.5-ASR | `XiaomiMiMo/MiMo-V2.5-ASR` | ✅︎ | ✅︎ | | |
| `Flux2Pipeline` | FLUX.2-dev | `black-forest-labs/FLUX.2-dev` | ✅︎ | ✅︎ | | |
| `FishSpeechSlowARForConditionalGeneration` | Fish Speech S2 Pro | `fishaudio/s2-pro` | ✅︎ | ✅︎ | | |
| `DreamIDOmniPipeline` | DreamID-Omni | `XuGuo699/DreamID-Omni` | ✅︎ | ✅︎ | | |
| `SenseNovaU1Pipeline` | SenseNova-U1 (DiT-only) | `SenseNova/SenseNova-U1-8B-MoT` | ✅︎ | | | |
| `LancePipeline` | Lance | `bytedance-research/Lance` | ✅︎ | | | |
| `HunyuanVideo15Pipeline` | HunyuanVideo-1.5-T2V | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v`, `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v` | ✅︎ | ✅︎ | | |
| `HunyuanVideo15ImageToVideoPipeline` | HunyuanVideo-1.5-I2V | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v`, `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v` | ✅︎ | ✅︎ | | |
| `VoxtralTTSForConditionalGeneration` | Voxtral TTS | `mistralai/Voxtral-4B-TTS-2603` | ✅︎ | ✅︎ | | |
| `CovoAudioForConditionalGeneration` | Covo-Audio-Chat | `tencent/Covo-Audio-Chat` | ✅︎ | | | |
|`DyninOmniForConditionalGeneration` | Dynin-Omni | `snu-aidas/Dynin-Omni` | ✅︎ | | | |
| `MiniCPMO45OmniForConditionalGeneration` | MiniCPM-o 4.5 | `openbmb/MiniCPM-o-4_5` | ✅︎ | | ✅︎ | |
| `ErnieImagePipeline` | ERNIE-Image | `baidu/ERNIE-Image`, `baidu/ERNIE-Image-Turbo` | ✅︎ | ✅︎ | ✅︎ | ✅︎ |
|`HiDreamImagePipeline` | HiDream-I1-Full | `HiDream-ai/HiDream-I1-Full` | ✅︎ | ✅︎ | | |

✅︎ indicates the model is supported on that backend. Empty cells mean not listed as supported on that backend.
