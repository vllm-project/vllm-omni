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

| Architecture | Models | Example HF Models | NVIDIA GPU | AMD GPU | Ascend NPU | Intel GPU | Recipe |
|--------------|--------|-------------------|------------|---------|-----|-----------|--------|
| `Qwen3OmniMoeForConditionalGeneration` | Qwen3-Omni | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | ✅︎ | | | | [Repository](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Qwen/Qwen3-Omni.md) |
| `Qwen2_5OmniForConditionalGeneration` | Qwen2.5-Omni | `Qwen/Qwen2.5-Omni-7B`, `Qwen/Qwen2.5-Omni-3B` | ✅︎ | ✅︎ | ✅︎ | ✅︎ | — |
| `MingFlashOmniForConditionalGeneration` + `MingImagePipeline` | Ming-flash-omni-2.0 (omni-speech + imagegen<sup>1</sup>) | `Jonathan1909/Ming-flash-omni-2.0` | ✅︎ | | | | — |
| `BagelForConditionalGeneration` | BAGEL (DiT-only) | `ByteDance-Seed/BAGEL-7B-MoT` | ✅︎ | ✅︎ | | ✅︎ | — |
| `InternVLAA1Pipeline` | InternVLA-A1 | `InternRobotics/InternVLA-A1-3B` | ✅︎ | ✅︎ | | | — |
| `Gr00tN1d7Pipeline` | GR00T N1.7 | `nvidia/GR00T-N1.7-3B` | ✅︎ | | | | — |
| `HunyuanImage3ForCausalMM` | HunyuanImage3.0 (DiT-only) | `tencent/HunyuanImage-3.0`, `tencent/HunyuanImage-3.0-Instruct` | ✅︎ | ✅︎ | ✅︎ | ✅︎ | — |
| `QwenImagePipeline` | Qwen-Image | `Qwen/Qwen-Image` | ✅︎ | | | ✅︎ | [Published](https://recipes.vllm.ai/Qwen/Qwen-Image) |
| `QwenImagePipeline` | Qwen-Image-2512 | `Qwen/Qwen-Image-2512` | ✅︎ | | | ✅︎ | [Repository](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Qwen/Qwen-Image-2512.md) |
| `QwenImageEditPipeline` | Qwen-Image-Edit | `Qwen/Qwen-Image-Edit` | ✅︎ | ✅︎ | ✅︎ | ✅︎ | — |
| `QwenImageEditPlusPipeline` | Qwen-Image-Edit-2509 | `Qwen/Qwen-Image-Edit-2509` | ✅︎ | ✅︎ | ✅︎ | ✅︎ | — |
| `QwenImageLayeredPipeline` | Qwen-Image-Layered | `Qwen/Qwen-Image-Layered` | ✅︎ | ✅︎ | ✅︎ | ✅︎ | — |
| `QwenImageEditPlusPipeline` | Qwen-Image-Edit-2511 | `Qwen/Qwen-Image-Edit-2511` | ✅︎ | ✅︎ | ✅︎ | ✅︎ | — |
| `GlmImagePipeline` | GLM-Image | `zai-org/GLM-Image` | ✅︎ | ✅︎ | | | — |
| `ZImagePipeline` | Z-Image | `Tongyi-MAI/Z-Image-Turbo` | ✅︎ | ✅︎ | ✅︎ | ✅︎ | — |
| `Krea2Pipeline` | Krea 2 (Raw + Turbo) | `krea/Krea-2-Raw`, `krea/Krea-2-Turbo` | ✅︎ | | | ✅︎ | — |
| `WanPipeline` | Wan2.1-T2V, Wan2.2-T2V, Wan2.2-TI2V | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, `Wan-AI/Wan2.1-T2V-14B-Diffusers`, `Wan-AI/Wan2.2-T2V-A14B-Diffusers`, `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | ✅︎ | ✅︎ | ✅︎ | ✅︎ | — |
| `WanImageToVideoPipeline` | Wan2.2-I2V | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | ✅︎ | ✅︎ | ✅︎ | ✅︎ | — |
| `LingBotWorldCausalDMDPipeline` | LingBot-World 2.0 v1 (experimental) | `robbyant/lingbot-world-v2-14b-causal-fast-diffusers` | Experimental | | | | — |
| `Cosmos3OmniDiffusersPipeline` | Cosmos3 T2I, T2V, I2V, V2V, T2V with sound, action policy | `nvidia/Cosmos3-Nano`, `nvidia/Cosmos3-Super` | ✅︎ | ✅︎ | ✅︎ | ✅︎ | — |
| `SanaWmPipeline` | SANA-WM | `BBBBruce/SANA-WM_bidirectional-stage1-diffusers` | ✅︎ | | | | [Repository](https://github.com/vllm-project/vllm-omni/blob/main/recipes/NVIDIA/SANA-WM.md) |
| `Wan22S2VPipeline` | Wan2.2-S2V | `Wan-AI/Wan2.2-S2V-14B` | ✅︎ | ✅︎ | ✅︎ | ✅︎ | — |
| `Wan22VACEPipeline` | Wan2.1-VACE | `Wan-AI/Wan2.1-VACE-1.3B-diffusers`, `Wan-AI/Wan2.1-VACE-14B-diffusers` | ✅︎ | ✅︎ | ✅︎ | ✅︎ | — |
| `Wan22VACEPipeline` | Wan2.2-VACE | `Pyros13/Wan2.2-VACE-Fun-A14B-Diffusers` | ✅︎ |   |   | ✅︎ | — |
| `LTX2Pipeline` | LTX-2 / LTX-2.3 one-stage T2V and I2V | `Lightricks/LTX-2`, `diffusers/LTX-2.3-Diffusers` | ✅︎ | ✅︎ | | | — |
| `LTX2TwoStagePipeline` | LTX-2 / LTX-2.3 ordinary two-stage T2V and I2V | `Lightricks/LTX-2`, `diffusers/LTX-2.3-Diffusers` + matching Lightricks LoRA and upsampler | ✅︎ | ✅︎ | | | — |
| `LTX2DistilledOneStagePipeline` | LTX-2 / LTX-2.3 merged-distilled one-stage T2V and I2V | `rootonchair/LTX-2-19b-distilled`, `diffusers/LTX-2.3-Distilled-Diffusers` | ✅︎ | ✅︎ | | | — |
| `LTX2DistilledTwoStagePipeline` | LTX-2 / LTX-2.3 merged-distilled two-stage T2V and I2V | `rootonchair/LTX-2-19b-distilled`, `diffusers/LTX-2.3-Distilled-Diffusers` + matching Lightricks upsampler | ✅︎ | ✅︎ | | | — |
| `LTX2Pipeline` | LTX-2.5 Full/SFT one-stage T2V and I2V | `Lightricks/LTX-2.5-Diffusers` | ✅︎ | | | | [Repository](https://github.com/vllm-project/vllm-omni/blob/main/recipes/LTX/LTX-2.5.md) |
| `LTX2TwoStagePipeline` | LTX-2.5 Full/SFT two-stage T2V and I2V | `Lightricks/LTX-2.5-Diffusers` + `Lightricks/LTX-2.5` LoRA | ✅︎ | | | | [Repository](https://github.com/vllm-project/vllm-omni/blob/main/recipes/LTX/LTX-2.5.md) |
| `LTX2DistilledOneStagePipeline` | LTX-2.5 merged-distilled one-stage T2V and I2V | `Lightricks/LTX-2.5-Diffusers` | ✅︎ | | | | [Repository](https://github.com/vllm-project/vllm-omni/blob/main/recipes/LTX/LTX-2.5.md) |
| `LTX2DistilledTwoStagePipeline` | LTX-2.5 merged-distilled two-stage T2V and I2V | `Lightricks/LTX-2.5-Diffusers` | ✅︎ | | | | [Repository](https://github.com/vllm-project/vllm-omni/blob/main/recipes/LTX/LTX-2.5.md) |
| `LingBotVideoPipeline` | LingBot-Video dense and MoE T2I, T2V, TI2V | `robbyant/lingbot-video-dense-1.3b`, `robbyant/lingbot-video-moe-30b-a3b` | ✅︎ | | | | — |
| `SanaVideoPipeline` | SANA-Video-2B T2V (native 480p/720p; Diffusers adapter validated at both resolutions) | `Efficient-Large-Model/SANA-Video_2B_480p_diffusers`, `Efficient-Large-Model/SANA-Video_2B_720p_diffusers` | ✅︎ | | | | [Repository](https://github.com/vllm-project/vllm-omni/blob/main/recipes/NVIDIA/SANA-Video-2B.md) |
| `SanaImageToVideoPipeline` | SANA-Video-2B I2V (native and Diffusers adapter validated at 480p/720p) | `Efficient-Large-Model/SANA-Video_2B_480p_diffusers`, `Efficient-Large-Model/SANA-Video_2B_720p_diffusers` | ✅︎ | | | | [Repository](https://github.com/vllm-project/vllm-omni/blob/main/recipes/NVIDIA/SANA-Video-2B.md) |
| `MiniMaxH3Pipeline` | MiniMax H3 T2VA, first/last-keyframe FL2VA, and mixed-reference Ref2VA | `MiniMaxAI/MiniMax-H3` | ✅︎ | ✅︎ | | | [Published](https://recipes.vllm.ai/MiniMaxAI/MiniMax-H3) |
| `DreamZeroPipeline` | DreamZero-DROID | `GEAR-Dreams/DreamZero-DROID` | ✅︎ | ✅︎ | ✅︎ | ✅︎ | — |
| `Pi0Pipeline` | π0 (Pi-Zero) VLA | `lerobot/pi0_base` | ✅︎ | | | | [Repository](https://github.com/vllm-project/vllm-omni/blob/main/recipes/lerobot/Pi0.md) |
| `HeliosPipeline`, `HeliosPyramidPipeline` | Helios | `BestWishYsh/Helios-Base`, `BestWishYsh/Helios-Mid`, `BestWishYsh/Helios-Distilled` | ✅︎ | ✅︎ | ✅︎ | | — |
| `OvisImagePipeline` | Ovis-Image | `OvisAI/Ovis-Image` | ✅︎ | ✅︎ | | ✅︎ | — |
| `LongcatImagePipeline` | LongCat-Image | `meituan-longcat/LongCat-Image` | ✅︎ | ✅︎ | ✅︎ | ✅︎ | — |
| `LongCatImageEditPipeline` | LongCat-Image-Edit | `meituan-longcat/LongCat-Image-Edit` | ✅︎ | ✅︎ | ✅︎ | ✅︎ | — |
| `LongCatVideoAvatarPipeline` | LongCat-Video-Avatar-1.5 A2V/AI2V (native single-speaker and multi-speaker AI2V/AVC) | `meituan-longcat/LongCat-Video-Avatar-1.5` | ✅︎ | | | | [Repository](https://github.com/vllm-project/vllm-omni/blob/main/recipes/meituan-longcat/LongCat-Video-Avatar-1.5.md) |
| `BooguImagePipeline` | Boogu-Image | `Boogu/Boogu-Image-0.1-Base`, `Boogu/Boogu-Image-0.1-Edit`, `Boogu/Boogu-Image-0.1-Edit-Turbo` | ✅︎ | | | ✅︎ | [Repository](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Boogu/Boogu-Image.md) |
| `StableDiffusionXLPipeline` | Stable-Diffusion-XL | `stabilityai/stable-diffusion-xl-base-1.0` | ✅︎ | ✅︎ | ✅︎ | ✅︎ | — |
| `StableDiffusion3Pipeline` | Stable-Diffusion-3 | `stabilityai/stable-diffusion-3.5-medium` | ✅︎ | ✅︎ | | ✅︎ | — |
| `CosyVoice3Model` | CosyVoice3 | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | ✅︎ | ✅︎ | | ✅︎ | — |
| `OmniVoiceModel` | OmniVoice | `k2-fsa/OmniVoice` | ✅︎ | | | ✅︎ | — |
| `VoxCPM2TalkerForConditionalGeneration` | VoxCPM2 | `openbmb/VoxCPM2` | ✅︎ | | | ✅︎ | — |
| `DotsTTSForConditionalGeneration` | dots.tts | `rednote-hilab/dots.tts-soar` | ✅︎ | | | | — |
| `MammothModa2ForConditionalGeneration` | MammothModa2-Preview | `bytedance-research/MammothModa2-Preview` | ✅︎ | ✅︎ | | | — |
| `MammothModa2ForConditionalGeneration` | MammothModa2-Dev (AR-only image understanding) | `bytedance-research/MammothModa2-Dev` | ✅︎ | | | | — |
| `Flux2KleinPipeline` | FLUX.2-klein | `black-forest-labs/FLUX.2-klein-4B`, `black-forest-labs/FLUX.2-klein-9B` | ✅︎ | ✅︎ | ✅︎ | ✅︎ | — |
| `FluxKontextPipeline` | FLUX.1-Kontext-dev | `black-forest-labs/FLUX.1-Kontext-dev` | ✅︎ | ✅︎ | | | — |
| `FluxPipeline` | FLUX.1-dev | `black-forest-labs/FLUX.1-dev` | ✅︎ | ✅︎ | | ✅︎ | — |
| `FluxPipeline` | FLUX.1-schnell | `black-forest-labs/FLUX.1-schnell` | ✅︎ | ✅︎ | | ✅︎ | — |
| `OmniGen2Pipeline` | OmniGen2 | `OmniGen2/OmniGen2` | ✅︎ | ✅︎ | | ✅︎ | — |
| `StableAudioPipeline` | Stable-Audio-Open | `stabilityai/stable-audio-open-1.0` | ✅︎ | ✅︎ | | ✅︎ | — |
| `MiniMaxMusic3ForConditionalGeneration` | MiniMax Music 3 (text-to-music) | `MiniMaxAI/MiniMax-Music3` | ✅︎ | | | | — |
| `Qwen3TTSForConditionalGeneration` | Qwen3-TTS-12Hz-1.7B-CustomVoice | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | ✅︎ | | | ✅︎ | [Published](https://recipes.vllm.ai/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) |
| `Qwen3TTSForConditionalGeneration` | Qwen3-TTS-12Hz-1.7B-VoiceDesign | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | ✅︎ | ✅︎ | ✅︎ | ✅︎ | — |
| `Qwen3TTSForConditionalGeneration` | Qwen3-TTS-12Hz-1.7B-Base | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | ✅︎ | ✅︎ | ✅︎ | ✅︎ | — |
| `MingTTSForConditionalGeneration` | Ming-omni-tts dense 0.5B | `inclusionAI/Ming-omni-tts-0.5B` | ✅︎ | | | ✅︎ | — |
| `GLMTTSForConditionalGeneration` | GLM-TTS | `zai-org/GLM-TTS` | ✅︎ | | | | — |
| `MossTTSNanoForCausalLM` | MOSS-TTS-Nano | `OpenMOSS-Team/MOSS-TTS-Nano` | ✅︎ | | | | — |
| `MossTTSDelayModel` | MOSS-TTS, MOSS-TTSD, MOSS-SoundEffect, MOSS-VoiceGenerator | `OpenMOSS-Team/MOSS-VoiceGenerator` | ✅︎ | | | | — |
| `MossTTSRealtime` | MOSS-TTS-Realtime | `OpenMOSS-Team/MOSS-TTS-Realtime` | ✅︎ | | | | — |
| `HiggsAudioV2ForConditionalGeneration` | Higgs-Audio v2 | `bosonai/higgs-audio-v2-generation-3B-base` | ✅︎ | | | | — |
| `HiggsMultimodalQwen3ForConditionalGeneration` | Higgs-Audio v3 (TTS) | `bosonai/higgs-audio-v3-tts-4b` | ✅︎ | | | | [Published](https://recipes.vllm.ai/bosonai/higgs-audio-v3-tts-4b) |
| `IndexTTS2TalkerForConditionalGeneration` | IndexTTS-2 | `IndexTeam/IndexTTS-2` | ✅︎ | | | | — |
| `IndexTTS25TalkerForConditionalGeneration` | IndexTTS-2.5 | native `checkpoints/` bundle | ✅︎ | | | | [Repository](https://github.com/vllm-project/vllm-omni/blob/main/recipes/IndexTeam/IndexTTS-2_5.md) |
| `NextStep11Pipeline` | NextStep-1.1 | `stepfun-ai/NextStep-1.1` | ✅︎ | ✅︎ | | ✅︎ | — |
| `MiMoAudioModel` | MiMo-Audio-7B-Instruct | `XiaomiMiMo/MiMo-Audio-7B-Instruct` | ✅︎ | ✅︎ | | | — |
| `MiMoV2ASRForCausalLM` | MiMo-V2.5-ASR | `XiaomiMiMo/MiMo-V2.5-ASR` | ✅︎ | ✅︎ | | | — |
| `Flux2Pipeline` | FLUX.2-dev | `black-forest-labs/FLUX.2-dev` | ✅︎ | ✅︎ | | | — |
| `FishSpeechSlowARForConditionalGeneration` | Fish Speech S2 Pro | `fishaudio/s2-pro` | ✅︎ | ✅︎ | | ✅︎ | — |
| `SenseNovaU1Pipeline` | SenseNova-U1 (DiT-only), SenseNova-U1.5 | `SenseNova/SenseNova-U1-8B-MoT`, `sensenova/SenseNova-U1.5-8B-MoT` | ✅︎ | | | | [Repository](https://github.com/vllm-project/vllm-omni/blob/main/recipes/SenseNova/SenseNova-U1.5.md) |
| `LancePipeline` | Lance | `bytedance-research/Lance` | ✅︎ | | | | — |
| `HunyuanVideo15Pipeline` | HunyuanVideo-1.5-T2V | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v`, `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v` | ✅︎ | ✅︎ | | | — |
| `HunyuanVideo15ImageToVideoPipeline` | HunyuanVideo-1.5-I2V | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v`, `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v` | ✅︎ | ✅︎ | | | — |
| `VoxtralTTSForConditionalGeneration` | Voxtral TTS | `mistralai/Voxtral-4B-TTS-2603` | ✅︎ | ✅︎ | | | — |
| `CovoAudioForConditionalGeneration` | Covo-Audio-Chat | `tencent/Covo-Audio-Chat` | ✅︎ | | | | — |
|`DyninOmniForConditionalGeneration` | Dynin-Omni | `snu-aidas/Dynin-Omni` | ✅︎ | | | ✅︎ | — |
| `MiniCPMO45OmniForConditionalGeneration` | MiniCPM-o 4.5 | `openbmb/MiniCPM-o-4_5` | ✅︎ | | ✅︎ | | — |
| `ErnieImagePipeline` | ERNIE-Image | `baidu/ERNIE-Image`, `baidu/ERNIE-Image-Turbo` | ✅︎ | ✅︎ | ✅︎ | ✅︎ | — |
| `GepardTalkerForConditionalGeneration` | Gepard-1.0 | `nineninesix/gepard-1.0` | ✅︎ | | | ✅︎ | — |
|`HiDreamImagePipeline` | HiDream-I1-Full | `HiDream-ai/HiDream-I1-Full` | ✅︎ | ✅︎ | | | — |
| `HiDreamO1ImagePipeline` | HiDream-O1-Image | `HiDream-ai/HiDream-O1-Image` | ✅︎ | | | | — |

The `Recipe` column links the corresponding deployment recipe. `Published` is
preferred when a corresponding `recipes.vllm.ai` page exists; otherwise the
link points to the repository recipe. `—` means that no recipe link has been
added for that model row yet.

For rows with an audited recipe link, the hardware checkmarks reflect hardware
explicitly documented in that recipe. Other rows retain the implementation
support metadata until their recipe is audited.

✅︎ indicates the model is supported on that backend. Empty cells mean not listed as supported on that backend.

See the [LingBot-World 2.0 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Robbyant/LingBot-World-2.0.md)
for offline and experimental realtime usage, supported controls, validation,
and current limitations.

<sup>H3</sup> MiniMax H3 AMD GPU support is validated on gfx942 (MI300X) and
gfx950 (MI350) in BF16, using the AITER `FLASH_ATTN` diffusion attention backend.
Other AMD SKUs (e.g. MI325X) are not listed until their
own validation evidence is added. See the
[MiniMax H3 recipe](https://recipes.vllm.ai/MiniMaxAI/MiniMax-H3) for
per-architecture serving commands.
