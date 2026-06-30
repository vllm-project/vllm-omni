# Community Recipes

This directory contains community-maintained recipes for answering a
practical user question:

> How do I run model X on hardware Y for task Z?

Add recipes for this repository under this in-repo `recipes/` directory. To
keep naming and layout consistent, organize recipes by model vendor in a way
that is aligned with
[`vllm-project/recipes`](https://github.com/vllm-project/recipes), but treat
that external repository as a reference for structure rather than the place to
add files for this repo. Use one Markdown file per model family by default.

> **Note:** The vLLM-Omni TTS recipes (12 models across 8 providers) are also
> published on the rendered recipes site at
> [recipes.vllm.ai](https://recipes.vllm.ai) (see
> [vllm-project/recipes#554](https://github.com/vllm-project/recipes/pull/554)).
> Individual pages resolve at `https://recipes.vllm.ai/<provider>/<model-id>`,
> e.g. [`bosonai/higgs-audio-v3-tts-4b`](https://recipes.vllm.ai/bosonai/higgs-audio-v3-tts-4b).

Example layout:

```text
recipes/
  Qwen/
    Qwen3-Omni.md
    Qwen3-TTS.md
  Tencent/
    Covo-Audio-Chat.md
```

## Available Recipes

| Recipe | Task | Hardware |
|--------|------|----------|
| [`audiox/AudioX.md`](./audiox/AudioX.md) | Offline + online unified text/video→audio diffusion | 1x L4 24GB |
| [`Baidu/ERNIE-Image.md`](./Baidu/ERNIE-Image.md) | Text-to-image online serving (ERNIE-Image 8B) | 1x or 2x RTX 4090 24GB |
| [`Bagel/BAGEL-7B-MoT.md`](./Bagel/BAGEL-7B-MoT.md) | Text-to-image with shared online/offline examples | 1x A100 80GB / 2x CUDA GPUs |
| [`BosonAI/Higgs-Audio-V3-TTS.md`](./BosonAI/Higgs-Audio-V3-TTS.md) | Online + offline multilingual TTS with voice cloning | 1x H100 80GB |
| [`ByteDance/Lance.md`](./ByteDance/Lance.md) | Unified AR+diffusion: text/img/video gen + understanding (Lance 3B) | 1x B300 / A100 80GB |
| [`fishaudio/Fish-Speech-S2-Pro.md`](./fishaudio/Fish-Speech-S2-Pro.md) | Online serving for TTS | 1x A800 80GB |
| [`Helios/Helios.md`](./Helios/Helios.md) | Text-to-video, image-to-video, and video-to-video generation | 1x NVIDIA H20 |
| [`inclusionAI/Ming-flash-omni-2.0.md`](./inclusionAI/Ming-flash-omni-2.0.md) | Online serving for multimodal chat + standalone TTS | 4x H100 / 1x H100 80GB |
| [`inclusionAI/Ming-omni-tts.md`](./inclusionAI/Ming-omni-tts.md) | Offline + online dense Ming TTS/audio generation | 1x H100 80GB / 1x AMD MI300X (ROCm 7.2) |
| [`IndexTeam/IndexTTS-2.md`](./IndexTeam/IndexTTS-2.md) | Online serving for voice-cloned TTS with optional emotion control | 1x L4 24GB or larger CUDA GPU |
| [`LTX/LTX-2.md`](./LTX/LTX-2.md) | Text-to-video and image-to-video serving | 1x H200 141GB |
| [`LTX/LTX-2.3.md`](./LTX/LTX-2.3.md) | Text-to-video with audio generation (22B) | 1x GPU (96GB VRAM) |
| [`MammothModa2/MammothModa2-Preview.md`](./MammothModa2/MammothModa2-Preview.md) | Text-to-image with the shared offline image example (AR → DiT) | 1x L40S 48GB / 1x ≥40GB GPU |
| [`mistralai/Voxtral-TTS.md`](./mistralai/Voxtral-TTS.md) | Online serving for TTS | 1x RTX 4090 24GB |
| [`cosmos3/Cosmos3-Nano.md`](./cosmos3/Cosmos3-Nano.md) | Text-to-image, text-to-video, image-to-video, video-to-video generation, text to video with sound, action policy | 1x H200 141GB / B300 |
| [`cosmos3/Cosmos3-Super.md`](./cosmos3/Cosmos3-Super.md) | 64B T2I / T2V / I2V / V2V generation (+ optional audio) / Action policy | 8x H200/H100/A100 / 2x H200 / B300 |
| [`OpenBMB/MiniCPM-o-4_5.md`](./OpenBMB/MiniCPM-o-4_5.md) | Online serving for omni multimodal chat (text / image / audio / video → text + 24 kHz speech) | 2x A100/H100 80GB / 3x mid-tier GPU / 8x RTX 4090 24GB |
| [`OpenBMB/VoxCPM2.md`](./OpenBMB/VoxCPM2.md) | Online + offline TTS with native AR pipeline (48 kHz, 30+ languages) | 1x RTX 4090 24GB |
| [`OpenMOSS/MOSS-TTS.md`](./OpenMOSS/MOSS-TTS.md) | Online + offline multilingual TTS (MOSS-TTS family, 8B) | 1x H100 80GB |
| [`Qwen/Qwen-Image.md`](./Qwen/Qwen-Image.md) | Text-to-image serving with step-wise continuous batching replay and ModelOpt mixed FP8/NVFP4 | 1x A100 80GB / 2x B200 |
| [`Qwen/Qwen-Image-2512.md`](./Qwen/Qwen-Image-2512.md) | Text-to-image serving with step-wise continuous batching replay and ModelOpt FP8 / mixed FP8/NVFP4 | 1x A800 80GB / 2x B200 |
| [`Qwen/Qwen-Image-Edit.md`](./Qwen/Qwen-Image-Edit.md) | Text-guided single-image editing | 1x or 2x H200 141GB |
| [`Qwen/Qwen3-Omni.md`](./Qwen/Qwen3-Omni.md) | Online serving for multimodal chat | 1x A100 80GB |
| [`Qwen/Qwen3-TTS.md`](./Qwen/Qwen3-TTS.md) | Text-to-speech serving (CustomVoice / VoiceDesign / Base) | 1x H100/A100 80GB |
| [`SenseNova/SenseNova-U1.md`](./SenseNova/SenseNova-U1.md) | Unified image generation and understanding | 1x H200 (144GB) |
| [`StabilityAI/Stable-Audio-Open.md`](./StabilityAI/Stable-Audio-Open.md) | Offline + online text-to-audio generation (Stable Audio Open) | 1x RTX 4090 24GB |
| [`Tencent/Covo-Audio-Chat.md`](./Tencent/Covo-Audio-Chat.md) | Online serving for audio chat | 1x A100 80GB |
| [`Tencent/HunyuanImage-3.0-Instruct.md`](./Tencent/HunyuanImage-3.0-Instruct.md) | DiT-only text-to-image serving and benchmark, including ModelOpt mixed FP8/NVFP4 | 4x H100/H800 80GB / 2x B200 |
| [`Wan-AI/Wan2.2-I2V.md`](./Wan-AI/Wan2.2-I2V.md) | Image-to-video serving (Wan2.2 14B) | 8x Ascend NPU (A2/A3) |
| [`Wan-AI/Wan2.2-S2V.md`](./Wan-AI/Wan2.2-S2V.md) | Speech-to-video serving (Wan2.2 14B) | 2x A100/H100 80GB |
| [`Wan-AI/Wan2.1-VACE.md`](./Wan-AI/Wan2.1-VACE.md) | Unified T2V, I2V, V2LF, FLF2V, inpaint, and R2V | 1x RTX 5090 (1.3B) / 1x L40S 48GB with layerwise offload (14B) |
| [`StabilityAI/Stable-Diffusion-3.5.md`](./StabilityAI/Stable-Diffusion-3.5.md) | Text-to-image serving (SD 3.5-medium and SD 3.5-large) | 1x RTX A6000 48GB |
| [`zai-org/GLM-TTS.md`](./zai-org/GLM-TTS.md) | Online serving for Chinese/English zero-shot voice-cloned TTS | 1x A40 48GB |
| [`GLM/GLM-Image.md`](./GLM/GLM-Image.md) | Online serving for image generation | 1x A800 80GB / 2x A800 80GB |
| [`JD/JoyAI-VL-Interaction.md`](./JD/JoyAI-VL-Interaction.md) | Real-time streaming video-language interaction (proactive speak/silence/delegate) | 1x GPU 24GB+ |

Within a single recipe file, include different hardware support sections such
as `GPU`, `ROCm`, and `NPU`, and add concrete tested configurations like
`1x A100 80GB` or `2x L40S` inside those sections when applicable.

See [TEMPLATE.md](./TEMPLATE.md) for the recommended format.
