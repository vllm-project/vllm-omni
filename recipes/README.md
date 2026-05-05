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

Example layout:

```text
recipes/
  Qwen/
    Qwen3-Omni.md
    Qwen3-TTS.md
  Tencent-Hunyuan/
    HunyuanVideo.md
```

## Available Recipes

- [`Qwen/Qwen3-Omni.md`](./Qwen/Qwen3-Omni.md): online serving recipe for
  multimodal chat on `1x A100 80GB`
- [`Wan-AI/Wan2.2-I2V.md`](./Wan-AI/Wan2.2-I2V.md): image-to-video serving
  recipe for Wan2.2 14B on `8x Ascend NPU (A2/A3)`
- [`Tencent-Hunyuan/HunyuanImage-3.0-Instruct.md`](./Tencent-Hunyuan/HunyuanImage-3.0-Instruct.md):
  DiT-only text-to-image serving and benchmark recipe for HunyuanImage-3.0-Instruct
  on `4x H100/H800 80GB`
- [`inclusionAI/Ming-flash-omni-2.0.md`](./inclusionAI/Ming-flash-omni-2.0.md):
  online serving recipe for multimodal chat (`4x H100 80GB`) and standalone TTS (`1x H100 80GB`)
- [`Baidu/ERNIE-Image.md`](./Baidu/ERNIE-Image.md): text-to-image serving
  online serving recipe for ERNIE-Image 8B on `1x RTX 4090 24GB` or `2x RTX 4090 24GB`
- [`fishaudio/Fish-Speech-S2-Pro.md`](./fishaudio/Fish-Speech-S2-Pro.md): online serving recipe for TTS on `1x A800 80GB`
- [`audiox/AudioX.md`](./audiox/AudioX.md): offline + online recipe for AudioX
  unified text/video→audio diffusion on `1x L4 24GB`

Within a single recipe file, include different hardware support sections such
as `GPU`, `ROCm`, and `NPU`, and add concrete tested configurations like
`1x A100 80GB` or `2x L40S` inside those sections when applicable.

See [TEMPLATE.md](./TEMPLATE.md) for the recommended format.
