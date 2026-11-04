# FastH3

Choose a release by task, then follow its recipe for installation and serving:

| Release | Input → output | Steps | Recipe |
| --- | --- | ---: | --- |
| **FastH3** (Preview V1 adapter) | Text → video and audio | 4 | [FastH3](FastH3.md) |
| **FastH3 V2** (full checkpoint) | Text → video and audio | 8 | [FastH3 V2](FastH3-V2.md) |
| **FastH3 Omni** (full checkpoint) | Text and references → video and audio | 8 | [Single card](<FastH3 RefOmni Single Card.md>) · [Multi-GPU](FastH3-Omni.md) |

Omni requires at least one image or video and also accepts audio references.
The single-card guide covers RTX 4090, RTX 5090, DGX Spark, and RTX PRO 6000
Blackwell. All releases use the H3 video HTTP API.
It defaults to NVFP4 QAD on Blackwell and BF16 with online FP8 on RTX 4090;
serialized FP8 is also available on Blackwell.

Steps count transformer forwards. Keep each release's trained schedule and
attention settings; do not apply the four-step adapter to either full checkpoint.

## References

- [Omni NVFP4 video decoder](FastH3-Omni.md#optional-nvfp4-video-decoder): choose the 8-layer or 36-layer decoder.
- [8× RTX PRO 5000 Omni benchmark](FastH3-Omni.md#benchmark-8-nvidia-rtx-pro-5000): generation time and memory use.
- [Native MiniMax H3](../MiniMaxAI/MiniMax-H3.md): base-model inputs and API fields.
- [VSA backend](../../docs/user_guide/diffusion/attention_backends/fastvideo_vsa.md): provider selection.
- [ComfyUI integration](../../docs/features/comfyui.md): generic video workflows.
- [All community recipes](../README.md).
