# Lance: Offline inference

[Lance](https://huggingface.co/bytedance-research/Lance) is a 3B unified
autoregressive + diffusion multimodal model on a Qwen2.5-VL backbone. It is
**BAGEL-lineage** (ByteDance Mixture-of-Transformers): the released `Lance_3B`
checkpoint uses the same `*_moe_gen` MoT weight layout as BAGEL, so vLLM-Omni
implements it by reusing the BAGEL transformer core and specializing only the
ViT (Qwen2.5-VL vision), the VAE (Wan2.2) and the checkpoint layout.

This example covers all six Lance modalities from the upstream HF model card:
`t2i`, `t2v`, `image_edit`, `video_edit`, `x2t_image` (image understanding) and
`x2t_video` (video understanding).

## Hardware

Single NVIDIA GPU with 16 GB+ VRAM in BF16 (we test on B300 / A100). CUDA ≥ 12.4.

## Run

```bash
# Text-to-image
python examples/offline_inference/lance/end2end.py \
    --model bytedance-research/Lance \
    --prompts "a corgi astronaut on the moon, cinematic" \
    --steps 30 --cfg-text-scale 4.0 --timestep-shift 3.5 \
    --height 1024 --width 1024 \
    --output ./out

# Text-to-video (uses the Lance_3B_Video subfolder; see ``--modality``
# choices for all six task variants)
python examples/offline_inference/lance/end2end.py \
    --model bytedance-research/Lance/Lance_3B_Video --modality text2video \
    --num-frames 25 --video-height 480 --video-width 768 \
    --prompts "a cat playing piano, cinematic" \
    --steps 30 --fps 8 --output ./out
```

`video_edit` requires `--model bytedance-research/Lance/Lance_3B_Video` so
the 3-D `latent_pos_embed` table is loaded; the other paths can point at
the top-level `bytedance-research/Lance` repo and resolve the right
sub-checkpoint automatically.

The HF repo bundles everything (`Lance_3B/`, `Lance_3B_Video/`,
`Qwen2.5-VL-ViT/`, `Wan2.2_VAE.pth`); no separate downloads are required.

## Defaults

Matches upstream `inference_lance.sh`: 30 denoising steps, timestep-shift 3.5,
text CFG 4.0, seed 42, 1024×1024 (override with `--height` / `--width`).  For
the understanding paths (`img2text` / `video2text`), sampling is enabled by
default at `--text-temperature 0.8` because Lance's greedy decoder emits an
immediate EOS for many prompts.
