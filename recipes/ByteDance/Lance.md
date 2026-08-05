# Lance

> Unified autoregressive + diffusion multimodal (text / image / video)

## Summary

- Vendor: ByteDance
- Model: [`bytedance-research/Lance`](https://huggingface.co/bytedance-research/Lance) (`Lance_3B`, `Lance_3B_Video`)
- Task: text2img, text2video, img2img (image edit), video2video (video edit),
  image2video, img2text (image understanding), video2text (video understanding)
- Mode: Offline inference, Online serving (OpenAI-compatible API)
- Maintainer: Community

Lance is a 3B unified autoregressive + diffusion multimodal model on a
Qwen2.5-VL backbone. Each modality runs through the standard task example, and
model-specific knobs are passed via `--extra-body`. `text2video` and
`video2video` need the `Lance_3B_Video` checkpoint for the 3-D
`latent_pos_embed` table.

## Common setup

Lance is registered in the pipeline registry (`model_type: lance`) and ships a
bundled default deploy [`vllm_omni/deploy/lance.yaml`](../../vllm_omni/deploy/lance.yaml)
that carries the engine knobs (`max_num_batched_tokens=32768`, `max_num_seqs=1`,
`enforce_eager`, `trust_remote_code`, `enable_prefix_caching=false`,
`async_chunk=false`). The model id resolves the pipeline and its default deploy
is loaded automatically, so **`--deploy-config` is optional**. The examples below
pass it explicitly for clarity; `--model bytedance-research/Lance` alone works too.

`--extra-body` knobs: `cfg_text_scale`, `cfg_img_scale`, `cfg_interval`,
`cfg_renorm_type`, `cfg_renorm_min`, `negative_prompt`, `timestep_shift`,
`num_frames`, `video_height`, `video_width`, `origin_fps`, `max_think_tokens`,
`do_sample`, `text_temperature`, `system_prompt`, `user_text`.

Defaults: 30 denoising steps, `timestep_shift=3.5`, text CFG 4.0, seed 42.

## Hardware

Lance_3B has a small BF16 footprint (~7 GB: LLM + Qwen2.5-VL ViT + Wan2.2 VAE),
so a single NVIDIA GPU with **>= 16 GB** VRAM (e.g. RTX 4090, L40S, A10) is
enough. Linux, Python 3.12, CUDA >= 12.4.

## Offline inference

### text-to-image

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
    --model bytedance-research/Lance \
    --deploy-config vllm_omni/deploy/lance.yaml \
    --prompt "a corgi astronaut on the moon, cinematic" \
    --num-inference-steps 30 \
    --height 1024 --width 1024 --seed 42 \
    --extra-body '{"cfg_text_scale": 4.0, "timestep_shift": 3.5}' \
    --output ./out/lance_t2i.png
```

### image edit (img2img)

```bash
python examples/offline_inference/image_to_image/image_edit.py \
    --model bytedance-research/Lance \
    --deploy-config vllm_omni/deploy/lance.yaml \
    --image /path/to/input.png \
    --prompt "Convert this into a vibrant cartoon-style illustration" \
    --num-inference-steps 30 \
    --extra-args '{"cfg_text_scale": 4.0, "timestep_shift": 3.5}' \
    --output ./out/lance_edit.png
```

Note: `image_edit.py` uses `--extra-args` (not `--extra-body`) for the JSON passthrough.

### image-to-video

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
    --model bytedance-research/Lance/Lance_3B_Video \
    --deploy-config vllm_omni/deploy/lance.yaml \
    --image /path/to/first_frame.png \
    --prompt "the scene comes to life with smooth, natural motion" \
    --height 480 --width 848 --num-frames 61 \
    --num-inference-steps 30 \
    --extra-body '{"cfg_text_scale": 4.0, "timestep_shift": 3.5}' \
    --output ./out/lance_i2v.mp4
```

The trailing `Lance_3B_Video` component is resolved as a subfolder of the valid
`bytedance-research/Lance` Hugging Face repo, so the 3-D `latent_pos_embed`
table is loaded. The standard `--height`, `--width`, and `--num-frames` flags
are forwarded to Lance; explicit `video_height`, `video_width`, or `num_frames`
values in `--extra-body` take precedence.

### image understanding (img2text)

```bash
python examples/offline_inference/x_to_text/x_to_text.py \
    --model bytedance-research/Lance \
    --deploy-config vllm_omni/deploy/lance.yaml \
    --trust-remote-code \
    --image /path/to/photo.jpg \
    --prompt "Describe this image in detail." \
    --max-tokens 512 --temperature 0.8
```

Lance's greedy decoder often emits an immediate EOS, so pass `--temperature 0.8`.

### text-to-video, video-to-video, video-to-text (direct API)

These three paths run through the `Omni.generate` dict-prompt API.
`render_lance_prompt` lives in
`vllm_omni/diffusion/models/lance/prompts.py`.

```python
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.diffusion.models.lance.prompts import (
    render_lance_prompt, VISION_START, VIDEO_PAD, VISION_END,
)

VISION_BLOCK = f"{VISION_START}{VIDEO_PAD}{VISION_END}"
omni = Omni(model="bytedance-research/Lance/Lance_3B_Video",
            deploy_config="vllm_omni/deploy/lance.yaml")

# text-to-video
t2v = {
    "prompt": render_lance_prompt("t2v", "a cat playing piano, cinematic"),
    "modalities": ["video"],
    "extra_args": {"num_frames": 25, "video_height": 480, "video_width": 768,
                   "cfg_text_scale": 4.0, "timestep_shift": 3.5},
}

# video-to-video (video edit): multi_modal_data.video = path or (T,H,W,3) array
v2v = {
    "prompt": render_lance_prompt("video_edit", "make it snowy", vision_token=VISION_BLOCK),
    "multi_modal_data": {"video": "/path/to/clip.mp4"},
    "modalities": ["video"],
    "extra_args": {"num_frames": 25, "video_height": 480, "video_width": 768},
}

# video-to-text (understanding): multi_modal_data.video = path or (T,H,W,3) array
v2t = {
    "prompt": render_lance_prompt("x2t_video", "What is happening in this video?",
                                  vision_token=VISION_BLOCK),
    "multi_modal_data": {"video": "/path/to/clip.mp4"},
    "modalities": ["text"],
    "extra_args": {"do_sample": True, "text_temperature": 0.8, "max_think_tokens": 256},
}

params = list(omni.default_sampling_params_list)
params[0].num_inference_steps = 30
outputs = list(omni.generate(prompts=[t2v], sampling_params_list=params))
```

## Online serving

Lance serves single-stage modalities via the OpenAI-compatible
`/v1/chat/completions` API. Launch:

```bash
vllm-omni serve bytedance-research/Lance --omni \
    --deploy-config vllm_omni/deploy/lance.yaml --port 8091
```

The deploy config carries the pipeline selector and all engine knobs
(`max_num_batched_tokens`, `max_num_seqs`, `enforce_eager`, `trust_remote_code`,
`enable_prefix_caching=false`, `async_chunk=false`), so no extra flags are needed.

For `text2video` / `video2video`, use the `bytedance-research/Lance/Lance_3B_Video`
subfolder form shown above.

### Verification

```bash
pytest -s -v tests/e2e/online_serving/test_lance.py
```

## Notes

- `video_edit` output is more abstract than `text2video` at the same
  resolution, but is functionally correct end-to-end.
