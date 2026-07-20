# Lance

> Unified autoregressive + diffusion multimodal (text / image / video)

## Summary

- Vendor: ByteDance
- Model: [`bytedance-research/Lance`](https://huggingface.co/bytedance-research/Lance) (`Lance_3B`, `Lance_3B_Video`)
- Task: text2img, text2video, img2img (image edit), video2video (video edit),
  img2text (image understanding), video2text (video understanding)
- Mode: Offline inference, Online serving (OpenAI-compatible API)
- Maintainer: Community

## When to use this recipe

Use this recipe to run Lance — a 3B unified AR + diffusion multimodal model
on a Qwen2.5-VL backbone — via vLLM-Omni. Lance is **BAGEL-lineage**: the
released checkpoint uses the same `*_moe_gen` Mixture-of-Transformers weight
layout as BAGEL, so vLLM-Omni reuses BAGEL's transformer core and specializes
only the ViT (Qwen2.5-VL vision), the VAE (Wan2.2) and the checkpoint layout.

All six single-stage modalities are supported on the `Lance_3B` checkpoint;
`text2video` and `video2video` additionally need `Lance_3B_Video` for the 3-D
`latent_pos_embed` table. The two-stage AR + DiT topology is deferred and
needs `LanceConfig` / `LanceProcessor` registered in the `vllm` package (a
separate upstream PR).

## References

- Offline example:
  [`examples/offline_inference/lance/end2end.py`](../../examples/offline_inference/lance/end2end.py)
- Online serving:
  [`examples/online_serving/lance/`](../../examples/online_serving/lance/)
- Deploy config:
  [`vllm_omni/deploy/lance.yaml`](../../vllm_omni/deploy/lance.yaml)
- E2E tests:
  [`tests/e2e/online_serving/test_lance.py`](../../tests/e2e/online_serving/test_lance.py)
- HuggingFace model page:
  [bytedance-research/Lance](https://huggingface.co/bytedance-research/Lance)
- PR: [vllm-project/vllm-omni#3710](https://github.com/vllm-project/vllm-omni/pull/3710)
  (closes [#3697](https://github.com/vllm-project/vllm-omni/issues/3697))

## Hardware Support

## GPU

### 1x NVIDIA B300 / A100 80GB

#### Environment

- OS: Linux
- Python: 3.12
- Driver / runtime: CUDA ≥ 12.4
- vLLM-Omni version: 0.18.x.dev

#### Model-specific parameters (`extra_body`)

Lance's generation knobs are declared in
[`vllm_omni/model_extras/lance.py`](../../vllm_omni/model_extras/lance.py) and
passed as a JSON object to the shared task examples. Keys are filtered against
that declaration, so an undeclared key is **silently dropped** rather than
raising:

| Key | Default | Notes |
| --- | --- | --- |
| `cfg_text_scale` | `4.0` | text CFG scale |
| `cfg_img_scale` | `1.0` | image/ViT CFG scale |
| `cfg_interval` | `[0.4, 1.0]` | CFG-active timestep range |
| `cfg_renorm_type` | `"global"` | CFG renorm strategy |
| `cfg_renorm_min` | `0.0` | CFG renorm floor |
| `timestep_shift` | `3.5` | flow-match timestep shift |
| `negative_prompt` | – | also settable via `--negative-prompt` |
| `think` | `false` | reasoning pass before generation (inherited BAGEL path) |
| `max_think_tokens` | `1000` | think / x2t decode budget |
| `do_sample` | `false` | sampling vs greedy for text decode |
| `text_temperature` | `0.3` | text decode temperature |
| `system_prompt` | task default | overrides the task system prompt |
| `num_frames` | `25` | video only (max 121) |
| `video_height` / `video_width` | `480` / `768` | video only |
| `origin_fps` | – | source fps for video inputs |

Prompt rendering (the Qwen chat template plus Lance's per-task system prompt)
is applied automatically by the registry — pass a plain prompt.

#### Command — text-to-image (default)

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
    --model bytedance-research/Lance \
    --deploy-config vllm_omni/deploy/lance.yaml \
    --prompt "a corgi astronaut on the moon, cinematic" \
    --height 1024 --width 1024 \
    --num-inference-steps 30 --seed 42 \
    --extra-body '{"cfg_text_scale": 4.0, "timestep_shift": 3.5}' \
    --output /tmp/lance_t2i.png
```

Defaults match upstream `inference_lance.sh`: 30 denoising steps,
`timestep_shift 3.5`, text CFG 4.0, seed 42, 1024×1024. `lance.yaml` supplies
the engine knobs (`pipeline: lance`, `trust_remote_code`, `max_num_seqs: 1`,
`enforce_eager`) — Lance's `config.json` is descriptive metadata, so the
pipeline cannot always be auto-detected from the model dir alone.

#### Command — image edit (img2img)

```bash
python examples/offline_inference/image_to_image/image_edit.py \
    --model bytedance-research/Lance \
    --deploy-config vllm_omni/deploy/lance.yaml \
    --image /path/to/input.png \
    --prompt "Convert this into a vibrant cartoon-style illustration" \
    --num-inference-steps 30 --seed 42 \
    --extra-args '{"cfg_text_scale": 4.0, "timestep_shift": 3.5}' \
    --output /tmp/lance_i2i.png
```

> **Note**: `image_edit.py` spells this flag `--extra-args`, while the other
> task examples use `--extra-body`. Same mechanism, different flag name.

The Lance-native VAE prefill scatters Wan2.2 latents into the LLM query
sequence; no separate image encoder is needed.

#### Command — text-to-video

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
    --model bytedance-research/Lance/Lance_3B_Video \
    --model-class-name LancePipeline \
    --prompt "a cat playing piano, cinematic" \
    --height 480 --width 768 --num-frames 25 \
    --num-inference-steps 30 --fps 8 --seed 42 \
    --extra-body '{"cfg_text_scale": 4.0, "timestep_shift": 3.5}' \
    --output /tmp/lance_t2v.mp4
```

#### Command — image-to-video

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
    --model bytedance-research/Lance/Lance_3B_Video \
    --model-class-name LancePipeline \
    --image /path/to/first.png \
    --prompt "the scene comes to life with smooth camera motion" \
    --height 480 --width 768 --num-frames 25 \
    --num-inference-steps 30 --fps 8 --seed 42 \
    --extra-body '{"cfg_text_scale": 4.0, "timestep_shift": 3.5}' \
    --output /tmp/lance_i2v.mp4
```

Use the `Lance_3B_Video` subfolder for any video path so the 3-D
`latent_pos_embed` table is loaded; image / understanding paths can point at
the top-level repo and resolve the right sub-checkpoint automatically.

> **Note**: the video task examples select the pipeline with
> `--model-class-name` because they do not (yet) accept `--deploy-config`, and
> they have no flag for `trust_remote_code` / `max_num_seqs`. If loading fails,
> fall back to the dedicated example below.

#### Command — video edit and understanding (dedicated example)

`video2video`, `img2text` and `video2text` have no shared task example to run
through — there is no `video_to_video` example, and the two `x2t` paths are
autoregressive text generation rather than diffusion. They keep using the
dedicated example below:

```bash
# Video edit (video2video)
python examples/offline_inference/lance/end2end.py \
    --model bytedance-research/Lance/Lance_3B_Video --modality video2video \
    --video-path /path/to/clip.mp4 \
    --prompts "make it look like a watercolor painting" \
    --num-frames 25 --video-height 480 --video-width 768 \
    --steps 30 --output ./out
```

```bash
# Image → text (caption / VQA)
python examples/offline_inference/lance/end2end.py \
    --model bytedance-research/Lance --modality img2text \
    --image-path /path/to/photo.jpg \
    --prompts "Describe this image in detail." \
    --do-sample --text-temperature 0.8

# Video → text
python examples/offline_inference/lance/end2end.py \
    --model bytedance-research/Lance --modality video2text \
    --video-path /path/to/clip.mp4 \
    --prompts "What is happening in this video?"
```

Sampling is enabled by default at `--text-temperature 0.8` for the
understanding paths because Lance's greedy decoder emits an immediate EOS for
many prompts.

#### Verification

```bash
pytest -s -v tests/e2e/online_serving/test_lance.py
```

#### Notes

- BF16 footprint: ~7 GB LLM + Qwen2.5-VL ViT + Wan2.2 VAE; comfortably fits
  on a single 16 GB+ GPU for `Lance_3B`.
- `rope_scaling = {"type": "mrope", "mrope_section": [16, 24, 24]}` is wired
  through `BagelRotaryEmbedding`. `text2img` uses scalar position ids
  (BAGEL-equivalent); `img2text` / `video` / edit paths thread per-axis 3-D
  position ids.
- `video_edit` quality is more abstract than `text2video` at the same
  resolution (known position-id offset between VAE-prefill and gen-latent
  blocks). Functionally correct end-to-end.

## Online Serving

Lance supports all single-stage modalities via the OpenAI-compatible
`/v1/chat/completions` API.

### Launch

```bash
bash examples/online_serving/lance/run_server.sh
# or, with overrides
MODEL=bytedance-research/Lance \
DEPLOY_CONFIG=vllm_omni/deploy/lance.yaml \
PORT=8091 \
    bash examples/online_serving/lance/run_server.sh
```

For `text2video` / `video2video`, set
`MODEL=bytedance-research/Lance/Lance_3B_Video`.

### Send requests

```bash
# Text-to-image
python examples/online_serving/lance/openai_chat_client.py \
    --prompt "A cute corgi astronaut on the moon, cinematic" \
    --modality text2img --output corgi.png

# Image edit
python examples/online_serving/lance/openai_chat_client.py \
    --prompt "Convert this into a vibrant cartoon-style illustration" \
    --modality img2img --image-url path/to/photo.png \
    --output edited.png

# Image understanding
python examples/online_serving/lance/openai_chat_client.py \
    --prompt "Describe this image" \
    --modality img2text --image-url photo.jpg
```

The client is shared with BAGEL — same OpenAI message format, same
`modalities` / `num_inference_steps` / `seed` / `height` / `width` knobs.
