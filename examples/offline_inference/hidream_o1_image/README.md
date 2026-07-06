# HiDream-O1-Image (offline inference)

[HiDream-O1-Image](https://github.com/HiDream-ai/HiDream-O1-Image) is a
Pixel-level Unified Transformer: a single Qwen3-VL-derived transformer that
performs flow-matching diffusion directly on raw pixel patches (no VAE, no
separate text encoder). Supports text-to-image, instruction-based editing,
multi-reference personalization, and layout-bbox spatial grounding.

## Scripts

| Script | Description |
|---|---|
| `text_to_image_hidream_o1.py` | Text-to-image generation |
| `image_edit_hidream_o1.py` | Image editing (1 ref) and multi-reference personalization (2+ refs) |
| `layout_control_hidream_o1.py` | Layout-bbox spatial grounding with bounding boxes |

## Run

### Text-to-image

```bash
# Dev variant (28 steps, no CFG — fast)
python text_to_image_hidream_o1.py \
    --model HiDream-ai/HiDream-O1-Image-Dev \
    --prompt "A dog holds a sign that says hello." \
    --model-type dev \
    --output output_dev.png

# Full variant (50 steps, guidance 5.0 — highest quality)
python text_to_image_hidream_o1.py \
    --model HiDream-ai/HiDream-O1-Image \
    --prompt "A cute cat sitting on a windowsill at sunset." \
    --height 1024 --width 1024 \
    --model-type full \
    --output output_full.png
```

### Image editing (1 reference image)

```bash
python image_edit_hidream_o1.py \
    --model HiDream-ai/HiDream-O1-Image-Dev \
    --model-type dev \
    --ref-images /path/to/photo.jpg \
    --prompt "Make the background a snowy mountain landscape" \
    --output edited.png
```

### Multi-reference personalization (2+ reference images)

```bash
python image_edit_hidream_o1.py \
    --model HiDream-ai/HiDream-O1-Image-Dev \
    --model-type dev \
    --ref-images person1.jpg person2.jpg person3.jpg \
    --prompt "The person is sitting at a café in Paris" \
    --output personalized.png
```

### Layout-bbox spatial grounding

Bboxes specify where each subject appears in the output. Format: `[x1, x2, y1, y2]`
in xxyy order, values in `[0, 1]` (relative) or `[0, 100]` (percentage).

```bash
# Inline JSON string (person left, dog right)
python layout_control_hidream_o1.py \
    --model HiDream-ai/HiDream-O1-Image-Dev \
    --model-type dev \
    --ref-images person.jpg dog.jpg \
    --layout-bboxes '[[0.0, 0.45, 0.1, 0.9], [0.55, 1.0, 0.1, 0.9]]' \
    --prompt "A person and a dog sitting in a sunny park" \
    --output layout_output.png

# JSON file
python layout_control_hidream_o1.py \
    --model HiDream-ai/HiDream-O1-Image-Dev \
    --ref-images subject.jpg \
    --layout-bboxes /path/to/bboxes.json \
    --prompt "The subject is standing in front of the Eiffel Tower" \
    --output output.png
```

## Key arguments

| Argument | Description |
|---|---|
| `--model-type` | `full` (50 steps, guidance 5.0, shift 3.0) or `dev` (28 steps, no CFG, shift 1.0). |
| `--num-inference-steps` / `--guidance-scale` / `--shift` | Override the `--model-type` preset. |
| `--scheduler` | `default` (`FlowUniPCMultistepScheduler`) or `flow_match` (diffusers `FlowMatchEulerDiscreteScheduler`). |
| `--height` / `--width` | Snapped to the nearest 32-px-aligned resolution internally. |
| `--ref-images` | One or more reference image paths. 1 = editing; 2+ = personalization. |
| `--layout-bboxes` | JSON string or `.json` file with bbox list in xxyy relative coords. |

## Requirements

- `transformers >= 4.57.1` (for `Qwen3VLForConditionalGeneration`).
- A CUDA GPU with ≥ 40 GB VRAM for 1024×1024 (H100 recommended; L4 works at lower resolutions).
- flash-attn is recommended but not required — the pipeline builds a dense attention mask as a
  fallback for non-flash backends (`DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA` to force it).
