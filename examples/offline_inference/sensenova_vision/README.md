# SenseNova-Vision-7B-MoT: Offline inference

Offline end-to-end example for
[`sensenova/SenseNova-Vision-7B-MoT`](https://huggingface.co/sensenova/SenseNova-Vision-7B-MoT),
covering the full input/output modality matrix. SenseNova-Vision is a
Bagel-fork Mixture-of-Transformers (MoT) model with a two-stage topology:
Stage 0 (Thinker, AR) for text/understanding and Stage 1 (DiT, diffusion)
for image generation.

## Requirements

- A CUDA GPU with ~80 GB VRAM for the full two-stage model (see the deploy YAML).
- Install vllm-omni from this repository (`pip install -e .`).
- The model checkpoint is downloaded from Hugging Face on first run.

## Supported modalities

| `--modality` | Input | Output | SenseNovaVision mode |
| :----------- | :---- | :----- | :------------- |
| `text2text` | text | text | `understanding` |
| `img2text` | image | text | `understanding` |
| `dense_detection` | image | text (parsed with `parse_bbox`) | `dense_detection` |
| `dense_OCR` | image | text (parsed OCR boxes) | `dense_OCR` |
| `text2img` | text | image | `generate` |
| `img2img` | image | image | `edit` |
| `img2dense` | image | decoded depth / normal / segmentation maps | `dense_perception` |
| `multi-img2text` | multiple images | text (parsed with `parse_camera_pose`) | `understanding` |
| `recon3d` | multiple images | per-view point maps (+ optional text) | `recon3d` |
| `mixed` | image | image + intermediate caption text | `caption_generate` |

Run from the repository root:

```bash
python examples/offline_inference/sensenova_vision/end2end.py \
    --modality <modality> \
    [--image-path path1 [path2 ...]] \
    [--prompts "text prompt"] \
    --output ./out
```

The example uses the official SenseNova-Vision prompts when `--prompts` is
omitted, and writes deterministic outputs (`<modality>_<index>.png`,
`<modality>_<index>.txt`, and `.npy` maps for dense/recon3d) into `--output`.

## Examples

### Text to text (text2text)

```bash
python examples/offline_inference/sensenova_vision/end2end.py \
    --modality text2text \
    --prompts "What is the capital of France?"
```

### Image to text (img2text)

```bash
python examples/offline_inference/sensenova_vision/end2end.py \
    --modality img2text \
    --image-path /path/to/photo.jpg \
    --prompts "What are the main objects in this scene and their relationships?"
```

### Dense detection (dense_detection)

```bash
python examples/offline_inference/sensenova_vision/end2end.py \
    --modality dense_detection \
    --image-path /path/to/photo.jpg
```

The detection text is saved to `dense_detection_0.txt` and also parsed into
normalized `<label>: [[x0, y0, x1, y1], ...]` boxes with `parse_bbox` on the
console.

### Dense OCR (dense_OCR)

```bash
python examples/offline_inference/sensenova_vision/end2end.py \
    --modality dense_OCR \
    --image-path /path/to/text.jpg
```

### Text to image (text2img)

```bash
python examples/offline_inference/sensenova_vision/end2end.py \
    --modality text2img \
    --prompts "A cute corgi astronaut on the moon, cinematic" \
    --height 1024 --width 1024 \
    --output ./out
```

### Image to image (img2img)

```bash
python examples/offline_inference/sensenova_vision/end2end.py \
    --modality img2img \
    --image-path /path/to/photo.jpg \
    --prompts "Turn this image into a vibrant cartoon-style illustration."
```

### Dense perception (img2dense: depth / normal / segmentation)

```bash
# depth
python examples/offline_inference/sensenova_vision/end2end.py \
    --modality img2dense --dense-task depth \
    --image-path /path/to/photo.jpg

# normals
python examples/offline_inference/sensenova_vision/end2end.py \
    --modality img2dense --dense-task normal \
    --image-path /path/to/photo.jpg

# segmentation
python examples/offline_inference/sensenova_vision/end2end.py \
    --modality img2dense --dense-task segmentation \
    --image-path /path/to/photo.jpg
```

The raw prediction image is saved as `img2dense_<i>.png` and the decoded map
as `img2dense_<i>_{depth,normal,segmentation}.npy` (float32 depth/normal,
int32/uint8 class-index segmentation).

### Multi-view camera pose (multi-img2text)

```bash
python examples/offline_inference/sensenova_vision/end2end.py \
    --modality multi-img2text \
    --image-path /path/to/view1.png /path/to/view2.png /path/to/view3.png
```

The tagged `<quat>/<offset>/<scale>` text is saved and parsed with
`parse_camera_pose` into `rotation` (N×4 quaternions) and `translation` (N×3)
lists on the console.

### Multi-view 3D reconstruction (recon3d)

```bash
python examples/offline_inference/sensenova_vision/end2end.py \
    --modality recon3d \
    --image-path /path/to/view1.png /path/to/view2.png /path/to/view3.png
```

Per-view point maps are decoded with `decode_point_map` and saved as
`recon3d_<i>_view<j>.npy` (H×W×3 float32). If the pipeline also emits text,
it is saved as `recon3d_<i>.txt`.

### Mixed text + image (mixed / caption_generate)

```bash
python examples/offline_inference/sensenova_vision/end2end.py \
    --modality mixed \
    --image-path /path/to/photo.jpg
```

The `caption_generate` mode returns both the generated segmentation image and
the intermediate caption text; both are saved.

## Configuration

The example builds the two-stage engine with the default deploy config
`vllm_omni/deploy/sensenova_vision.yaml` (Thinker + DiT sharing GPU 0). To use a
custom topology or device layout, pass `--deploy-config /path/to/config.yaml`.

Generation knobs mirror the SenseNovaVision per-mode `BASE_PARAMS`: `--steps`,
`--seed`, `--height`, `--width`, `--cfg-text-scale`, `--cfg-img-scale`,
`--timestep-shift`, `--max-think-tokens`, and `--extra-args` (JSON object for
keys such as `cfg_interval`, `cfg_renorm_type`, `cfg_renorm_min`, `think`,
`do_sample`, `text_temperature`). When omitted, the SenseNovaVision pipeline fills
the per-mode defaults.
