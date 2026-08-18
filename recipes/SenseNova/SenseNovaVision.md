# SenseNova-Vision-7B-MoT

> SenseNova-Vision-7B-MoT unified image understanding, generation, dense perception, and 3D reconstruction through the model-specific offline example and the OpenAI-compatible online serving example.

## Summary

- Vendor: SenseNovaVision
- Model: `sensenova/SenseNova-Vision-7B-MoT`
- Task: Text-to-image, image-to-image, image-to-text, text-to-text, dense perception (depth / normal / segmentation), dense detection, dense OCR, multi-view camera pose estimation, and multi-view 3D reconstruction
- Mode: Offline inference and OpenAI-compatible online serving
- Maintainer: Community

## When to use this recipe

Use this recipe when you want to run SenseNova-Vision-7B-MoT through the
model-specific example clients. The offline example
(`examples/offline_inference/sensenova_vision/end2end.py`) covers the full modality
matrix with `--modality` flags, official per-mode default prompts, and
client-side decoders for dense and 3D outputs. The online example
(`examples/online_serving/sensenova_vision/openai_chat_client.py`) demonstrates the
mixed `caption_generate` mode (image + intermediate text) plus image
understanding and image generation through the OpenAI chat completions API.

## References

- Upstream model:
  [`sensenova/SenseNova-Vision-7B-MoT`](https://huggingface.co/sensenova/SenseNova-Vision-7B-MoT)
- Related offline example:
  [`examples/offline_inference/sensenova_vision/end2end.py`](../../examples/offline_inference/sensenova_vision/end2end.py)
- Related online example:
  [`examples/online_serving/sensenova_vision/openai_chat_client.py`](../../examples/online_serving/sensenova_vision/openai_chat_client.py)
- Default deploy config:
  [`vllm_omni/deploy/sensenova_vision.yaml`](../../vllm_omni/deploy/sensenova_vision.yaml)

## Hardware Support

This recipe documents the CUDA layout used by the in-repo SenseNovaVision deploy
config. The default two-stage config shares one 80 GB GPU; for more headroom,
move the diffusion stage to a second GPU in a custom deploy config.

## GPU

### 1x A100 80GB

#### Environment

- OS: Linux
- Python: Match the repository requirements for your checkout
- Driver / runtime: NVIDIA CUDA environment with one A100 80 GB GPU
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Offline Commands

Run the full modality matrix from the repository root with the model-specific
offline example. The default deploy config is `vllm_omni/deploy/sensenova_vision.yaml`
(two-stage Thinker + DiT sharing GPU 0); override with `--deploy-config` for
custom topologies.

Text-to-text:

```bash
python examples/offline_inference/sensenova_vision/end2end.py \
  --modality text2text \
  --prompts "What is the capital of France?" \
  --output /tmp/sensenova_vision
```

Image-to-text (captioning):

```bash
python examples/offline_inference/sensenova_vision/end2end.py \
  --modality img2text \
  --image-path /path/to/photo.jpg \
  --output /tmp/sensenova_vision
```

Dense detection (structured `<bbox>` text parsed with `parse_bbox`):

```bash
python examples/offline_inference/sensenova_vision/end2end.py \
  --modality dense_detection \
  --image-path /path/to/photo.jpg \
  --output /tmp/sensenova_vision
```

Dense OCR (structured text parsed with `parse_points`):

```bash
python examples/offline_inference/sensenova_vision/end2end.py \
  --modality dense_OCR \
  --image-path /path/to/photo.jpg \
  --output /tmp/sensenova_vision
```

Text-to-image:

```bash
python examples/offline_inference/sensenova_vision/end2end.py \
  --modality text2img \
  --prompts "A cute corgi astronaut on the moon, cinematic" \
  --height 1024 \
  --width 1024 \
  --steps 50 \
  --output /tmp/sensenova_vision
```

Image-to-image (edit):

```bash
python examples/offline_inference/sensenova_vision/end2end.py \
  --modality img2img \
  --image-path /path/to/photo.jpg \
  --prompts "Turn this image into a vibrant cartoon-style illustration." \
  --output /tmp/sensenova_vision
```

Image-to-dense (depth / normal / segmentation decoded with
`decode_depth` / `decode_normal` / `decode_segmentation`):

```bash
python examples/offline_inference/sensenova_vision/end2end.py \
  --modality img2dense \
  --dense-task depth \
  --image-path /path/to/photo.jpg \
  --output /tmp/sensenova_vision

python examples/offline_inference/sensenova_vision/end2end.py \
  --modality img2dense \
  --dense-task normal \
  --image-path /path/to/photo.jpg \
  --output /tmp/sensenova_vision

python examples/offline_inference/sensenova_vision/end2end.py \
  --modality img2dense \
  --dense-task segmentation \
  --image-path /path/to/photo.jpg \
  --output /tmp/sensenova_vision
```

Multi-view camera pose estimation (at least two images; parsed with
`parse_camera_pose`):

```bash
python examples/offline_inference/sensenova_vision/end2end.py \
  --modality multi-img2text \
  --image-path /path/to/view1.png /path/to/view2.png /path/to/view3.png \
  --output /tmp/sensenova_vision
```

Multi-view 3D reconstruction (per-view point maps decoded with
`decode_point_map`):

```bash
python examples/offline_inference/sensenova_vision/end2end.py \
  --modality recon3d \
  --image-path /path/to/view1.png /path/to/view2.png /path/to/view3.png \
  --output /tmp/sensenova_vision
```

Mixed text + image (`caption_generate` returns an image plus the intermediate
caption text):

```bash
python examples/offline_inference/sensenova_vision/end2end.py \
  --modality mixed \
  --image-path /path/to/photo.jpg \
  --output /tmp/sensenova_vision
```

Outputs are written to `--output` with deterministic names (`text2img_0.png`,
`img2text_0.txt`, `img2dense_0_depth.npy`, `recon3d_0_view0.npy`, ...).

Model-specific generation parameters can be forwarded with the dedicated CLI
flags (`--cfg-text-scale`, `--cfg-img-scale`, `--timestep-shift`,
`--max-think-tokens`) or as a JSON object through `--extra-args`:

```bash
python examples/offline_inference/sensenova_vision/end2end.py \
  --modality text2img \
  --prompts "A cute corgi astronaut on the moon, cinematic" \
  --height 1024 \
  --width 1024 \
  --steps 50 \
  --seed 42 \
  --extra-args '{"cfg_text_scale": 4.0, "cfg_img_scale": 1.5, "timestep_shift": 3.0}' \
  --output /tmp/sensenova_vision
```

#### Online Commands

Start the OpenAI-compatible server:

```bash
bash examples/online_serving/sensenova_vision/run_server.sh
```

Equivalently, run `vllm serve` directly:

```bash
vllm serve sensenova/SenseNova-Vision-7B-MoT \
  --omni \
  --port 8092 \
  --deploy-config vllm_omni/deploy/sensenova_vision.yaml
```

Send a mixed text + image request (`caption_generate` returns both an image
and the intermediate caption text) with the Python client:

```bash
cd examples/online_serving/sensenova_vision
python openai_chat_client.py \
  --modality mixed \
  --image-url /path/to/photo.jpg \
  --prompt "<image> Please briefly describe the contents of the image. Please respond with interleaved segmentation masks for the corresponding parts of the answer." \
  --output /tmp/sensenova_vision_mixed.png
```

Image understanding (img2text):

```bash
python openai_chat_client.py \
  --modality img2text \
  --image-url /path/to/photo.jpg \
  --prompt "What are the main objects in this scene and their relationships?"
```

Text-to-image (text2img):

```bash
python openai_chat_client.py \
  --modality text2img \
  --prompt "A cute corgi astronaut on the moon, cinematic" \
  --output /tmp/sensenova_vision_text2img.png
```

The `--cfg-text-scale`, `--cfg-img-scale`, `--timestep-shift`, and
`--max-think-tokens` flags forward SenseNovaVision-specific parameters through the
pipeline-declared `extra_args` contract.

Equivalent curl for text2img with SenseNovaVision-specific generation parameters:

```bash
curl http://localhost:8092/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "A cute corgi astronaut on the moon, cinematic"}
        ]
      }
    ],
    "modalities": ["image"],
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "cfg_text_scale": 4.0,
    "cfg_img_scale": 1.5,
    "seed": 42
  }'
```

The important part is that model-specific keys such as `cfg_text_scale`,
`cfg_img_scale`, `cfg_interval`, `cfg_renorm_type`, `cfg_renorm_min`, and
`timestep_shift` belong in the request body. The serving layer routes the
declared keys into `OmniDiffusionSamplingParams.extra_args` for the SenseNovaVision
pipeline.

#### Verification

Decode the returned data URL into an image:

```bash
curl http://localhost:8092/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "A ceramic teapot on a wooden table"}
        ]
      }
    ],
    "modalities": ["image"],
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 25,
    "seed": 42
  }' | jq -r '.choices[]?.message.content[]? | select(.image_url.url) | .image_url.url' \
    | head -n 1 \
    | cut -d',' -f2- \
    | base64 -d > /tmp/sensenova_vision_online.png

ls -lh /tmp/sensenova_vision_online.png
```

### 2x CUDA GPUs

Create a custom deploy config from `vllm_omni/deploy/sensenova_vision.yaml` and move
the diffusion stage to GPU 1:

```yaml
stages:
  - stage_id: 0
    devices: "0"
    # keep the remaining stage-0 settings from sensenova_vision.yaml
  - stage_id: 1
    devices: "1"
    # keep the remaining stage-1 settings from sensenova_vision.yaml
```

Then start serving with that config:

```bash
vllm serve sensenova/SenseNova-Vision-7B-MoT \
  --omni \
  --port 8092 \
  --deploy-config /path/to/custom_sensenova_vision_2gpu.yaml
```

Use the online curl request from the `1x A100 80GB` section to verify that the
server returns an image.
