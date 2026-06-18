# BAGEL-7B-MoT

> BAGEL image generation through the shared online and offline image examples

## Summary

- Vendor: ByteDance Seed
- Model: `ByteDance-Seed/BAGEL-7B-MoT`
- Task: Text-to-image and image-to-image generation
- Mode: Offline inference and OpenAI-compatible online serving
- Maintainer: Community

## When to use this recipe

Use this recipe when you want to run BAGEL through the shared image examples
instead of model-specific example clients. The generic examples can format
BAGEL text-to-image and image-to-image prompts, select the image output
modality, attach reference images, and forward BAGEL-specific generation
parameters through the pipeline-declared `extra_args` contract.

## References

- Upstream model:
  [`ByteDance-Seed/BAGEL-7B-MoT`](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT)
- Related offline example:
  [`examples/offline_inference/text_to_image/text_to_image.py`](../../examples/offline_inference/text_to_image/text_to_image.py)
- Related offline image-to-image example:
  [`examples/offline_inference/image_to_image/image_edit.py`](../../examples/offline_inference/image_to_image/image_edit.py)
- Related online example:
  [`examples/online_serving/text_to_image/openai_chat_client.py`](../../examples/online_serving/text_to_image/openai_chat_client.py)
- Related online image-to-image example:
  [`examples/online_serving/image_to_image/openai_chat_client.py`](../../examples/online_serving/image_to_image/openai_chat_client.py)
- Default deploy configs:
  [`vllm_omni/deploy/bagel.yaml`](../../vllm_omni/deploy/bagel.yaml),
  [`vllm_omni/deploy/bagel_single_stage.yaml`](../../vllm_omni/deploy/bagel_single_stage.yaml)

## Hardware Support

This recipe documents the CUDA layouts used by the in-repo BAGEL deploy
configs. The default two-stage config shares one 80 GB GPU; for more headroom,
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

Run text-to-image with the shared offline example from the repository root.
Pick the topology with `--stage-configs-path` and forward BAGEL-specific
generation parameters as a JSON object through `--extra-body`:

```bash
# Two-stage (Thinker + DiT), shares one GPU by default
python examples/offline_inference/text_to_image/text_to_image.py \
  --model ByteDance-Seed/BAGEL-7B-MoT \
  --stage-configs-path vllm_omni/deploy/bagel.yaml \
  --prompt "A beautiful sunset over mountains" \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 50 \
  --negative-prompt "blurry, low quality" \
  --seed 42 \
  --extra-body '{"timestep_shift": 3.0, "cfg_text_scale": 4.0, "cfg_img_scale": 1.5, "cfg_interval": [0.4, 1.0], "cfg_renorm_type": "global", "cfg_renorm_min": 0.0, "think": false}' \
  --output /tmp/bagel_text2img.png
```

```bash
# Single-stage (DiT only, with internal LLM/ViT/VAE)
python examples/offline_inference/text_to_image/text_to_image.py \
  --model ByteDance-Seed/BAGEL-7B-MoT \
  --stage-configs-path vllm_omni/deploy/bagel_single_stage.yaml \
  --prompt "A beautiful sunset over mountains" \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 50 \
  --seed 42 \
  --extra-body '{"timestep_shift": 3.0, "cfg_text_scale": 4.0, "cfg_interval": [0.4, 1.0], "cfg_renorm_type": "global", "cfg_renorm_min": 0.0}' \
  --output /tmp/bagel_text2img_single.png
```

The `--extra-body` JSON forwards BAGEL-specific parameters into
`OmniDiffusionSamplingParams.extra_args`. Keys are filtered against the model's
declared `extra_body_params` (see
[`vllm_omni/model_extras/bagel.py`](../../vllm_omni/model_extras/bagel.py)), so
unknown keys for BAGEL are silently dropped. This is the preferred way to pass
model-specific params — `timestep_shift` (use `3.0` for 1024×1024; the
deprecated `--timesteps-shift` flag defaults to `1.0` and is NextStep-oriented),
`cfg_text_scale`, `cfg_img_scale`, `cfg_interval`, `cfg_renorm_type`,
`cfg_renorm_min`, and `think`. `--seed` makes generation reproducible.

> **Note**: the two-stage `bagel.yaml` runs both stages on GPU 0. If you hit
> CUDA OOM during warmup, lower stage-0 `gpu_memory_utilization` in the YAML or
> split the stages across two GPUs (see the `2x CUDA GPUs` section).

Run image-to-image with the shared offline image-to-image example:

```bash
python examples/offline_inference/image_to_image/image_edit.py \
  --model ByteDance-Seed/BAGEL-7B-MoT \
  --prompt "Make the scene look like a watercolor painting" \
  --image /tmp/bagel_text2img.png \
  --height 512 \
  --width 512 \
  --extra-args '{"cfg_text_scale": 4.0, "cfg_img_scale": 1.5}' \
  --output /tmp/bagel_img2img.png
```

The `--extra-args` JSON forwards BAGEL-specific parameters (e.g. `cfg_text_scale`,
`cfg_img_scale`, `cfg_interval`, `cfg_renorm_type`) into
`OmniDiffusionSamplingParams.extra_args` via the model-extras registry.

#### Online Commands

Start the OpenAI-compatible server:

```bash
vllm serve ByteDance-Seed/BAGEL-7B-MoT --omni --port 8091
```

To use the single-stage topology online:

```bash
vllm serve ByteDance-Seed/BAGEL-7B-MoT \
  --omni \
  --port 8091 \
  --deploy-config vllm_omni/deploy/bagel_single_stage.yaml
```

Send a text-to-image request with BAGEL-specific generation parameters:

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "<|im_start|>A beautiful sunset over mountains<|im_end|>"}
        ]
      }
    ],
    "modalities": ["image"],
    "extra_body": {
      "height": 512,
      "width": 512,
      "num_inference_steps": 50,
      "cfg_text_scale": 4.0,
      "cfg_img_scale": 1.5,
      "negative_prompt": "blurry, low quality",
      "seed": 42
    }
  }'
```

The important part is that BAGEL-specific keys such as `cfg_text_scale`,
`cfg_img_scale`, `cfg_interval`, `cfg_renorm_type`, and `cfg_renorm_min` belong
in `extra_body`. The serving layer routes the declared keys into
`OmniDiffusionSamplingParams.extra_args` for the BAGEL pipeline.

#### Verification

Decode the returned data URL into an image:

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "<|im_start|>A ceramic teapot on a wooden table<|im_end|>"}
        ]
      }
    ],
    "modalities": ["image"],
    "extra_body": {
      "height": 512,
      "width": 512,
      "num_inference_steps": 25,
      "cfg_text_scale": 4.0,
      "seed": 42
    }
  }' | jq -r '.choices[]?.message.content[]? | select(.image_url.url) | .image_url.url' \
    | head -n 1 \
    | cut -d',' -f2- \
    | base64 -d > /tmp/bagel_online.png

ls -lh /tmp/bagel_online.png
```

### 2x CUDA GPUs

Create a custom deploy config from `vllm_omni/deploy/bagel.yaml` and move the
diffusion stage to GPU 1:

```yaml
stages:
  - stage_id: 0
    devices: "0"
    # keep the remaining stage-0 settings from bagel.yaml
  - stage_id: 1
    devices: "1"
    # keep the remaining stage-1 settings from bagel.yaml
```

Then start serving with that config:

```bash
vllm serve ByteDance-Seed/BAGEL-7B-MoT \
  --omni \
  --port 8091 \
  --deploy-config /path/to/custom_bagel_2gpu.yaml
```

Use the online curl request from the `1x A100 80GB` section to verify that the
server returns an image.
