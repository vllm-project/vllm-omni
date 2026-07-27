# MammothModa2

> MammothModa2-Preview and MammothModa2-Dev unified understanding and generation

## Summary

- Vendor: ByteDance Research
- Models: `bytedance-research/MammothModa2-Preview`, `bytedance-research/MammothModa2-Dev`
- Tasks: Preview and Dev text-to-image (AR → DiT); Dev text/image understanding
- Mode: Offline inference
- Maintainer: Community

## When to use this recipe

Use this recipe to run MammothModa2-Preview through shared task-oriented
offline examples. Text-to-image uses the shared image example (`text_to_image.py`) instead of a model-specific script.
The generic example formats the AR prompt, drives the AR → DiT stage pipeline,
and forwards MammothModa2-specific generation parameters through the
pipeline-declared `extra_body` contract.

MammothModa2's DiT stage consumes its inputs through the multi-stage kwargs
interface (not `OmniDiffusionRequest`), so its generation knobs
(`text_guidance_scale`, `cfg_range`, `num_inference_steps`) are passed via
`--extra-body` rather than the standard `--num-inference-steps` / `--cfg-scale`
flags. Image size uses the standard `--height` / `--width` flags.

## References

- Upstream model:
  [`bytedance-research/MammothModa2-Preview`](https://huggingface.co/bytedance-research/MammothModa2-Preview)
- Dev model:
  [`bytedance-research/MammothModa2-Dev`](https://huggingface.co/bytedance-research/MammothModa2-Dev)
- Related offline example:
  [`examples/offline_inference/text_to_image/text_to_image.py`](../../examples/offline_inference/text_to_image/text_to_image.py)
- Related T2T/I2T example:
  [`examples/offline_inference/x_to_text/x_to_text.py`](../../examples/offline_inference/x_to_text/x_to_text.py)
- Declared parameters:
  [`vllm_omni/model_extras/mammothmodal2_preview.py`](../../vllm_omni/model_extras/mammothmodal2_preview.py)
- Deploy config:
  [`vllm_omni/deploy/mammoth_moda2.yaml`](../../vllm_omni/deploy/mammoth_moda2.yaml)

## Hardware Support

The default deploy config runs both the AR and DiT stages on a single GPU
(`devices: "0"`). The committed `gpu_memory_utilization` split (stage-0 AR `0.5`,
stage-1 DiT `0.3`) is sized for an ~80 GB GPU. The model also fits on a 48 GB GPU
after rebalancing the split so the AR weights (~23 GB) leave room for the KV
cache — see the note under *1x L40S 48GB*.

## GPU

### 1x L40S 48GB

> **48 GB config adjustment:** the committed
> `vllm_omni/deploy/mammoth_moda2.yaml` uses
> `gpu_memory_utilization` 0.5 / 0.3 (sized for ~80 GB). To fit on a 48 GB L40S,
> set the stage-0 (AR) value to `0.8` and the stage-1 (DiT) value to `0.16`
> before running. (On an ~80 GB GPU, leave the defaults unchanged.)

#### Environment

- OS: Linux
- Python: Match the repository requirements for your checkout
- Driver / runtime: NVIDIA CUDA environment with one L40S 48 GB (verified) or an
  ~80 GB GPU for the default config
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Offline Commands

Download the model:

```bash
hf download bytedance-research/MammothModa2-Preview --local-dir ./MammothModa2-Preview
```

Run text-to-image with the shared offline example from the repository root. The
deploy config sets `trust_remote_code`, so no extra flag is needed. Forward the
MammothModa2 generation parameters as a JSON object through `--extra-body`:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model ./MammothModa2-Preview \
  --stage-configs-path vllm_omni/deploy/mammoth_moda2.yaml \
  --prompt "A stylish woman riding a motorcycle in NYC, movie poster style" \
  --height 1024 \
  --width 1024 \
  --extra-body '{"text_guidance_scale": 4.0, "cfg_range": [0.0, 1.0], "num_inference_steps": 50}' \
  --output mammoth_t2i.png
```

The `--extra-body` JSON forwards MammothModa2-specific parameters into
`OmniDiffusionSamplingParams.extra_args`. Keys are filtered against the model's
declared `extra_body_params` (see
[`vllm_omni/model_extras/mammothmodal2_preview.py`](../../vllm_omni/model_extras/mammothmodal2_preview.py)),
so unknown keys for MammothModa2 are silently dropped:

- `text_guidance_scale` — classifier-free guidance scale for the DiT stage
  (default `9.0`; CFG is active only when `> 1.0`).
- `cfg_range` — relative step range `[start, end]` over which CFG is applied
  (default `[0.0, 1.0]`).
- `num_inference_steps` — number of DiT denoising steps (default `50`).

`--height` and `--width` must be multiples of 16.

Run text-to-text through the shared understanding example. It recognizes the
MammothModa2 checkpoint and automatically selects `mammoth_moda2_ar.yaml`:

```bash
python examples/offline_inference/x_to_text/x_to_text.py \
  --model ./MammothModa2-Preview \
  --prompt "Explain multimodal generation in three sentences."
```

Add an image for image-to-text or image summarization. The shared example
uses MammothModa2's chat and vision-token template:

```bash
python examples/offline_inference/x_to_text/x_to_text.py \
  --model ./MammothModa2-Preview \
  --image /path/to/input.jpg \
  --prompt "Please summarize the content of this image."
```

#### Verification

The example writes the generated image to the `--output` path. Confirm the file
exists and is a valid image:

```bash
ls -lh mammoth_t2i.png
python -c "from PIL import Image; print(Image.open('mammoth_t2i.png').size)"
```

## MammothModa2-Dev unified inference

MammothModa2-Dev uses a Qwen3-VL AR backbone, while MammothModa2-Preview uses
Qwen2.5-VL. vLLM-Omni selects the matching implementation from the nested
`llm_config.model_type`; no checkpoint edits or `trust_remote_code` flag are
required.

Text-to-text and image-to-text use the AR-only deploy. Text-to-image loads the
Qwen3 generation experts (`gen_mlp`), extra visual vocabulary and image head,
then sends the generated visual tokens and hidden states to the DiT stage.

Download the checkpoint:

```bash
hf download bytedance-research/MammothModa2-Dev --local-dir ./MammothModa2-Dev
```

Run text-to-text through the shared understanding example. It recognizes the
Dev checkpoint as MammothModa2 and automatically selects
`mammoth_moda2_ar.yaml`:

```bash
python examples/offline_inference/x_to_text/x_to_text.py \
  --model ./MammothModa2-Dev \
  --prompt "Explain multimodal generation in three sentences."
```

Add an image for image-to-text or image summarization:

```bash
python examples/offline_inference/x_to_text/x_to_text.py \
  --model ./MammothModa2-Dev \
  --image ./image.png \
  --prompt "Please summarize the content of this image."
```

The Dev checkpoint is approximately 47.55 GiB on disk. In the verified AR-only
run, loaded model weights used approximately 16.97 GiB of GPU memory before KV
and encoder caches. Allow additional GPU memory for those caches and the input
image.
