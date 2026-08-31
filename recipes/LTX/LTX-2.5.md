# LTX-2.5

> Text-to-video and first-frame image-to-video generation with synchronized audio

## Overview

vLLM-Omni supports the gated
[`Lightricks/LTX-2.5-Diffusers`](https://huggingface.co/Lightricks/LTX-2.5-Diffusers)
checkpoint for T2V and first-frame I2V. Every output includes synchronized
48 kHz stereo audio.

The raw [`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5)
repository is not directly loadable with `--model`. It supplies the canonical
Diffusion VAE checkpoint and the official upsampler and LoRA sidecars used by
the Full/SFT two-stage pipeline, so accept both model licenses and authenticate
before first use.

The vLLM-Omni integration code is provided under the project's
[Apache-2.0 license](../../LICENSE). The LTX-2.5 model weights are separate
artifacts covered by the
[LTX-2.x Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE.md),
not the vLLM-Omni license. Users are responsible for reviewing and complying
with the model license before downloading or using the weights.

## Choose a pipeline

| Pipeline | Mode | Default size | Schedule |
| --- | --- | ---: | ---: |
| `LTX2Pipeline` | Full/SFT one-stage | 960x544 | 30 steps |
| `LTX2TwoStagePipeline` | Full/SFT two-stage | 1920x1088 | 30 + 3 steps |
| `LTX2DistilledOneStagePipeline` | Distilled one-stage | 960x544 | 8 steps |
| `LTX2DistilledTwoStagePipeline` | Distilled two-stage | 1920x1088 | 8 + 3 steps |

Both two-stage pipelines generate at half resolution, apply the official x2
latent upsampler, and run a three-step refinement stage. Select the class with
`--model-class-name`; no `--task-type` flag is required. Supplying one initial
image selects I2V, while omitting it selects T2V.

## Video decoder

LTX-2.5 uses the canonical Native Diffusion VAE decoder by default. To opt in
to the legacy convolutional VAE, set the startup-only stage-0 model extra
`ltx2_use_conv_vae: true`. The choice applies to all requests and all four
pipeline classes above. DiffVAE loads its NATTEN kernel from Hugging Face Hub
during startup. `decode_timestep` and `decode_noise_scale` condition only the
legacy ConvVAE and have no effect when DiffVAE is selected.

For offline Python usage:

```python
omni = Omni(
    model="Lightricks/LTX-2.5-Diffusers",
    model_class_name="LTX2Pipeline",
    stage_overrides='{"0":{"extras":{"ltx2_use_conv_vae":true}}}',
)
```

For online serving:

```bash
vllm serve Lightricks/LTX-2.5-Diffusers \
  --omni \
  --model-class-name LTX2Pipeline \
  --stage-overrides '{"0":{"extras":{"ltx2_use_conv_vae":true}}}'
```

Both decoders are untiled by default. Set `vae_use_tiling` for memory-saving
serial tiling; DiffVAE tiles only above 80 frames or 768 pixels in either spatial
dimension. DiffVAE also supports distributed VAE decode; see the
[VAE Parallelism Guide](../../docs/user_guide/diffusion/parallelism/vae_parallelism.md).
DiffVAE is decoder-only, so I2V still uses the convolutional VAE encoder.

## Prerequisites

```bash
hf auth login
export MODEL=Lightricks/LTX-2.5-Diffusers
```

Install matching vLLM and vLLM-Omni versions, and ensure `ffmpeg` and
`ffprobe` are on `PATH`. I2V requires PyAV backed by an FFmpeg build with the
`libx264` encoder.

## Hardware

| GPU | Status | Recommended scope |
| --- | --- | --- |
| NVIDIA B300 | Verified | All four canonical pipelines |
| NVIDIA B200 or H200 | Capacity-based recommendation; not yet verified | All four canonical pipelines |
| NVIDIA GB200 or GB300 | Capacity-based recommendation; not yet verified | All four canonical pipelines |
| NVIDIA H100 80 GB | FP8 recipe | Distilled one-stage at 960x544 |

The 1920x1088 two-stage examples require about 114 GB of peak GPU memory.
80 GB GPUs do not have enough safety margin for the canonical two-stage
configuration; use a larger GPU or reduce the output configuration.

### H100 80 GB FP8

Use the distilled one-stage pipeline with FP8 and cuDNN attention:

```bash
vllm serve Lightricks/LTX-2.5-Diffusers \
  --omni \
  --model-class-name LTX2DistilledOneStagePipeline \
  --quantization fp8 \
  --diffusion-attention-backend CUDNN_ATTN \
  --host 0.0.0.0 \
  --port 8000
```

Use the T2V or I2V request below after the server is ready.

## Offline inference

Choose values from the pipeline table. For example, the distilled two-stage
path uses:

```bash
export PIPELINE=LTX2DistilledTwoStagePipeline
export WIDTH=1920
export HEIGHT=1088
export STEPS=8
```

### Text-to-video

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model "${MODEL}" \
  --model-class-name "${PIPELINE}" \
  --prompt "A cinematic shot of a red fox walking through a snowy forest at dawn, the camera tracking alongside, snow crunching underfoot." \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --num-frames 121 \
  --num-inference-steps "${STEPS}" \
  --frame-rate 24 \
  --fps 24 \
  --seed 42 \
  --output ltx25-t2v.mp4
```

### First-frame image-to-video

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
  --model "${MODEL}" \
  --model-class-name "${PIPELINE}" \
  --image /absolute/path/to/first-frame.png \
  --prompt "The red fox walks forward while the camera tracks alongside." \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --num-frames 121 \
  --num-inference-steps "${STEPS}" \
  --frame-rate 24 \
  --fps 24 \
  --seed 42 \
  --output ltx25-i2v.mp4
```

LTX-2.5 uses the official CRF-18 first-frame conditioning path by default.

## Online serving

Start one server for the selected pipeline:

```bash
vllm serve "${MODEL}" \
  --omni \
  --model-class-name "${PIPELINE}" \
  --host 0.0.0.0 \
  --port 8000 \
  --stage-init-timeout 900
```

T2V request:

```bash
curl -sS --fail-with-body \
  -X POST http://127.0.0.1:8000/v1/videos/sync \
  -F 'prompt=A cinematic shot of a red fox walking through a snowy forest at dawn, the camera tracking alongside, snow crunching underfoot.' \
  -o ltx25-online-t2v.mp4
```

For I2V, add exactly one first frame:

```bash
export FIRST_FRAME=/absolute/path/to/first-frame.png

curl -sS --fail-with-body \
  -X POST http://127.0.0.1:8000/v1/videos/sync \
  -F "input_reference=@${FIRST_FRAME};type=image/png" \
  -F 'prompt=The red fox walks forward while the camera tracks alongside.' \
  -o ltx25-online-i2v.mp4
```

Restart the server after changing `PIPELINE` because each class selects a
different weight layout and execution topology.

## Constraints

- Online `width`, `height`, `num_frames`, `fps`, and `num_inference_steps`
  are optional and use the selected pipeline defaults when omitted. `seed` is
  optional and random when omitted.
- `num_frames` must be `8k+1` when overridden.
- One-stage width and height must be divisible by 32; two-stage final
  dimensions must be divisible by 64.
- Full/SFT pipelines accept negative prompts. Distilled pipelines are
  positive-only and reject negative prompts.
- I2V accepts exactly one initial image per prompt.
- The supported `--model` value is `Lightricks/LTX-2.5-Diffusers`.

## References

- [LTX-2.5-Diffusers model card](https://huggingface.co/Lightricks/LTX-2.5-Diffusers)
- [Official LTX-2 repository](https://github.com/Lightricks/LTX-2)
- [Generic T2V example](../../examples/offline_inference/text_to_video/text_to_video.py)
- [Generic I2V example](../../examples/offline_inference/image_to_video/image_to_video.py)
