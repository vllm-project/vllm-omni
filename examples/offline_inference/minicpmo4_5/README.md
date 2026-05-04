# MiniCPM-o 4.5 Offline Inference

This example runs MiniCPM-o 4.5 with the vLLM-Omni offline `Omni` API for text,
image, and video inputs with audio output.

## Setup

Install vLLM-Omni and the MiniCPM Token2Wav dependency:

```bash
pip install --no-build-isolation 'minicpmo-utils[all]'
pip install soundfile pillow
```

The default deploy config is:

```text
vllm_omni/deploy/minicpmo4_5.yaml
```

The default config places the thinker stage on GPU 0 and the talker/code2wav
stages on GPU 1, so use two visible GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1
```

## Run Examples

From the repository root:

```bash
bash examples/offline_inference/minicpmo4_5/run_text_to_audio.sh
```

For image input, either use the built-in synthetic image:

```bash
bash examples/offline_inference/minicpmo4_5/run_image_to_audio.sh
```

or pass a local image:

```bash
IMAGE_PATH=/path/to/image.png \
bash examples/offline_inference/minicpmo4_5/run_image_to_audio.sh
```

For video input, either use the built-in synthetic video:

```bash
bash examples/offline_inference/minicpmo4_5/run_video_to_audio.sh
```

or pass a local video:

```bash
VIDEO_PATH=/path/to/video.mp4 \
bash examples/offline_inference/minicpmo4_5/run_video_to_audio.sh
```

## Direct Python Usage

```bash
python examples/offline_inference/minicpmo4_5/end2end.py \
  --model-path openbmb/MiniCPM-o-4_5 \
  --deploy-config vllm_omni/deploy/minicpmo4_5.yaml \
  --query-type use_image \
  --modalities audio \
  --output-wav minicpmo45_image_to_audio.wav
```

Supported query types:

| Query type | Input | Default output |
| ---------- | ----- | -------------- |
| `text` | Text | Audio |
| `use_image` | Image + text | Audio |
| `use_video` | Video + text | Audio |

Set `--modalities text` to request text output, or `--modalities text,audio`
to request both text and audio when supported by the deploy config.

## Notes

- The text-to-audio path uses MiniCPM's TTS chat template through
  `tokenizer.apply_chat_template(..., use_tts_template=True)`.
- The image/video-to-audio paths use the MiniCPM prompt prefix expected by the
  current non-async deploy config.
- The public `py_generator=True` API closes `Omni` after iteration. This example
  uses the normal one-request `generate` path and closes the engine explicitly at
  the end.
