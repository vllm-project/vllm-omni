# LongCat-Video Avatar

Offline inference example for LongCat-Video-Avatar-1.5 A2V and AI2V generation.

## Setup

Install the LongCat Avatar optional runtime dependencies before running the example:

```bash
pip install "vllm-omni[longcat-video-avatar]"
```

The example uses the `imageio_ffmpeg` executable installed by vLLM-Omni's
`imageio[ffmpeg]` dependency.

## Quick Start

```bash
cd examples/offline_inference/longcat_video
```

Audio-to-video:

```bash
python end2end.py \
  --model meituan-longcat/LongCat-Video-Avatar-1.5 \
  --stage at2v \
  --audio /path/to/speech.wav \
  --prompt "A person speaks calmly while facing the camera." \
  --num-frames 93 \
  --num-inference-steps 8 \
  --output longcat_avatar_at2v.mp4
```

Audio-and-image-to-video:

```bash
python end2end.py \
  --model meituan-longcat/LongCat-Video-Avatar-1.5 \
  --stage ai2v \
  --audio /path/to/speech.wav \
  --image /path/to/reference.jpg \
  --prompt "A person speaks calmly while facing the camera." \
  --num-frames 93 \
  --num-inference-steps 8 \
  --output longcat_avatar_ai2v.mp4
```

## Sampling Parameters

`end2end.py` maps its generation settings to
`OmniDiffusionSamplingParams` as follows:

| Parameter | LongCat pipeline value | Description |
| --- | --- | --- |
| `sampling_params.num_frames` / `--num-frames` | `num_frames` | Number of frames generated per segment. The default is 93 frames, or about 3.72 seconds at 25 FPS. LongCat normalizes other values to the temporal-VAE-compatible `4k+1` form. |
| `sampling_params.guidance_scale` | `text_guidance_scale` | Classifier-free guidance applied to the text prompt. Higher values strengthen prompt conditioning. |
| `sampling_params.guidance_scale_2` | `audio_guidance_scale` | Classifier-free guidance applied to the speech audio. Higher values strengthen audio conditioning. |
| `sampling_params.num_inference_steps` / `--num-inference-steps` | `steps` | Number of denoising steps. The distilled Avatar path uses 8 steps. |
| `sampling_params.fps` / `--fps` | `save_fps` | Output frame rate and the rate used to align audio embeddings with video frames. The default is 25 FPS. |

With distilled LoRA enabled (the default), the pipeline uses 8 inference steps
and sets both guidance scales to `1.0`. With `--no-use-distill`, this example
sets both guidance scales to `4.0`. For AVC continuation, each segment contains
`num_frames` frames, but every segment after the first reuses
`num_cond_frames` conditioning frames. The final unique frame count is
`num_frames + (num_segments - 1) * (num_frames - num_cond_frames)`.

## Official Example Assets

Download the official Avatar example assets directly:

```bash
mkdir -p longcat_avatar_assets/assets/avatar/single longcat_avatar_assets/assets/avatar/multi

curl -L -o longcat_avatar_assets/assets/avatar/single/man.mp3 \
  https://raw.githubusercontent.com/meituan-longcat/LongCat-Video/main/assets/avatar/single/man.mp3
curl -L -o longcat_avatar_assets/assets/avatar/single/man.png \
  https://raw.githubusercontent.com/meituan-longcat/LongCat-Video/main/assets/avatar/single/man.png
curl -L -o longcat_avatar_assets/assets/avatar/single_example_1.json \
  https://raw.githubusercontent.com/meituan-longcat/LongCat-Video/main/assets/avatar/single_example_1.json

curl -L -o longcat_avatar_assets/assets/avatar/multi/sing.png \
  https://raw.githubusercontent.com/meituan-longcat/LongCat-Video/main/assets/avatar/multi/sing.png
curl -L -o longcat_avatar_assets/assets/avatar/multi/sing_man.WAV \
  https://raw.githubusercontent.com/meituan-longcat/LongCat-Video/main/assets/avatar/multi/sing_man.WAV
curl -L -o longcat_avatar_assets/assets/avatar/multi/sing_woman.WAV \
  https://raw.githubusercontent.com/meituan-longcat/LongCat-Video/main/assets/avatar/multi/sing_woman.WAV
curl -L -o longcat_avatar_assets/assets/avatar/multi/introduce.png \
  https://raw.githubusercontent.com/meituan-longcat/LongCat-Video/main/assets/avatar/multi/introduce.png
curl -L -o longcat_avatar_assets/assets/avatar/multi/introduce_man.mp3 \
  https://raw.githubusercontent.com/meituan-longcat/LongCat-Video/main/assets/avatar/multi/introduce_man.mp3
curl -L -o longcat_avatar_assets/assets/avatar/multi/introduce_woman.mp3 \
  https://raw.githubusercontent.com/meituan-longcat/LongCat-Video/main/assets/avatar/multi/introduce_woman.mp3
curl -L -o longcat_avatar_assets/assets/avatar/multi_example_1.json \
  https://raw.githubusercontent.com/meituan-longcat/LongCat-Video/main/assets/avatar/multi_example_1.json
curl -L -o longcat_avatar_assets/assets/avatar/multi_example_2.json \
  https://raw.githubusercontent.com/meituan-longcat/LongCat-Video/main/assets/avatar/multi_example_2.json

export LONGCAT_VIDEO_ASSET_DIR="$PWD/longcat_avatar_assets/assets/avatar"
```

## Official Asset Smoke Test

The downloaded asset set includes a single-speaker Avatar example. Set these
paths to the downloaded asset files:

```bash
export AUDIO_PATH="$LONGCAT_VIDEO_ASSET_DIR/single/man.mp3"
export IMAGE_PATH="$LONGCAT_VIDEO_ASSET_DIR/single/man.png"
export PROMPT_JSON="$LONGCAT_VIDEO_ASSET_DIR/single_example_1.json"
export AVATAR_MODEL=meituan-longcat/LongCat-Video-Avatar-1.5
# Use the Modal validation cache path, or replace it with your local
# LongCat-Video base checkpoint path. Remove --base-model-dir from the
# commands below to download these base components from Hugging Face.
export BASE_MODEL_DIR=/cache/longcat_models/LongCat-Video
export PROMPT="$(python -c 'import json, os; print(json.load(open(os.environ["PROMPT_JSON"]))["prompt"])')"
```

Audio-to-video with the official audio:

```bash
python end2end.py \
  --model "$AVATAR_MODEL" \
  --base-model-dir "$BASE_MODEL_DIR" \
  --stage at2v \
  --audio "$AUDIO_PATH" \
  --prompt "$PROMPT" \
  --num-frames 93 \
  --num-inference-steps 8 \
  --output official_asset_longcat_avatar_at2v.mp4
```

Audio-and-image-to-video with the official audio and image:

```bash
python end2end.py \
  --model "$AVATAR_MODEL" \
  --base-model-dir "$BASE_MODEL_DIR" \
  --stage ai2v \
  --audio "$AUDIO_PATH" \
  --image "$IMAGE_PATH" \
  --prompt "$PROMPT" \
  --num-frames 93 \
  --num-inference-steps 8 \
  --output official_asset_longcat_avatar_ai2v.mp4
```

## Support Matrix

| Mode | Speakers | Image required | AVC continuation | Validation |
| --- | --- | --- | --- | --- |
| AT2V | Single | No | Yes | Example smoke, native path |
| AI2V | Single | Yes | Yes | Example smoke, e2e, full-audio Modal validation |
| AT2V | Multi | Not supported | Not supported | Guarded with an explicit error |
| AI2V | Multi | Yes | Yes | Example smoke, e2e, official multi-example validation |

Multi-speaker Avatar requires AI2V because speaker bounding boxes and masks
are defined on a reference image. AVC continuation uses full-audio embeddings
only when `--num-segments` is greater than `1` or set to `auto`; single-segment
AT2V/AI2V keeps the shorter first-segment audio path.

## Official Single-Speaker AVC Continuation

Use `--num-segments 5` with the official single-speaker JSON case to match
the official Avatar 1.5 AVC example style.

```bash
python end2end.py \
  --model "$AVATAR_MODEL" \
  --base-model-dir "$BASE_MODEL_DIR" \
  --input-json "$LONGCAT_VIDEO_ASSET_DIR/single_example_1.json" \
  --num-frames 93 \
  --num-cond-frames 13 \
  --num-segments 5 \
  --num-inference-steps 8 \
  --output official_asset_longcat_avatar_single_example_1_ai2v_5seg.mp4
```

### Optional Full-Audio AVC Validation

This runs AVC continuation until the full official audio is covered. It is a
slow archival validation path, not a regular smoke test.

```bash
python end2end.py \
  --model "$AVATAR_MODEL" \
  --base-model-dir "$BASE_MODEL_DIR" \
  --input-json "$LONGCAT_VIDEO_ASSET_DIR/single_example_1.json" \
  --num-frames 93 \
  --num-cond-frames 13 \
  --num-segments auto \
  --num-inference-steps 8 \
  --output official_asset_longcat_avatar_single_example_1_ai2v_auto_full.mp4
```

Observed on Modal H100: about 24m28s end to end for the full official
`single_example_1.json` audio with 8 inference steps.

## Official Multi-Speaker AI2V Smoke Test

The downloaded asset set also includes multi-speaker Avatar cases in
`multi_example_*.json`. These examples use the native AI2V path with multiple
audio tracks, speaker masks, and optional bounding boxes.

The commands below generate only the first 93-frame segment, matching the
first call to official `generate_ai2v()`.

Parallel-speaking example:

```bash
python end2end.py \
  --model "$AVATAR_MODEL" \
  --base-model-dir "$BASE_MODEL_DIR" \
  --input-json "$LONGCAT_VIDEO_ASSET_DIR/multi_example_1.json" \
  --num-frames 93 \
  --num-inference-steps 8 \
  --output official_asset_longcat_avatar_multi_example_1_ai2v.mp4
```

Sequential-speaking example:

```bash
python end2end.py \
  --model "$AVATAR_MODEL" \
  --base-model-dir "$BASE_MODEL_DIR" \
  --input-json "$LONGCAT_VIDEO_ASSET_DIR/multi_example_2.json" \
  --num-frames 93 \
  --num-inference-steps 8 \
  --output official_asset_longcat_avatar_multi_example_2_ai2v.mp4
```

## Official Multi-Speaker AVC Continuation

Use `--num-segments auto` to cover the full official multi-speaker audio.
This runs the first AI2V segment and then continues with the native AVC path
using the previous segment as video conditioning.

Parallel-speaking full example:

```bash
python end2end.py \
  --model "$AVATAR_MODEL" \
  --base-model-dir "$BASE_MODEL_DIR" \
  --input-json "$LONGCAT_VIDEO_ASSET_DIR/multi_example_1.json" \
  --num-frames 93 \
  --num-cond-frames 13 \
  --num-segments auto \
  --num-inference-steps 8 \
  --output official_asset_longcat_avatar_multi_example_1_ai2v_full.mp4
```

Sequential-speaking full example:

```bash
python end2end.py \
  --model "$AVATAR_MODEL" \
  --base-model-dir "$BASE_MODEL_DIR" \
  --input-json "$LONGCAT_VIDEO_ASSET_DIR/multi_example_2.json" \
  --num-frames 93 \
  --num-cond-frames 13 \
  --num-segments auto \
  --num-inference-steps 8 \
  --output official_asset_longcat_avatar_multi_example_2_ai2v_full.mp4
```

## Options

- `--stage`: `at2v` for audio-to-video, `ai2v` for audio-and-image-to-video.
- `--resolution`: `480p` or `720p`.
- `--input-json`: official LongCat Avatar JSON case. This is used for single-speaker and multi-speaker AI2V examples.
- `--num-segments`: number of AVC segments, or `auto` to cover the full input audio.
- `--num-cond-frames`: previous-frame conditioning window for AVC continuation.
- `--use-int8` / `--no-use-int8`: load the official INT8 Avatar DiT weights by default, or full precision weights.
- `--use-distill` / `--no-use-distill`: enable the official distilled LoRA path by default.
- `--build-components-on-gpu`: build large Avatar components directly on GPU for faster startup. This requires more peak VRAM and is disabled by default.
- `--base-model-dir`: optional local LongCat-Video base model directory for tokenizer, text encoder, and VAE components. Omit it to download the base components from Hugging Face.

## Peak GPU Memory

The following LongCat-Video-Avatar-1.5 measurements were observed on a Modal
H100 while running the official 93-frame AI2V example with the INT8 DiT,
distilled LoRA, and 8 inference steps:

| Component build | CLI option | Peak GPU memory | Model loading time |
| --- | --- | ---: | ---: |
| CPU (default) | None | About 41.0 GiB | About 130s |
| GPU | `--build-components-on-gpu` | About 56.8 GiB | About 33s |

The reported peak covers the end-to-end example run. CPU component build only
reduces initialization memory; denoising still runs on GPU. These measurements
are memory guidance rather than a performance benchmark.

The script saves an mp4 file with the generated frames muxed with the input audio.
