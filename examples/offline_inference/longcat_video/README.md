# LongCat-Video Avatar

Offline inference example for LongCat-Video-Avatar-1.5 A2V and AI2V generation.

## Setup

Install the LongCat Avatar optional runtime dependencies before running the example:

```bash
pip install "vllm-omni[longcat-video-avatar]"
```

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
  --output longcat_avatar_ai2v.mp4
```

## Official Asset Smoke Test

The official LongCat-Video repository includes a single-speaker Avatar example
under `assets/avatar/single/`. Set these paths to your local checkout and model
cache:

```bash
export LONGCAT_VIDEO_REPO=/path/to/LongCat-Video
export AVATAR_MODEL=meituan-longcat/LongCat-Video-Avatar-1.5
export BASE_MODEL_DIR=/path/to/LongCat-Video
export PROMPT="$(python -c 'import json, os; print(json.load(open(os.path.join(os.environ["LONGCAT_VIDEO_REPO"], "assets/avatar/single_example_1.json")))["prompt"])')"
```

Audio-to-video with the official audio:

```bash
python end2end.py \
  --model "$AVATAR_MODEL" \
  --base-model-dir "$BASE_MODEL_DIR" \
  --stage at2v \
  --audio "$LONGCAT_VIDEO_REPO/assets/avatar/single/man.mp3" \
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
  --audio "$LONGCAT_VIDEO_REPO/assets/avatar/single/man.mp3" \
  --image "$LONGCAT_VIDEO_REPO/assets/avatar/single/man.png" \
  --prompt "$PROMPT" \
  --num-frames 93 \
  --num-inference-steps 8 \
  --output official_asset_longcat_avatar_ai2v.mp4
```

## Options

- `--stage`: `at2v` for audio-to-video, `ai2v` for audio-and-image-to-video.
- `--resolution`: `480p` or `720p`.
- `--use-int8` / `--no-use-int8`: load the official INT8 Avatar DiT weights by default, or full precision weights.
- `--use-distill` / `--no-use-distill`: enable the official distilled LoRA path by default.
- `--base-model-dir`: optional local LongCat-Video base model directory for tokenizer, text encoder, and VAE components.

The script saves an mp4 file with the generated frames muxed with the input audio.
