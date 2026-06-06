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

## Official Example Assets

Download the official Avatar example assets directly:

```bash
mkdir -p longcat_avatar_assets/single longcat_avatar_assets/multi

curl -L -o longcat_avatar_assets/single/man.mp3 \
  https://raw.githubusercontent.com/meituan-longcat/LongCat-Video/main/assets/avatar/single/man.mp3
curl -L -o longcat_avatar_assets/single/man.png \
  https://raw.githubusercontent.com/meituan-longcat/LongCat-Video/main/assets/avatar/single/man.png
curl -L -o longcat_avatar_assets/single_example_1.json \
  https://raw.githubusercontent.com/meituan-longcat/LongCat-Video/main/assets/avatar/single_example_1.json

curl -L -o longcat_avatar_assets/multi/sing.png \
  https://raw.githubusercontent.com/meituan-longcat/LongCat-Video/main/assets/avatar/multi/sing.png
curl -L -o longcat_avatar_assets/multi/sing_man.WAV \
  https://raw.githubusercontent.com/meituan-longcat/LongCat-Video/main/assets/avatar/multi/sing_man.WAV
curl -L -o longcat_avatar_assets/multi/sing_woman.WAV \
  https://raw.githubusercontent.com/meituan-longcat/LongCat-Video/main/assets/avatar/multi/sing_woman.WAV
curl -L -o longcat_avatar_assets/multi/introduce.png \
  https://raw.githubusercontent.com/meituan-longcat/LongCat-Video/main/assets/avatar/multi/introduce.png
curl -L -o longcat_avatar_assets/multi/introduce_man.mp3 \
  https://raw.githubusercontent.com/meituan-longcat/LongCat-Video/main/assets/avatar/multi/introduce_man.mp3
curl -L -o longcat_avatar_assets/multi/introduce_woman.mp3 \
  https://raw.githubusercontent.com/meituan-longcat/LongCat-Video/main/assets/avatar/multi/introduce_woman.mp3
curl -L -o longcat_avatar_assets/multi_example_1.json \
  https://raw.githubusercontent.com/meituan-longcat/LongCat-Video/main/assets/avatar/multi_example_1.json
curl -L -o longcat_avatar_assets/multi_example_2.json \
  https://raw.githubusercontent.com/meituan-longcat/LongCat-Video/main/assets/avatar/multi_example_2.json

export LONGCAT_VIDEO_ASSET_DIR="$PWD/longcat_avatar_assets"
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
- `--input-json`: official LongCat Avatar JSON case. This is used for multi-speaker AI2V examples.
- `--num-segments`: number of AVC segments, or `auto` to cover the full multi-speaker audio.
- `--num-cond-frames`: previous-frame conditioning window for AVC continuation.
- `--use-int8` / `--no-use-int8`: load the official INT8 Avatar DiT weights by default, or full precision weights.
- `--use-distill` / `--no-use-distill`: enable the official distilled LoRA path by default.
- `--base-model-dir`: optional local LongCat-Video base model directory for tokenizer, text encoder, and VAE components. Omit it to download the base components from Hugging Face.

The script saves an mp4 file with the generated frames muxed with the input audio.
