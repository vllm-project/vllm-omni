# LongCat-Video-Avatar-1.5

> Audio-driven avatar video generation (AT2V / AI2V)

## Summary

- Vendor: meituan-longcat
- Model: `meituan-longcat/LongCat-Video-Avatar-1.5`
- Task: Audio-to-video (AT2V) and audio-and-image-to-video (AI2V) avatar generation
- Mode: Offline inference
- Maintainer: Community

## When to use this recipe

Use this recipe to drive a talking or singing avatar from a speech clip, either
from text alone (AT2V) or from a reference image (AI2V), with the native
LongCat-Video-Avatar-1.5 pipeline in vLLM-Omni. It covers:

1. **Single-speaker AT2V / AI2V** — one 93-frame segment from one audio track.
2. **AVC continuation** — chained segments that cover a longer audio track.
3. **Multi-speaker AI2V** — several audio tracks placed on a reference image
   through speaker bounding boxes.

The example saves an mp4 with the generated frames muxed with the input audio.

## References

- Upstream model card: <https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5>
- Upstream repository and example assets: <https://github.com/meituan-longcat/LongCat-Video>
- Related example under `examples/`: [`longcat_video/end2end.py`](../../examples/offline_inference/longcat_video/end2end.py)

## Hardware Support

## CUDA

### 1× NVIDIA H100 (80 GB)

#### Environment

- OS: Linux
- Python: 3.10+
- Driver: NVIDIA driver with CUDA 12.x
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Prerequisites

Install the LongCat Avatar optional runtime dependencies:

```bash
pip install "vllm-omni[longcat-video-avatar]"
```

The example uses the `imageio_ffmpeg` executable installed by vLLM-Omni's
`imageio[ffmpeg]` dependency.

#### Command

Audio-to-video (AT2V):

```bash
python examples/offline_inference/longcat_video/end2end.py \
  --model meituan-longcat/LongCat-Video-Avatar-1.5 \
  --stage at2v \
  --audio /path/to/speech.wav \
  --prompt "A person speaks calmly while facing the camera." \
  --num-frames 93 \
  --num-inference-steps 8 \
  --output longcat_avatar_at2v.mp4
```

Audio-and-image-to-video (AI2V):

```bash
python examples/offline_inference/longcat_video/end2end.py \
  --model meituan-longcat/LongCat-Video-Avatar-1.5 \
  --stage ai2v \
  --audio /path/to/speech.wav \
  --image /path/to/reference.jpg \
  --prompt "A person speaks calmly while facing the camera." \
  --num-frames 93 \
  --num-inference-steps 8 \
  --output longcat_avatar_ai2v.mp4
```

#### Official Example Assets

Download the official Avatar example assets:

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
export AVATAR_MODEL=meituan-longcat/LongCat-Video-Avatar-1.5
export PROMPT="$(python -c 'import json, os; print(json.load(open(os.environ["LONGCAT_VIDEO_ASSET_DIR"] + "/single_example_1.json"))["prompt"])')"
```

`--base-model-dir` points at a local LongCat-Video base checkpoint for the
tokenizer, text encoder and VAE; omit it to download those components from
Hugging Face.

```bash
export BASE_MODEL_DIR=/cache/longcat_models/LongCat-Video
```

#### Official Asset Smoke Test

Audio-to-video with the official audio:

```bash
python examples/offline_inference/longcat_video/end2end.py \
  --model "$AVATAR_MODEL" \
  --base-model-dir "$BASE_MODEL_DIR" \
  --stage at2v \
  --audio "$LONGCAT_VIDEO_ASSET_DIR/single/man.mp3" \
  --prompt "$PROMPT" \
  --num-frames 93 \
  --num-inference-steps 8 \
  --output official_asset_longcat_avatar_at2v.mp4
```

Audio-and-image-to-video with the official audio and image:

```bash
python examples/offline_inference/longcat_video/end2end.py \
  --model "$AVATAR_MODEL" \
  --base-model-dir "$BASE_MODEL_DIR" \
  --stage ai2v \
  --audio "$LONGCAT_VIDEO_ASSET_DIR/single/man.mp3" \
  --image "$LONGCAT_VIDEO_ASSET_DIR/single/man.png" \
  --prompt "$PROMPT" \
  --num-frames 93 \
  --num-inference-steps 8 \
  --output official_asset_longcat_avatar_ai2v.mp4
```

#### Official Single-Speaker AVC Continuation

`--num-segments` chains AVC segments; `auto` covers the full input audio.

```bash
python examples/offline_inference/longcat_video/end2end.py \
  --model "$AVATAR_MODEL" \
  --base-model-dir "$BASE_MODEL_DIR" \
  --input-json "$LONGCAT_VIDEO_ASSET_DIR/single_example_1.json" \
  --num-frames 93 \
  --num-cond-frames 13 \
  --num-segments 5 \
  --num-inference-steps 8 \
  --output official_asset_longcat_avatar_single_example_1_ai2v_5seg.mp4
```

Replace `--num-segments 5` with `--num-segments auto` to run until the full
official audio is covered. That is a slow archival validation path rather than a
regular smoke test: about 24m28s on a Modal H100 for the full
`single_example_1.json` audio with 8 inference steps.

#### Official Multi-Speaker AI2V

The multi-speaker cases use several audio tracks with speaker masks and optional
bounding boxes. The command below generates the first 93-frame segment, matching
the first call to the official `generate_ai2v()`.

```bash
python examples/offline_inference/longcat_video/end2end.py \
  --model "$AVATAR_MODEL" \
  --base-model-dir "$BASE_MODEL_DIR" \
  --input-json "$LONGCAT_VIDEO_ASSET_DIR/multi_example_1.json" \
  --num-frames 93 \
  --num-inference-steps 8 \
  --output official_asset_longcat_avatar_multi_example_1_ai2v.mp4
```

Use `multi_example_2.json` for the sequential-speaking case, and add
`--num-cond-frames 13 --num-segments auto` to cover the full audio with AVC
continuation.

#### Verification

```bash
ffprobe -hide_banner longcat_avatar_ai2v.mp4
```

The output must report one video stream and one audio stream: the example muxes
the generated frames with the input audio.

#### Notes

**Support matrix**

| Mode | Speakers | Image required | AVC continuation |
| --- | --- | --- | --- |
| AT2V | Single | No | Yes |
| AI2V | Single | Yes | Yes |
| AT2V | Multi | Not supported | Not supported |
| AI2V | Multi | Yes | Yes |

Multi-speaker Avatar requires AI2V because speaker bounding boxes and masks are
defined on a reference image; AT2V with several tracks fails with an explicit
error. AVC continuation uses full-audio embeddings only when `--num-segments` is
greater than `1` or set to `auto`; single-segment AT2V/AI2V keeps the shorter
first-segment audio path.

**Sampling parameters**

| Parameter | LongCat pipeline value | Description |
| --- | --- | --- |
| `--num-frames` | `num_frames` | Frames generated per segment. The default is 93 frames, or about 3.72 seconds at 25 FPS. LongCat normalizes other values to the temporal-VAE-compatible `4k+1` form. |
| `--num-inference-steps` | `steps` | Denoising steps. The distilled Avatar path uses 8 steps. |
| `--fps` | `save_fps` | Output frame rate, and the rate used to align audio embeddings with video frames. The default is 25 FPS. |
| `sampling_params.guidance_scale` | `text_guidance_scale` | Classifier-free guidance applied to the text prompt. |
| `sampling_params.guidance_scale_2` | `audio_guidance_scale` | Classifier-free guidance applied to the speech audio. |

With the distilled LoRA enabled (the default), the pipeline uses 8 inference
steps and sets both guidance scales to `1.0`. With `--no-use-distill`, the
example sets both to `4.0`. For AVC continuation every segment holds
`num_frames` frames, but each segment after the first reuses `num_cond_frames`
conditioning frames, so the unique frame count is
`num_frames + (num_segments - 1) * (num_frames - num_cond_frames)`.

**Key flags**

- `--stage`: `at2v` for audio-to-video, `ai2v` for audio-and-image-to-video.
- `--resolution`: `480p` or `720p`.
- `--input-json`: official LongCat Avatar JSON case, used for the single-speaker
  and multi-speaker AI2V examples.
- `--num-segments`: number of AVC segments, or `auto` to cover the full audio.
- `--num-cond-frames`: previous-frame conditioning window for AVC continuation.
- `--use-int8` / `--no-use-int8`: load the official INT8 Avatar DiT weights by
  default, or full precision weights.
- `--use-distill` / `--no-use-distill`: enable the official distilled LoRA path
  by default.
- `--build-components-on-gpu`: build large Avatar components directly on GPU for
  faster startup, at a higher peak VRAM cost. Disabled by default.
- `--base-model-dir`: local LongCat-Video base model directory for the
  tokenizer, text encoder and VAE components.

**Memory usage**

Measured on a Modal H100 running the official 93-frame AI2V example with the
INT8 DiT, distilled LoRA and 8 inference steps:

| Component build | CLI option | Peak GPU memory | Model loading time |
| --- | --- | ---: | ---: |
| CPU (default) | None | About 41.0 GiB | About 130s |
| GPU | `--build-components-on-gpu` | About 56.8 GiB | About 33s |

The reported peak covers the end-to-end run. Building components on CPU only
reduces initialization memory; denoising still runs on GPU. These numbers are
memory guidance rather than a performance benchmark.

**Known limitations**

- Single GPU only. Cache acceleration, sequence/CFG/tensor parallelism, HSDP,
  CPU offload, VAE patch parallelism and quantization are not wired up for this
  pipeline yet; see the diffusion feature matrix in
  [`docs/user_guide/diffusion_features.md`](../../docs/user_guide/diffusion_features.md).
- Offline inference only. Online serving is not supported yet.
