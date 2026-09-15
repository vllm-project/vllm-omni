# MiniMax H3 online serving — timeline guides (GUIDE-01)

This directory contains server startup and request scripts for the MiniMax H3
HTTP API with ordered timeline guide conditioning.

## Prerequisites

Download the four official input images (used by the curl scripts):

```bash
BASE=https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input
wget -P ./input/ \
  "${BASE}/h3_frame_ref_1.png" \
  "${BASE}/h3_frame_ref_2.png" \
  "${BASE}/h3_frame_ref_3.png" \
  "${BASE}/h3_frame_ref_4.png"
```

Expected SHA-256:
```
h3_frame_ref_1.png: d6bcff697b0a1c3fc17692c3644daf150edfe0be5966dcbbc8df817f82c2e48c
h3_frame_ref_2.png: 259f46eb04e977036a34bfa784c9cfacf4d91a255af1508899f72c62b81c6230
h3_frame_ref_3.png: 14314a54df5beb54b15f56a1d1c2c592b347087c0df3ac9e4f163ee2374b9dcb
h3_frame_ref_4.png: 4a853f271b8ff83d4f6f2da5693f23342b90c982860a9e16047f45d49ab933a5
```

## Files

| Script | Purpose |
|---|---|
| `run_server_minimax_h3.sh` | Start the MiniMax H3 Ref2VA server on port 8099 |
| `run_curl_minimax_h3_guides_async.sh` | GUIDE-01 four-image demo via async `POST /v1/videos` + poll + download |
| `run_curl_minimax_h3_guides_sync.sh` | Same demo via blocking `POST /v1/videos/sync` (one-shot latency) |

## Start server

```bash
# Ref2VA partition, port 8099, 4-way USP + text-encoder TP + VAE patch parallel
MODEL=/path/to/MiniMax-H3/Ref2VA bash run_server_minimax_h3.sh

# FL2VA partition for t2va / fl2va guides
MODEL=/path/to/MiniMax-H3/FL2VA PORT=8099 bash run_server_minimax_h3.sh

# Fewer GPUs
USP=2 TEXT_ENCODER_TP=2 VAE_PP=2 CUDA_VISIBLE_DEVICES=0,1 bash run_server_minimax_h3.sh
```

H3 needs sequence/tensor parallelism to fit; without `--usp` and
`--text-encoder-tp-size` the server OOMs. Overridable: `MODEL`, `PORT`, `USP`,
`TEXT_ENCODER_TP`, `VAE_PP`, `CUDA_VISIBLE_DEVICES`.

Timeline guides require cache-free execution, requested per call with the
`quality=lossless` **form field** — it is not a `vllm serve` argument. Guide
requests are rejected for any model that does not declare
`supports_timeline_guides` in its metadata.

## Run the GUIDE-01 demo

### Async (job-based)

```bash
INPUT_DIR=./input \
BASE_URL=http://localhost:8099 \
OUTPUT_PATH=./output/h3_guide01_async.mp4 \
bash run_curl_minimax_h3_guides_async.sh
```

Creates a job, polls until `completed`, downloads the MP4.

### Sync (one-shot)

```bash
INPUT_DIR=./input \
BASE_URL=http://localhost:8099 \
OUTPUT_PATH=./output/h3_guide01_sync.mp4 \
bash run_curl_minimax_h3_guides_sync.sh
```

Blocks until generation finishes and writes the raw MP4 bytes.
`X-Inference-Time-S` in the response headers reports server-side latency.

## What the demo exercises

This is the GUIDE-01 reference case, matching the ComfyUI multi-frame reference
workflow (`user/default/workflows/minimax_h3_multiframe_reference.json`).

| Upload | Role | Frame |
|---|---|---|
| `h3_frame_ref_1.png` | Ordinary `input_references` — enters Qwen as `<Picture 1>` and the VAE reference path | — |
| `h3_frame_ref_2.png` | `guide_files[0]` — image guide at frame 36 | 36 |
| `h3_frame_ref_3.png` | `guide_files[1]` — image guide at frame 72 | 72 |
| `h3_frame_ref_4.png` | `guide_files[2]` — image guide at frame 120 | 120 |

Guide files are referenced by `upload_index` in the `timeline_guides` manifest:

```json
[
  {"frame_index": 36,  "image": {"upload_index": 0}},
  {"frame_index": 72,  "image": {"upload_index": 1}},
  {"frame_index": 120, "image": {"upload_index": 2}}
]
```

Guides do not enter Qwen presentation or count against the ordinary reference
budget (at most 9 images, 12 total). They are timeline conditioning anchors:
they suppress denoising at their placement row but are never decoded as output.

Expected output: 864×480, 124 frames, 24 FPS, stereo 32 kHz audio (~5.17 s).

## Reproducing a result exactly

Two things must match exactly to reproduce a result for a given seed.

### 1. The prompt must be byte-identical

Both scripts read the same file, `multiframe_reference_prompt.txt`, next to
them in this directory. Do not inline a shortened prompt: its
`overall_soundscape` and `non_diegetic_music` sections are what condition H3's
audio branch, so dropping them changes the generated audio substantially even
when every other parameter is identical. Override with `PROMPT_FILE=...` only
to point at a different prompt deliberately.

### 2. H3 parameters must travel through `extra_params`

MiniMax H3 resolves `task` and both sigma shifts from `extra_args`, not from
top-level fields — see `extra.get("flow_shift")` / `extra.get("audio_flow_shift")`
in `pipeline_minimax_h3.py`. There is no top-level `task` or `audio_flow_shift`
form field, so sending them with `-F` is **silently ignored** by FastAPI:

```bash
# WRONG - silently dropped, falls back to pipeline defaults
-F "task=ref2va" -F "audio_flow_shift=3" -F "flow_shift=12"

# CORRECT
-F 'extra_params={"task":"ref2va","aspect_ratio":"16:9","flow_shift":12.0,"audio_flow_shift":3.0}'
```

Parameters that *are* real top-level form fields (`width`, `height`,
`num_frames`, `fps`, `num_inference_steps`, `quality`, `seed`) stay as `-F`
fields.

`quality=lossless` is also a per-request field, not a server flag — passing it
to `vllm serve` fails with `unrecognized arguments`.

## API reference

The two new multipart form fields on `POST /v1/videos` and `POST /v1/videos/sync`:

| Field | Type | Description |
|---|---|---|
| `timeline_guides` | JSON string | Ordered manifest. Each entry: `frame_index` (strict int) + at least one of `image`, `video`, `audio`, each `{"upload_index": N}`. `image` and `video` cannot coexist. |
| `guide_files` | repeated file upload | Binary files in manifest order. Image: JPEG/PNG/WebP. Video: MP4/MOV. Audio: WAV/MP3/FLAC. |

Guides require:
- A guide-capable model (`MiniMaxH3Pipeline` or `MiniMaxH3ModularPipeline`)
- `quality=lossless` (or no `quality` with a lossless server default)
- No active LoRA/Turbo adapter

See `docs/serving/videos_api.md` for the full placement and normalization rules,
admission limits, and cancellation semantics.
