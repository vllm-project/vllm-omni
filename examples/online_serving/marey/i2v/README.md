# Marey Image-to-Video (Multi-Keyframe)

Online serving example for the Marey 30B (Flux-30B-control-v2, distilled) multi-keyframe image-to-video pipeline. Two conditioning keyframes constrain a 128-frame 1920x1080 generation; the model interpolates motion between them.


## Start the server

`run_server.sh` is the same script t2v uses; the 30B server already supports multi-keyframe i2v via the `frame_conditions` form field on `POST /v1/videos`.

Required env vars:
- `MODEL` — path to the Marey checkpoint directory (with `config.yaml`)
- `MOONVALLEY_AI_PATH` — path to the moonvalley_ai checkout containing `open_sora/`

```bash
HF_HOME=/mnt/localdisk/vllm_omni_hf_cache/ \
VLLM_OMNI_STORAGE_PATH=/mnt/localdisk/vllm_omni_storage \
MODEL=/app/hf_checkpoints/marey-distilled-0100/ \
MOONVALLEY_AI_PATH=${PATH_TO_MOONVALLEY_AI} \
bash examples/online_serving/marey/i2v/run_server.sh
```

If you already have a t2v server running on the same port, you can skip re-launching — the same `MareyPipeline` handles both t2v and multi-keyframe i2v.

## Input artifacts

Sidecar files in this directory:

| File                  | Contents                                           |
| --------------------- | -------------------------------------------------- |
| `prompt.txt`          | Production prompt (the moonvalley payload's text)  |
| `negative_prompt.txt` | Canonical Marey negative prompt                    |
| `frame_0.webp`        | Keyframe at output frame index 0                   |
| `frame_127.webp`      | Keyframe at output frame index 127                 |

## Submit a request

```bash
SEED=1997074405 bash examples/online_serving/marey/i2v/run_curl_image_to_video.sh
```

Per-script env knobs: `BASE_URL` (default `http://localhost:8098`), `SEED` (default `1997074405`, matches the production payload), `OUTPUT_PATH`, `POLL_INTERVAL`, `PROMPT_FILE`, `NEGATIVE_PROMPT_FILE`, `PROMPT`, `NEGATIVE_PROMPT`.

The request body sends:

```text
prompt=<contents of PROMPT_FILE>
negative_prompt=<contents of NEGATIVE_PROMPT_FILE>
size=1920x1080
num_frames=128
num_inference_steps=33
guidance_scale=4.5
fps=24
flow_shift=3.0
seed=${SEED}
frame_conditions=<JSON dict, see below>
```

## frame_conditions JSON

The `frame_conditions` form field is a JSON dict keyed by the target output frame index (as a string), with values that mirror the OpenAI chat-completions `image_url` schema. Accepted URL schemes: `data:image/...;base64,...`, `http(s)://`, and `file:///abs/path` (server-side filesystem read; only works when the client and server share the same machine).

```json
{
  "0": {
    "image_url": {
      "url": "file:///abs/path/to/frame_0.webp",
      "detail": "auto"
    }
  },
  "127": {
    "image_url": {
      "url": "file:///abs/path/to/frame_127.webp",
      "detail": "auto"
    }
  }
}
```

Constraints:
- Keys must be unique non-negative integers.
- Each entry must have a non-empty `image_url.url`.
- Dense keyframe groups (4 consecutive frames starting at a multiple of 4) are rejected — they would require the VAE-block encoding path, which is not implemented yet. Use sparse keyframes (e.g. every 4+ frames apart).

`run_curl_image_to_video.sh` builds this dict at runtime from the local `frame_*.webp` files via a small inline Python helper, then sends it as a form field. To use a different keyframe set, edit the `[(idx, path), ...]` list inside the `FRAME_CONDITIONS_JSON` block in the script.

## Storage

Generated `.mp4` files are persisted by the async video API:

- `VLLM_OMNI_STORAGE_PATH` — output directory (default: `/tmp/storage`)
- `VLLM_OMNI_STORAGE_MAX_CONCURRENCY` — concurrent save/delete ops (default: 4)

## Loading note

Simultaneous HF model loads from multiple ranks over a shared filesystem can fail with errors like `OSError: google/ul2 does not appear to have a file named pytorch_model-00001-of-00004.bin.` Set `HF_HOME` to a node-local path (as in the launch example above) to avoid this.
