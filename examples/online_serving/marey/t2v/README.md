# Marey Text-to-Video

Online serving example for the Marey 30B (Flux-30B-control-v2, distilled) text-to-video pipeline.

## Start the server

`run_server.sh` wraps `vllm-omni serve` with the Marey-specific flags (`--model-class-name MareyPipeline`, `--ulysses-degree 8`, etc.). The model checkpoint must already be prepared per the top-level repo `README.md`.

Required env vars:

- `MODEL` — path to the Marey checkpoint directory (with `config.yaml`)
- `MOONVALLEY_AI_PATH` — path to the moonvalley_ai checkout containing `open_sora/`

Common optional env vars: `PORT` (default 8098), `ULYSSES_DEGREE` (8), `GPU_MEMORY_UTILIZATION` (0.98), `HF_HOME`, `VLLM_OMNI_STORAGE_PATH`.

`flow_shift` is no longer a server flag — each curl client sends its own per-request value (`3.0` for the 30B distilled, `0` for the 7B), which overrides the pipeline's built-in default. One server can serve both.

```bash
HF_HOME=/mnt/localdisk/vllm_omni_hf_cache/ \
VLLM_OMNI_STORAGE_PATH=/mnt/localdisk/vllm_omni_storage \
MODEL=/app/hf_checkpoints/marey-distilled-0100/ \
MOONVALLEY_AI_PATH=${PATH_TO_MOONVALLEY_AI} \
bash examples/online_serving/marey/t2v/run_server.sh
```

## Prompt files

The prompt and negative prompt live as plain-text sidecar files so they can be edited without touching the curl script:


| File                  | Contents                                      |
| --------------------- | --------------------------------------------- |
| `prompt.txt`          | Default eagle prompt (matches the parity ref) |
| `negative_prompt.txt` | Canonical Marey negative prompt               |


Both curl scripts auto-load these from the directory they live in. To use a different prompt without editing the defaults:

```bash
# Point at a different file
PROMPT_FILE=/path/to/my_prompt.txt \
NEGATIVE_PROMPT_FILE=/path/to/my_negative.txt \
bash examples/online_serving/marey/t2v/run_curl_text_to_video.sh

# Or pass the strings inline (skips file read)
PROMPT="A cat dancing in the rain" \
NEGATIVE_PROMPT="" \
bash examples/online_serving/marey/t2v/run_curl_text_to_video.sh
```

`negative_prompt.txt` mirrors `DEFAULT_NEGATIVE_PROMPT` in `vllm_omni/diffusion/models/marey/pipeline_marey.py` and the moonvalley `marey_inference.py --negative-prompt` reference. The pipeline applies that default automatically whenever `guidance_scale > 1.0`, so dropping the field gives the same output today — pinning it client-side makes the example parity-stable across pipeline versions.

## Submit a request

The curl client posts to `POST /v1/videos`, polls `GET /v1/videos/{id}` until the job reaches `completed`, then downloads the result from `/content`.

```bash
SEED=0 bash examples/online_serving/marey/t2v/run_curl_text_to_video.sh
```

Per-script env knobs: `BASE_URL` (default `http://localhost:8098`), `SEED`, `OUTPUT_PATH`, `POLL_INTERVAL`, `PROMPT_FILE`, `NEGATIVE_PROMPT_FILE`, `PROMPT`, `NEGATIVE_PROMPT`.

The request body sends:

```text
prompt=<contents of PROMPT_FILE>
negative_prompt=<contents of NEGATIVE_PROMPT_FILE>
size=1920x1080
num_frames=128
num_inference_steps=33
guidance_scale=3.5
flow_shift=3.0
seed=${SEED}
```

## Storage

Generated `.mp4` files are persisted by the async video API:

- `VLLM_OMNI_STORAGE_PATH` — output directory (default: `/tmp/storage`)
- `VLLM_OMNI_STORAGE_MAX_CONCURRENCY` — concurrent save/delete ops (default: 4)

## Loading note

Simultaneous HF model loads from multiple ranks over a shared filesystem can fail with errors like `OSError: google/ul2 does not appear to have a file named pytorch_model-00001-of-00004.bin.` Set `HF_HOME` to a node-local path (as in the launch example above) to avoid this.
