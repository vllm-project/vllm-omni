# higgs-audio v2 online example

This directory contains the online serving examples for boson-ai's higgs-audio v2 as integrated by vllm-omni.

## v1 scope

Plain text -> 24 kHz speech only. The server validator rejects every other speech-control field defined by the OpenAI-compatible schema (voice / instructions / task_type / language / ref_audio / ref_text / x_vector_only_mode / speaker_embedding / non-1.0 speed) and multi-speaker `[SPEAKERn]` tags inside the input text. Each rejection returns a 4xx response whose error message names `higgs_audio_v2` and the offending field.

## Files

- `run_server.sh` — launch the vllm-omni server with `vllm_omni/deploy/higgs_audio_v2.yaml`.
- `batch_speech_client.py` — send a fixed list of prompts to `/v1/audio/speech` and save WAV (or raw PCM) files.
- `gradio_demo.py` — minimal Gradio UI: plain-text input -> playable 24 kHz WAV.

## Launching the server

```bash
GPUS=6,7 PORT=8094 ./examples/online_serving/text_to_speech/higgs_audio_v2/run_server.sh
```

Environment overrides:
- `MODEL` — HF id of the talker (default `bosonai/higgs-audio-v2-generation-3B-base`).
- `PORT` — server port (default `8094`).
- `GPUS` — `CUDA_VISIBLE_DEVICES` value (default `6,7`).
- `GPU_UTIL` — `--gpu-memory-utilization` (default `0.4`).

The deploy YAML's `async_chunk` flag controls streaming; flip it on once the talker AR loop is ready.

## Driving the server

### Batch client

```bash
python examples/online_serving/text_to_speech/higgs_audio_v2/batch_speech_client.py \
    --base-url http://localhost:8094 \
    --output-dir /tmp/higgs_audio_v2_batch
```

### Gradio demo

```bash
python examples/online_serving/text_to_speech/higgs_audio_v2/gradio_demo.py \
    --base-url http://localhost:8094 --port 7861
```
