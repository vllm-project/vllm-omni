# higgs-audio v2 online example

This directory contains the online-serving entry points for boson-ai's
higgs-audio v2 as integrated by vllm-omni.

## v1 scope

Plain text -> 24 kHz speech only. The server validator rejects every other
speech-control field defined by the OpenAI-compatible schema
(`voice` / `instructions` / `task_type` / `language` / `ref_audio` /
`ref_text` / `x_vector_only_mode` / `speaker_embedding` / non-1.0 `speed`)
and multi-speaker `[SPEAKERn]` tags inside the input text. Each rejection
returns a 4xx response whose error message names `higgs_audio_v2` and the
offending field.

## Files

- `run_server.sh` — launch the vllm-omni server with the bundled
  `vllm_omni/deploy/higgs_audio_v2.yaml` deploy config.
- `batch_speech_client.py` — send a list of prompts to
  `/v1/audio/speech` and save the returned WAV / PCM bytes to a directory.

## Launching the server

```bash
GPUS=6,7 PORT=8094 bash examples/online_serving/text_to_speech/higgs_audio_v2/run_server.sh
```

Environment overrides:
- `MODEL` — HF id of the talker (default `bosonai/higgs-audio-v2-generation-3B-base`).
- `PORT` — server port (default `8094`).
- `GPUS` — `CUDA_VISIBLE_DEVICES` value (default `6,7`).
- `GPU_UTIL` — `--gpu-memory-utilization` (default `0.4`).

The script also exports `VLLM_USE_DEEP_GEMM=0` / `VLLM_MOE_USE_DEEP_GEMM=0`
so the example works on images without the optional `deep_gemm` backend.

The deploy YAML ships with `async_chunk: false` and `codec_streaming: true`,
i.e. Stage 0 finishes its codec frames before Stage 1 starts decoding, and
Stage 1 streams WAV/PCM bytes to the client chunk-by-chunk.

## Driving the server

```bash
python examples/online_serving/text_to_speech/higgs_audio_v2/batch_speech_client.py \
    --base-url http://localhost:8094 \
    --model bosonai/higgs-audio-v2-generation-3B-base \
    --output-dir /tmp/higgs_audio_v2_batch \
    --prompts "Hello world." \
              "The quick brown fox jumps over the lazy dog."
```
