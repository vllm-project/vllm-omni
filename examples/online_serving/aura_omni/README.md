# AURA Omni: Online serving

`aura_omni` serves AURA as a native multi-stage vLLM-Omni pipeline:

```text
Qwen3-ASR -> AURA/Qwen3-VL -> Qwen3-TTS Talker -> Qwen3-TTS Code2Wav
```

**AURA's supported online path is Realtime duplex** (`/v1/realtime?duplex=1`).
The default deploy profile sets `session_mode: duplex` because the pipeline
declares `duplex_plugin` and `DuplexOmniEngine` requires that mode.

This is AURA v1 / Qwen3-VL (silent / ChatML ids `151669` / `151645`), not
AURA v2 / Qwen3.5-VL.

## Duplex Realtime (only supported online path)

Stage2→3 codec handoff needs `async_chunk: true`. The default
`vllm_omni/deploy/aura_omni.yaml` keeps `async_chunk: false` as a safer
baseline (turn-based `OmniOrchestrator` never sees the duplex async-chunk
gate). For Realtime, use the smoke deploy (already `async_chunk: true`) or
set that flag yourself:

```bash
bash examples/online_serving/aura_omni/run_duplex_smoke_serve.sh
python examples/online_serving/aura_omni/smoke_duplex_realtime_client.py
```

Smoke deploy file: `examples/online_serving/aura_omni/aura_omni_duplex_smoke.yaml`.

Equivalent serve with the default topology and async chunks enabled:

```bash
# Copy or edit so async_chunk: true, then:
vllm serve aurateam/AURA \
  --omni \
  --port 8091 \
  --deploy-config examples/online_serving/aura_omni/aura_omni_duplex_smoke.yaml \
  --served-model-name aurateam/AURA \
  --trust-remote-code
```

Connect clients to `/v1/realtime?duplex=1`. Silent Stage1 outputs gate TTS (no
audio for that turn). Overlapped input and vision-follow are AURA duplex
capabilities.

### Browser UI

AURA has no client VAD: control is **push-to-talk** on the shared shell from
[#7585](https://github.com/vllm-project/vllm-omni/pull/7585)
([`examples/online_serving/realtime_web/`](../realtime_web/README.md)), profile
`aura-ptt` (hold = `is_speech=true` PCM + sticky frames; release = `commit`;
silent+frames for vision-follow). Thin wrapper:

```bash
python -m examples.online_serving.aura_omni.realtime_web \
    --ws-backend ws://127.0.0.1:8099 --model aurateam/AURA --port 7862
```

Do **not** fork a separate `aura_omni/realtime_web` asset tree; MiniCPM / Qwen
profiles are unchanged and hide the PTT control.

### Per-stage models

Edit `model` on each stage in `vllm_omni/deploy/aura_omni.yaml` (or the smoke
YAML) for local checkpoints:

- Stage 0 ASR: `Qwen/Qwen3-ASR-1.7B`
- Stage 1 AURA: `aurateam/AURA`
- Stage 2/3 TTS: `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (or CustomVoice)

### GPU utilization

Tune `gpu_memory_utilization` per stage. Baseline on one large GPU:

- Stage 0 (ASR): `0.10`
- Stage 1 (AURA): `0.40`
- Stage 2 (Talker): `0.20`
- Stage 3 (Code2Wav): `0.20`

## Old chat / Gradio / curl scripts (unsupported)

`openai_chat_completion_client.py`, `run_curl_multimodal_generation.sh`, and
`run_gradio_demo.sh` target the older turn-based Omni path. They do **not**
work against the shipped `session_mode: duplex` profile.

That turn-based path was an incomplete June-era stand-in: vLLM-Omni did not
yet support streaming I/O for this stack, and adding session history then would
have required a much larger orchestrator change. It is not a supported AURA
mode. Do not run those scripts against the default deploy. Prefer the duplex
smoke client above.

## Offline

For offline inference, see
[`examples/offline_inference/aura_omni`](../../offline_inference/aura_omni/).
