# AURA Omni: Online serving

`aura_omni` serves AURA as a native multi-stage vLLM-Omni pipeline:

```text
Qwen3-ASR -> AURA/Qwen3-VL -> Qwen3-TTS Talker -> Qwen3-TTS Code2Wav
```

**Primary online path is Realtime duplex** (`/v1/realtime?duplex=1`), not
turn-based `chat/completions`. The default deploy profile sets
`session_mode: duplex` because the pipeline declares `duplex_plugin` and
`DuplexOmniEngine` requires that mode.

This is AURA v1 / Qwen3-VL (silent / ChatML ids `151669` / `151645`), not
AURA v2 / Qwen3.5-VL.

## Duplex Realtime (primary)

Start with the default deploy profile:

```bash
vllm serve aurateam/AURA \
  --omni \
  --port 8091 \
  --deploy-config vllm_omni/deploy/aura_omni.yaml \
  --served-model-name aurateam/AURA \
  --trust-remote-code
```

Local-weight smoke (Stage1 path baked into the smoke YAML):

```bash
bash examples/online_serving/aura_omni/run_duplex_smoke_serve.sh
python examples/online_serving/aura_omni/smoke_duplex_realtime_client.py
```

Smoke deploy file: `examples/online_serving/aura_omni/aura_omni_duplex_smoke.yaml`.

Connect clients to `/v1/realtime?duplex=1`. Silent Stage1 outputs gate TTS (no
audio for that turn). Overlapped input and vision-follow are AURA duplex
capabilities; see the PR / RFC for behaviour.

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

## Turn-based chat / Gradio / curl (not primary)

The OpenAI chat-completions client, curl helper, and Gradio demo were written
for the older **turn-based** Omni serve path (one HTTP request ≈ one turn).
They are **not** the supported primary online path for this duplex profile:
with `session_mode: duplex`, serve is Realtime-oriented.

Keep these scripts for offline-adjacent debugging or historical reference only.
Prefer the duplex smoke client above for online checks.

```bash
# Legacy / debug only — not the duplex Realtime path
python examples/online_serving/aura_omni/openai_chat_completion_client.py --help
bash examples/online_serving/aura_omni/run_curl_multimodal_generation.sh
bash examples/online_serving/aura_omni/run_gradio_demo.sh
```

## Offline

For offline inference, see
[`examples/offline_inference/aura_omni`](../../offline_inference/aura_omni/).
