# AURA Omni: Online serving

`aura_omni` serves AURA as a native multi-stage vLLM-Omni pipeline:

```text
Qwen3-ASR -> AURA thinker -> Qwen3-TTS Talker -> Qwen3-TTS Code2Wav
```

**AURA's supported online path is Realtime duplex** (`/v1/realtime?duplex=1`).
The default deploy profile sets `session_mode: duplex` because the pipeline
declares `duplex_plugin` and `DuplexOmniEngine` requires that mode.

Two thinkers share this pipeline. The deploy file picks one:

| Thinker | Architecture | silent / ChatML end ids | Deploy |
| --- | --- | --- | --- |
| AURA v1 (Qwen3-VL) | `AuraQwen3VLForConditionalGeneration` | `151669` / `151645` | `vllm_omni/deploy/aura_omni.yaml`, smoke YAML below |
| AURA_v2 (Qwen3.5) | `AuraQwen3_5ForConditionalGeneration` | `248070` / `248046` | `vllm_omni/deploy/aura_omni_v2.yaml`, `vllm_omni/deploy/aura_omni_v2_2gpu.yaml` |

## AURA_v2

Two GPUs (Stage1 on GPU 1, CUDA graphs, up to 16 sessions):

```bash
vllm serve aurateam/AURA_2 --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/aura_omni_v2_2gpu.yaml --port 8091
```

One GPU (eager, one session):

```bash
vllm serve aurateam/AURA_2 --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/aura_omni_v2.yaml --port 8091
```

Both profiles already set `async_chunk: true`. The server is ready after
`Duplex warmup finished`. Check it with the smoke client:

```bash
python examples/online_serving/aura_omni/smoke_duplex_realtime_client.py \
  --url ws://127.0.0.1:8091/v1/realtime --model aurateam/AURA_2
```

AURA_v2 starts each reply with `<|silent|>` (no speech this turn) or
`<|respond|>`. Frames-only commits let it speak up on its own when something
it was asked to watch for appears. It also emits tool calls; see
[Browser UI](#browser-ui). Hardware, numbers and limits:
[`recipes/AURA/AURA_v2-H200.md`](../../../recipes/AURA/AURA_v2-H200.md).
Design: [`docs/design/aura_v2_duplex.md`](../../../docs/design/aura_v2_duplex.md).

## AURA v1

### Duplex Realtime (only supported online path)

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

For AURA_v2, point `--ws-backend` at the v2 server and pass
`--model aurateam/AURA_2`.

Do **not** fork a separate `aura_omni/realtime_web` asset tree; MiniCPM / Qwen
profiles are unchanged and hide the PTT control.

The wrapper turns on function calling. The proxy advertises a tool set from
[`realtime_web/tools.py`](../realtime_web/tools.py), runs each call the model
emits, and returns the result as `function_call_output`; the model then speaks
the answer. By default it uses the vision set (`VisionToolExecutor`):
`deepseek_vlm_search` and `visual_web_search` receive the latest camera frame,
and they and `DeepSeek` need `DEEPSEEK_API_KEY`.
`AURA_DEMO_TOOLS=safe` switches to the allowlisted set instead: `WebSearch` needs
`SERPER_API_KEY`; `DeepSeek` returns a mock without `DEEPSEEK_API_KEY`.
`BraveSearch` (`BRAVE_SEARCH_API_KEY`), `duckduckgo_search` and `WebFetch` stay
off unless `VLLM_REALTIME_WEB_TOOL_BRAVE`, `VLLM_REALTIME_WEB_TOOL_DDG` or
`VLLM_REALTIME_WEB_TOOL_WEBFETCH` is set to `1`. `AURA_DEMO_TOOLS=0` serves the
page without tools.

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
