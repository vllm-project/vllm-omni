# JoyAI-VL-Interaction

> Real-time streaming video-language interaction (proactive speak / silence / delegate)

## Summary

- Vendor: JD (Joy Future Academy)
- Model: [`jdopensource/JoyAI-VL-Interaction-Preview`](https://huggingface.co/jdopensource/JoyAI-VL-Interaction-Preview) (8B, Qwen3-VL architecture with weights retrained for streaming interaction)
- Task: Per-tick proactive interaction over a live video stream — the model decides
  on its own each second to speak, stay silent, or delegate a hard question
- Mode: Online serving — an OpenAI-compatible interaction orchestrator in front of
  a plain `vllm serve` backend
- Maintainer: Community

## When to use this recipe

Use this to stand up the streaming-interaction serving layer (`vllm_omni/experimental/fullduplex/`).
The model is served unchanged by `vllm serve`; this layer adds session state, 3-tier
summary memory, the per-tick decision, and pluggable ASR / TTS / delegation. For the
framework internals and how to add another full-duplex model, see
[`vllm_omni/experimental/fullduplex/README.md`](../../vllm_omni/experimental/fullduplex/README.md).

## Environment

- OS: Linux
- Python: 3.10+
- Hardware: 1x GPU. The default `T_s=100` frame window + `--max-model-len 131072` wants
  ≈48GB+; for ~24GB, lower `--chunk-frames`, `--max-model-len`, and the image limit together,
  and/or load the weights in fp8 (see "Smaller GPU: fp8 weights" below — ~9.9 GiB vs 16.8 GiB)
- vLLM / vLLM-Omni: versions from your current checkout

## Start server

From repository root:

```bash
# 1. Serve the model (plain `vllm serve`, NOT --omni; it uses the Qwen3-VL architecture).
#    The image limit must cover the short-term frame window (chunk_frames, default 100 = T_s);
#    prefix caching keeps the accumulating window cheap. Lower both for smaller GPUs.
vllm serve jdopensource/JoyAI-VL-Interaction-Preview \
  --served-model-name JoyAI-VL-Interaction-Preview --port 8061 \
  --max-model-len 131072 --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":256,"video":1}'

# 2. Interaction orchestrator (OpenAI-compatible, :8070)
python -m vllm_omni.experimental.fullduplex.joyvl.serving.server --port 8070 \
  --main-backend-url http://127.0.0.1:8061/v1 --main-model JoyAI-VL-Interaction-Preview
```

### Smaller GPU: fp8 weights

Add `--quantization fp8` to the step-1 `vllm serve` to load the weights in fp8 online
from the bf16 checkpoint (no separate quantized checkpoint needed; uses vLLM's generic
fp8 path, since the model is a standard Qwen3-VL VLM):

```bash
vllm serve jdopensource/JoyAI-VL-Interaction-Preview \
  --served-model-name JoyAI-VL-Interaction-Preview --port 8061 \
  --quantization fp8 --max-model-len 131072 --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":256,"video":1}'
```

Measured here: model weights **9.9 GiB** vs **16.8 GiB** bf16 (**−41%**), which lets the 8B
fit a much smaller card. This is a **memory** option, not a speed one: at one request per
tick (the streaming regime) fp8 was **not** faster (≈19% slower in our test) because
single-stream decode is memory-latency-bound and pays fp8's dynamic-scale overhead; its
throughput gain needs batching. Decisions stay coherent, but fp8 can shift the
speak/silence boundary, so for timing-critical alerting validate output against bf16 first.

Optional one-shot launch (model + orchestrator + JD webui + ASR/TTS/background,
all env-configurable). The JD webui frontend is external — set `WEBUI_DIR` to point at it:

```bash
bash examples/online_serving/joyvl_interaction/scripts/start_all.sh
```

## Using the model

The orchestrator is OpenAI-compatible: send **one user turn per video frame** (~1 fps) to
`/v1/chat/completions` with an `x-session-id` header, and attach an optional **standing
instruction** as a text part. Each reply carries an `interaction` block — `action` is
`silence` / `response` / `delegate` and `text` is what to say:

```bash
curl -s http://127.0.0.1:8070/v1/chat/completions \
  -H 'x-session-id: s1' -H 'content-type: application/json' -d '{
    "messages": [{"role": "user", "content": [
      {"type": "text", "text": "Alert me if a fire breaks out"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]}]}' | jq .interaction
```

Send the standing instruction once (it persists for the session); subsequent turns send
just the frame. Ready-made headless client: `cli/run_cli_demo.py`. Reset a session with
`POST /reset {"session_id": "..."}`.

**What to ask it** — give a standing task and let it act on its own each second:

- **Proactive alerting** — "tell me when someone enters", "alert me if a fire breaks out"
- **Streaming Q&A** — ask about what's on screen; it answers once it has the evidence
- **Time & memory** — "how many people have walked past?", "what did you see earlier?"
  (the 3-tier summary memory lets it recall beyond the live frame window)
- **Read & translate** — "translate the on-screen text as it appears"
- **Delegate hard questions** — handed to a background brain (see below)
- **Live commentary** — `--persona talkative` for continuous, danmaku-style narration

**Personas** (`--persona`): `default` (speak on meaningful events or to answer), `silent`
(answer only when asked, never delegate), `talkative` (proactively narrate).

**Tuning:** `--chunk-frames` (short-term window `T_s`), `--response-dedup-threshold`
(lower drops more near-duplicate narration), and `--no-memory` to disable the 3-tier
summaries. The model stays silent until the first instruction arrives
(`force_silence_before_query`, on by default).

## Delegation (background brain)

When the model judges a question too hard, it emits `</delegation> <question>` and the
orchestrator hands it to a **background brain** — any OpenAI-compatible endpoint you
self-host. Enable it by pointing the orchestrator at one:

```bash
python -m vllm_omni.experimental.fullduplex.joyvl.serving.server --port 8070 \
  --main-backend-url http://127.0.0.1:8061/v1 --main-model JoyAI-VL-Interaction-Preview \
  --delegation-backend-url <brain-endpoint>/v1 \
  --delegation-model <brain-model> --delegation-kind chat
```

`--delegation-kind` picks the bridge:

- `chat` — a stronger text/VL model answers (`/chat/completions`)
- `image` — a text-to-image model generates a picture (`/images/generations`, e.g. Qwen-Image)
- `edit` — an image-edit model restyles the current frame (e.g. Qwen-Image-Edit)
- `router` — dispatch each request to chat / image / edit by inspecting it (set
  `--delegation-image-url` / `--delegation-edit-url` for the latter two)
- `stub` — canned answers for tests/demos only (no backend needed)

`chat`/`image`/`edit`/`router` each need a backend URL — **without one, delegation stays
off** (the model's delegate note is still spoken, but nothing is folded back). The brain is
**bring-your-own**: a larger vLLM you serve, or any OpenAI-compatible API (e.g.
`--delegation-backend-url https://api.anthropic.com/v1/ --delegation-model claude-...
--delegation-api-key …`). The reference deployment instead drives the `codex` CLI as the
brain via a separate background-agent service; that agent runs with its own credentials
and bypasses its sandbox, so it is **not bundled here** — self-host a plain
OpenAI-compatible endpoint instead.

## Host the WebUI demo

For the full browser experience — live webcam / RTSP input, voice (ASR/TTS), and the
per-tick decision stream — run JD's official WebUI in front of the orchestrator. Clone the
model repo and start its WebUI pointed at the orchestrator (`:8070`):

```bash
git clone https://github.com/jd-opensource/JoyAI-VL-Interaction.git
cd JoyAI-VL-Interaction/services/webui
uv venv && uv pip install -e .
bash scripts/start_server.sh --api-base http://127.0.0.1:8070/v1
```

Open the printed HTTPS URL, allow the camera (or enter an RTSP URL), and give a standing
instruction. `examples/online_serving/joyvl_interaction/scripts/start_all.sh` can launch
the model + orchestrator + WebUI together — point `WEBUI_DIR` at your clone.

## Verification

```bash
# headless: stream a clip and print the per-tick decision timeline
# (the CLI reads video frames via OpenCV: `uv pip install opencv-python`)
python examples/online_serving/joyvl_interaction/cli/run_cli_demo.py \
  path/to/video.mp4 --query "Alert me if a fire breaks out"

pytest tests/fullduplex   # framework + JoyVL unit tests
```

## Testing with an RTSP stream (optional)

RTSP is a **webui-side input** — the browser pulls the stream and feeds frames to the
orchestrator over the normal API; no serving-layer code is involved. To simulate an RTSP
camera from a local video file (no physical IP camera needed), use the helper scripts in
[`examples/online_serving/joyvl_interaction/rtsp/`](../../examples/online_serving/joyvl_interaction/rtsp/),
which wrap [MediaMTX](https://github.com/bluenviron/mediamtx/releases) + `ffmpeg`:

```bash
cd examples/online_serving/joyvl_interaction/rtsp

# 1. Local RTSP server (MediaMTX, listens on :8554)
bash ./mediamtx.sh

# 2. Push a local video file as an RTSP stream (another terminal)
bash ./rtsp.sh ./videos/example.mp4 rtsp://127.0.0.1:8554/fire1

# 3. In the WebUI RTSP box, enter:  rtsp://127.0.0.1:8554/fire1
#    (replace 127.0.0.1 with the MediaMTX host IP if the webui runs on another machine)
```

See the directory's `README.md` for streaming a whole video folder (`rtsp_all.sh`) and
the audio-track caveat.

## Notes

- `--omni` is **not** used: the model keeps the Qwen3-VL architecture (only the
  weights are retrained), so stock `vllm serve` runs the forward pass; this recipe
  only adds the interaction/serving layer.
- On a host without `nvcc` / `ninja`, `vllm serve` of the 8B can crash engine-core in the
  FlashInfer sampler JIT (`FileNotFoundError: 'ninja'`) during `profile_run`. Set
  `VLLM_USE_FLASHINFER_SAMPLER=0` (or install `ninja`) to work around it.
- `force_silence_before_query` is on by default — the model stays silent until an
  instruction arrives; give a standing task (e.g. "translate the on-screen text")
  to arm proactive output.
- **Frame resolution is the main latency knob.** The system/memory prefix is prefix-cached,
  so per-tick *new* compute is dominated by the new frame's vision tokens. Measured on the
  8B: ~256×192 frames hit Qwen3-VL's min-pixel floor (~72 vision tokens, ~17 ms/tick) vs
  ~302 tokens / ~38 ms at 640×480 — about 2× cheaper. Downsample frames to ~256×192 for the
  tightest latency / highest concurrency; one GPU then sustains ~150–180 concurrent 1 fps
  streams with p95 < 200 ms. Per-tick latency is already far inside the 1 fps budget, so
  serving is rarely the bottleneck — resolution is the lever if you need more headroom.
- Speech is external and pluggable: point `ASR_URL` / `TTS_URL` at the bridges in
  `examples/online_serving/joyvl_interaction/bridges/` or any compatible service.
- The decision prompts, sampling, and 3-tier summary memory (`T_s=100`, mid→long every
  5 chunks, `key_frames=0` = summarize all chunk frames) are aligned to the JoyVL
  reference adapter so per-tick behavior matches the released model; the framework only
  supplies the serving structure.
