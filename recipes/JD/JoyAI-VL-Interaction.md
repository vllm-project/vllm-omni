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

## Quick start

Serve the model in fp8 (fits small GPUs without OOM), then put JD's official WebUI on top.

```bash
# 1. Serve the model — fp8 shrinks the 8B to ~10 GiB of weights
#    (drop `--quantization fp8` on 48GB+ GPUs for full precision)
vllm serve jdopensource/JoyAI-VL-Interaction-Preview \
  --served-model-name JoyAI-VL-Interaction-Preview --port 8061 \
  --quantization fp8 --max-model-len 131072 --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":256,"video":1}'

# 2. Start the interaction server (OpenAI-compatible, port 8070)
python -m vllm_omni.experimental.fullduplex.joyvl.serving.server --port 8070 \
  --main-backend-url http://127.0.0.1:8061/v1 --main-model JoyAI-VL-Interaction-Preview

# 3. Run JD's official WebUI, pointed at the interaction server
git clone https://github.com/jd-opensource/JoyAI-VL-Interaction.git
cd JoyAI-VL-Interaction/services/webui
uv venv && uv pip install -e .
bash scripts/start_server.sh --api-base http://127.0.0.1:8070/v1
```

Open the printed HTTPS URL, allow the camera (or enter an RTSP URL), and type a standing
instruction — e.g. "Alert me if a fire breaks out". The model then decides each second on
its own to stay silent, speak, or delegate a hard question.

One-shot alternative for steps 1–3:
`bash examples/online_serving/joyvl_interaction/scripts/start_all.sh` (set `WEBUI_DIR`
to your WebUI clone). For webui-side issues, see the upstream
[Troubleshooting Guide](https://github.com/jd-opensource/JoyAI-VL-Interaction/blob/main/doc/troubleshooting.md).

## Environment

- OS: Linux
- Python: 3.10+
- Hardware: 1x GPU. The fp8 quick start fits much smaller cards (~10 GiB weights vs
  16.8 GiB bf16); bf16 at the default settings wants ≈48GB+. On small cards also lower
  `--chunk-frames`, `--max-model-len`, and the image limit together (see "Serving notes")
- vLLM / vLLM-Omni: versions from your current checkout

## Serving notes

- Use plain `vllm serve`, **not** `--omni`.
- fp8 loads online from the bf16 checkpoint — nothing extra to download. It saves
  memory (~10 GiB vs 16.8 GiB weights), not speed; for timing-critical alerting,
  validate output against bf16 first.
- Keep the `--limit-mm-per-prompt` image limit ≥ `--chunk-frames` (default 100).
  On small GPUs lower `--chunk-frames`, `--max-model-len`, and the image limit together.

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
off**. The brain is bring-your-own: a larger vLLM you serve, or any OpenAI-compatible
API (e.g. `--delegation-backend-url https://api.anthropic.com/v1/
--delegation-model claude-... --delegation-api-key …`).

## Verification

```bash
# headless smoke test: stream a clip and print the per-second decisions
# (needs: uv pip install opencv-python)
python examples/online_serving/joyvl_interaction/cli/run_cli_demo.py \
  path/to/video.mp4 --query "Alert me if a fire breaks out"

pytest tests/fullduplex   # framework + JoyVL unit tests
```

## Testing with an RTSP stream (optional)

To simulate an RTSP camera from a local video file (no physical IP camera needed),
enter its stream URL in the WebUI RTSP box, using the helper scripts in
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

- If `vllm serve` crashes with `FileNotFoundError: 'ninja'` (FlashInfer sampler JIT),
  set `VLLM_USE_FLASHINFER_SAMPLER=0` or install `ninja`.
- The model stays silent until the first instruction arrives — give a standing task
  (e.g. "translate the on-screen text") to arm proactive output.
- Downsample frames to ~256×192 for the lowest latency and highest concurrency
  (~2× cheaper per tick than 640×480; one GPU sustains ~150–180 concurrent 1 fps
  streams with p95 < 200 ms).
- Speech is pluggable: point `ASR_URL` / `TTS_URL` at the bridges in
  `examples/online_serving/joyvl_interaction/bridges/` or any compatible service.
