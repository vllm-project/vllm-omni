# MiniCPM-o 4.5: Online serving

This directory contains the MiniCPM-o 4.5 full-duplex serving demos for
vLLM-Omni. A MiniCPM-o 4.5 server is a **duplex-only** server: it speaks the
OpenAI Realtime protocol over `/v1/realtime?duplex=1` (alias `/v1/duplex`),
plus `/v1/models` and `/health`. The turn-based HTTP routes
(`/v1/chat/completions`, speech, batch, ...) are not served by a duplex model;
turn-based use of MiniCPM-o 4.5 stays available offline through the Python
API (`Omni` / `AsyncOmni`, see
[`examples/offline_inference/minicpmo/`](../../offline_inference/minicpmo/)).

For the duplex framework architecture, lifecycle invariants, capability
boundary, and validation scope, see
[`docs/design/fullduplex.md`](../../../docs/design/fullduplex.md); for the
`DuplexClient` / `InlineDuplexClient` API and the `/v1/realtime?duplex=1`
wire protocol, see
[`docs/serving/realtime_duplex_api.md`](../../../docs/serving/realtime_duplex_api.md).

## Installation

Install vLLM-Omni with the MiniCPM-o talker dependencies:

```bash
pip install stepaudio2-minicpmo
```

The `minicpmo` extra installs `stepaudio2-minicpmo` and its audio dependencies,
including `librosa`.

## Start the backend server

The deploy config auto-loads via `--omni`.
The default `vllm_omni/deploy/minicpmo_4_5.yaml` keeps all three stages on
logical device 0 with memory budgets of 55%, 15%, and 18%, leaving headroom
for runtime kernels such as the HiFi-GAN vocoder's cuDNN workspace. The
profile admits at most four sequences per stage and bounds Talker context
to 4096 tokens. For throughput,
`minicpmo_4_5_2gpu.yaml` gives the Thinker
GPU 0 (90%) and colocates the Talker (55%) and Code2Wav (35%) on GPU 1. That
profile admits at most four concurrent sequences per stage.

| deploy config | GPUs | Notes |
| --- | --- | --- |
| `minicpmo_4_5.yaml` (default) | 1 | Memory-constrained compatibility layout. |
| `minicpmo_4_5_2gpu.yaml` | 2 | Recommended continuous-batching layout; Talker and Code2Wav share GPU 1. |
| `minicpmo_4_5_3gpu.yaml` | 3 | One GPU per stage. |
| `minicpmo_4_5_8x4090.yaml` | 8 | Full 8x4090 layout. |

Every profile sets `session_mode: duplex`; `vllm-omni serve` detects the
pipeline's `duplex_plugin` and runs the model through `DuplexOmni`, so the
whole process serves duplex sessions only.

The split pipeline preserves native-duplex epoch/turn identity, segment text,
turn completion, reference voice, and terminal-audio metadata through
Code2Wav. Focused CPU regressions cover this envelope; run the Realtime
scenario below for live barge-in validation on the target GPU.

Default:

```bash
vllm serve openbmb/MiniCPM-o-4_5 \
    --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
```

For local ModelScope checkpoints, replace `openbmb/MiniCPM-o-4_5` with the
checkpoint path. Clients connect to `/v1/realtime?duplex=1` on this server
(`vllm_omni.clients.duplex.DuplexClient`, the CLI demo below, or the browser
client). To drive the model in-process without a server, use
`vllm_omni.clients.inline_duplex.InlineDuplexClient` over a `DuplexOmni`
(`examples/online_serving/barge_in_client.py --inline`).

### Per-stage overrides

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni --trust-remote-code --port 8099 \
    --stage-overrides '{"0": {"gpu_memory_utilization": 0.55}}'
```

## Run the Realtime duplex CLI demo

After the server is running, stream one WAV through the Realtime
WebSocket endpoint:

```bash
python examples/online_serving/minicpmo/realtime_duplex_demo.py \
    --url ws://localhost:8099/v1/realtime?duplex=1 \
    --model openbmb/MiniCPM-o-4_5 \
    --input-wav /path/to/input_16k_mono_pcm16.wav \
    --ref-audio /path/to/MiniCPM-o-Demo/assets/ref_audio/ref_minicpm_signature.wav \
    --output-dir /tmp/minicpmo_realtime_duplex_demo
```

Video input uses the same session. PyAV demuxes JPEG frames at `--video-fps`
(default 1.0) and vLLM `load_audio` extracts a 16 kHz mono WAV unless
`--input-wav` overrides the soundtrack. Frames keep their capture size
(`--frame-max-side 0`); the server `process_image` normalizes at 448.

`--stack-frames N` raises the visual refresh rate the way official duplex does:
each 1 s unit also samples the `N-1` sub-frames captured inside it and tiles
them into a single composite image sent next to that unit's base frame. The
audio cadence never changes — a unit is always one second — and the wire always
carries 2 images per unit however large `N` is, because the sub-frames share one
composite. Official uses 5 for high refresh rate mode:

```bash
python examples/online_serving/minicpmo/realtime_duplex_demo.py \
    --url ws://localhost:8099/v1/realtime?duplex=1 \
    --model openbmb/MiniCPM-o-4_5 \
    --input-video /path/to/clip.mp4 \
    --stack-frames 5 \
    --ref-audio /path/to/MiniCPM-o-Demo/assets/ref_audio/ref_minicpm_signature.wav \
    --output-dir /tmp/minicpmo_realtime_duplex_video_demo
```

Detail inside a composite is capped by `scale_resolution=448` at
`max_slice_nums=1`: official suggests HD slicing (`max_slice_nums=[2, 1]`) for
stacked frames, which the MiniCPM-o duplex plugin does not implement yet. Reading small
text or digits out of a wide scene is limited by that, not by frame timing.

## Open the experimental browser client

The browser UI serves the page and proxies the same-origin Realtime WebSocket to
the backend:

```bash
python -m examples.online_serving.minicpmo.realtime_web \
    --port 7862 \
    --ws-backend ws://127.0.0.1:8099 \
    --ref-audio /path/to/MiniCPM-o-Demo/assets/ref_audio/ref_minicpm_signature.wav
```

Open `http://<host>:7862/`. When using a reverse proxy, open the URL mapped to
port `7862`; the browser derives its WebSocket endpoint relative to that URL.

If the page proxy serves HTTP but does not forward WebSocket upgrades, point the
browser at a separately exposed Realtime endpoint:

```bash
python -m examples.online_serving.minicpmo.realtime_web \
    --port 7862 \
    --ws-backend ws://127.0.0.1:8099 \
    --public-realtime-url wss://public.example/v1/realtime
```

## Validate soft-interrupt behavior

```bash
python tests/e2e/online_serving/run_minicpmo_realtime_duplex_soft_interrupt.py \
    --url ws://localhost:8099/v1/realtime?duplex=1 \
    --input-wav /path/to/two_response_16k.wav \
    --ref-audio /path/to/ref_audio.wav
```

The soft-interrupt E2E driver defaults to `--validation-mode model-policy`,
which checks lifecycle and streaming invariants for arbitrary input audio. The
stronger `response-required` mode is diagnostic: it requires a purpose-built
two-response WAV, its `--input-sha256`, and an
`--expect-second-response-substring` value.

## Run Omni-DuplexEval

Omni-DuplexEval generation uses the vLLM-Omni native MiniCPM-o duplex
endpoint. Evaluation uses a separate OpenAI-compatible multimodal judge
served from the same vLLM-Omni environment. This validates the vLLM-Omni
duplex client and local judge integration; it does not reproduce the paper's
original MiniCPM-o inference implementation.

Start the duplex generation server:

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni --trust-remote-code \
    --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml \
    --served-model-name openbmb/MiniCPM-o-4_5 \
    --host 0.0.0.0 --port 8099
```

Start a separate multimodal judge. The allowed path must contain any local
videos passed with `--judge-video-mode video_url`:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
    --allowed-local-media-path /data/omni-duplex-eval \
    --host 0.0.0.0 --port 8000
```

Generate, evaluate, and summarize:

```bash
vllm bench omni-duplex-eval --omni generate \
    --url ws://127.0.0.1:8099/v1/realtime?duplex=1 \
    --model openbmb/MiniCPM-o-4_5 \
    --ref-audio /data/ref.wav \
    --dataset Hothan/Omni-DuplexEval \
    --concurrency 2 \
    --response-root /data/omni-duplex-eval/responses

vllm bench omni-duplex-eval --omni evaluate \
    --dataset Hothan/Omni-DuplexEval \
    --response-root /data/omni-duplex-eval/responses \
    --score-root /data/omni-duplex-eval/scores \
    --judge-base-url http://127.0.0.1:8000 \
    --judge-model Qwen/Qwen2.5-VL-7B-Instruct \
    --judge-video-mode video_url \
    --eval-workers 4

vllm bench omni-duplex-eval --omni summarize \
    --score-root /data/omni-duplex-eval/scores
```

The dataset names such as `RTD_OCR` and `PR_correction` are Hugging Face
splits, not dataset configurations. Pass one with `--split`, or omit it to
run all nine splits. `--limit 1` is useful for a smoke test. Realtime
generation records `clock=media`; artifacts generated with
`--pace as-fast-as-possible` record `clock=invalid` and evaluation rejects
them unless `--allow-invalid-clock` is explicit.

## Related examples

- [Offline MiniCPM-o inference](../../offline_inference/minicpmo/)
- [MiniCPM-o 4.5 recipe](../../../recipes/OpenBMB/MiniCPM-o-4_5.md)

## Pipeline notes

- Stage 1 performs request-owned AR continuous batching. Stage 2 keeps
  request-owned Flow/HiFT caches and batches exact-shape-compatible chunks.
- Reference audio travels with the first codec chunk; Stage 2 owns its
  temporary prompt WAV and evicts prompt features when the request finishes.
- Codec sampling reads the checkpoint `tts_config` (default deterministic
  seed 42). Stage-1 YAML sampling parameters govern only the binary
  continue/stop token exposed to vLLM.
- `StageRequestStats.batch_size` is request-scoped and does not report the
  scheduler's execution batch.
- Stage 0 and Stage 1 use vLLM CUDA Graph capture. Stage 2 remains eager until
  a dedicated exact-shape graph wrapper owns static I/O buffers and copies
  request cache state outside capture.
- Co-locating all three stages minimizes hardware requirements but makes their
  CUDA contexts contend for one GPU. Use the 8x4090 layout or a custom
  multi-GPU deploy config when throughput is the primary goal.
- Output audio streams as base64 PCM16 (24 kHz mono) in `response.output_audio.delta`.
- Offline counterpart:
  [`examples/offline_inference/minicpmo/`](../../offline_inference/minicpmo/)
- Recipe:
  [`recipes/OpenBMB/MiniCPM-o-4_5.md`](../../../recipes/OpenBMB/MiniCPM-o-4_5.md)
