# Mage-VL

> Codec-native image/video understanding plus proactive full-duplex streaming.

## Summary

- Vendor: Microsoft
- Model: [`microsoft/Mage-VL`](https://huggingface.co/microsoft/Mage-VL)
- Task: Image understanding, video understanding, codec-native video understanding,
  and event-gated streaming commentary
- Mode: Full-duplex WebSocket server and session adapter in
  `vllm_omni/experimental/fullduplex/mage_vl`
- Maintainer: Community

## Status

This experimental integration provides the Mage-VL adapter contract: bounded causal
visual windows, codec-window input, rolling gate evaluation, explicit query handling,
per-session state isolation, and stale-output cancellation through the shared
full-duplex runtime.

The production transport exposes `WebSocket /v1/mage-vl/duplex`, with bearer-token
authentication, bounded concurrent sessions and message sizes, idle timeouts,
disconnect cancellation, per-session state cleanup, and serialized access to the
shared Transformers checkpoint. Native vLLM model execution remains follow-up work.

The default `frames` backend uses decoded frames and has no codec preprocessing
requirement. The optional `codec` backend requires `ffmpeg` and `ffprobe`, plus the
`cv-preinfer` executable provided by `codec-video-prep>=0.2.5`, on `PATH`. Server
startup fails with an actionable error when `cv-preinfer` is unavailable; use
`--video-backend frames` when codec preprocessing is not installed.

## Serving

```bash
python -m vllm_omni.experimental.fullduplex.mage_vl.serving.server \
  --model microsoft/Mage-VL --host 0.0.0.0 --port 8090 \
  --auth-token "$MAGE_VL_API_KEY" --max-sessions 32 \
  --video-backend frames --num-frames 8 --target-fps 1 \
  --gate-threshold 0.5 --attn-impl sdpa

python examples/online_serving/mage_vl/duplex_client.py \
  --url ws://127.0.0.1:8090/v1/mage-vl/duplex \
  --auth-token "$MAGE_VL_API_KEY" \
  --video /path/to/segment.mp4 \
  --prompt "Describe this video segment in detail."
```

Each input window is a JSON `input.append` event whose `data.data.video_base64`
contains an MP4 segment. A text `input.append` before the video requests an explicit
answer; without text, the StreamMind gate decides whether to respond proactively.
The server emits `session.created`, `response.created`, one or more
`response.delta`, and `response.done` events. Clients can send `response.cancel` for
barge-in and `close` for graceful shutdown.

The adapter can still be wired to another backend through `--adapter module:factory`,
including:

- a Transformers `trust_remote_code=True` backend for offline/local testing;
- the model author's SGLang-compatible serving path;
- a future native vLLM `mage_vl` backend.

## Adapter Shape

```python
from vllm_omni.experimental.fullduplex.core.session import DuplexSession, DuplexSessionConfig
from vllm_omni.experimental.fullduplex.mage_vl import MageVLDuplexAdapter, MageVLDuplexRuntime


adapter = MageVLDuplexAdapter(gate=run_mage_gate, generate=run_mage_decoder)
session = DuplexSession(
    "stream-1",
    DuplexSessionConfig(
        input_modalities=("codec_window", "video", "image", "text"),
        output_modalities=("text",),
        proactive=True,
    ),
)
runtime = MageVLDuplexRuntime(session, adapter)
await runtime.run(input_events, emit_event)
```

Input events use the shared full-duplex protocol. For decoded video/image, pass the frame
payload directly. For codec-native streaming, pass `modality="codec_window"` and include
fields such as `segment_id`, `pts_ms`, `duration_ms`, `codec`, and `metadata`.

```python
{
    "type": "input.append",
    "modality": "codec_window",
    "data": {
        "kind": "h264",
        "segment_id": "seg-42",
        "pts_ms": 42000,
        "duration_ms": 1000,
        "codec": {
            "motion_vectors": ...,
            "residual_energy": ...,
        },
        "metadata": {"gop": "P"},
    },
}
```

The gate callable receives the current rolling window and returns either a
`MageVLGateDecision`, a mapping with `should_respond` / `text` / `event_id`, a bool, or
a text string. Repeated `event_id` values are de-duplicated at the session adapter.

## Verification

```bash
pytest tests/e2e/features/fullduplex/test_mage_vl_adapter.py \
  tests/examples/offline_inference/test_mage_vl.py
```
