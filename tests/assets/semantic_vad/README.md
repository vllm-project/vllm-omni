# Semantic VAD regression audio

Fixed synthetic Mandarin speech, mono 16 kHz PCM16, generated with
Microsoft Edge TTS `zh-CN-XiaoxiaoNeural` at `-5%` rate. These are authored
synthetic prompts, not recordings of users. Each full sentence was synthesized
once, split at TTS word timestamps, trimmed, and joined with the prescribed
digital silence. A 300 ms leading pad is included. No trailing silence is
included; the WebSocket test sends another five seconds of silence.
`manifest.json` records the text, speech intervals, voice, and SHA-256 hashes.
Tests load the checked-in WAVs and never call a TTS service.

| File | Input plan | Expected commit |
| --- | --- | --- |
| `complete.wav` | [不用帮我叫车了，我自己坐地铁回去。] | Once, about 1.2 s after the last speech |
| `short_pause.wav` | [帮我点一杯, pause 0.8 s, 少冰不加糖的拿铁。] | Once, after the continuation; never during the pause |
| `thinking_pause.wav` | [我今天不能去开会，因为, pause 3.8 s, 家里临时有点事情。] | Once, after the continuation; Smart Turn must hold the incomplete prefix |
| `abandoned.wav` | [我今天不能去开会，因为, remain silent] | Once, about 4 s after speech, by the maximum-silence fallback |

These cases guard the agreed `semantic_vad` / `eagerness=auto` behavior with
CPU INT8 Smart Turn and real Silero. The 0.8 s pause guards the 1.2 s candidate
setting; the 3.8 s pause exceeds that setting and therefore also exercises the
semantic decision. On the pinned model the incomplete prefix scores about
0.013, and the complete/continued sentences score above 0.83. These scores
explain fixture selection; tests assert externally visible commits and replies,
not exact floating-point model outputs. They are representative regressions,
not a general semantic-accuracy benchmark.

The E2E cases live in
`tests/entrypoints/openai_api/test_qwen3_omni_realtime_websocket.py`, reuse its
class-scoped real Qwen server, and inherit the existing merge-CI
`advanced_model and cuda and H100` route. They check one start/stop/commit/reply,
event ordering, input item ownership, input-clock endpoint timing, and completed
text/audio output. Timing tolerances allow acoustic offsets and Silero frame
rounding; they do not include network or model-generation latency.
