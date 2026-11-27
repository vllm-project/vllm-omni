# Sophon-Realtime (Tag, Flash) on one H100 80 GB

## Summary

- Vendor: Adventists AI
- Models: [`adventists-ai/Sophon-Realtime-32B-Flash`](https://huggingface.co/adventists-ai/Sophon-Realtime-32B-Flash),
  [`adventists-ai/Sophon-Realtime-32B-Tag`](https://huggingface.co/adventists-ai/Sophon-Realtime-32B-Tag) (Apache-2.0)
- Task: speech-to-speech. Full duplex over `/v1/realtime`, half duplex over `/v1/chat/completions`.
- Pipeline: DuplexJev thinker (speech + optional image in, text out) → talker → vocoder, three stages on one GPU,
  streamed with `async_chunk`.
    - Flash: Qwen3-TTS talker + code2wav.
    - Tag: CosyVoice3 talker (bistream text input) + flow + HiFT.
- Turn-taking: the thinker itself decides when the user has finished (a one-token turn question at pauses), not a
  VAD timeout. Speech over a reply is told apart as a barge-in, a back-channel or a late continuation of the turn.
- Maintainer: [`@BKtensor`](https://github.com/BKtensor)
- Design discussion: [RFC #10406](https://github.com/vllm-project/vllm-omni/issues/10406)

## Hardware Support

Verified on one NVIDIA H100 80 GB (vLLM 0.34.0). The thinker checkpoint is FP8; the three stages share the GPU
through a private MPS daemon (`cuda_mps: true` in the deploy config).

## GPU

### 1x H100 80 GB

#### Command

The model argument is the Hugging Face id or a local download of it.

```bash
# Flash
vllm serve adventists-ai/Sophon-Realtime-32B-Flash --omni \
  --deploy-config vllm_omni/deploy/sophon_realtime_flash.yaml --port 8000

# Tag
vllm serve adventists-ai/Sophon-Realtime-32B-Tag --omni \
  --deploy-config vllm_omni/deploy/sophon_realtime_tag.yaml --port 8000
```

Endpoints:

- `WS /v1/realtime`: full duplex. Send `session.update` with `turn_detection: {"type": "server_vad"}` and
  `extra_body: {"auto_response": true}`, then stream 16 kHz microphone audio with `input_audio_buffer.append`.
  The server decides when to answer; the client never commits.
- `POST /v1/chat/completions`: half duplex (spoken question in, text + audio out).

Optional per-session settings in `extra_body` (validated; out-of-range values are rejected at `session.update`):

| key | range | meaning |
| --- | --- | --- |
| `turn_fallback_s` | 0.3 – 10 | reply anyway after this much silence |
| `min_release_s` | 0 – 5 | minimum pause before speaking once the user is judged finished |
| `min_release_after_s` | 0 – 30 | apply `min_release_s` only after this much user speech |

#### Verification

```bash
# CPU: turn policy, turn question, bistream schedule, thinker -> talker streaming
pytest tests/model_executor/models/sophon_realtime tests/model_executor/stage_input_processors/test_sophon_realtime.py -q
```

#### Measured (H100 80 GB, Flash, vLLM 0.34.0)

Half duplex, 16 spoken questions (zh/en), temperature 0, time from request to the first audio chunk:

| concurrency | median | max |
| --- | --- | --- |
| 1 | 0.19 s | 0.20 s |
| 4 | 0.25 s | 0.31 s |
| 8 | 0.31 s | 0.38 s |

Full duplex, scripted 16 kHz client streaming in real time (barge-in 2 s into a reply, a back-channel during a
reply, a question split by a 1 s pause, three talk-overs in a row; each run twice): no errors, every reply has
audio, and the last question is answered 0.1 – 0.6 s after the user stops speaking.
