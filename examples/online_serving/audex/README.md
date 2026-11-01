# Audex (Nemotron-Labs-Audex-2B) online serving

One checkpoint, four deployment modes. `run_server.sh` starts the server
for a given `MODE`; `client.py --mode <mode>` tests it. The capability
matrix:

| mode (`vllm_omni/deploy/audex_<mode>.yaml`) | audio in | text out | speech out | general audio out | endpoint |
|---|---|---|---|---|---|
| `tts` | ❌ | ❌ | ✅ | ❌ | `/v1/audio/speech` |
| `tta` | ❌ | ❌ | ❌ | ✅ | `/v1/audio/speech` |
| `thinker_only` | ✅ | ✅ | ❌ | ❌ | `/v1/chat/completions` |
| `s2s` | ✅ | ✅ | ✅ | ❌ | both |

Audio-input modes of `client.py` fall back to vLLM's public
`mary_had_lamb` asset when `--audio-file` is omitted; CFG defaults follow
the official settings per mode (tts/s2s 1.5, tta 3.0; `--cfg-scale 1.0`
disables).

## tts — text → speech

Text-only thinker (`checkpoint_folder_audiogen`) emits `<speechcodec_N>`
tokens; the streaming causal decoder produces 16 kHz WAVs. This is also
the DEFAULT pipeline when the repo root is served without a deploy config.

```bash
./run_server.sh                       # MODE=tts, port 8097

python client.py --mode tts --text "Hello there." --output hello.wav
# or raw curl:
curl -s http://localhost:8097/v1/audio/speech \
    -H 'Content-Type: application/json' \
    -d '{"model": "nvidia/Nemotron-Labs-Audex-2B", "input": "Hello there.", "response_format": "wav", "extra_params": {"cfg_scale": 1.5}}' \
    -o hello.wav
```

## tta — caption → general audio

Same thinker over the 4-codebook `<audiocodec_N>` RVQ block, decoded by
the external XCodec1 checkpoint (auto-downloaded; override with
`XCODEC1_PATH`). Clips are capped at 10 s.

```bash
MODE=tta ./run_server.sh

python client.py --mode tta --caption "Heavy rain falling on a tin roof." \
    --output rain.wav
```

## thinker_only — audio (+ instruction) → text

Single-stage audio understanding on the full checkpoint (NV-Whisper
encoder + projector + LM). Send the clip as `input_audio` chat content.

```bash
MODE=thinker_only ./run_server.sh

python client.py --mode thinker_only                      # transcribe the demo asset
python client.py --mode thinker_only --audio-file a.wav \
    --question "What language is being spoken?"
```

## s2s — spoken question → spoken answer

The audio-capable thinker plus the speech decoder in one deployment.
`client.py --mode s2s` runs the official three passes: ASR (chat with
`input_audio`) → chat on the transcript → TTS on the answer. Text-final
passes carry `"modalities": ["text"]`; only the TTS pass streams through
the decoder.

```bash
MODE=s2s PORT=8098 ./run_server.sh

python client.py --mode s2s --port 8098 \
    --audio-file question.wav --output answer.wav
```

Offline counterparts (no HTTP, one `Omni` engine per script) live in
`examples/offline_inference/audex/`.

## 30B-A3B (nvidia/Nemotron-Labs-Audex-30B-A3B)

`SIZE=30b` switches run_server.sh to the 30B yamls and model id:

    SIZE=30b ./run_server.sh                 # 30B tts on port 8097
    SIZE=30b MODE=s2s PORT=8098 ./run_server.sh

Pass `--model nvidia/Nemotron-Labs-Audex-30B-A3B` to the client — the server
validates the model id and 404s on a mismatch. Single-H100 defaults; see the
yaml comments for the TP2 fallback. First launch downloads ~60 GB.
