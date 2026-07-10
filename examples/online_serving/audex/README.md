# Audex (Nemotron-Labs-Audex-2B) online serving

One checkpoint, four deployment pipelines. Pick the deploy yaml that matches
the task; the capability matrix:

| pipeline (`vllm_omni/deploy/<name>.yaml`) | audio in | text out | speech out | general audio out |
|---|---|---|---|---|
| `audex_tts` | ❌ | ❌ | ✅ | ❌ |
| `audex_tta` | ❌ | ❌ | ❌ | ✅ |
| `audex_thinker_only` | ✅ | ✅ | ❌ | ❌ |
| `audex_s2s` | ✅ | ✅ | ✅ | ❌ |

- **`audex_tts`** — text → speech. Text-only thinker
  (`checkpoint_folder_audiogen`) emits `<speechcodec_N>` tokens; the
  streaming causal speech decoder turns them into 16 kHz waveforms. Serve
  via `/v1/audio/speech`; classifier-free guidance through
  `extra_params.cfg_scale` (1.0 disables). This is the DEFAULT pipeline
  when the repo root is served without a deploy config.
- **`audex_tta`** — caption → general audio (rain, barking, …). Same
  audiogen thinker but over the interleaved 4-codebook `<audiocodec_N>`
  RVQ block, decoded by the external XCodec1 checkpoint. CFG is
  effectively mandatory (default scale 3.0). Serve via `/v1/audio/speech`.
- **`audex_thinker_only`** — audio (+ instruction) → text (ASR / audio
  QA). Single stage: the audio-capable full checkpoint (NV-Whisper
  encoder + projector + LM). Serve via `/v1/chat/completions` with
  `input_audio` content.
- **`audex_s2s`** — the cascaded speech-to-speech deployment: the
  audio-capable thinker plus the same speech decoder. Per-request
  `modalities` route the official three passes — `["text"]` requests
  (ASR, chat) finish at stage 0, only `["audio"]` requests (TTS) stream
  through the decoder.

## Quick start

TTS (default pipeline — no deploy config needed):

```bash
./run_server.sh                       # vllm-omni serve nvidia/Nemotron-Labs-Audex-2B --omni
curl -s http://localhost:8097/v1/audio/speech \
    -H 'Content-Type: application/json' \
    -d '{"model": "nvidia/Nemotron-Labs-Audex-2B", "input": "Hello there.", "response_format": "wav", "extra_params": {"cfg_scale": 1.5}}' \
    -o hello.wav
```

Speech-to-speech (three-pass cascade against one server):

```bash
vllm-omni serve nvidia/Nemotron-Labs-Audex-2B --omni --port 8098 \
    --trust-remote-code \
    --stage-configs-path vllm_omni/deploy/audex_s2s.yaml

python client.py --audio-file question.wav --port 8098 --output answer.wav
```

Text-to-audio / audio understanding: serve with
`--stage-configs-path vllm_omni/deploy/audex_tta.yaml` (then
`/v1/audio/speech` with the caption as `input`) or
`vllm_omni/deploy/audex_thinker_only.yaml` (then `/v1/chat/completions`
with `input_audio`). Offline counterparts for all four live in
`examples/offline_inference/audex/`.
