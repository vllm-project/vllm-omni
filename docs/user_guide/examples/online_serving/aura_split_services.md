# AURA Split Services

This deployment splits the semantic AURA pipeline into three independently
served HTTP services:

```text
Client/orchestrator -> ASR service -> AURA service -> TTS service
```

The TTS service still uses the native Qwen3-TTS two-stage topology:

```text
Qwen3-TTS Talker -> Code2Wav
```

Because TTS runs as its own `qwen3_tts` service, its deploy profile can keep
`async_chunk: true` by default. The ASR and AURA services remain simple
single-stage text-producing services. The split-service examples default to
Qwen3-TTS CustomVoice mode with speaker `Vivian`; Base voice-clone mode is an
explicit override.

## Workflow

1. Start ASR with `vllm_omni/deploy/aura_asr_service.yaml`.
2. Start AURA with `vllm_omni/deploy/aura_vl_service.yaml`.
3. Start TTS with `vllm_omni/deploy/aura_tts_service.yaml`.
4. The client sends audio to ASR and receives a transcript.
5. The client sends transcript plus video to AURA and receives response text.
6. If AURA returns `<|silent|>`, the client stops.
7. Otherwise the client sends response text to TTS `/v1/audio/speech`.
8. TTS streams PCM chunks and the client writes a WAV file.

## Sequence

```mermaid
sequenceDiagram
    participant U as User/Benchmark
    participant C as Split Client
    participant ASR as ASR Service
    participant AURA as AURA Service
    participant TTS as TTS Service
    participant Talker as TTS Talker
    participant Code2Wav as Code2Wav

    U->>C: audio + video
    C->>ASR: /v1/chat/completions(audio)
    ASR-->>C: transcript text
    C->>AURA: /v1/chat/completions(video + transcript)
    AURA-->>C: response text or <|silent|>
    alt response text
        C->>TTS: /v1/audio/speech(stream=true)
        TTS->>Talker: text -> codec tokens
        Talker-->>Code2Wav: async chunks via shared memory
        Code2Wav-->>TTS: PCM chunks
        TTS-->>C: streamed audio bytes
        C-->>U: text + wav
    else <|silent|>
        C-->>U: text only; no TTS call
    end
```

## Start Services

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-ASR-1.7B \
  --omni \
  --port 8661 \
  --deploy-config vllm_omni/deploy/aura_asr_service.yaml \
  --served-model-name Qwen/Qwen3-ASR-1.7B \
  --allowed-local-media-path /data/ \
  --trust-remote-code

CUDA_VISIBLE_DEVICES=1 vllm serve aurateam/AURA \
  --omni \
  --port 8662 \
  --deploy-config vllm_omni/deploy/aura_vl_service.yaml \
  --served-model-name aurateam/AURA \
  --allowed-local-media-path /data/ \
  --trust-remote-code

CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --omni \
  --port 8663 \
  --deploy-config vllm_omni/deploy/aura_tts_service.yaml \
  --served-model-name Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --allowed-local-media-path /data/ \
  --trust-remote-code
```

For local checkpoints, set `ASR_MODEL`, `AURA_MODEL`, and `TTS_MODEL` before
running `/data/yrr/rein_test/start_aura_split_services.sh`. The service YAMLs do
not hardcode model paths, so the command-line model remains the source of truth.

## Run Client

```bash
python examples/online_serving/aura_omni/split_services_client.py \
  --audio-path /data/models/datasets/OmniInteract/data/1q1a/audios/0038_2.wav \
  --video-path /data/models/datasets/OmniInteract/data/1q1a/videos/0038.mp4 \
  --output-dir output_aura_split_services
```

The default TTS request is:

```text
task_type=CustomVoice, language=Chinese, voice=Vivian, stream=true
```

To run Base voice-clone mode instead, start the TTS service with a Base
checkpoint and pass reference audio/text:

```bash
python examples/online_serving/aura_omni/split_services_client.py \
  --audio-path /data/models/datasets/OmniInteract/data/1q1a/audios/0038_2.wav \
  --video-path /data/models/datasets/OmniInteract/data/1q1a/videos/0038.mp4 \
  --tts-model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --tts-task-type Base \
  --tts-ref-audio /data/yrr/vllm-omni/tests/assets/qwen3_tts/clone_2.wav \
  --tts-ref-text "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
```
