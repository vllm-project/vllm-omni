# IndexTTS-2 for voice-cloned TTS on 1x GPU

## Summary

- Vendor: IndexTeam
- Model: `IndexTeam/IndexTTS-2`
- Task: Text-to-speech with voice cloning and optional emotion control
- Mode: Online serving with the OpenAI-compatible `/v1/audio/speech` API
- Maintainer: Community

## When to use this recipe

Use this recipe to serve IndexTTS-2 as a two-stage TTS system for
voice-cloned speech synthesis. Each request provides synthesis text plus either
a reference audio clip or an uploaded audio voice; the model returns 22.05 kHz
mono speech. IndexTTS-2 also supports optional emotion conditioning through
model-specific `extra_params`.

IndexTTS-2 is **not an async-chunk streaming model** in the bundled deployment
configuration: the S2Mel flow-matching stage consumes the full mel-code sequence
from the AR talker before it produces audio. The OpenAI-compatible endpoint can
still be called with `stream=true` as an HTTP compatibility path, but that should
not be interpreted as incremental model streaming.

## References

- Upstream model:
  [IndexTeam/IndexTTS-2 on Hugging Face](https://huggingface.co/IndexTeam/IndexTTS-2)
- Online serving example:
  [`examples/online_serving/text_to_speech/README.md#indextts-2`](../../examples/online_serving/text_to_speech/README.md#indextts-2)
- Offline inference example:
  [`examples/offline_inference/text_to_speech/README.md#indextts-2`](../../examples/offline_inference/text_to_speech/README.md#indextts-2)
- OpenAI-compatible client:
  [`examples/online_serving/text_to_speech/indextts2/speech_client.py`](../../examples/online_serving/text_to_speech/indextts2/speech_client.py)
- Deploy config:
  [`vllm_omni/deploy/indextts2.yaml`](../../vllm_omni/deploy/indextts2.yaml)

## Hardware Support

### GPU

### 1x L4 24GB or larger CUDA GPU

#### Environment

- OS: Linux
- Python: 3.10+
- Driver / runtime: NVIDIA CUDA environment
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Command

Start the server from the repository root:

```bash
vllm serve IndexTeam/IndexTTS-2 --omni --trust-remote-code --port 8092
```

The bundled deployment config is selected automatically for the `indextts2`
model type. To make the config choice explicit, pass it directly:

```bash
vllm serve IndexTeam/IndexTTS-2 --omni --trust-remote-code --port 8092 \
  --deploy-config vllm_omni/deploy/indextts2.yaml
```

#### Verification

Run the bundled client. Local audio paths passed to this client are converted to
base64 data URLs before they are sent to the HTTP API:

```bash
python examples/online_serving/text_to_speech/indextts2/speech_client.py \
  --api-base http://localhost:8092 \
  --text "你好，这是 IndexTTS-2 语音合成测试。" \
  --ref-audio /path/to/reference.wav \
  --output indextts2.wav
```

For a raw HTTP request, pass `ref_audio` as an HTTP(S) URL, a base64 data URL,
or a `file://` URI allowed by the server's `--allowed-local-media-path`. Do not
pass an arbitrary local filesystem path directly in JSON unless your client
converts it first:

```bash
curl http://localhost:8092/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "IndexTeam/IndexTTS-2",
    "input": "你好，这是 IndexTTS-2 语音合成测试。",
    "response_format": "wav",
    "ref_audio": "data:audio/wav;base64,<BASE64_ENCODED_AUDIO>"
  }' \
  --output indextts2.wav
```

Optional emotion conditioning is passed through `extra_params`. Use one emotion
source mode at a time:

```bash
curl http://localhost:8092/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "IndexTeam/IndexTTS-2",
    "input": "今天心情很好！",
    "response_format": "wav",
    "ref_audio": "data:audio/wav;base64,<BASE64_ENCODED_REFERENCE_AUDIO>",
    "extra_params": {
      "emo_vector": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      "emo_alpha": 0.8
    }
  }' \
  --output indextts2_happy.wav
```

#### Notes

- Hardware scope: the default config is a single-GPU CUDA deployment. Stage 0
  and Stage 1 share GPU 0 by default.
- Audio output: 22.05 kHz mono WAV.
- Voice cloning: `ref_audio` is required for the documented raw request path.
  Alternatively, pass `voice` only when it names an uploaded audio voice from
  `/v1/audio/voices`; IndexTTS-2 does not provide a built-in text-only preset
  voice in this serving recipe.
- Reference audio format: use an HTTP(S) URL, a base64 data URL, or a `file://`
  URI allowed by the server's `--allowed-local-media-path` for raw API calls.
  The bundled Python client accepts local paths and converts them to base64 data
  URLs.
- Emotion control: `extra_params.emo_audio`, `extra_params.emo_vector`, and
  `extra_params.use_emo_text` are alternative conditioning modes. Official
  precedence is `use_emo_text` > `emo_vector` > `emo_audio` > same emotion as
  the speaker reference. The `emo_vector` order is `[happy, angry, sad, afraid,
  disgusted, melancholic, surprised, calm]`.
- Streaming limitation: `async_chunk` is disabled in
  `vllm_omni/deploy/indextts2.yaml`; S2Mel requires the full mel-code sequence
  before audio generation.
- First request can be slower because reference encoders and auxiliary models
  are lazily loaded and cached.
