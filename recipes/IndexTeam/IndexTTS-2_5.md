# IndexTTS-2.5 for multilingual voice-cloned TTS on 1x GPU

## Summary

- Vendor: IndexTeam
- Model: `IndexTeam/IndexTTS-2.5`
- Task: Multilingual text-to-speech with voice cloning, emotion control, and
  native speed control
- Mode: Online serving with the OpenAI-compatible `/v1/audio/speech` API, or
  offline inference through `Omni`
- Maintainer: Community

## When to use this recipe

Use this recipe to serve IndexTTS-2.5 as a two-stage TTS system. Stage 0 is an
autoregressive talker; Stage 1 uses EnhancedCodec, S2Mel CFM/DiT, and BigVGAN
to produce 22.05 kHz mono speech. Each request supplies synthesis text and
reference audio, or an uploaded audio voice, for zero-shot voice cloning.

IndexTTS-2.5 adds multilingual text processing and model-native speed control
to the IndexTTS-2 serving contract.

## References

- Upstream model:
  [IndexTeam/IndexTTS-2.5 on Hugging Face](https://huggingface.co/IndexTeam/IndexTTS-2.5)
- Online serving example:
  [`examples/online_serving/text_to_speech/README.md#indextts-2-and-indextts-25`](../../examples/online_serving/text_to_speech/README.md#indextts-2-and-indextts-25)
- Offline inference example:
  [`examples/offline_inference/text_to_speech/README.md#indextts-2-and-indextts-25`](../../examples/offline_inference/text_to_speech/README.md#indextts-2-and-indextts-25)
- OpenAI-compatible client:
  [`examples/online_serving/text_to_speech/indextts2/speech_client.py`](../../examples/online_serving/text_to_speech/indextts2/speech_client.py)
- Standard deploy config:
  [`vllm_omni/deploy/indextts2_5.yaml`](../../vllm_omni/deploy/indextts2_5.yaml)

## Hardware Support

### GPU

### 1x NVIDIA H20 96GB

#### Environment

- OS: Linux
- Python: 3.10+
- Driver / runtime: NVIDIA CUDA environment
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

Install vLLM-Omni with the IndexTTS text-processing dependencies:

```bash
pip install 'vllm-omni[indextts2]'
```

Obtain the native IndexTTS-2.5 bundle and point `MODEL` at its root. The model
loader accepts the upstream nested `checkpoints/` layout.

#### Command

Start the standard code-only server from the repository root:

```bash
MODEL_VERSION=2.5 \
MODEL=/path/to/indextts-2.5 \
bash examples/online_serving/text_to_speech/indextts2/run_server.sh
```

This selects `vllm_omni/deploy/indextts2_5.yaml`, which uses
`use_gpt_latent=false`. To launch directly:

```bash
vllm serve /path/to/indextts-2.5 \
  --omni \
  --trust-remote-code \
  --port 8092 \
  --deploy-config vllm_omni/deploy/indextts2_5.yaml
```

#### Optional: NVIDIA MPS for higher single-GPU throughput

Stage 0 and Stage 1 run in separate processes on the same GPU. Enabling
NVIDIA Multi-Process Service (MPS) can improve steady-state throughput by
allowing work from both stages to overlap more effectively. MPS is an opt-in
deployment optimization: this recipe does not start it automatically, and the
gain depends on the request mix and concurrency, so benchmark it with the
intended workload before enabling it in production.

Pay attention to CUDA device renumbering. For example, if the MPS control
daemon is started with physical `CUDA_VISIBLE_DEVICES=1`, that GPU is exposed
to MPS clients as logical device `0`; launch the server with
`CUDA_VISIBLE_DEVICES=0`, not `1`. Use a dedicated MPS pipe/log directory when
the host is shared with other services, and stop only the MPS daemon owned by
this deployment.

#### Verification

Send a multilingual voice-cloning request with the bundled client. Local audio
paths are converted to base64 data URLs before transmission:

```bash
python examples/online_serving/text_to_speech/indextts2/speech_client.py \
  --api-base http://localhost:8092 \
  --model-version 2.5 \
  --model /path/to/indextts-2.5 \
  --lang zh \
  --text "你好，这是 IndexTTS-2.5 语音合成测试。" \
  --ref-audio /path/to/reference.wav \
  --output indextts2_5.wav
```

##### Native speed control

The public `speed` field is handled natively. Its accepted range is
`[0.5, 2.0]`; values above `1.0` produce shorter, faster speech. The serving
adapter converts the API convention to the model-native control with
`duration_factor = 1.0 / speed`, so `speed=1.25` uses
`duration_factor=0.8`:

```bash
python examples/online_serving/text_to_speech/indextts2/speech_client.py \
  --api-base http://localhost:8092 \
  --model-version 2.5 \
  --model /path/to/indextts-2.5 \
  --speed 1.25 \
  --text "这是使用模型原生语速控制的测试。" \
  --ref-audio /path/to/reference.wav \
  --output indextts2_5_speed.wav
```

##### Upload and reuse a named voice

Upload a reference recording once when callers should reuse the same voice
without sending `ref_audio` on every request. `consent` is required by the
voice-storage API:

```bash
curl -X POST http://localhost:8092/v1/audio/voices \
  -F "audio_sample=@/path/to/reference.wav" \
  -F "consent=user-consent-id" \
  -F "name=indextts_demo_voice" \
  -F "speaker_description=IndexTTS-2.5 demonstration voice"
```

Confirm that the voice was stored, then synthesize with its name instead of a
reference-audio payload:

```bash
curl http://localhost:8092/v1/audio/voices

python examples/online_serving/text_to_speech/indextts2/speech_client.py \
  --api-base http://localhost:8092 \
  --model-version 2.5 \
  --model /path/to/indextts-2.5 \
  --voice indextts_demo_voice \
  --lang zh \
  --text "你好，这是复用已上传音色的语音合成测试。" \
  --output indextts2_5_named_voice.wav
```

There are no built-in text-only preset voices. Each named IndexTTS voice must
first be created from an uploaded reference recording.

##### Multilingual synthesis and text normalization

The client accepts `zh` (Mandarin), `en` (English), `zhen` (mixed Chinese and
English), `ja` (Japanese), and `yue` (Cantonese). The following commands use
the same reference voice across all supported language modes:

```bash
CLIENT=examples/online_serving/text_to_speech/indextts2/speech_client.py
MODEL=/path/to/indextts-2.5
REF_AUDIO=/path/to/reference.wav

python "$CLIENT" --model-version 2.5 --model "$MODEL" \
  --ref-audio "$REF_AUDIO" --lang zh \
  --text "你好，这是中文语音合成测试。" --output indextts2_5_zh.wav

python "$CLIENT" --model-version 2.5 --model "$MODEL" \
  --ref-audio "$REF_AUDIO" --lang en \
  --text "Hello, this is an English speech synthesis test." \
  --output indextts2_5_en.wav

python "$CLIENT" --model-version 2.5 --model "$MODEL" \
  --ref-audio "$REF_AUDIO" --lang zhen \
  --text "Hello，欢迎使用 IndexTTS 二点五。" --output indextts2_5_zhen.wav

python "$CLIENT" --model-version 2.5 --model "$MODEL" \
  --ref-audio "$REF_AUDIO" --lang ja \
  --text "こんにちは、これは日本語の音声合成テストです。" \
  --output indextts2_5_ja.wav

python "$CLIENT" --model-version 2.5 --model "$MODEL" \
  --ref-audio "$REF_AUDIO" --lang yue \
  --text "你好，呢段係粵語語音合成測試。" --output indextts2_5_yue.wav
```

Text normalization is enabled by default. Disable it only when the input is
already written exactly as it should be spoken:

```bash
python examples/online_serving/text_to_speech/indextts2/speech_client.py \
  --model-version 2.5 --model /path/to/indextts-2.5 \
  --ref-audio /path/to/reference.wav --lang en --no-text-normalization \
  --text "The temperature is twenty five degrees." \
  --output indextts2_5_pre_normalized.wav
```

For raw HTTP requests, put the same language and text-normalization controls
under `extra_params`:

```bash
curl http://localhost:8092/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/indextts-2.5",
    "input": "This is an IndexTTS-2.5 test.",
    "response_format": "wav",
    "speed": 1.0,
    "ref_audio": "data:audio/wav;base64,<BASE64_ENCODED_AUDIO>",
    "extra_params": {
      "lang": "en",
      "text_normalization": true
    }
  }' \
  --output indextts2_5_en.wav
```

##### Emotion control

IndexTTS-2.5 supports an explicit eight-value emotion vector, an emotion text
description, or a separate emotion-reference recording. Use one source at a
time. Vector values are ordered as `happy`, `angry`, `sad`, `afraid`,
`disgusted`, `melancholic`, `surprised`, and `calm`; each value must be in
`[0, 1.2]`. `emo_alpha` controls the strength in `[0, 1]`.

Use an explicit emotion vector for deterministic control:

```bash
python examples/online_serving/text_to_speech/indextts2/speech_client.py \
  --model-version 2.5 --model /path/to/indextts-2.5 \
  --ref-audio /path/to/reference.wav --lang zh \
  --text "今天真是令人开心的一天！" \
  --emo-vector 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
  --emo-alpha 0.8 --output indextts2_5_happy_vector.wav
```

Infer the emotion from a natural-language description with `use_emo_text`:

```bash
python examples/online_serving/text_to_speech/indextts2/speech_client.py \
  --model-version 2.5 --model /path/to/indextts-2.5 \
  --ref-audio /path/to/reference.wav --lang zh \
  --text "我们终于完成了这项工作！" \
  --use-emo-text --emo-text "开心、兴奋而且充满活力" \
  --emo-alpha 0.8 --output indextts2_5_happy_text.wav
```

Transfer emotion from a recording while retaining the speaker identity from
`ref_audio`:

```bash
python examples/online_serving/text_to_speech/indextts2/speech_client.py \
  --model-version 2.5 --model /path/to/indextts-2.5 \
  --ref-audio /path/to/reference.wav --lang zh \
  --text "请用情绪参考音频中的表达方式朗读这句话。" \
  --emo-audio /path/to/emotion_reference.wav \
  --emo-alpha 0.8 --output indextts2_5_emotion_audio.wav
```

If multiple emotion sources are supplied, the official precedence is
`use_emo_text` > `emo_vector` > `emo_audio` > the emotion in the speaker
reference. `--use-random` is also available for exploratory random emotion
prototypes, but should not be used when reproducible conditioning is required.

For offline inference:

```bash
python examples/offline_inference/text_to_speech/indextts2/end2end.py \
  --model /path/to/indextts-2.5 \
  --model-version 2.5 \
  --lang zh \
  --speed 1.0 \
  --text "你好，这是离线语音合成测试。" \
  --ref-audio /path/to/reference.wav
```

#### Notes

- Hardware scope: the standard recipe has been exercised on one NVIDIA H200.
  It does not claim an unverified throughput or quality result.
- Audio output: 22.05 kHz mono WAV.
- Voice cloning: reference audio is required on the documented raw request
  path. Alternatively, `voice` may name an uploaded audio voice; there is no
  built-in text-only preset voice.
- Native speed: serving maps `speed` to the model's duration factor and skips
  generic waveform speed adjustment, so speed is applied exactly once.
- Languages: common codes include `zh`, `en`, `zhen` (mixed Chinese/English),
  `ja`, and `yue`; `Mandarin` is accepted as an alias for `zh`.
- Japanese normalization: `ja` tokenization does not automatically expand
  numbers, dates, or percentages. Write these inputs as readable Japanese text
  before inference.
- Emotion controls: `use_emo_text`, `emo_vector`, and `emo_audio` are
  alternative conditioning modes. Their precedence is `use_emo_text` >
  `emo_vector` > `emo_audio` > the speaker-reference emotion.
- Sampling difference: Stage 0 uses plain vLLM sampling, not the upstream
  default `num_beams=3` beam search. Use upstream `num_beams=1` for parity
  comparisons; output may differ from the official beam-search result.
