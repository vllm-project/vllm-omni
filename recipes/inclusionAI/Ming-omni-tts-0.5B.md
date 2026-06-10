# Ming-omni-tts 0.5B

> Online serving for voice synthesis, cloning, and design

## Summary

- Vendor: inclusionAI
- Model: `inclusionAI/Ming-omni-tts-0.5B`
- Task: Text-to-speech with style, dialect, cloning, and multi-speaker controls
- Mode: Online serving via the OpenAI-compatible `/v1/audio/speech` API
- Maintainer: Community

## References

- [Model card](https://huggingface.co/inclusionAI/Ming-omni-tts-0.5B)
- [Upstream repository](https://github.com/inclusionAI/Ming-omni-tts)
- [Offline example](../../examples/offline_inference/text_to_speech/ming_tts/)
- [Online example](../../examples/online_serving/text_to_speech/ming_tts/)

## Hardware Support

This recipe documents a validated ROCm configuration and a CUDA configuration for the dense 0.5B two-stage TTS pipeline deployment.
Other hardware is welcome as community validation lands.

## CUDA

### 1x H100 80GB - TTS

#### Environment

- OS: Linux
- Python: 3.10+
- CUDA Driver Version: 590.48.01
- CUDA: 13.0
- vLLM version: 0.22.0
- vLLM-Omni version or commit: 0342827d

#### Command

Launch the two-stage talker:

```bash
vllm-omni serve inclusionAI/Ming-omni-tts-0.5B \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --omni \
    --port 8091 \
    --enforce-eager
```

#### Verification

Basic synthesis (save the WAV bytes):

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
      "model": "inclusionAI/Ming-omni-tts-0.5B",
      "input": "你好，这是 Ming 在线语音合成测试。",
      "response_format": "wav"
    }' --output ming_tts_basic.wav
```

For voice cloning / design, download the upstream assets from the vendor repo's [`inclusionAI/Ming-omni-tts/tree/main/data/wavs`](https://github.com/inclusionAI/Ming-omni-tts/tree/main/data/wavs) directory — the same cookbook assets used by the examples:

```bash
BASE=https://raw.githubusercontent.com/inclusionAI/Ming-omni-tts/main/data/wavs
curl -LO "$BASE/10002287-00000094.wav"                 # zero-shot cloning prompt
curl -LO "$BASE/CTS-CN-F2F-2019-11-11-423-012-A.wav"   # podcast speaker 1
curl -LO "$BASE/CTS-CN-F2F-2019-11-11-423-012-B.wav"   # podcast speaker 2
```

Reference-audio zero-shot cloning (supply both `ref_audio` and `ref_text`):

```bash
REF_AUDIO=$(base64 -w0 10002287-00000094.wav)
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
      "model": "inclusionAI/Ming-omni-tts-0.5B",
      "input": "我们的愿景是构建未来服务业的数字化基础设施，为世界带来更多微小而美好的改变。",
      "ref_audio": "data:audio/wav;base64,'"$REF_AUDIO"'",
      "ref_text": "在此奉劝大家别乱打美白针。",
      "max_new_tokens": 200,
      "response_format": "wav"
    }' --output ming_tts_zero_shot.wav
```

Podcast-style multi-speaker generation (one reference clip plus one transcript per speaker; `ref_audio` becomes a JSON array, and the `speaker_N` labels in `input`/`ref_text` line up with the reference order):

```bash
REF_A=$(base64 -w0 CTS-CN-F2F-2019-11-11-423-012-A.wav)
REF_B=$(base64 -w0 CTS-CN-F2F-2019-11-11-423-012-B.wav)
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
      "model": "inclusionAI/Ming-omni-tts-0.5B",
      "input": " speaker_1:你可以说一下，就大概说一下，可能虽然我也不知道，我看过那部电影没有。\n speaker_2:就是那个叫什么，变相一节课的嘛。\n speaker_1:嗯。\n speaker_2:一部搞笑的电影。\n speaker_1:一部搞笑的。",
      "ref_audio": ["data:audio/wav;base64,'"$REF_A"'", "data:audio/wav;base64,'"$REF_B"'"],
      "ref_text": " speaker_1:并且我们还要进行每个月还要考核 笔试的话还要进行笔试，做个，当服务员还要去笔试了\n speaker_2:对啊，这真的很奇怪，就是 单纯的因，单纯自己工资不高，只是因为可能人家那个店比较出名一点，就对你苛刻要求",
      "response_format": "wav"
    }' --output ming_tts_podcast_style.wav
```

Streaming PCM (set `"stream": true` and request the `pcm` format):

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
      "model": "inclusionAI/Ming-omni-tts-0.5B",
      "input": "你好，这是流式输出测试。",
      "stream": true,
      "response_format": "pcm"
    }' --output ming_tts_stream.pcm
```

## ROCm

### 1x AMD `gfx942`

#### Environment

- OS: Ubuntu 22.04.5 LTS, x86_64
- Python: 3.12.13
- ROCm / HIP: 7.2.53211
- vLLM version: 0.22.0
- vLLM-Omni version or commit: 0.1.dev1873 / `99c35c410`
- Docker image: `vllm/vllm-omni-rocm:v0.22.0`

#### Command

From the vLLM-Omni repository root:

```bash
docker run --rm \
    --group-add=video \
    --ipc=host \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device /dev/kfd \
    --device /dev/dri \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v "$PWD":/app/vllm-omni \
    -w /app/vllm-omni \
    -e VLLM_ROCM_USE_AITER=0 \
    -p 8091:8091 \
    vllm/vllm-omni-rocm:v0.22.0 \
    --model inclusionAI/Ming-omni-tts-0.5B \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --omni \
    --port 8091 \
    --enforce-eager
```

#### Verification

```bash
python examples/online_serving/text_to_speech/ming_tts/openai_speech_client.py \
    --text "我觉得社会企业同个人都有责任" \
    --instruction-json '{"方言":"广粤话"}' \
    --ref-audio /path/to/yue_prompt.wav \
    --max-new-tokens 200 \
    --output dialect.wav
```

`--ref-audio` matches upstream `use_spk_emb=True`; do not add `--ref-text`
for the dialect case.

## Notes

- The official ROCm image includes the platform dependencies.
- See the [ROCm installation guide](../../docs/getting_started/installation/gpu.md) for interactive and source-build workflows.
- The reference clips above are a subset of the upstream [`inclusionAI/Ming-omni-tts/tree/main/data/wavs`](https://github.com/inclusionAI/Ming-omni-tts/tree/main/data/wavs) cookbook fixtures.
- The tested environment uses `--enforce-eager`.
- Non-streaming responses return WAV bytes; streaming responses return PCM.
- Output is mono 44.1 kHz audio.
