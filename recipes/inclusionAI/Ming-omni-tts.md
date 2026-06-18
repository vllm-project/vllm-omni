# Ming-omni-tts

> Online serving for voice synthesis, cloning, and design

## Summary

- Vendor: inclusionAI
- Model: `inclusionAI/Ming-omni-tts-0.5B`
- Task: Text-to-speech with style, dialect, cloning, and multi-speaker controls
- Mode: Online serving via the OpenAI-compatible `/v1/audio/speech` API
- Maintainer: Community

## References

- [Huggingface Model card](https://huggingface.co/inclusionAI/Ming-omni-tts-0.5B)
- Upstream repository [inclusionAI/Ming-omni-tts](https://github.com/inclusionAI/Ming-omni-tts)


## Hardware Support

This recipe documents a validated ROCm configuration and a CUDA configuration for the dense 0.5B two-stage TTS pipeline deployment.
Other hardware is welcome as community validation lands.

## CUDA

### 1x H100 80GB

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

The following requests condition on reference clips from upstream audio assets. The server resolves `ref_audio` as an `http(s)://` URL, a `file://` path, or an inline base64 `data:` URL, so the examples point it straight at the public cookbook fixtures in the vendor repo's [`inclusionAI/Ming-omni-tts/tree/main/data/wavs`](https://github.com/inclusionAI/Ming-omni-tts/tree/main/data/wavs) directory:

```bash
BASE=https://raw.githubusercontent.com/inclusionAI/Ming-omni-tts/main/data/wavs
```

And note that the server latency will include automatically resolving and downloading input files.

Reference-audio zero-shot cloning:

```bash
REF_AUDIO="$BASE/10002287-00000094.wav"
jq -n --arg ref_audio "$REF_AUDIO" \
    '{
       model: "inclusionAI/Ming-omni-tts-0.5B",
       input: "我们的愿景是构建未来服务业的数字化基础设施，为世界带来更多微小而美好的改变。",
       ref_audio: $ref_audio,
       ref_text: "在此奉劝大家别乱打美白针。",
       max_new_tokens: 200,
       response_format: "wav"
     }' \
| curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d @- --output ming_tts_zero_shot.wav
```

Podcast-style multi-speaker generation uses one reference clip plus one transcript per speaker, with the `speaker_N` labels in `input`/`ref_text` lined up with the reference order:

```bash
REF_A="$BASE/CTS-CN-F2F-2019-11-11-423-012-A.wav"
REF_B="$BASE/CTS-CN-F2F-2019-11-11-423-012-B.wav"

INPUT=' speaker_1:你可以说一下，就大概说一下，可能虽然我也不知道，我看过那部电影没有。
 speaker_2:就是那个叫什么，变相一节课的嘛。
 speaker_1:嗯。
 speaker_2:一部搞笑的电影。
 speaker_1:一部搞笑的。'

REF_TEXT=' speaker_1:并且我们还要进行每个月还要考核 笔试的话还要进行笔试，做个，当服务员还要去笔试了
 speaker_2:对啊，这真的很奇怪，就是 单纯的因，单纯自己工资不高，只是因为可能人家那个店比较出名一点，就对你苛刻要求'

jq -n \
    --arg input "$INPUT" \
    --arg ref_text "$REF_TEXT" \
    --arg a "$REF_A" \
    --arg b "$REF_B" \
    '{
       model: "inclusionAI/Ming-omni-tts-0.5B",
       input: $input,
       ref_audio: [$a, $b],
       ref_text: $ref_text,
       response_format: "wav"
     }' \
| curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d @- --output ming_tts_podcast_style.wav
```

## ROCm

### 1x AMD MI300X

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
- To clone from local audio instead of a URL, pass a `file://` URI (launch the server with `--allowed-local-media-path <dir>`), or base64-encode the clip into a `data:` URL written to a payload file and send it with `curl -d @payload.json`. Putting the base64 directly on the `curl` command line overflows the OS argument limit (`bash: /usr/bin/curl: Argument list too long`).
- The tested environment uses `--enforce-eager`.
- Non-streaming responses return WAV bytes; streaming responses return PCM.
- Output is mono 44.1 kHz audio.
