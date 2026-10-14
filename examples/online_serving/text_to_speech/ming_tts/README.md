# Ming-omni-tts Online Serving

Serve the dense `inclusionAI/Ming-omni-tts-0.5B` two-stage TTS model through
the OpenAI-compatible `/v1/audio/speech` endpoint.

## Start Server

```bash
vllm-omni serve inclusionAI/Ming-omni-tts-0.5B \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --omni \
    --port 8091 \
    --enforce-eager
```

Or:

```bash
cd examples/online_serving/text_to_speech/ming_tts
./run_server.sh
```

The tested ROCm environment is summarized in the
[Ming recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/inclusionAI/Ming-omni-tts-0.5B.md).

## Send Requests

The Python client targets `http://localhost:8091/v1` with `api_key=EMPTY`; it
does not call OpenAI's hosted API.

```bash
python openai_speech_client.py \
    --text "你好，这是 Ming 在线语音合成测试。" \
    --max-new-tokens 200
```

Style or dialect controls can be plain text or Ming JSON. The upstream
dialect example also uses `yue_prompt.wav` for speaker conditioning:

```bash
python openai_speech_client.py \
    --text "我觉得社会企业同个人都有责任" \
    --instruction-json '{"方言":"广粤话"}' \
    --ref-audio /path/to/yue_prompt.wav \
    --max-new-tokens 200
```

When `--ref-audio` is supplied without `--ref-text`, the server extracts the
Ming speaker embedding, matching upstream `use_spk_emb=True`, without using the
audio as a zero-shot prompt.

Reference-audio cloning:

```bash
python openai_speech_client.py \
    --text "我们的愿景是构建未来服务业的数字化基础设施，为世界带来更多微小而美好的改变。" \
    --ref-audio /path/to/10002287-00000094.wav \
    --ref-text "在此奉劝大家别乱打美白针。" \
    --max-new-tokens 200
```

Podcast-style multi-speaker prompt:

```bash
python openai_speech_client.py \
    --text " speaker_1:你可以说一下，就大概说一下，可能虽然我也不知道，我看过那部电影没有。
 speaker_2:就是那个叫什么，变相一节课的嘛。
 speaker_1:嗯。
 speaker_2:一部搞笑的电影。
 speaker_1:一部搞笑的。" \
    --ref-audio /path/to/CTS-CN-F2F-2019-11-11-423-012-A.wav \
    --ref-audio /path/to/CTS-CN-F2F-2019-11-11-423-012-B.wav \
    --ref-text " speaker_1:并且我们还要进行每个月还要考核 笔试的话还要进行笔试，做个，当服务员还要去笔试了
 speaker_2:对啊，这真的很奇怪，就是 单纯的因，单纯自己工资不高，只是因为可能人家那个店比较出名一点，就对你苛刻要求"
```

Streaming PCM:

```bash
python openai_speech_client.py \
    --text "你好，这是流式输出测试。" \
    --stream \
    --output ming_output.pcm
```

`run_curl.sh` keeps small smoke checks:

```bash
./run_curl.sh basic
REF_AUDIO=/path/to/reference.wav REF_TEXT="在此奉劝大家别乱打美白针。" ./run_curl.sh zero_shot
./run_curl.sh stream
```

## Request Fields

| Field | Ming meaning |
|-------|--------------|
| `input` | target text |
| `instructions` | plain style text, or JSON object for structured Ming controls |
| `voice` | Ming IP voice label unless it resolves to an uploaded speaker |
| `language` | Ming `方言` control |
| `ref_audio` | speaker reference; with `ref_text`, also supplies the prompt waveform |
| `ref_text` | transcript enabling zero-shot or podcast prompt-latent conditioning |
| `speaker_embedding` | 192-d Ming speaker embedding |
| `max_new_tokens` | Ming `max_decode_steps` |

## Notes

- `ref_audio` accepts local paths through the client, remote URLs, `file://`,
  or `data:` URLs.
- Non-streaming responses return WAV bytes; streaming responses return PCM.
- Music-only `bgm` generation is offline-only until the API exposes Ming
  prompt-mode selection.
