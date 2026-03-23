# Online serving Example of vLLM-Omni for MiMo-Audio

## 🛠️ Installation

Please refer to [README.md](../../../README.md)

## Run examples (MiMo-Audio)

### Launch the Server
```bash
export MIMO_AUDIO_TOKENIZER_PATH="XiaomiMiMo/MiMo-Audio-Tokenizer"

vllm-omni serve XiaomiMiMo/MiMo-Audio-7B-Instruct --omni \
--served-model-name "MiMo-Audio-7B-Instruct"  \
--port 18091 --stage-configs-path ./vllm_omni/model_executor/stage_configs/mimo_audio.yaml \
--chat-template ./examples/online_serving/mimo_audio/chat_template.jinja
```
> ⚠️ **Important**  
> **MiMo-Audio is not compatible with the default chat template.**  
> The provided `chat_template.jinja` implements MiMo-specific role, audio token, and instruction formatting and **must be used for all inference**.


### Send Multi-modal Request

Get into the example folder
```bash
cd examples/online_serving/mimo_audio
```

####  Send request via python

```bash
# Audio dialogue task
python openai_chat_completion_client_for_multimodal_generation.py \
--query-type multi_audios \
--message-json ../../offline_inference/mimo_audio/message_base64_wav.json
```

The Python client supports the following command-line arguments:

- `--query-type` (or `-q`): Query type (default: `multi_audios`)
  - Options: `multi_audios`, `text`
- `--message-json` (or `-m`): Path to `base64` multi rounds audio messages json file
  - Do not pass any value for "text" query type
  - Supports local file paths (automatically encoded to base64) or HTTP/HTTPS URLs, only for "Are these two audio clips the same?" task
  - Example: `---message-json ./examples/offline_inference/mimo_audio/message_base64_wav.json`
- `--prompt` (or `-p`): Custom text prompt/question, only for query type is "text"(TTS task)
  - Attention! Do not pass any value for "multi_audios" query type
  - Example: `--prompt "What are the main activities shown in this video?"`


For example, to use multi rounds audios with local files:

```bash
python openai_chat_completion_client_for_multimodal_generation.py \
--query-type multi_audios \
--message-json ../../offline_inference/mimo_audio/message_base64_wav.json
```
