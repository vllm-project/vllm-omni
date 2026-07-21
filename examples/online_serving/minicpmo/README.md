# MiniCPM-o 4.5: Online serving

OpenAI-compatible `/v1/chat/completions` serving for **MiniCPM-o 4.5**, plus a
Gradio UI and curl / Python clients.

Inputs: text / image / audio / video. Outputs: text and optional **24 kHz** speech.

## Installation

Please refer to [README.md](../../../README.md). Install the talker extra:

```bash
pip install 'vllm-omni[minicpmo]'
```

## Launch the server

The deploy config auto-loads via `--omni`; the default
`vllm_omni/deploy/minicpmo_4_5.yaml` targets a single-GPU layout (thinker
and talker + t2w co-located on GPU 0).  For other hardware layouts pick
one of the deploy variants below.

| deploy config | GPUs | Notes |
|---|---|---|
| `minicpmo_4_5.yaml` (default) | 1 | Thinker and talker+t2w co-located on GPU0. |
| `minicpmo_4_5_2gpu.yaml` | 2 | Thinker on GPU0, talker+t2w on GPU1. |
| `minicpmo_4_5_3gpu.yaml` | 3 | Thinker 2-way TP on GPU0/1, talker+t2w share GPU2. |
| `minicpmo_4_5_8x4090.yaml` | 8 | Full 8x4090 layout. |

Default (single-GPU):

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
```

Other layouts:

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5_3gpu.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
```

### Stage-based CLI (optional)

Stage 0 (thinker + API) and stage 1 (talker) can run in separate processes:

```bash
# Stage 0
CUDA_VISIBLE_DEVICES=0 vllm serve openbmb/MiniCPM-o-4_5 --omni \
    --trust-remote-code --port 8099 --stage-id 0 \
    --omni-master-address 127.0.0.1 --omni-master-port 26000

# Stage 1 (headless)
CUDA_VISIBLE_DEVICES=1 vllm serve openbmb/MiniCPM-o-4_5 --omni \
    --trust-remote-code --stage-id 1 --headless \
    --omni-master-address 127.0.0.1 --omni-master-port 26000
```

### Per-stage overrides

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni --trust-remote-code --port 8099 \
    --stage-overrides '{"0": {"gpu_memory_utilization": 0.65}}'
```

## Send multimodal requests

```bash
cd examples/online_serving/minicpmo
```

### curl

```bash
bash run_curl_multimodal_generation.sh text
bash run_curl_multimodal_generation.sh use_image
bash run_curl_multimodal_generation.sh use_audio '["text"]'   # text-only
```

Text + speech smoke test (TTS needs top-level `chat_template_kwargs`):

```bash
curl http://localhost:8099/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "openbmb/MiniCPM-o-4_5",
        "messages": [{"role": "user", "content": "Say hello, then introduce vLLM in one sentence."}],
        "modalities": ["text", "audio"],
        "chat_template_kwargs": {"use_tts_template": true}
    }'
```

### Python client

```bash
python openai_chat_completion_client_for_multimodal_generation.py \
    --query-type use_image \
    --port 8099 \
    --host localhost

# Text-only (faster; no <|tts_bos|>)
python openai_chat_completion_client_for_multimodal_generation.py \
    --query-type text \
    --modalities text \
    --prompt "Briefly introduce yourself."
```

Shared helpers also work if you pass MiniCPM defaults yourself:

```bash
python ../openai_chat_completion_client_for_multimodal_generation.py \
    --model openbmb/MiniCPM-o-4_5 \
    --query-type text \
    --port 8099
```

(Note: the shared client does **not** set `use_tts_template`; prefer the
MiniCPM-specific client above for speech.)

### Gradio demo

```bash
bash run_gradio_demo.sh

# Or:
python gradio_demo.py \
    --minicpmo45-api-base http://localhost:8099/v1 \
    --minicpmo45-model openbmb/MiniCPM-o-4_5 \
    --port 7862
```

Open `http://<host>:7862`. Uncheck **"Generate speech output (TTS)"** for
text-only responses.

## Modality control

| Modalities | Output |
|---|---|
| `["text"]` | Text only (no TTS bos) |
| `["text", "audio"]` / unset | Text + 24 kHz speech |

Speech requires `chat_template_kwargs.use_tts_template=true` so the chat
template appends `<|tts_bos|>`. For **curl**, put that field at the request
root; nested `extra_body` is ignored. The OpenAI Python SDK may use
`extra_body` because it merges those fields into the root.

## Notes

- Stage 1 is capped at `max_num_seqs: 1` in the deploy YAML (talker shares
  request-0 audio metadata).
- Output audio is base64 WAV in `message.audio.data` (24 kHz mono).
- Offline counterpart:
  [`examples/offline_inference/minicpmo/`](../../offline_inference/minicpmo/)
- Recipe:
  [`recipes/OpenBMB/MiniCPM-o-4_5.md`](../../../recipes/OpenBMB/MiniCPM-o-4_5.md)
