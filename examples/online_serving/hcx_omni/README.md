# HyperCLOVAX-SEED-Omni-8B with vLLM-Omni

[HyperCLOVAX-SEED-Omni-8B](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B)
is an omni-modal model by NAVER Cloud that supports:

| Input  | Output          |
|--------|-----------------|
| Text   | Text            |
| Audio  | Text + Audio    |
| Image  | Text            |
| Text   | Text + Image    |
| Audio  | Text + Audio + Image |

## Architecture

The model uses a 3-stage pipeline:

```
Stage 0 (Thinker) ──→ Stage 1 (Vision Decoder, diffusion)
         │
         └──────────→ Stage 2 (Audio Decoder, unit-BigVGAN)
```

- **Thinker**: Qwen2.5-VL vision encoder + Qwen2Audio encoder + HyperCLOVAX language model.
  Outputs text tokens and discrete audio/vision codes in the vocabulary.
- **Vision Decoder**: Diffusion-based image generation from 729 discrete TA-Tok codes.
- **Audio Decoder**: Unit-BigVGAN vocoder from CosyVoice2 FSQ discrete audio codes.

## Hardware Requirements

| Setup     | GPUs                                        |
|-----------|---------------------------------------------|
| Default   | 6 × GPU ≥24 GB (4 for thinker TP, 1+1 for decoders) |
| Minimal   | 3 × GPU ≥24 GB (1 for thinker, 1+1 for decoders) |

## Quick Start

### 1. Start the Server

```bash
# 6-GPU setup (production)
./run_server.sh --model naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B

# Custom GPU allocation
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./run_server.sh
```

### 2. Run the Client Demo

```bash
# All modes: text-only, text-to-vision, speech-to-speech
python client_demo.py --base-url http://localhost:8000/v1

# Speech-to-Speech with your own audio file
python client_demo.py --mode s2s --audio-file /path/to/speech.wav

# Text-to-Vision
python client_demo.py --mode t2v --prompt "고양이 그림을 그려줘"
```

### 3. Use the OpenAI API Directly

**Speech-to-Speech:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B",
    "modalities": ["text", "audio"],
    "messages": [{
      "role": "user",
      "content": [
        {"type": "input_audio", "input_audio": {"data": "<base64-wav>", "format": "wav"}},
        {"type": "text", "text": "이 오디오에 무슨 내용이 있나요?"}
      ]
    }]
  }'
```

**Text-to-Vision:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B",
    "modalities": ["text", "image"],
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "귀여운 강아지 한 마리가 공원에서 뛰노는 그림을 그려줘."}
      ]
    }]
  }'
```

## System Prompt (Required for Audio/Image Generation)

The thinker model decides whether to emit discrete audio or image tokens based
on context. **A system prompt is required** to reliably activate audio/image
generation. Without it, the model typically responds in text only.

```python
SYSTEM_PROMPT = {
    "role": "system",
    "content": [
        {
            "type": "text",
            "text": (
                "당신은 CLOVA X입니다. 네이버가 만든 AI 어시스턴트로서 "
                "오디오와 이미지를 인식하고 텍스트, 음성, 이미지를 생성할 수 있습니다."
            ),
        }
    ],
}
```

Include it as the first message in every request that expects audio or image output.

## Mode Activation Conditions

### Speech-to-Speech (S2S)

**Requirements:**
- `modalities: ["text", "audio"]`
- Audio input via `input_audio` content block (base64-encoded WAV/MP3)
- System prompt included

The thinker generates discrete audio unit tokens (`<|audio0000|>` … `<|audio6560|>`)
in its output, which are routed to the audio decoder (BigVGAN). The audio
response is in `choices[N].message.audio.data` (base64 WAV).

```python
response = client.chat.completions.create(
    model=MODEL,
    modalities=["text", "audio"],
    messages=[
        SYSTEM_PROMPT,
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_b64, "format": "wav"},
                },
                {"type": "text", "text": "이 오디오에 무슨 내용이 있나요?"},
            ],
        },
    ],
)

# The response may have two choices: one with text, one with audio
for choice in response.choices:
    if choice.message.audio:
        wav_bytes = base64.b64decode(choice.message.audio.data)
```

### Text-to-Vision (T2V)

**Requirements:**
- `modalities: ["text", "image"]`
- Text-only user message (no audio input)
- System prompt included

The thinker generates 729 discrete vision codes (`<|vision00000|>` … `<|vision65535|>`,
27×27 TA-Tok tokens), which are routed to the vision decoder (diffusion, 50 steps by
default). The image is returned in `choices[N].message.content` as an
`image_url` item with a `data:image/png;base64,...` URL.

```python
response = client.chat.completions.create(
    model=MODEL,
    modalities=["text", "image"],
    messages=[
        SYSTEM_PROMPT,
        {
            "role": "user",
            "content": [{"type": "text", "text": "귀여운 강아지가 공원에서 뛰노는 그림을 그려줘."}],
        },
    ],
)

# Parse raw JSON to access image_url content
import json, httpx
raw = json.loads(response._raw_response.content)
for choice in raw["choices"]:
    content = choice["message"].get("content", [])
    if isinstance(content, list):
        for item in content:
            if item.get("type") == "image_url":
                data_url = item["image_url"]["url"]   # "data:image/png;base64,..."
                img_bytes = base64.b64decode(data_url.split(",", 1)[1])
```

### Text-to-Text (T2T)

No special requirements. The thinker responds in text only.

```python
response = client.chat.completions.create(
    model=MODEL,
    modalities=["text"],
    messages=[{"role": "user", "content": "대한민국의 수도는 어디인가요?"}],
)
print(response.choices[0].message.content)
```

## Response Structure

| Mode | `choices[i].message` field | Content |
|------|--------------------------|---------|
| T2T  | `content` (str)          | Text response |
| S2S  | `content` (str)          | Text transcript |
| S2S  | `audio.data` (str)       | base64 WAV |
| T2V  | `content` (list)         | `[{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]` |

> **Note:** S2S responses typically contain two `choices` entries — one with
> the text and one with the audio. Iterate over all choices to collect both.

## Stage Config

The default stage config is at
`vllm_omni/model_executor/stage_configs/hcx_omni.yaml`.

Key parameters:

| Stage | Type      | `model_arch` / `model_class_name`  | GPU   |
|-------|-----------|------------------------------------|-------|
| 0     | LLM       | `HCXVisionV2ForCausalLM`           | 0-3   |
| 1     | Diffusion | `HyperCLOVAXVisionPipeline`        | 4     |
| 2     | Diffusion | `HyperCLOVAXAudioPipeline`         | 5     |

## Benchmarks

See [`benchmarks/hcx-omni/`](../../../benchmarks/hcx-omni/) for latency and
throughput measurement scripts.
