# Stable Audio Online Serving

Generate audio from text prompts using Stable Audio models via an OpenAI-compatible API endpoint.

## Features

- **OpenAI-compatible API**: Use `/v1/audio/generate` endpoint
- **Flexible control**: Adjust audio length, guidance scale, inference steps
- **Quality control**: Use negative prompts to avoid unwanted characteristics
- **Reproducible**: Set random seed for deterministic generation

## Quick Start

### 1. Start the Server

```bash
vllm-omni serve stabilityai/stable-audio-open-1.0 \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --enforce-eager \
    --omni
```

### 2. Generate Audio

#### Using curl

```bash
curl -X POST http://localhost:8000/v1/audio/generate \
    -H "Content-Type: application/json" \
    -d '{
        "input": "The sound of a cat purring",
        "audio_length": 10.0
    }' --output cat.wav
```

#### Using Python Client

```bash
python stable_audio_client.py \
    --text "The sound of a cat purring" \
    --audio_length 10.0 \
    --output cat.wav
```

#### Using Bash Script

```bash
bash curl_examples.sh
```

## API Reference

### Endpoint

```
POST /v1/audio/generate
```

### Request Body

```json
{
    "input": "Text description of the audio",
    "audio_length": 10.0,
    "audio_start": 0.0,
    "negative_prompt": "Low quality",
    "guidance_scale": 7.0,
    "num_inference_steps": 100,
    "seed": 42,
    "response_format": "wav"
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | **required** | Text prompt describing the audio to generate |
| `audio_length` | float | ~47s | Audio duration in seconds (max ~47s for stable-audio-open-1.0) |
| `audio_start` | float | 0.0 | Audio start time in seconds |
| `negative_prompt` | string | null | Text describing what to avoid in generation |
| `guidance_scale` | float | 7.0 | Classifier-free guidance scale (higher = more adherence to prompt) |
| `num_inference_steps` | int | 50 | Number of denoising steps (higher = better quality, slower) |
| `seed` | int | null | Random seed for reproducibility |
| `response_format` | string | "wav" | Output format: wav, mp3, flac, pcm |

### Response

Returns audio data in the requested format (default: WAV).

## Usage Examples

### Basic Generation

```bash
curl -X POST http://localhost:8000/v1/audio/generate \
    -H "Content-Type: application/json" \
    -d '{
        "input": "The sound of ocean waves"
    }' --output ocean.wav
```

### Custom Duration

```bash
curl -X POST http://localhost:8000/v1/audio/generate \
    -H "Content-Type: application/json" \
    -d '{
        "input": "A dog barking",
        "audio_length": 5.0
    }' --output dog_5s.wav
```

### High Quality with Negative Prompt

```bash
curl -X POST http://localhost:8000/v1/audio/generate \
    -H "Content-Type: application/json" \
    -d '{
        "input": "A piano playing a gentle melody",
        "audio_length": 10.0,
        "negative_prompt": "Low quality, distorted, noisy",
        "guidance_scale": 8.0,
        "num_inference_steps": 150
    }' --output piano_hq.wav
```

### Reproducible Generation

```bash
curl -X POST http://localhost:8000/v1/audio/generate \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Thunder and rain sounds",
        "audio_length": 15.0,
        "seed": 42
    }' --output thunder.wav
```

### Quick Generation (Fewer Steps)

For faster generation with slightly lower quality:

```bash
curl -X POST http://localhost:8000/v1/audio/generate \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Birds chirping in a forest",
        "audio_length": 8.0,
        "num_inference_steps": 50
    }' --output birds_quick.wav
```

## Python Client Examples

### Simple Generation

```bash
python stable_audio_client.py \
    --text "The sound of a cat purring"
```

### Custom Parameters

```bash
python stable_audio_client.py \
    --text "Thunder and rain" \
    --audio_length 15.0 \
    --negative_prompt "Low quality" \
    --guidance_scale 7.0 \
    --num_inference_steps 100 \
    --seed 42 \
    --output thunder.wav
```

### Different Output Format

```bash
python stable_audio_client.py \
    --text "Guitar playing" \
    --response_format mp3 \
    --output guitar.mp3
```

## Tips

1. **Audio Length**: Keep under 47 seconds for `stable-audio-open-1.0`
2. **Quality vs Speed**:
   - 50 steps: Fast, decent quality
   - 100 steps: Good balance (default)
   - 150+ steps: High quality, slower
3. **Guidance Scale**:
   - Lower (3-5): More creative/varied
   - Default (7): Good balance
   - Higher (10+): More literal to prompt
4. **Negative Prompts**: Use to avoid "Low quality", "distorted", "noisy", etc.
5. **Seeds**: Use same seed for reproducible results

## Performance

| Inference Steps | Quality | Speed | Use Case |
|----------------|---------|-------|----------|
| 50 | Good | Fast | Quick previews |
| 100 (default) | Very Good | Medium | Production |
| 150+ | Excellent | Slow | Final/critical audio |

## Troubleshooting

### Server not responding
- Check if server is running: `curl http://localhost:8000/health`
- Check server logs for errors

### Audio quality issues
- Increase `num_inference_steps` (e.g., 150)
- Add negative prompts: `"Low quality, distorted, noisy"`
- Increase `guidance_scale` for more prompt adherence

### Generation timeout
- Reduce `num_inference_steps`
- Reduce `audio_length`
- Check GPU memory with `nvidia-smi`

### Wrong audio length
- Ensure `audio_length` is within model limits (~47s max)
- Adjust `audio_start` if trimming is needed

## See Also

- [Offline Inference Example](../../offline_inference/text_to_audio/README.md)
- [Stable Audio Model Card](https://huggingface.co/stabilityai/stable-audio-open-1.0)
- [vLLM-Omni Documentation](https://github.com/vllm-project/vllm-omni)
