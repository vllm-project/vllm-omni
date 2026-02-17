# Text-To-Audio

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/stable_audio>.

This example demonstrates how to deploy Stable Audio models for online text-to-audio generation using vLLM-Omni.

## Supported Models

| Model | Description |
|-------|-------------|
| `stabilityai/stable-audio-open-1.0` | Open-source audio generation, up to ~47 seconds, 44.1 kHz stereo |

## Start Server

### Basic Start

```bash
vllm-omni serve stabilityai/stable-audio-open-1.0 \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --enforce-eager \
    --omni
```

## API Calls

### Method 1: Using curl

```bash
# Run all curl examples
bash curl_examples.sh

# Or execute directly
curl -X POST http://localhost:8000/v1/audio/generate \
    -H "Content-Type: application/json" \
    -d '{
        "input": "The sound of a cat purring",
        "audio_length": 10.0
    }' --output cat.wav
```

### Method 2: Using Python Client

```bash
cd examples/online_serving/stable_audio

# Simple generation
python stable_audio_client.py \
    --text "The sound of a cat purring"

# With custom duration
python stable_audio_client.py \
    --text "A dog barking" \
    --audio_length 5.0

# With all parameters
python stable_audio_client.py \
    --text "Thunder and rain" \
    --audio_length 15.0 \
    --negative_prompt "Low quality" \
    --guidance_scale 7.0 \
    --num_inference_steps 100 \
    --seed 42 \
    --output thunder.wav
```

The Python client supports the following command-line arguments:

- `--api_url`: API endpoint URL (default: `http://localhost:8000/v1/audio/generate`)
- `--text`: Text prompt for audio generation (default: `"The sound of a cat purring"`)
- `--audio_length`: Audio length in seconds (default: `10.0`, max ~47s for `stable-audio-open-1.0`)
- `--audio_start`: Audio start time in seconds (default: `0.0`)
- `--negative_prompt`: Negative prompt for classifier-free guidance (default: `"Low quality"`)
- `--guidance_scale`: Guidance scale for diffusion (default: `7.0`)
- `--num_inference_steps`: Number of inference steps (default: `100`)
- `--seed`: Random seed for reproducibility (default: `None`)
- `--response_format`: Audio output format (default: `wav`). Options: `wav`, `mp3`, `flac`, `pcm`
- `--output`: Output file path (default: `stable_audio_output.wav`)

### Method 3: Using Python httpx

```python
import httpx

response = httpx.post(
    "http://localhost:8000/v1/audio/generate",
    json={
        "input": "The sound of ocean waves crashing on a beach",
        "audio_length": 10.0,
        "negative_prompt": "Low quality, distorted",
        "guidance_scale": 7.0,
        "num_inference_steps": 100,
    },
    timeout=300.0,
)

with open("ocean.wav", "wb") as f:
    f.write(response.content)
```

## Request Format

### Simple Generation

```json
{
    "input": "The sound of ocean waves"
}
```

### Generation with Parameters

```json
{
    "input": "A piano playing a gentle melody",
    "audio_length": 10.0,
    "negative_prompt": "Low quality, distorted, noisy",
    "guidance_scale": 8.0,
    "num_inference_steps": 150,
    "seed": 42,
    "response_format": "wav"
}
```

## API Reference

### Endpoint

```
POST /v1/audio/generate
Content-Type: application/json
```

### Generation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | **required** | Text prompt describing the audio to generate |
| `model` | string | server's model | Model to use (optional, should match server if specified) |
| `response_format` | string | "wav" | Audio format: wav, mp3, flac, pcm, aac, opus |
| `speed` | float | 1.0 | Playback speed (0.25 - 4.0) |
| `audio_length` | float | null | Audio duration in seconds (max ~47s for `stable-audio-open-1.0`) |
| `audio_start` | float | 0.0 | Audio start time in seconds |
| `negative_prompt` | string | null | Text describing what to avoid in generation |
| `guidance_scale` | float | model default | Classifier-free guidance scale (higher = more adherence to prompt) |
| `num_inference_steps` | int | model default | Number of denoising steps (higher = better quality, slower) |
| `seed` | int | null | Random seed for reproducible generation |

### Response Format

Returns binary audio data with appropriate `Content-Type` header (e.g., `audio/wav`).

## Tuning Tips

1. **Audio Length**: Keep under 47 seconds for `stable-audio-open-1.0`.
2. **Quality vs Speed**:
   - 50 steps: Fast, decent quality (quick previews)
   - 100 steps: Good balance (general purpose)
   - 150+ steps: High quality, slower (final / critical audio)
3. **Guidance Scale**:
   - Lower (3 - 5): More creative / varied output
   - Default (7): Good balance
   - Higher (10+): Strict adherence to the prompt
4. **Negative Prompts**: Use to avoid unwanted characteristics such as `"Low quality"`, `"distorted"`, `"noisy"`.
5. **Seeds**: Set a fixed seed to get deterministic, reproducible results.

## File Description

| File | Description |
|------|-------------|
| `curl_examples.sh` | Curl examples covering common use cases |
| `stable_audio_client.py` | Python client with full CLI argument support |

## Troubleshooting

1. **Audio generation model did not produce audio output**: Verify the server started successfully and the model loaded without errors.
2. **Connection refused**: Make sure the server is running on the correct port.
3. **Generation timeout**: Reduce `num_inference_steps` or `audio_length`, and check GPU memory with `nvidia-smi`.
4. **Out of memory**: Lower `--gpu-memory-utilization` or reduce `audio_length`.
5. **Audio quality issues**: Increase `num_inference_steps`, add a negative prompt, or raise `guidance_scale`.

## Example materials

??? abstract "stable_audio_client.py"
    ``````py
    --8<-- "examples/online_serving/stable_audio/stable_audio_client.py"
    ``````
??? abstract "curl_examples.sh"
    ``````sh
    --8<-- "examples/online_serving/stable_audio/curl_examples.sh"
    ``````
