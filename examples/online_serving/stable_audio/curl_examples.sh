#!/bin/bash
# Examples for using Stable Audio with curl via /v1/audio/generate endpoint

# Example 1: Simple request with default parameters
echo "Example 1: Simple request with default parameters"
curl -X POST http://localhost:8000/v1/audio/generate \
    -H "Content-Type: application/json" \
    -d '{
        "input": "The sound audience clapping and cheering in a stadium"
    }' --output stadium.wav

# Example 2: Request with custom audio_length
echo "Example 2: Custom audio length (5 seconds)"
curl -X POST http://localhost:8000/v1/audio/generate \
    -H "Content-Type: application/json" \
    -d '{
        "input": "The sound of a dog barking",
        "audio_length": 5.0
    }' --output dog_5s.wav

# Example 3: Request with negative prompt for quality control
echo "Example 3: With negative prompt"
curl -X POST http://localhost:8000/v1/audio/generate \
    -H "Content-Type: application/json" \
    -d '{
        "input": "A piano playing a gentle melody",
        "audio_length": 10.0,
        "negative_prompt": "Low quality, distorted, noisy"
    }' --output piano.wav

# Example 4: Full control with all parameters
echo "Example 4: Full control (custom length, guidance, steps, seed)"
curl -X POST http://localhost:8000/v1/audio/generate \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Thunder and rain sounds",
        "audio_length": 15.0,
        "negative_prompt": "Low quality",
        "guidance_scale": 7.0,
        "num_inference_steps": 100,
        "seed": 42
    }' --output thunder_rain.wav

# Example 5: Quick generation with fewer steps (faster but lower quality)
echo "Example 5: Quick generation (fewer steps)"
curl -X POST http://localhost:8000/v1/audio/generate \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Ocean waves crashing on a beach",
        "audio_length": 8.0,
        "num_inference_steps": 50
    }' --output ocean.wav

echo "All examples completed!"
