#!/bin/bash
# Stable Audio text-to-audio curl example

curl -X POST http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "The sound of a dog barking"}
    ],
    "extra_body": {
      "num_inference_steps": 100,
      "guidance_scale": 7.0,
      "audio_start_in_s": 0.0,
      "audio_end_in_s": 10.0,
      "seed": 42
    }
  }' | jq -r '.choices[0].message.content[0].audio_url.url' | cut -d',' -f2- | base64 -d > output.wav
