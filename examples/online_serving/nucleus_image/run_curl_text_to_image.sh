#!/bin/bash
# Nucleus-Image text-to-image curl example

curl -X POST http://localhost:8091/v1/images/generations   -H "Content-Type: application/json"   -d '{
    "prompt": "A weathered lighthouse on a rocky coastline at golden hour.",
    "size": "1024x1024",
    "num_inference_steps": 50,
    "guidance_scale": 4.0,
    "negative_prompt": "blurry, low quality",
    "seed": 42
  }' | jq -r '.data[0].b64_json' | base64 -d > nucleus_image_output.png
