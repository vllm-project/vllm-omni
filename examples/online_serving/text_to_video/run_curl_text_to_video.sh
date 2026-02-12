#!/bin/bash
# Wan2.2 text-to-video curl example

OUTPUT_PATH="wan22_output.mp4"

curl -X POST http://localhost:8098/v1/videos \
  -H "Accept: application/json" \
  -F "prompt=Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  -F "seconds=2" \
  -F "size=832x480" \
  -F "negative_prompt=色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
  -F "fps=16" \
  -F "num_inference_steps=40" \
  -F "guidance_scale=4.0" \
  -F "guidance_scale_2=4.0" \
  -F "boundary_ratio=0.875" \
  -F "flow_shift=5.0" \
  -F "seed=42" | jq -r '.data[0].b64_json' | base64 -d > "${OUTPUT_PATH}"

echo "Saved video to ${OUTPUT_PATH}"
