#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8091}"
MODEL="${MODEL:-aurateam/AURA}"
OUTPUT_DIR="${OUTPUT_DIR:-output_aura_omni_online}"
TTS_PASS_TOKEN_IDS="${TTS_PASS_TOKEN_IDS:-false}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_OMNI_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CLONE_REF_AUDIO="${VLLM_OMNI_ROOT}/tests/assets/qwen3_tts/clone_2.wav"
CLONE_REF_TEXT="Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
mkdir -p "$OUTPUT_DIR"

MARY_HAD_LAMB_AUDIO_URL="https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/mary_had_lamb.ogg"
SAMPLE_VIDEO_URL="https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4"

request_body=$(cat <<EOF
{
  "model": "$MODEL",
  "modalities": ["text", "audio"],
  "sampling_params_list": [
    {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 256, "seed": 42},
    {"temperature": 0.5, "top_p": 1.0, "top_k": -1, "max_tokens": 256, "seed": 42, "repetition_penalty": 1.0},
    {"temperature": 0.9, "top_k": 50, "max_tokens": 4096, "seed": 42, "detokenize": false, "repetition_penalty": 1.05, "stop_token_ids": [2150]},
    {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 65536, "seed": 42, "repetition_penalty": 1.0}
  ],
  "additional_information": {
    "aura_system_prompt": "You are receiving a live video stream where the final frame is the present moment. Respond only when a response is needed. Otherwise output '<|silent|>'. Respond in English.",
    "tts_task_type": "Base",
    "tts_ref_audio": "file://${CLONE_REF_AUDIO}",
    "tts_ref_text": "${CLONE_REF_TEXT}",
    "tts_language": "English",
    "tts_speaker": "Vivian",
    "tts_instruct": "",
    "tts_pass_token_ids": ${TTS_PASS_TOKEN_IDS}
  },
  "messages": [{
    "role": "user",
    "content": [
      {"type": "audio_url", "audio_url": {"url": "$MARY_HAD_LAMB_AUDIO_URL"}},
      {"type": "video_url", "video_url": {"url": "$SAMPLE_VIDEO_URL"}},
      {"type": "text", "text": "Use the audio and video together to decide whether a reply is needed. If needed, respond briefly in English."}
    ]
  }]
}
EOF
)

response=$(curl -sS --retry 3 --retry-delay 3 --retry-connrefused \
  -X POST "http://localhost:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$request_body")

echo "$response" | jq '.choices[].message.content'

audio_b64=$(echo "$response" | jq -r '.choices[]?.message.audio.data // empty' | head -n 1)
if [[ -n "$audio_b64" ]]; then
  echo "$audio_b64" | base64 -d > "${OUTPUT_DIR}/aura_omni_output.wav"
  echo "Audio saved to ${OUTPUT_DIR}/aura_omni_output.wav"
fi
