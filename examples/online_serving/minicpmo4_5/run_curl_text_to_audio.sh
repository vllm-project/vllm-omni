#!/usr/bin/env bash
set -euo pipefail

SERVER="${SERVER:-http://localhost:8091}"
MODEL="${MODEL:-openbmb/MiniCPM-o-4_5}"
OUTPUT="${OUTPUT:-minicpmo45_text_to_audio.wav}"
PROMPT="${1:-Please read this sentence aloud: vLLM Omni is testing MiniCPM online serving.}"
REQUEST_JSON="$(mktemp)"
RESPONSE_JSON="$(mktemp)"
AUDIO_B64="$(mktemp)"
trap 'rm -f "${REQUEST_JSON}" "${RESPONSE_JSON}" "${AUDIO_B64}"' EXIT

python3 - "${MODEL}" "${PROMPT}" "${REQUEST_JSON}" <<'PY'
import json
import sys

model, prompt, output_path = sys.argv[1:]
payload = {
    "model": model,
    "messages": [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are MiniCPM, a helpful multimodal assistant. "
                        "When audio output is requested, reply with speech only."
                    ),
                }
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ],
    "modalities": ["audio"],
    "chat_template_kwargs": {
        "use_tts_template": True,
        "enable_thinking": False,
    },
}
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(payload, f)
PY

HTTP_CODE="$(curl -sS -w '%{http_code}' -o "${RESPONSE_JSON}" "${SERVER}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data-binary "@${REQUEST_JSON}")"

if [[ "${HTTP_CODE}" != 2* ]]; then
  echo "Request failed with HTTP ${HTTP_CODE}" >&2
  jq -r '.error.message? // .error? // .' "${RESPONSE_JSON}" >&2 || cat "${RESPONSE_JSON}" >&2
  exit 1
fi

jq -r '[.choices[]?.message.audio.data // empty][0] // empty' "${RESPONSE_JSON}" > "${AUDIO_B64}"
if [[ ! -s "${AUDIO_B64}" ]]; then
  echo "Response did not include audio data" >&2
  jq . "${RESPONSE_JSON}" >&2 || cat "${RESPONSE_JSON}" >&2
  exit 1
fi

base64 -d "${AUDIO_B64}" > "${OUTPUT}"

echo "Audio saved to ${OUTPUT}"
