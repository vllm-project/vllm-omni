#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <image_path> [prompt] [output_wav]" >&2
  exit 1
fi

SERVER="${SERVER:-http://localhost:8091}"
MODEL="${MODEL:-openbmb/MiniCPM-o-4_5}"
IMAGE_PATH="$1"
PROMPT="${2:-Describe the image in one short spoken sentence.}"
OUTPUT="${3:-minicpmo45_image_to_audio.wav}"
REQUEST_JSON="$(mktemp)"
RESPONSE_JSON="$(mktemp)"
AUDIO_B64="$(mktemp)"
trap 'rm -f "${REQUEST_JSON}" "${RESPONSE_JSON}" "${AUDIO_B64}"' EXIT

python3 - "${IMAGE_PATH}" "${MODEL}" "${PROMPT}" "${REQUEST_JSON}" <<'PY'
import base64
import json
import mimetypes
import sys
from pathlib import Path

image_path, model, prompt, output_path = sys.argv[1:]
path = Path(image_path)
mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
image_url = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
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
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ],
        },
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
