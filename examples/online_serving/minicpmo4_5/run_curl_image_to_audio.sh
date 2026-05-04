#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <image_path> [prompt] [output_wav]" >&2
  exit 1
fi

SERVER="${SERVER:-http://localhost:8091}"
MODEL="${MODEL:-openbmb/MiniCPM-o-4_5}"
IMAGE_PATH="$1"
PROMPT="${2:-Describe the image in one single detailed spoken sentence of at least sixty words, mentioning every visible shape, its color, its approximate size, its position relative to the other shapes, the plain background, and the overall layout, and keep the answer natural but long enough to last more than ten seconds.}"
OUTPUT="${3:-minicpmo45_image_to_audio.wav}"
REF_AUDIO_PATH="${REF_AUDIO_PATH:-}"
REQUEST_JSON="$(mktemp)"
RESPONSE_JSON="$(mktemp)"
AUDIO_B64="$(mktemp)"
trap 'rm -f "${REQUEST_JSON}" "${RESPONSE_JSON}" "${AUDIO_B64}"' EXIT

python3 - "${IMAGE_PATH}" "${MODEL}" "${PROMPT}" "${REF_AUDIO_PATH}" "${REQUEST_JSON}" <<'PY'
import base64
import json
import mimetypes
import sys
from pathlib import Path

image_path, model, prompt, ref_audio_path, output_path = sys.argv[1:]
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
                        "When audio output is requested, reply with speech only "
                        "and follow any requested length constraints."
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
if ref_audio_path:
    import numpy as np
    import soundfile as sf

    wav, sr = sf.read(Path(ref_audio_path).expanduser(), dtype="float32", always_2d=False)
    wav_np = np.asarray(wav, dtype=np.float32)
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=-1)
    payload["additional_information"] = {
        "ref_audio": {
            "wav": wav_np.reshape(-1).tolist(),
            "sr": int(sr),
        }
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
