#!/usr/bin/env bash
set -euo pipefail

SERVER="${SERVER:-http://localhost:8091}"
MODEL="${MODEL:-openbmb/MiniCPM-o-4_5}"
OUTPUT="${OUTPUT:-minicpmo45_text_to_audio.wav}"
TEXT_OUTPUT="${TEXT_OUTPUT:-${OUTPUT%.*}.txt}"
PROMPT="${1:-${PROMPT:-Please read this single long sentence aloud exactly once without shortening it: vLLM Omni is running a benchmark for MiniCPM speech generation, and this sentence intentionally includes enough detail about streaming text to audio generation, multimodal reasoning, stage connectors, careful benchmarking, and stable speech synthesis behavior to last well over ten seconds when spoken at a natural pace.}}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-When audio output is requested, reply with speech only and follow any requested length constraints.}"
REF_AUDIO_PATH="${REF_AUDIO_PATH:-}"
REQUEST_JSON="$(mktemp)"
RESPONSE_JSON="$(mktemp)"
AUDIO_B64="$(mktemp)"
trap 'rm -f "${REQUEST_JSON}" "${RESPONSE_JSON}" "${AUDIO_B64}"' EXIT

python3 - "${MODEL}" "${PROMPT}" "${SYSTEM_PROMPT}" "${REF_AUDIO_PATH}" "${REQUEST_JSON}" <<'PY'
import json
import sys
from pathlib import Path

model, prompt, system_prompt, ref_audio_path, output_path = sys.argv[1:]
payload = {
    "model": model,
    "messages": [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                }
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ],
    "modalities": ["text", "audio"],
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
jq -r '[.choices[]?.message.content // empty][0] // empty' "${RESPONSE_JSON}" > "${TEXT_OUTPUT}"

echo "Audio saved to ${OUTPUT}"
echo "Text saved to ${TEXT_OUTPUT}"
