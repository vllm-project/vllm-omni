#!/bin/bash
set -euo pipefail

MODE="${1:-basic}"
HOST="${HOST:-localhost}"
PORT="${PORT:-8091}"
MODEL="${MODEL:-inclusionAI/Ming-omni-tts-0.5B}"
API_URL="http://${HOST}:${PORT}/v1/audio/speech"
TEXT="${TEXT:-你好，这是 Ming 在线语音合成测试。}"
OUTPUT="${OUTPUT:-ming_output.wav}"
STREAM_OUTPUT="${STREAM_OUTPUT:-ming_output.pcm}"
REF_AUDIO="${REF_AUDIO:-}"
REF_TEXT="${REF_TEXT:-}"

post_json() {
    local payload="$1"
    local output_path="$2"
    curl -X POST "$API_URL" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer EMPTY" \
        -d "$payload" \
        --output "$output_path"
}

case "$MODE" in
    basic)
        post_json "{
            \"model\": \"${MODEL}\",
            \"input\": \"${TEXT}\",
            \"response_format\": \"wav\"
        }" "$OUTPUT"
        ;;
    zero_shot)
        if [ -z "$REF_AUDIO" ] || [ -z "$REF_TEXT" ]; then
            echo "zero_shot requires REF_AUDIO and REF_TEXT" >&2
            exit 1
        fi
        python - <<'PY' > /tmp/ming_zero_shot_payload.json
import base64
import json
import mimetypes
import os
from pathlib import Path

path = Path(os.environ["REF_AUDIO"])
mime_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
payload = {
    "model": os.environ["MODEL"],
    "input": os.environ["TEXT"],
    "ref_audio": f"data:{mime_type};base64,{base64.b64encode(path.read_bytes()).decode('utf-8')}",
    "ref_text": os.environ["REF_TEXT"],
    "response_format": "wav",
}
print(json.dumps(payload, ensure_ascii=False))
PY
        curl -X POST "$API_URL" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer EMPTY" \
            --data-binary @/tmp/ming_zero_shot_payload.json \
            --output "$OUTPUT"
        rm -f /tmp/ming_zero_shot_payload.json
        ;;
    stream)
        post_json "{
            \"model\": \"${MODEL}\",
            \"input\": \"${TEXT}\",
            \"stream\": true,
            \"response_format\": \"pcm\"
        }" "$STREAM_OUTPUT"
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        echo "Supported sanity checks: basic, zero_shot, stream" >&2
        exit 1
        ;;
esac
