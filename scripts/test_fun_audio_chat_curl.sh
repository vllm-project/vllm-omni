#!/usr/bin/env bash
# Smoke-test Fun-Audio-Chat-8B S2S via vllm-omni's OpenAI-compatible API.
#
# Usage:
#   ./scripts/test_fun_audio_chat_curl.sh
#
# Overridable env:
#   API_BASE  (default http://localhost:8091)
#   AUDIO     (default src/funaudiochat/examples/ck7vv9ag.wav)
#   OUT_WAV   (default /tmp/fac_serve_reply.wav)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${API_BASE:=http://localhost:8091}"
: "${AUDIO:=$REPO/src/funaudiochat/examples/ck7vv9ag.wav}"
: "${OUT_WAV:=/tmp/fac_serve_reply.wav}"
: "${MODEL:=/home/jovyan/ye/vllm-omni/pretrained_models/Fun-Audio-Chat-8B}"

[[ -f "$AUDIO" ]] || { echo "missing $AUDIO"; exit 1; }

SYS='You are asked to generate both text and speech tokens at the same time. 你的名字是小云。你是一位来自杭州的温柔友善的女孩，声音甜美，举止亲切。你的回复简短，通常只有一到三句话。'

echo "[test] building request payload from $AUDIO"
REQ_FILE=$(mktemp --suffix=.json)
trap 'rm -f "$REQ_FILE"' EXIT
python3 - "$MODEL" "$SYS" "$AUDIO" "$REQ_FILE" <<'PY'
import json, sys, base64
model, sysprompt, audio_path, outpath = sys.argv[1:5]
with open(audio_path, "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("ascii")
req = {
    "model": model,
    "messages": [
        {"role": "system", "content": sysprompt},
        {"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
        ]},
    ],
    "temperature": 0.0,
    "max_tokens": 512,
    "modalities": ["text", "audio"],
    "audio": {"voice": "default", "format": "wav"},
}
with open(outpath, "w") as f:
    json.dump(req, f)
PY

echo "[test] POST $API_BASE/v1/chat/completions ($(wc -c <"$REQ_FILE") bytes)"
RESP_FILE=/tmp/last_resp.json
curl -sS -m 300 "$API_BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer EMPTY' \
    --data-binary @"$REQ_FILE" -o "$RESP_FILE"
RESP=$(head -c 500 "$RESP_FILE" || true)

echo "[test] raw response (first 500 chars):"
printf '%s' "$RESP" | head -c 500
echo

# Try to extract text + audio.
# Fun-Audio-Chat's pipeline produces TWO choices per response:
#   choices[0] = text (message.content)
#   choices[1] = audio (message.audio.{id, data, transcript, expires_at})
python3 - <<'PY'
import json, base64, os, sys
with open("/tmp/last_resp.json") as f:
    r = json.load(f)
# Merge into a single view.
text_content = None
audio_b64 = None
for c in r.get("choices", []):
    m = c.get("message") or {}
    if m.get("content") and text_content is None:
        text_content = m["content"]
    audio = m.get("audio") or {}
    if audio.get("data") and audio_b64 is None:
        audio_b64 = audio["data"]

if text_content:
    msg = {"content": text_content, "audio": {"data": audio_b64}}
else:
    msg = (r.get("choices") or [{}])[0].get("message", {})
msg = (r.get("choices") or [{}])[0].get("message", {})
print(f"[test] text: {msg.get('content') or '(empty)'}")
audio = msg.get("audio", {}).get("data") if isinstance(msg.get("audio"), dict) else None
out = os.environ.get("OUT_WAV", "/tmp/fac_serve_reply.wav")
if audio:
    with open(out, "wb") as f:
        f.write(base64.b64decode(audio))
    sz = os.path.getsize(out)
    print(f"[test] wav  : {out} ({sz} bytes)")
else:
    print("[test] wav  : (no audio in response)")
PY
