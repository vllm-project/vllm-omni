#!/usr/bin/env bash
# 用 seed-tts 数据集里的参考音 + 文本，向正在运行的 TTS 服务发一条
# /v1/audio/speech 请求，把返回的音频保存成 wav。
#
# 用法:
#   bash gen_audio.sh [locale] [row_index] [out_wav]
#   bash gen_audio.sh zh 0 /root/test_out.wav      # 默认
#   bash gen_audio.sh en 3 /root/en3.wav
#
# 依赖: 服务已在 $HOST:$PORT 上跑 (bash start_pd.sh <topo> 起的那个)。
set -euo pipefail

LOCALE="${1:-zh}"
ROW_IDX="${2:-0}"
OUT="${3:-/root/test_out.wav}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8091}"
MODEL="${MODEL:-/root/models/Qwen3-TTS-12Hz-0.6B-Base}"
ROOT="${SEED_TTS_ROOT:-/root/datasets/seed-tts-eval}"
INPUT_TEXT="${INPUT_TEXT:-}"   # 留空则用数据集该行的 target_text
SEED="${SEED:-0}"

META="$ROOT/$LOCALE/meta.lst"
[[ -f "$META" ]] || { echo "!! meta.lst 不存在: $META"; exit 1; }

# 取第 ROW_IDX 条有效行 (跳过空行/注释)
LINE=$(grep -v '^#' "$META" | sed '/^[[:space:]]*$/d' | sed -n "$((ROW_IDX+1))p")
[[ -n "$LINE" ]] || { echo "!! 第 $ROW_IDX 行为空 (meta 行数不够)"; exit 1; }

IFS='|' read -r UTT REF_TEXT WAV_REL TARGET <<< "$LINE"
WAV="$ROOT/$LOCALE/$WAV_REL"
[[ -f "$WAV" ]] || { echo "!! 参考音频不存在: $WAV"; exit 1; }

[[ -n "$INPUT_TEXT" ]] && TARGET="$INPUT_TEXT"
LANG_NAME=$([[ "$LOCALE" == "en" ]] && echo English || echo Chinese)

echo "==> utt=$UTT"
echo "==> ref_wav=$WAV"
echo "==> ref_text=$REF_TEXT"
echo "==> input(target)=$TARGET"

# 用 python 组 JSON (安全转义 + inline base64 参考音)，再交给 curl
python3 - "$MODEL" "$TARGET" "$REF_TEXT" "$LANG_NAME" "$WAV" "$SEED" > /tmp/_tts_req.json <<'PY'
import base64, json, sys
model, target, ref_text, lang, wav, seed = sys.argv[1:7]
with open(wav, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("ascii")
print(json.dumps({
    "model": model,
    "input": target,
    "task_type": "Base",
    "bot_task": "think",
    "language": lang,
    "ref_audio": f"data:audio/wav;base64,{b64}",
    "ref_text": ref_text,
    "seed": int(seed),
}))
PY

echo "==> POST http://$HOST:$PORT/v1/audio/speech -> $OUT"
curl -s "http://$HOST:$PORT/v1/audio/speech" \
    -H "Content-Type: application/json" \
    -d @/tmp/_tts_req.json \
    --output "$OUT"

# 校验输出
MAGIC=$(head -c 4 "$OUT" 2>/dev/null || true)
if [[ "$MAGIC" == "RIFF" ]]; then
    echo "✅ OK: $(ls -lh "$OUT" | awk '{print $5}')  $OUT"
    command -v soxi >/dev/null 2>&1 && soxi "$OUT" || true
else
    echo "!! 返回的不是 wav，内容如下 (多半是 JSON 报错):"
    head -c 400 "$OUT"; echo
    exit 1
fi
