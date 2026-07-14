#!/usr/bin/env bash
# Usage:
#   bash bench_pd.sh                                  # 默认: 1p1d, c=8, 50 prompts
#   bash bench_pd.sh 1p1d                             # 指定 topology tag (用于结果目录命名)
#   bash bench_pd.sh 1p1d 8                           # topology, concurrency
#   bash bench_pd.sh 1p1d 8 50                        # topology, concurrency, num_prompts
#   bash bench_pd.sh 1p1d 8 50 zh                     # +locale (zh/en)
#   bash bench_pd.sh 1p1d 8 2 zh smoke                # smoke 模式：不算 warmup 不存盘
#
# Special:
#   bash bench_pd.sh 1p1d sweep                       # 自动扫 c=1,8,16,32,64,128，每档 50 prompts
#
# 每档压测跑完后，会自动生成一条“内容自描述”的验证语音，用来听感校验正确性，
# 例如 3p1d + c=8 会合成朗读 “这是3个编码1个解码并行数为8的测试语音”，
# 存到对应结果目录下的 check_audio_c<并发>.wav。设 CHECK_AUDIO=0 可关闭。
#
# Env override:
#   MODEL=/root/models/Qwen3-TTS-12Hz-0.6B-Base
#   HOST=127.0.0.1
#   PORT=8091
#   DATASET=/root/datasets/seed-tts-eval
#   RESULT_ROOT=<this_dir>/results
#   CHECK_AUDIO=1                                     # 1=压测后生成校验语音(默认), 0=关闭

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- 默认参数 ----------
TAG="${1:-1p1d}"
ARG2="${2:-8}"
NUM_PROMPTS="${3:-50}"
LOCALE="${4:-zh}"
MODE="${5:-}"

MODEL="${MODEL:-/root/models/Qwen3-TTS-12Hz-0.6B-Base}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8091}"
DATASET="${DATASET:-/root/datasets/seed-tts-eval}"
RESULT_ROOT="${RESULT_ROOT:-$SCRIPT_DIR/results}"
CHECK_AUDIO="${CHECK_AUDIO:-1}"

# 从 topology tag 里解析 “编码(prefill)/解码(decode)” 个数，用于生成自描述文本。
# 支持形如 1p3d / 3p1d / 1p1d / ar_3p1d / custom_xxx_1p3d 等；解析不到则回退 1/1。
parse_pd_counts() {
    local tag="$1"
    if [[ "$tag" =~ ([0-9]+)p([0-9]+)d ]]; then
        P_CNT="${BASH_REMATCH[1]}"
        D_CNT="${BASH_REMATCH[2]}"
    else
        P_CNT=1
        D_CNT=1
    fi
}

# 生成一条“内容自描述”的验证语音：朗读 “这是X个编码Y个解码并行数为Z的测试语音”。
# 便于人工听感校验：一听内容就知道对应哪套拓扑 + 并发。
gen_check_audio() {
    local concurrency="$1"
    local result_dir="$2"
    [[ "$CHECK_AUDIO" == "1" ]] || return 0

    parse_pd_counts "$TAG"
    local text="这是${P_CNT}个编码${D_CNT}个解码并行数为${concurrency}的测试语音"
    local out_wav="$result_dir/check_audio_c${concurrency}.wav"

    echo "==> [check] 生成校验语音: \"$text\" -> $out_wav"
    if HOST="$HOST" PORT="$PORT" MODEL="$MODEL" SEED_TTS_ROOT="$DATASET" \
        INPUT_TEXT="$text" \
        bash "$SCRIPT_DIR/gen_audio.sh" "$LOCALE" 0 "$out_wav"; then
        echo "==> [check] ✅ 校验语音已保存: $out_wav"
    else
        echo "==> [check] !! 校验语音生成失败 (不影响压测结果)"
    fi
}

# ---------- 清代理 ----------
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="localhost,127.0.0.1,::1"

# ---------- 单次 bench 的封装 ----------
run_one() {
    local concurrency="$1"
    local n_prompts="$2"
    local mode="$3"   # "" or "smoke"
    local result_dir="$RESULT_ROOT/${TAG}_${LOCALE}_c${concurrency}"
    mkdir -p "$result_dir"

    local warmups=2
    local extra_save=(--save-result --result-dir "$result_dir")
    if [[ "$mode" == "smoke" ]]; then
        warmups=0
        extra_save=()   # smoke 不存盘
    fi

    echo "==> bench: tag=$TAG locale=$LOCALE concurrency=$concurrency num_prompts=$n_prompts warmups=$warmups mode=${mode:-normal}"
    echo "==> result_dir=$result_dir"

    vllm bench serve --omni \
        --host "$HOST" --port "$PORT" \
        --model "$MODEL" \
        --backend openai-audio-speech \
        --endpoint /v1/audio/speech \
        --dataset-name seed-tts \
        --dataset-path "$DATASET" \
        --seed-tts-locale "$LOCALE" \
        --num-prompts "$n_prompts" --num-warmups "$warmups" \
        --extra-body '{"task_type":"Base"}' \
        --max-concurrency "$concurrency" --request-rate inf \
        --percentile-metrics ttft,e2el,audio_rtf,audio_ttfp,audio_duration,audio_underrun \
        --trust-remote-code \
        "${extra_save[@]}"

    # 压测后做语音正确性校验（smoke 模式跳过）
    if [[ "$mode" != "smoke" ]]; then
        gen_check_audio "$concurrency" "$result_dir"
    fi
}

# ---------- 健康检查 ----------
check_server_alive() {
    # /health
    if ! curl -sS -m 3 --noproxy '*' "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
        return 1
    fi
    # /v1/models（更全面，能检测 stage 是否真的就绪）
    if ! curl -sS -m 5 --noproxy '*' "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
        return 1
    fi
    # 检查 pid 文件对应进程是否存活
    for pid_file in "$RESULT_ROOT"/server_*.pid /tmp/server_*.pid; do
        if [[ -f "$pid_file" ]]; then
            local pid
            pid=$(cat "$pid_file" 2>/dev/null)
            if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
                return 0
            fi
        fi
    done
    # 没 pid 文件但 /health 通也算 OK（可能用户手动启动）
    return 0
}

if ! check_server_alive; then
    echo "!! server at ${HOST}:${PORT} not ready or already crashed. Start it first:"
    echo "   bash $SCRIPT_DIR/start_pd.sh $TAG"
    echo "!! 看 server 日志定位原因:"
    echo "   tail -100 $RESULT_ROOT/server_${TAG}.log"
    exit 1
fi

# ---------- 分发：sweep / 单次 ----------
if [[ "$ARG2" == "sweep" ]]; then
    echo "==> SWEEP mode: c in {1, 8, 16, 32, 64, 128}, num_prompts=$NUM_PROMPTS each"
    for c in 1 8 16 32 64 128; do
        run_one "$c" "$NUM_PROMPTS" ""
        echo
    done
    echo "==> ✅ sweep done. Results under: $RESULT_ROOT/${TAG}_${LOCALE}_c*/"
    ls -d "$RESULT_ROOT"/${TAG}_${LOCALE}_c*/ 2>/dev/null || true
else
    run_one "$ARG2" "$NUM_PROMPTS" "$MODE"
    echo
    echo "==> ✅ done"
fi