#!/usr/bin/env bash
# ==============================================================================
# 一键跑全部拓扑压测: single -> 1p1d -> 3p1d -> 1p3d -> 1p6d
#
# 对每一套拓扑:
#   1) 用 start_pd.sh 起服务, 等 /health OK
#   2) 依次按并发档 (CONCURRENCIES) 压测, 每档:
#        - 完整 vllm bench 输出存到 <topo>/raw_c<并发>.log
#        - 关键指标块抽取汇总到 <topo>/summary.txt (标出并发)
#        - JSON 原始结果 (--save-result) 存到 <topo>/
#        - 生成一条"内容自描述"校验语音 <topo>/check_audio_c<并发>.wav
#   3) 用 stop_pd.sh 干净关服务, 释放显存/端口
# 最后把所有拓扑的关键指标汇总成一张总表 SUMMARY_ALL.txt
#
# 用法:
#   bash run_all_bench.sh                 # 默认全跑
#   TOPOS="1p1d 3p1d" bash run_all_bench.sh          # 只跑指定拓扑
#   CONCURRENCIES="1 8 32" bash run_all_bench.sh      # 自定义并发档
#   NUM_PROMPTS=30 LOCALE=en bash run_all_bench.sh    # 每档条数 / 语种
#
# Env override (与 start_pd.sh / bench_pd.sh 一致):
#   MODEL   (默认 /root/models/Qwen3-TTS-12Hz-0.6B-Base)
#   HOST    (默认 127.0.0.1)
#   PORT    (默认 8091)
#   DATASET (默认 /root/datasets/seed-tts-eval)
#   TOPOS         (默认 "single 1p1d 3p1d 1p3d 1p6d")
#   CONCURRENCIES (默认 "1 8 16 32")
#   NUM_PROMPTS   (默认 50)
#   LOCALE        (默认 zh)
#   CHECK_AUDIO   (默认 1; 0=不生成校验语音)
#
# 结果: benchmark/results/all_bench_<时间戳>/
#         SUMMARY_ALL.txt          <- 拓扑横向对比总表 (先看这个)
#         single/  1p1d/  3p1d/  1p3d/  1p6d/
#             summary.txt          <- 该拓扑所有并发档的指标 (标出并发)
#             raw_c<并发>.log       <- 完整 bench 输出
#             check_audio_c<并发>.wav
#             *.json               <- vllm --save-result 原始数据
# ==============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- 参数 ----------
MODEL="${MODEL:-/root/models/Qwen3-TTS-12Hz-0.6B-Base}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8091}"
DATASET="${DATASET:-/root/datasets/seed-tts-eval}"
TOPOS="${TOPOS:-single 1p1d 3p1d 1p3d 1p6d}"
CONCURRENCIES="${CONCURRENCIES:-1 8 16 32}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
LOCALE="${LOCALE:-zh}"
CHECK_AUDIO="${CHECK_AUDIO:-1}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/results/all_bench_$TS}"
SUMMARY_ALL="$OUT_ROOT/SUMMARY_ALL.txt"
mkdir -p "$OUT_ROOT"

# ---------- 清代理 ----------
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="localhost,127.0.0.1,::1"

# 把 topology tag 解析成 "编码P / 解码D" 个数, 用于自描述校验语音文本
parse_pd_counts() {
    local tag="$1"
    if [[ "$tag" =~ ([0-9]+)p([0-9]+)d ]]; then
        P_CNT="${BASH_REMATCH[1]}"; D_CNT="${BASH_REMATCH[2]}"
    else
        P_CNT=1; D_CNT=1
    fi
}

# 生成一条内容自描述的校验语音, 便于人工听感核对
gen_check_audio() {
    local topo="$1" concurrency="$2" out_dir="$3"
    [[ "$CHECK_AUDIO" == "1" ]] || return 0
    local text
    if [[ "$topo" == "single" ]]; then
        text="这是单机基线并行数为${concurrency}的测试语音"
    else
        parse_pd_counts "$topo"
        text="这是${P_CNT}个编码${D_CNT}个解码并行数为${concurrency}的测试语音"
    fi
    local out_wav="$out_dir/check_audio_c${concurrency}.wav"
    echo "    [check] 生成校验语音: \"$text\" -> $out_wav"
    HOST="$HOST" PORT="$PORT" MODEL="$MODEL" SEED_TTS_ROOT="$DATASET" \
        INPUT_TEXT="$text" \
        bash "$SCRIPT_DIR/gen_audio.sh" "$LOCALE" 0 "$out_wav" \
        >"$out_dir/check_audio_c${concurrency}.log" 2>&1 \
        && echo "    [check] OK: $out_wav" \
        || echo "    [check] !! 校验语音生成失败 (见 check_audio_c${concurrency}.log, 不影响压测数据)"
}

# 从一次 bench 的 raw log 里抽 "Serving Benchmark Result" 结果块, 追加到 summary
extract_metrics() {
    local raw_log="$1" summary="$2" topo="$3" concurrency="$4"
    {
        echo "################################################################"
        echo "# 拓扑: $topo   并发(max-concurrency): $concurrency   条数: $NUM_PROMPTS   语种: $LOCALE"
        echo "################################################################"
        # 抽取从结果标题行到文件末尾的指标块; 抽不到就退回关键行 grep
        if grep -q "Serving Benchmark Result" "$raw_log"; then
            awk '/Serving Benchmark Result/{f=1} f' "$raw_log"
        else
            grep -iE "Successful requests|Benchmark duration|throughput|TTFT|TPOT|ITL|E2EL|RTF|TTFP|underrun|duration" "$raw_log" \
                || echo "(未抓到指标, 见 raw log, 可能该档 bench 失败)"
        fi
        echo
    } >> "$summary"
}

# 从 summary 里为总表抽一行 (吞吐 + 关键延迟), 尽力而为
one_line_for_all() {
    local raw_log="$1" topo="$2" concurrency="$3"
    local reqput mean_ttfp mean_e2el rtf
    # 用结果块里的精确列名, 并用 tail -1 取"最终结果块"(在日志末尾),
    # 避开压测中途刷出的实时进度行(那些行里 rtf/throughput 常是 0.00)。
    # 注意: TTS 后端没有 "Mean TTFT", 首包指标叫 "Mean AUDIO_TTFP (ms)";
    #       RTF 列名是 "Mean AUDIO_RTF"。
    reqput=$(grep -E "Request throughput" "$raw_log" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    mean_ttfp=$(grep -E "Mean AUDIO_TTFP" "$raw_log" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    mean_e2el=$(grep -E "Mean E2EL" "$raw_log" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    rtf=$(grep -E "Mean AUDIO_RTF" "$raw_log" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    printf "%-8s %8s %14s %14s %16s %12s\n" \
        "$topo" "$concurrency" "${reqput:-NA}" "${mean_ttfp:-NA}" "${mean_e2el:-NA}" "${rtf:-NA}" \
        >> "$SUMMARY_ALL"
}

# ---------- 总表表头 ----------
{
    echo "=============================================================================="
    echo " Qwen3-TTS 全拓扑压测总表   time=$TS"
    echo " model=$MODEL"
    echo " dataset=$DATASET  locale=$LOCALE  num_prompts/档=$NUM_PROMPTS"
    echo " 拓扑=[$TOPOS]  并发档=[$CONCURRENCIES]"
    echo "=============================================================================="
    printf "%-8s %8s %14s %14s %16s %12s\n" \
        "TOPO" "CONC" "ReqThru(r/s)" "MeanTTFP(ms)" "MeanE2EL(ms)" "MeanRTF"
    echo "------------------------------------------------------------------------------"
} > "$SUMMARY_ALL"

echo "=============================================================================="
echo "==> 输出目录: $OUT_ROOT"
echo "==> 拓扑: $TOPOS | 并发: $CONCURRENCIES | 每档条数: $NUM_PROMPTS | 语种: $LOCALE"
echo "=============================================================================="

# ---------- 主循环 ----------
for topo in $TOPOS; do
    topo_dir="$OUT_ROOT/$topo"
    summary="$topo_dir/summary.txt"
    mkdir -p "$topo_dir"
    : > "$summary"

    echo
    echo "########################## [$topo] 起服务 ##########################"
    if ! MODEL="$MODEL" HOST="$HOST" PORT="$PORT" bash "$SCRIPT_DIR/start_pd.sh" "$topo"; then
        echo "!! [$topo] 服务启动失败, 跳过该拓扑 (看 results/server_*.log)"
        {
            echo "################################################################"
            echo "# 拓扑: $topo  ==> 启动失败, 跳过"
            echo "################################################################"
            echo
        } >> "$summary"
        printf "%-8s %8s %14s %14s %16s %12s\n" "$topo" "-" "START_FAIL" "-" "-" "-" >> "$SUMMARY_ALL"
        bash "$SCRIPT_DIR/stop_pd.sh" "$PORT" >/dev/null 2>&1
        continue
    fi

    for c in $CONCURRENCIES; do
        raw_log="$topo_dir/raw_c${c}.log"
        echo
        echo "-------- [$topo] bench 并发=$c 条数=$NUM_PROMPTS --------"
        set +e
        vllm bench serve --omni \
            --host "$HOST" --port "$PORT" \
            --model "$MODEL" \
            --backend openai-audio-speech \
            --endpoint /v1/audio/speech \
            --dataset-name seed-tts \
            --dataset-path "$DATASET" \
            --seed-tts-locale "$LOCALE" \
            --num-prompts "$NUM_PROMPTS" --num-warmups 2 \
            --extra-body '{"task_type":"Base"}' \
            --max-concurrency "$c" --request-rate inf \
            --percentile-metrics ttft,e2el,audio_rtf,audio_ttfp,audio_duration,audio_underrun \
            --trust-remote-code \
            --save-result --result-dir "$topo_dir" \
            2>&1 | tee "$raw_log"
        rc=${PIPESTATUS[0]}
        set -e
        if [[ $rc -ne 0 ]]; then
            echo "!! [$topo] c=$c bench 退出码 $rc (数据可能不完整)"
        fi

        extract_metrics "$raw_log" "$summary" "$topo" "$c"
        one_line_for_all "$raw_log" "$topo" "$c"
        gen_check_audio "$topo" "$c" "$topo_dir"
    done

    echo
    echo "########################## [$topo] 关服务 ##########################"
    bash "$SCRIPT_DIR/stop_pd.sh" "$PORT"

    echo "==> [$topo] 完成. 指标见: $summary"
done

echo "------------------------------------------------------------------------------" >> "$SUMMARY_ALL"

echo
echo "=============================================================================="
echo "==> 全部完成!"
echo "==> 总表 (先看这个): $SUMMARY_ALL"
echo "==> 各拓扑明细: $OUT_ROOT/<topo>/summary.txt"
echo "==> 校验语音:   $OUT_ROOT/<topo>/check_audio_c<并发>.wav"
echo "=============================================================================="
echo
cat "$SUMMARY_ALL"
