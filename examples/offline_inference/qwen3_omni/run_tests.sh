#!/usr/bin/env bash
# =============================================================================
# Qwen3-Omni end2end 功能测试脚本
# 使用默认内置素材，遍历各种 query-type × output-modalities 组合
# =============================================================================

set -euo pipefail

export VLLM_OMNI_USE_V2_RUNNER=1

MODEL_PATH="/workspace/fattysand/models/Qwen3-Omni-30B-A3B-Instruct"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
END2END="$SCRIPT_DIR/end2end.py"
LOG_DIR="$SCRIPT_DIR/test_logs"
SUMMARY_FILE="$LOG_DIR/summary.log"

rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"
> "$SUMMARY_FILE"

PASS=0
FAIL=0
TOTAL=0

run_test() {
    local name="$1"
    shift
    local log_file="$LOG_DIR/${name}.log"
    local output_dir="$LOG_DIR/${name}_output"

    TOTAL=$((TOTAL + 1))
    echo "========================================"
    echo "[${TOTAL}] Running: ${name}"
    echo "  Command: python $END2END $*"
    echo "========================================"

    local start_time
    start_time=$(date +%s)

    if python "$END2END" "$@" --output-dir "$output_dir" > "$log_file" 2>&1; then
        local end_time
        end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        echo "  => PASS (${elapsed}s)"
        echo "[PASS] ${name} (${elapsed}s)" >> "$SUMMARY_FILE"
        PASS=$((PASS + 1))
    else
        local exit_code=$?
        local end_time
        end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        echo "  => FAIL (exit code: ${exit_code}, ${elapsed}s)"
        echo "[FAIL] ${name} (exit code: ${exit_code}, ${elapsed}s)" >> "$SUMMARY_FILE"
        # 记录最后 30 行错误信息到 summary
        echo "  --- last 30 lines of log ---" >> "$SUMMARY_FILE"
        tail -n 30 "$log_file" >> "$SUMMARY_FILE"
        echo "  --- end ---" >> "$SUMMARY_FILE"
        FAIL=$((FAIL + 1))
    fi
    echo ""
}

# =============================================================================
# 测试用例
# =============================================================================

# --- 1. 纯文本输入 ---
run_test "text_to_text" \
    --model "$MODEL_PATH" --query-type text --modalities text

run_test "text_to_text_audio" \
    --model "$MODEL_PATH" --query-type text --modalities text,audio

run_test "text_to_audio" \
    --model "$MODEL_PATH" --query-type text --modalities audio

# --- 2. 图片输入 ---
run_test "image_to_text" \
    --model "$MODEL_PATH" --query-type use_image --modalities text

run_test "image_to_text_audio" \
    --model "$MODEL_PATH" --query-type use_image --modalities text,audio

# --- 3. 音频输入 ---
run_test "audio_to_text" \
    --model "$MODEL_PATH" --query-type use_audio --modalities text

run_test "audio_to_text_audio" \
    --model "$MODEL_PATH" --query-type use_audio --modalities text,audio

# --- 4. 视频输入 ---
run_test "video_to_text" \
    --model "$MODEL_PATH" --query-type use_video --modalities text

run_test "video_to_text_audio" \
    --model "$MODEL_PATH" --query-type use_video --modalities text,audio

# --- 5. 多音频输入 ---
run_test "multi_audios_to_text" \
    --model "$MODEL_PATH" --query-type use_multi_audios --modalities text

run_test "multi_audios_to_text_audio" \
    --model "$MODEL_PATH" --query-type use_multi_audios --modalities text,audio

# --- 6. 混合模态输入 (音频+图片+视频) ---
run_test "mixed_to_text" \
    --model "$MODEL_PATH" --query-type use_mixed_modalities --modalities text

run_test "mixed_to_text_audio" \
    --model "$MODEL_PATH" --query-type use_mixed_modalities --modalities text,audio

# --- 7. 视频中使用音频 ---
run_test "audio_in_video_to_text" \
    --model "$MODEL_PATH" --query-type use_audio_in_video --modalities text

run_test "audio_in_video_to_text_audio" \
    --model "$MODEL_PATH" --query-type use_audio_in_video --modalities text,audio

# --- 8. py-generator 模式 ---
run_test "text_to_text_generator" \
    --model "$MODEL_PATH" --query-type text --modalities text --py-generator

run_test "audio_to_text_audio_generator" \
    --model "$MODEL_PATH" --query-type use_audio --modalities text,audio --py-generator

# --- 9. 多 prompt 批量推理 ---
run_test "text_to_text_batch3" \
    --model "$MODEL_PATH" --query-type text --modalities text --num-prompts 3

# =============================================================================
# 汇总
# =============================================================================
echo "========================================" | tee -a "$SUMMARY_FILE"
echo "测试完成: 共 ${TOTAL} 项, 通过 ${PASS} 项, 失败 ${FAIL} 项" | tee -a "$SUMMARY_FILE"
echo "========================================" | tee -a "$SUMMARY_FILE"
echo ""
echo "详细日志目录: $LOG_DIR"
echo "汇总报告: $SUMMARY_FILE"

exit $FAIL
