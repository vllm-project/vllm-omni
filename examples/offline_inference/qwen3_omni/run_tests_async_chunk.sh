#!/usr/bin/env bash
# =============================================================================
# Qwen3-Omni end2end_async_chunk 功能测试脚本
# 使用默认内置素材，遍历 AsyncOmni 支持的 query-type × output-modalities 组合
# =============================================================================

set -euo pipefail

export VLLM_OMNI_USE_V2_RUNNER=1

MODEL_PATH="/workspace/fattysand/models/Qwen3-Omni-30B-A3B-Instruct"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
END2END="$SCRIPT_DIR/end2end_async_chunk.py"
STAGE_CONFIGS="$SCRIPT_DIR/../../../vllm_omni/model_executor/stage_configs/qwen3_omni_moe_async_chunk.yaml"
LOG_DIR="$SCRIPT_DIR/test_logs_async_chunk"
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
    --model "$MODEL_PATH" --stage-configs-path "$STAGE_CONFIGS" \
    --query-type text --modalities text

run_test "text_to_text_audio" \
    --model "$MODEL_PATH" --stage-configs-path "$STAGE_CONFIGS" \
    --query-type text --modalities text,audio

run_test "text_to_audio" \
    --model "$MODEL_PATH" --stage-configs-path "$STAGE_CONFIGS" \
    --query-type text --modalities audio

# --- 2. 图片输入 ---
run_test "image_to_text" \
    --model "$MODEL_PATH" --stage-configs-path "$STAGE_CONFIGS" \
    --query-type use_image --modalities text

run_test "image_to_text_audio" \
    --model "$MODEL_PATH" --stage-configs-path "$STAGE_CONFIGS" \
    --query-type use_image --modalities text,audio

# --- 3. 音频输入 ---
run_test "audio_to_text" \
    --model "$MODEL_PATH" --stage-configs-path "$STAGE_CONFIGS" \
    --query-type use_audio --modalities text

run_test "audio_to_text_audio" \
    --model "$MODEL_PATH" --stage-configs-path "$STAGE_CONFIGS" \
    --query-type use_audio --modalities text,audio

# --- 4. 视频输入 ---
run_test "video_to_text" \
    --model "$MODEL_PATH" --stage-configs-path "$STAGE_CONFIGS" \
    --query-type use_video --modalities text

run_test "video_to_text_audio" \
    --model "$MODEL_PATH" --stage-configs-path "$STAGE_CONFIGS" \
    --query-type use_video --modalities text,audio

# --- 5. 多 prompt 批量推理 ---
run_test "text_to_text_batch3" \
    --model "$MODEL_PATH" --stage-configs-path "$STAGE_CONFIGS" \
    --query-type text --modalities text --num-prompts 3

# --- 6. 并发请求 (max-in-flight) ---
run_test "text_to_text_audio_concurrent2" \
    --model "$MODEL_PATH" --stage-configs-path "$STAGE_CONFIGS" \
    --query-type text --modalities text,audio --num-prompts 2 --max-in-flight 2

# --- 7. stream-audio-to-disk 模式 ---
run_test "audio_to_text_audio_stream" \
    --model "$MODEL_PATH" --stage-configs-path "$STAGE_CONFIGS" \
    --query-type use_audio --modalities text,audio --stream-audio-to-disk

# --- 8. request-timeout 测试 (给足够长的超时，应该 PASS) ---
run_test "text_to_text_with_timeout" \
    --model "$MODEL_PATH" --stage-configs-path "$STAGE_CONFIGS" \
    --query-type text --modalities text --request-timeout-s 600

# --- 9. batch-timeout 测试 (给足够长的超时，应该 PASS) ---
run_test "text_to_text_audio_batch_timeout" \
    --model "$MODEL_PATH" --stage-configs-path "$STAGE_CONFIGS" \
    --query-type text --modalities text,audio --batch-timeout-s 600

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
