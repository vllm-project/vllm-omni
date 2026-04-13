#!/usr/bin/env bash
# =============================================================================
# Qwen3-TTS end2end functional test script
# Iterates over query-type and mode combinations for qwen3_tts/end2end.py
#
# Usage:
#   ./run_tests.sh --custom-voice-model /path/a --voice-design-model /path/b --base-model /path/c
#   ./run_tests.sh --custom-voice-model /path/a                # only run CustomVoice tests
#   ./run_tests.sh --base-model /path/c                        # only run Base tests
# =============================================================================

set -euo pipefail

export VLLM_OMNI_USE_V2_RUNNER=1
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# ---- Parse named arguments ----
CUSTOM_VOICE_MODEL=""
VOICE_DESIGN_MODEL=""
BASE_MODEL=""

usage() {
    echo "Usage: $0 [--custom-voice-model PATH] [--voice-design-model PATH] [--base-model PATH]"
    echo ""
    echo "Pass one or more model paths. Only the query types with a model provided will be tested."
    exit 1
}

if [[ $# -eq 0 ]]; then
    usage
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --custom-voice-model) CUSTOM_VOICE_MODEL="$2"; shift 2 ;;
        --voice-design-model) VOICE_DESIGN_MODEL="$2"; shift 2 ;;
        --base-model)         BASE_MODEL="$2";         shift 2 ;;
        -h|--help)            usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [[ -z "$CUSTOM_VOICE_MODEL" && -z "$VOICE_DESIGN_MODEL" && -z "$BASE_MODEL" ]]; then
    echo "Error: at least one model path must be provided."
    usage
fi

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
        echo "  --- last 30 lines of log ---" >> "$SUMMARY_FILE"
        tail -n 30 "$log_file" >> "$SUMMARY_FILE"
        echo "  --- end ---" >> "$SUMMARY_FILE"
        FAIL=$((FAIL + 1))
    fi
    echo ""
}

# =============================================================================
# Test cases — only run sections whose model path was provided
# =============================================================================

# --- 1. CustomVoice ---
if [[ -n "$CUSTOM_VOICE_MODEL" ]]; then
    run_test "custom_voice" \
        --model "$CUSTOM_VOICE_MODEL" --query-type CustomVoice

    run_test "custom_voice_batch_sample" \
        --model "$CUSTOM_VOICE_MODEL" --query-type CustomVoice --use-batch-sample

    run_test "custom_voice_py_generator" \
        --model "$CUSTOM_VOICE_MODEL" --query-type CustomVoice --py-generator

    run_test "custom_voice_streaming" \
        --model "$CUSTOM_VOICE_MODEL" --query-type CustomVoice --streaming

    run_test "custom_voice_batch4" \
        --model "$CUSTOM_VOICE_MODEL" --query-type CustomVoice --batch-size 4

    run_test "custom_voice_batch4_batch_sample" \
        --model "$CUSTOM_VOICE_MODEL" --query-type CustomVoice --batch-size 4 --use-batch-sample --stage-configs-path "$SCRIPT_DIR/../../../vllm_omni/model_executor/stage_configs/qwen3_tts_batch.yaml"

    run_test "custom_voice_num_prompts3" \
        --model "$CUSTOM_VOICE_MODEL" --query-type CustomVoice --num-prompts 3
fi

# --- 2. VoiceDesign ---
if [[ -n "$VOICE_DESIGN_MODEL" ]]; then
    run_test "voice_design" \
        --model "$VOICE_DESIGN_MODEL" --query-type VoiceDesign

    run_test "voice_design_batch_sample" \
        --model "$VOICE_DESIGN_MODEL" --query-type VoiceDesign --use-batch-sample

    run_test "voice_design_py_generator" \
        --model "$VOICE_DESIGN_MODEL" --query-type VoiceDesign --py-generator

    run_test "voice_design_streaming" \
        --model "$VOICE_DESIGN_MODEL" --query-type VoiceDesign --streaming
fi

# --- 3. Base (icl mode) ---
if [[ -n "$BASE_MODEL" ]]; then
    run_test "base_icl" \
        --model "$BASE_MODEL" --query-type Base --mode-tag icl

    run_test "base_icl_batch_sample" \
        --model "$BASE_MODEL" --query-type Base --mode-tag icl --use-batch-sample

    run_test "base_icl_py_generator" \
        --model "$BASE_MODEL" --query-type Base --mode-tag icl --py-generator

    run_test "base_icl_streaming" \
        --model "$BASE_MODEL" --query-type Base --mode-tag icl --streaming

    # --- 4. Base (xvec_only mode) ---
    run_test "base_xvec_only" \
        --model "$BASE_MODEL" --query-type Base --mode-tag xvec_only

    run_test "base_xvec_only_batch_sample" \
        --model "$BASE_MODEL" --query-type Base --mode-tag xvec_only --use-batch-sample
fi

# =============================================================================
# Summary
# =============================================================================
echo "========================================" | tee -a "$SUMMARY_FILE"
echo "Tests done: total ${TOTAL}, passed ${PASS}, failed ${FAIL}" | tee -a "$SUMMARY_FILE"
echo "========================================" | tee -a "$SUMMARY_FILE"
echo ""
echo "Detailed logs: $LOG_DIR"
echo "Summary report: $SUMMARY_FILE"

exit $FAIL
