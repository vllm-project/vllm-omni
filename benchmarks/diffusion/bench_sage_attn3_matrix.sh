#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# 2x2 timing matrix (backend x compile mode) for SAGE_ATTN_3 vs TORCH_SDPA,
# on HunyuanVideo-1.5 480p T2V and Wan 2.2 TI2V-5B. Run from the repo root:
#
#     bash benchmarks/diffusion/bench_sage_attn3_matrix.sh
#
# Requires: sageattn3 installed (for SAGE_ATTN_3 rows) and a Blackwell GPU.

set -u

STEPS=${STEPS:-50}
SEED=${SEED:-42}
PROMPT=${PROMPT:-"A dog running across a field of golden wheat."}
OUT_DIR=${OUT_DIR:-/tmp/bench_sage3}
mkdir -p "$OUT_DIR"

declare -A TOTAL PER_STEP TEXT_ENC TRANSFORMER VAE

run() {
    local name="$1"; shift
    local model="$1"; shift
    local h="$1"; shift
    local w="$1"; shift
    local f="$1"; shift
    local g="$1"; shift
    local log="$OUT_DIR/${name}.log"
    echo "[$(date +%T)] Running $name ..."
    env "$@" python examples/offline_inference/text_to_video/text_to_video.py \
        --model "$model" \
        --prompt "$PROMPT" \
        --height "$h" --width "$w" --num-frames "$f" \
        --num-inference-steps "$STEPS" --seed "$SEED" --guidance-scale "$g" \
        --enable-diffusion-pipeline-profiler \
        --output "$OUT_DIR/${name}.mp4" > "$log" 2>&1 || {
            echo "  FAILED — see $log"
            TOTAL[$name]="FAIL"; PER_STEP[$name]="FAIL"
            TEXT_ENC[$name]="-"; TRANSFORMER[$name]="-"; VAE[$name]="-"
            return
        }

    local wait_ms
    wait_ms=$(grep "add_req_and_wait=" "$log" | tail -1 | sed -nE 's/.*add_req_and_wait=([0-9.]+) ms.*/\1/p')
    local total_s
    total_s=$(grep -oE "Total generation time: [0-9.]+ seconds" "$log" | tail -1 | sed -nE 's/.*: ([0-9.]+) seconds.*/\1/p')
    local text_enc_s
    text_enc_s=$(grep -oE "text_encoder.forward took [0-9.]+s" "$log" | awk '{sub(/s$/,"",$NF); sum+=$NF} END {printf "%.3f", sum+0}')
    local vae_s
    vae_s=$(grep -oE "vae.decode took [0-9.]+s" "$log" | awk '{sub(/s$/,"",$NF); sum+=$NF} END {printf "%.3f", sum+0}')
    local pipeline_s
    pipeline_s=$(grep -oE "Pipeline\.forward took [0-9.]+s" "$log" | tail -1 | sed -nE 's/.*took ([0-9.]+)s.*/\1/p')
    local transformer_s
    transformer_s=$(awk "BEGIN {printf \"%.3f\", ${pipeline_s:-0} - ${vae_s:-0} - ${text_enc_s:-0}}")

    TOTAL[$name]="${total_s:-0}"
    PER_STEP[$name]=$(awk "BEGIN {printf \"%.3f\", ${wait_ms:-0} / 1000 / $STEPS}")
    TEXT_ENC[$name]="$text_enc_s"
    TRANSFORMER[$name]="$transformer_s"
    VAE[$name]="$vae_s"

    echo "  total=${TOTAL[$name]}s  per-step=${PER_STEP[$name]}s/it  transformer=${TRANSFORMER[$name]}s  vae=${VAE[$name]}s"
}

# ========== HunyuanVideo-1.5 480p T2V (480x832x33, CFG=6.0) ==========
HV_MODEL=hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v
run hv15_sdpa_eager     "$HV_MODEL" 480 832 33 6.0 TORCH_COMPILE_DISABLE=1
run hv15_sdpa_compiled  "$HV_MODEL" 480 832 33 6.0
run hv15_sage3_eager    "$HV_MODEL" 480 832 33 6.0 TORCH_COMPILE_DISABLE=1 DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN_3
run hv15_sage3_compiled "$HV_MODEL" 480 832 33 6.0                         DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN_3

# ========== Wan 2.2 TI2V-5B (704x1280x33, CFG=5.0) ==========
WAN_MODEL=Wan-AI/Wan2.2-TI2V-5B-Diffusers
run wan22_sdpa_eager     "$WAN_MODEL" 704 1280 33 5.0 TORCH_COMPILE_DISABLE=1
run wan22_sdpa_compiled  "$WAN_MODEL" 704 1280 33 5.0
run wan22_sage3_eager    "$WAN_MODEL" 704 1280 33 5.0 TORCH_COMPILE_DISABLE=1 DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN_3
run wan22_sage3_compiled "$WAN_MODEL" 704 1280 33 5.0                         DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN_3

print_table() {
    local title="$1"; shift
    echo
    echo "$title (${STEPS} steps, seed=${SEED})"
    echo
    printf "| %-23s | %-9s | %-15s | %-12s | %-15s | %-14s |\n" \
        "Config" "Total (s)" "Per-step (s/it)" "Text Enc (s)" "Transformer (s)" "VAE Decode (s)"
    printf "| %-23s | %-9s | %-15s | %-12s | %-15s | %-14s |\n" \
        "-----------------------" "---------" "---------------" "------------" "---------------" "--------------"
    for row in "$@"; do
        case "$row" in
            *_sdpa_eager)     label="SDPA + Eager" ;;
            *_sdpa_compiled)  label="SDPA + Compiled" ;;
            *_sage3_eager)    label="SAGE_ATTN_3 + Eager" ;;
            *_sage3_compiled) label="SAGE_ATTN_3 + Compiled" ;;
        esac
        printf "| %-23s | %-9s | %-15s | %-12s | %-15s | %-14s |\n" \
            "$label" "${TOTAL[$row]}" "${PER_STEP[$row]}" "${TEXT_ENC[$row]}" "${TRANSFORMER[$row]}" "${VAE[$row]}"
    done
}

print_table "HunyuanVideo-1.5 480p T2V, 832x480x33" \
    hv15_sdpa_eager hv15_sdpa_compiled hv15_sage3_eager hv15_sage3_compiled

print_table "Wan 2.2 TI2V-5B, 1280x704x33" \
    wan22_sdpa_eager wan22_sdpa_compiled wan22_sage3_eager wan22_sage3_compiled

echo
echo "Logs: $OUT_DIR/*.log"
