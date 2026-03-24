#!/bin/bash
# End-to-end test script for unified quantization framework (PR #1764).
#
# Tests FP8 quantization for:
#   1. Z-Image-Turbo  (single-stage, ~20GB VRAM)
#   2. Qwen-Image     (single-stage, ~20GB VRAM)
#   3. FLUX.1-dev     (single-stage, ~25GB VRAM with fp8)
#   4. BAGEL          (multi-stage LLM+DiT, ~55GB VRAM)
#
# Usage:
#   bash tests/e2e/offline_inference/run_quantization_e2e.sh [--skip-flux] [--skip-bagel]
#
# Expected: all runs produce images without errors.
# Key check for BAGEL: FP8 only applies to diffusion stage, NOT the LLM stage.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/test_quant_outputs"
mkdir -p "$OUTPUT_DIR"

SKIP_FLUX=false
SKIP_BAGEL=false
for arg in "$@"; do
    case "$arg" in
        --skip-flux) SKIP_FLUX=true ;;
        --skip-bagel) SKIP_BAGEL=true ;;
    esac
done

PASS=0
FAIL=0
SKIP=0

run_test() {
    local name="$1"
    shift
    echo ""
    echo "============================================================"
    echo "  TEST: $name"
    echo "============================================================"
    if "$@"; then
        echo "  PASS: $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $name (exit code $?)"
        FAIL=$((FAIL + 1))
    fi
}

# ─── 1. Z-Image-Turbo ────────────────────────────────────────────────────────
# Defaults from README: 1024x1024, 50 steps, cfg-scale 4.0, seed 42

run_test "Z-Image-Turbo BF16 (baseline)" \
    python "$REPO_ROOT/examples/offline_inference/text_to_image/text_to_image.py" \
        --model Tongyi-MAI/Z-Image-Turbo \
        --prompt "a cup of coffee on the table" \
        --seed 42 --num-inference-steps 50 \
        --height 1024 --width 1024 \
        --cfg-scale 4.0 \
        --output "$OUTPUT_DIR/zimage_bf16.png"

run_test "Z-Image-Turbo FP8" \
    python "$REPO_ROOT/examples/offline_inference/text_to_image/text_to_image.py" \
        --model Tongyi-MAI/Z-Image-Turbo \
        --prompt "a cup of coffee on the table" \
        --seed 42 --num-inference-steps 50 \
        --height 1024 --width 1024 \
        --cfg-scale 4.0 \
        --quantization fp8 \
        --output "$OUTPUT_DIR/zimage_fp8.png"

# ─── 2. Qwen-Image ───────────────────────────────────────────────────────────
# Defaults from README: 1024x1024, 50 steps, cfg-scale 4.0, seed 142

run_test "Qwen-Image BF16 (baseline)" \
    python "$REPO_ROOT/examples/offline_inference/text_to_image/text_to_image.py" \
        --model Qwen/Qwen-Image \
        --prompt "a cup of coffee on the table" \
        --seed 142 --num-inference-steps 50 \
        --height 1024 --width 1024 \
        --cfg-scale 4.0 \
        --output "$OUTPUT_DIR/qwen_bf16.png"

run_test "Qwen-Image FP8" \
    python "$REPO_ROOT/examples/offline_inference/text_to_image/text_to_image.py" \
        --model Qwen/Qwen-Image \
        --prompt "a cup of coffee on the table" \
        --seed 142 --num-inference-steps 50 \
        --height 1024 --width 1024 \
        --cfg-scale 4.0 \
        --quantization fp8 \
        --output "$OUTPUT_DIR/qwen_fp8.png"

# ─── 3. FLUX.1-dev ───────────────────────────────────────────────────────────
# Defaults from first test run: 1024x1024, 20 steps, guidance-scale 3.5

if [ "$SKIP_FLUX" = true ]; then
    echo ""
    echo "  SKIP: FLUX.1-dev (--skip-flux)"
    SKIP=$((SKIP + 1))
    SKIP=$((SKIP + 1))
else
    run_test "FLUX.1-dev BF16 (baseline)" \
        python "$REPO_ROOT/examples/offline_inference/text_to_image/text_to_image.py" \
            --model black-forest-labs/FLUX.1-dev \
            --prompt "a cup of coffee on the table" \
            --seed 42 --num-inference-steps 20 \
            --height 1024 --width 1024 \
            --guidance-scale 3.5 --cfg-scale 1.0 \
            --output "$OUTPUT_DIR/flux_bf16.png"

    run_test "FLUX.1-dev FP8" \
        python "$REPO_ROOT/examples/offline_inference/text_to_image/text_to_image.py" \
            --model black-forest-labs/FLUX.1-dev \
            --prompt "a cup of coffee on the table" \
            --seed 42 --num-inference-steps 20 \
            --height 1024 --width 1024 \
            --guidance-scale 3.5 --cfg-scale 1.0 \
            --quantization fp8 \
            --output "$OUTPUT_DIR/flux_fp8.png"
fi

# ─── 4. BAGEL (multi-stage) ──────────────────────────────────────────────────
# Defaults from README: 50 steps, cfg-text-scale 4.0, cfg-img-scale 1.5
# BAGEL end2end.py saves images as output_1_stage_None_0.png in cwd.
# We copy them to OUTPUT_DIR with proper names for LPIPS comparison.

if [ "$SKIP_BAGEL" = true ]; then
    echo ""
    echo "  SKIP: BAGEL (--skip-bagel)"
    SKIP=$((SKIP + 1))
    SKIP=$((SKIP + 1))
else
    BAGEL_WORKDIR=$(mktemp -d)

    run_test "BAGEL BF16 (baseline)" \
        bash -c "cd '$BAGEL_WORKDIR' && python '$REPO_ROOT/examples/offline_inference/bagel/end2end.py' \
            --model ByteDance-Seed/BAGEL-7B-MoT \
            --modality text2img \
            --prompts 'A cute cat' \
            --steps 50 \
            --cfg-text-scale 4.0 \
            --cfg-img-scale 1.5 \
            --seed 52 \
            && cp -f output_1_stage_None_0.png '$OUTPUT_DIR/bagel_bf16.png'"

    run_test "BAGEL FP8 (diffusion-only quantization)" \
        bash -c "cd '$BAGEL_WORKDIR' && python '$REPO_ROOT/examples/offline_inference/bagel/end2end.py' \
            --model ByteDance-Seed/BAGEL-7B-MoT \
            --modality text2img \
            --prompts 'A cute cat' \
            --steps 50 \
            --cfg-text-scale 4.0 \
            --cfg-img-scale 1.5 \
            --seed 52 \
            --quantization fp8 \
            && cp -f output_1_stage_None_0.png '$OUTPUT_DIR/bagel_fp8.png'"

    rm -rf "$BAGEL_WORKDIR"
fi

# ─── 5. LPIPS quality comparison ─────────────────────────────────────────────
# Compares each BF16/FP8 image pair using perceptual LPIPS distance.
# Requires: pip install lpips torchvision

echo ""
echo "============================================================"
echo "  LPIPS Quality Comparison"
echo "============================================================"

if python -c "import lpips" 2>/dev/null; then
    run_test "LPIPS BF16-vs-FP8 quality check" \
        python "$SCRIPT_DIR/compute_lpips.py" \
            --image-dir "$OUTPUT_DIR" \
            --threshold 0.1
else
    echo "  SKIP: lpips not installed (pip install lpips torchvision)"
    SKIP=$((SKIP + 1))
fi

# ─── Summary ─────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  SUMMARY"
echo "============================================================"
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
echo "  SKIP: $SKIP"
echo "  Output images: $OUTPUT_DIR/"
if [ -f "$OUTPUT_DIR/lpips_results.md" ]; then
    echo "  LPIPS report: $OUTPUT_DIR/lpips_results.md"
fi
echo "============================================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
