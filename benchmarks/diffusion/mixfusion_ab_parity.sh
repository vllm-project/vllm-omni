#!/usr/bin/env bash
# Full latent-parity matrix for the A/B endorsement run ("verify output
# counts, dimensions and numerical parity against the baseline with an
# explicit tolerance").
#
# Each case: 1 iteration, no warmup, --output-type latent, one .pt per request
# (dumped in request order). The harness runs both modes in that iteration, so
# each dump dir carries independent-NNN.pt and mixfusion_batch-NNN.pt; the
# batch-vs-batch comparison (the fair-comparison mode) uses mixfusion_batch-*.
#
# Control runs (A run twice) measure the noise floor of unchanged code -- the
# empirical basis for the "explicit tolerance". W1 and W3 are the two extremes
# (single request / mixed-resolution batch); W2's floor sits between them.
#
# Seeds, prompts and per-request seed offsets are identical across all cases,
# so files pair up by name.
#
# Resumable: a case with a `done` marker is skipped.
#
# Usage:
#   A_WT=... B_WT=... bash mixfusion_ab_parity.sh
#   GPU=0 A_WT=... B_WT=... PARITY=$PWD/parity_v3 bash mixfusion_ab_parity.sh
set -u

BENCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PARITY=${PARITY:-$PWD/parity_v3}
GPU=${GPU:-0}
STEPS=${STEPS:-50}
SEED=${SEED:-42}
MODEL=${MODEL:-Qwen/Qwen-Image}
GUIDANCE=${GUIDANCE:-4.0}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}

: "${A_WT:?set A_WT to the baseline (A-side) checkout path}"
: "${B_WT:?set B_WT to the candidate (B-side) checkout path}"

W1_SIZES="1024x1024"
W2_SIZES="1024x1024,1024x1024,1024x1024,1024x1024,1024x1024,1024x1024,1024x1024,1024x1024"
W3_SIZES="1024x1024,1024x1024,1024x768,1024x768,768x1024,768x1024,512x512,512x512"
mkdir -p "$PARITY"

dump() {
  local worktree=$1 name=$2 sizes=$3 mix=$4
  local dir=$PARITY/$name
  if [[ -f $dir/done ]]; then echo "[skip] parity $name"; return 0; fi
  mkdir -p "$dir"
  echo "[run ] parity $name  ($sizes $mix)"
  # Same driver code on both sides; VLLM_OMNI_ROOT selects the checkout.
  VLLM_OMNI_ROOT=$worktree PYTHONPATH=$worktree \
    CUDA_VISIBLE_DEVICES=$GPU DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN \
    python "$BENCH_DIR/mixfusion_benefit.py" \
      --family qwen --model "$MODEL" \
      --image-sizes "$sizes" \
      --steps "$STEPS" --iters 1 --warmup 0 \
      --seed "$SEED" --guidance-scale "$GUIDANCE" --max-num-seqs "$MAX_NUM_SEQS" \
      --dtype bfloat16 --enforce-eager --output-type latent \
      --dump-outputs-dir "$dir" \
      $mix \
      > "$PARITY/$name.log" 2>&1 \
    && touch "$dir/done" || echo "[FAIL] parity $name -- see $PARITY/$name.log"
}

# ---- controls: A vs A noise floor (same code, same seed, two runs) ----
dump $A_WT A-W1-ctrl1 "$W1_SIZES" --no-enable-mixfusion
dump $A_WT A-W1-ctrl2 "$W1_SIZES" --no-enable-mixfusion
dump $A_WT A-W3-ctrl1 "$W3_SIZES" --no-enable-mixfusion
dump $A_WT A-W3-ctrl2 "$W3_SIZES" --no-enable-mixfusion

# ---- A side ----
dump $A_WT A-W1 "$W1_SIZES" --no-enable-mixfusion
dump $A_WT A-W2 "$W2_SIZES" --no-enable-mixfusion
dump $A_WT A-W3 "$W3_SIZES" --no-enable-mixfusion

# ---- B side, mixfusion off (candidate code, flag off) ----
dump $B_WT B-W1-off "$W1_SIZES" --no-enable-mixfusion
dump $B_WT B-W2-off "$W2_SIZES" --no-enable-mixfusion
dump $B_WT B-W3-off "$W3_SIZES" --no-enable-mixfusion

# ---- B side, mixfusion on (packed flat varlen) ----
dump $B_WT B-W1-on "$W1_SIZES" --enable-mixfusion
dump $B_WT B-W2-on "$W2_SIZES" --enable-mixfusion
dump $B_WT B-W3-on "$W3_SIZES" --enable-mixfusion

echo
echo "parity dumps finished. Aggregate with:"
echo "  python $BENCH_DIR/mixfusion_ab_summarize.py --results ... --parity $PARITY ..."
