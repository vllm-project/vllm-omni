#!/usr/bin/env bash
# A/B timing benchmark driver for Qwen-Image MixFusion (real pipeline).
#
# Runs mixfusion_benefit.py against two repo checkouts:
#   A_WT: baseline checkout (e.g. main)
#   B_WT: candidate checkout (e.g. the PR head, based on the same commit)
# Both sides run the same driver code; only the imported tree differs
# (VLLM_OMNI_ROOT + PYTHONPATH, see the sys.path guard in mixfusion_benefit.py),
# so the comparison isolates the candidate's changes.
#
# Workloads:
#   W1: single request, 1024x1024
#   W2: 8 concurrent requests, all 1024x1024
#   W3: 8 concurrent requests, mixed 1024x1024 x2 / 1024x768 x2 / 768x1024 x2 / 512x512 x2
#
# Matrix: A W1-W3; B W1-W3 x {mixfusion off, on}. The A side and B-off runs
# form the no-regression controls; B-on carries the packed flat-varlen path.
#
# All cases default to --enforce-eager (EAGER=1): if one side auto-compiles
# graphs while the other does not, eager keeps the comparison clean. Each case
# is a fresh engine in its own process; results are resumable (an existing
# result.json skips the case unless FORCE=1).
#
# Record the measured SHAs when running:
#   git -C "$A_WT" rev-parse HEAD   /   git -C "$B_WT" rev-parse HEAD
#
# Usage:
#   A_WT=... B_WT=... bash mixfusion_ab_bench.sh
#   PILOT=1 A_WT=... B_WT=... bash mixfusion_ab_bench.sh   # steps=8 smoke run
#   ONLY=B-W3-on A_WT=... B_WT=... bash mixfusion_ab_bench.sh   # single case
#   python mixfusion_ab_summarize.py ...                    # aggregate afterwards
set -u

BENCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RESULTS=${RESULTS:-$PWD/results}
mkdir -p "$RESULTS"

: "${A_WT:?set A_WT to the baseline (A-side) checkout path}"
: "${B_WT:?set B_WT to the candidate (B-side) checkout path}"

GPU=${GPU:-0}
STEPS=${STEPS:-50}
ITERS=${ITERS:-5}
WARMUP=${WARMUP:-1}
SEED=${SEED:-42}
GUIDANCE=${GUIDANCE:-4.0}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
MODEL=${MODEL:-Qwen/Qwen-Image}
EAGER=${EAGER:-1}
ONLY=${ONLY:-}
FORCE=${FORCE:-0}

if [[ ${PILOT:-0} = 1 ]]; then
  STEPS=8; ITERS=2; WARMUP=1
fi
EAGER_FLAG=--enforce-eager
if [[ $EAGER != 1 ]]; then
  EAGER_FLAG=--no-enforce-eager
fi

W1_SIZES="1024x1024"
W2_SIZES="1024x1024,1024x1024,1024x1024,1024x1024,1024x1024,1024x1024,1024x1024,1024x1024"
W3_SIZES="1024x1024,1024x1024,1024x768,1024x768,768x1024,768x1024,512x512,512x512"

run_case() {
  local worktree=$1 name=$2 sizes=$3 mix=$4
  if [[ -n "$ONLY" && "$name" != *"$ONLY"* ]]; then return 0; fi
  local dir=$RESULTS/$name
  local out_json=$dir/result.json
  if [[ $FORCE != 1 && -f $out_json ]]; then
    echo "[skip] $name (result exists)"
    return 0
  fi
  mkdir -p "$dir"
  echo "[run ] $name  (steps=$STEPS iters=$ITERS warmup=$WARMUP)"
  # VLLM_OMNI_ROOT + PYTHONPATH make the engine import from the intended
  # checkout; the sys.path guard in mixfusion_benefit.py keeps any
  # sitecustomize-injected vllm-omni paths from winning.
  VLLM_OMNI_ROOT=$worktree PYTHONPATH=$worktree \
    CUDA_VISIBLE_DEVICES=$GPU DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN \
    python "$BENCH_DIR/mixfusion_benefit.py" \
      --family qwen --model "$MODEL" \
      --image-sizes "$sizes" \
      --steps "$STEPS" --iters "$ITERS" --warmup "$WARMUP" \
      --seed "$SEED" --guidance-scale "$GUIDANCE" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --dtype bfloat16 \
      $EAGER_FLAG \
      $mix \
      --json-output "$out_json" \
      > "$RESULTS/$name.log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[FAIL] $name rc=$rc -- see $RESULTS/$name.log"
  else
    echo "[done] $name"
  fi
  return 0
}

# ---- A side: baseline (no mixfusion code at all) ----
run_case $A_WT A-W1     "$W1_SIZES" --no-enable-mixfusion
run_case $A_WT A-W2     "$W2_SIZES" --no-enable-mixfusion
run_case $A_WT A-W3     "$W3_SIZES" --no-enable-mixfusion

# ---- B side, mixfusion off (batching-only control) ----
run_case $B_WT B-W1-off "$W1_SIZES" --no-enable-mixfusion
run_case $B_WT B-W2-off "$W2_SIZES" --no-enable-mixfusion
run_case $B_WT B-W3-off "$W3_SIZES" --no-enable-mixfusion

# ---- B side, mixfusion on (packed flat varlen) ----
run_case $B_WT B-W1-on  "$W1_SIZES" --enable-mixfusion
run_case $B_WT B-W2-on  "$W2_SIZES" --enable-mixfusion
run_case $B_WT B-W3-on  "$W3_SIZES" --enable-mixfusion

echo "All cases dispatched. Aggregate with: python $BENCH_DIR/mixfusion_ab_summarize.py --results $RESULTS ..."
