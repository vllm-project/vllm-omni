#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# π0.5 engine-overhead measurement matrix for one dtype on one GPU.
#
#   run_engine_overhead_matrix.sh <bf16|fp32> <out_root>
#
# Environment:
#   MODEL        checkpoint dir or repo id (default lerobot/pi05_base)
#   BEFORE_TREE  source tree of the reference revision; when set, the "before"
#                variant imports vllm_omni from it via PYTHONPATH
#   VARIANTS     default "before after" (needs BEFORE_TREE) -- or just one of them
#   RUNS=3 ROUNDS=3 REQUESTS=30 WARMUP=3 ITERS=30 PORT=8100
#   EAGER=0      1 = pass --enforce-eager everywhere (eager baseline path)
#
# Per variant and run: one clean-process standalone run (model-only, pipeline),
# one `uni` and one `mp` engine run (3 warmups + 30 timed requests per view
# count, traced). Then one fresh `vllm serve` per variant for the OpenPI E2E
# layer (3 client rounds x 3 warmups + 30 requests per view count), traced and untraced.
# Variant order alternates between runs so drift does not favour one side.
set -u
DTYPE=$1; ROOT=$2
MODEL=${MODEL:-lerobot/pi05_base}; read -r -a VARIANTS <<< "${VARIANTS:-before after}"
REVERSED=(); for variant in "${VARIANTS[@]}"; do REVERSED=("$variant" "${REVERSED[@]}"); done
RUNS=${RUNS:-3}; ROUNDS=${ROUNDS:-3}; REQUESTS=${REQUESTS:-30}; WARMUP=${WARMUP:-3}; ITERS=${ITERS:-30}; PORT=${PORT:-8100}
EAGER_FLAG=(); [ "${EAGER:-0}" = 1 ] && EAGER_FLAG=(--enforce-eager)
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
# Backend settings of the measurement.
export ATTENTION_BACKEND=torch DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA VLLM_DISABLE_COMPILE_CACHE=1
cd "$REPO" || exit 1
if [ "$DTYPE" = bf16 ]; then
  CFG=$(mktemp /tmp/pi05-bf16-XXXXXX.yaml); sed 's/^dtype: float32$/dtype: bfloat16/' vllm_omni/deploy/pi05.yaml > "$CFG"
else
  CFG=vllm_omni/deploy/pi05.yaml
fi

tree_env() { # variant -> TREE_ENV: the PYTHONPATH assignment of the before variant, else nothing
  TREE_ENV=()
  if [ "$1" = before ]; then
    [ -n "${BEFORE_TREE:-}" ] || { echo "BEFORE_TREE is required for the before variant" >&2; exit 2; }
    TREE_ENV=("PYTHONPATH=$BEFORE_TREE${PYTHONPATH:+:$PYTHONPATH}")
  fi
}

bench() { # variant label args...
  local variant=$1 label=$2; shift 2
  local out=$ROOT/$variant/$DTYPE; mkdir -p "$out"
  tree_env "$variant"
  env "${TREE_ENV[@]}" python "$HERE/bench_engine_overhead_pi05.py" --model "$MODEL" --deploy-config "$CFG" \
    "${EAGER_FLAG[@]}" --warmup "$WARMUP" --iters "$ITERS" --out "$out/$label.json" --save-actions "$out/$label.npz" "$@" \
    > "$out/$label.log" 2>&1
  echo "$(date +%T) $variant/$DTYPE $label rc=$? $(grep -cE 'Traceback' "$out/$label.log") traceback(s)"
}

e2e() { # variant traced(0|1)
  local variant=$1 traced=$2 name=e2e; [ "$traced" = 1 ] || name=e2e_notrace
  local out=$ROOT/$variant/$DTYPE/$name; rm -rf "$out"; mkdir -p "$out"
  local tdir; tdir=$(mktemp -d /tmp/pi05-trace-XXXXXX)
  local trace_env=(); [ "$traced" = 1 ] && trace_env=("VLLM_OMNI_LATENCY_TRACE_DIR=$tdir")
  tree_env "$variant"
  env "${TREE_ENV[@]}" "${trace_env[@]}" python -c "import vllm_omni, os; print(os.path.dirname(vllm_omni.__file__))" > "$out/source_tree.txt" 2>/dev/null
  env "${TREE_ENV[@]}" "${trace_env[@]}" vllm serve "$MODEL" --omni --host 127.0.0.1 --port "$PORT" \
    --deploy-config "$CFG" "${EAGER_FLAG[@]}" --disable-log-stats > "$out/server.log" 2>&1 &
  local spid=$!
  for _ in $(seq 1 300); do
    curl -s -o /dev/null -m 2 "http://127.0.0.1:$PORT/health" && break
    kill -0 $spid 2>/dev/null || { echo "$variant/$DTYPE $name: server died"; tail -20 "$out/server.log"; return 1; }
    sleep 2
  done
  for round in $(seq 1 "$ROUNDS"); do for views in 1 2 3; do
    python "$HERE/e2e_latency_pi05.py" --port "$PORT" --views "$views" --warmup "$WARMUP" --iters "$REQUESTS" \
      --request-seed 42 --no-memory-sampler --round "$round" --requests-out "$out/client.jsonl" --out "$out/round${round}_views$views.json" \
      --save-actions "$ROOT/$variant/$DTYPE/${name}_round${round}_views$views.npz" >> "$out/client.log" 2>&1
  done; done
  nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,clocks.max.sm,pstate \
    --format=csv > "$out/gpu_during.csv" 2>/dev/null
  sleep 1
  kill -INT $spid; for _ in $(seq 1 40); do kill -0 $spid 2>/dev/null || break; sleep 1; done
  pkill -KILL -P $spid 2>/dev/null; kill -KILL $spid 2>/dev/null
  if [ "$traced" = 1 ]; then mkdir -p "$out/trace"; cp "$tdir"/trace-*.tsv "$out/trace/" 2>/dev/null; fi
  rm -rf "$tdir"
  echo "$(date +%T) $variant/$DTYPE $name requests=$(wc -l < "$out/client.jsonl") errors=$(grep -c Error "$out/client.log")"
}

mkdir -p "$ROOT"
{ date; hostname; uname -m; nproc; lscpu | grep -E "Model name|^CPU\(s\)|Thread|MHz"; nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit --format=csv; } > "$ROOT/system_$DTYPE.txt" 2>&1

for run in $(seq 1 "$RUNS"); do
  order=("${VARIANTS[@]}")
  if [ $((run % 2)) -eq 0 ]; then order=("${REVERSED[@]}"); fi
  for variant in "${order[@]}"; do
    bench "$variant" "standalone_run$run" --mode standalone
    bench "$variant" "engine_uni_run$run" --mode engine --backend uni \
      --trace-dir "$ROOT/$variant/$DTYPE/trace_engine_uni_run$run"
    bench "$variant" "engine_mp_run$run" --mode engine --backend mp \
      --trace-dir "$ROOT/$variant/$DTYPE/trace_engine_mp_run$run"
  done
done
for variant in "${VARIANTS[@]}"; do e2e "$variant" 1; done
for variant in "${REVERSED[@]}"; do e2e "$variant" 0; done
echo "MATRIX_DONE $DTYPE"
