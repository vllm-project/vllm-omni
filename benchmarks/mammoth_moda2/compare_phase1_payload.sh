#!/usr/bin/env bash
# Compare MammothModa2's pre-Phase-1 payload path with the request-end path.
# Run from the optimized checkout on a two-GPU host.
#
# PROFILE_BACKEND=torch (default) writes child-worker PyTorch traces and reports
# stage-0 aten::to call counts. PROFILE_BACKEND=nsys performs a separate Nsight
# Systems run. Never enable both in one process tree: both subscribe to CUPTI.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
PYTHON_BIN="${PYTHON_BIN:-/data/vllm-workspace/.venv/bin/python}"
MODEL="${MODEL:-/data/vllm-workspace/models/MammothModa2-Preview}"
PHASE1_COMMIT="${PHASE1_COMMIT:-$(git -C "$REPO_ROOT" log --format=%H --fixed-strings --grep='Optimize MammothModa2 request-end AR to DiT payload' -1)}"
if [[ -z "$PHASE1_COMMIT" ]]; then
    echo "Set PHASE1_COMMIT or BASE_COMMIT: the Phase 1 implementation commit was not found." >&2
    exit 1
fi
BASE_COMMIT="${BASE_COMMIT:-${PHASE1_COMMIT}^}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results/mammoth_moda2_phase1_$(date +%Y%m%d_%H%M%S)}"
PROMPT="${PROMPT:-A small red cabin beside a quiet mountain lake at sunrise}"
PROFILE_BACKEND="${PROFILE_BACKEND:-torch}"
PAYLOAD_STATS="${VLLM_OMNI_MAMMOTH_MODA2_PAYLOAD_STATS:-0}"
REQUIRE_IDLE_GPUS="${REQUIRE_IDLE_GPUS:-1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

if [[ "$PROFILE_BACKEND" != "torch" && "$PROFILE_BACKEND" != "nsys" ]]; then
    echo "PROFILE_BACKEND must be 'torch' or 'nsys', got: $PROFILE_BACKEND" >&2
    exit 1
fi

require_idle_gpus() {
    if [[ "$REQUIRE_IDLE_GPUS" != "1" ]]; then
        return
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi is required for the idle-GPU safety check." >&2
        exit 1
    fi

    local active_processes
    active_processes="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
        --format=csv,noheader 2>/dev/null | grep -v 'No running processes found' || true)"
    if [[ -n "$active_processes" ]]; then
        echo "Refusing to start the benchmark because compute processes already own GPU memory:" >&2
        echo "$active_processes" >&2
        echo "Inspect and stop only the verified stale process, then rerun." >&2
        echo "Set REQUIRE_IDLE_GPUS=0 only when intentionally sharing the GPUs." >&2
        exit 1
    fi
}

require_idle_gpus

mkdir -p "$RESULTS_DIR"
BASE_WORKTREE="$(mktemp -d /tmp/vllm-omni-mammoth-baseline.XXXXXX)"
rmdir "$BASE_WORKTREE"

cleanup() {
    git -C "$REPO_ROOT" worktree remove --force "$BASE_WORKTREE" 2>/dev/null || true
}
trap cleanup EXIT

git -C "$REPO_ROOT" diff --check
git -C "$REPO_ROOT" rev-parse --verify "$BASE_COMMIT^{commit}" >/dev/null
git -C "$REPO_ROOT" worktree add --detach "$BASE_WORKTREE" "$BASE_COMMIT"

write_deploy_config() {
    local deploy_path="$1"
    local profiler_backend="$2"
    local label="$3"

    cat > "$deploy_path" <<'YAML'
async_chunk: false
pipeline: mammoth_moda2

stages:
  - stage_id: 0
    devices: "0"
    max_num_seqs: 1
    max_model_len: 2048
    max_num_batched_tokens: 2048
    gpu_memory_utilization: 0.85
    enforce_eager: true
    trust_remote_code: true
YAML

    if [[ "$profiler_backend" == "torch" ]]; then
        cat >> "$deploy_path" <<YAML
    profiler_config:
      profiler: torch
      torch_profiler_dir: $RESULTS_DIR/torch_${label}_stage0
      torch_profiler_record_shapes: false
      torch_profiler_with_memory: false
      torch_profiler_with_stack: false
YAML
    elif [[ "$profiler_backend" == "nsys" ]]; then
        cat >> "$deploy_path" <<'YAML'
    profiler_config:
      profiler: cuda
YAML
    fi

    cat >> "$deploy_path" <<'YAML'
    enable_prefix_caching: false

  - stage_id: 1
    devices: "1"
    max_num_seqs: 1
    gpu_memory_utilization: 0.3
    enforce_eager: true
    trust_remote_code: true
YAML

    if [[ "$profiler_backend" == "torch" ]]; then
        cat >> "$deploy_path" <<YAML
    profiler_config:
      profiler: torch
      torch_profiler_dir: $RESULTS_DIR/torch_${label}_stage1
      torch_profiler_record_shapes: false
      torch_profiler_with_memory: false
      torch_profiler_with_stack: false
YAML
    elif [[ "$profiler_backend" == "nsys" ]]; then
        cat >> "$deploy_path" <<'YAML'
    profiler_config:
      profiler: cuda
YAML
    fi

    cat >> "$deploy_path" <<'YAML'
    enable_prefix_caching: false
    default_sampling_params:
      extra_args:
        text_guidance_scale: 4.0
        cfg_range: [0.0, 1.0]
        num_inference_steps: 20
YAML
}

write_deploy_config "$RESULTS_DIR/deploy.yaml" "none" ""

TORCH_PROFILER_JSON_TEMPLATE='{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_record_shapes":false,"torch_profiler_with_memory":false,"torch_profiler_with_stack":false}'

run_case() {
    local label="$1"
    local checkout="$2"
    local mode="$3"
    local output="$RESULTS_DIR/${label}_${mode}.png"
    local log="$RESULTS_DIR/${label}_${mode}.log"
    local deploy_config="$RESULTS_DIR/deploy.yaml"
    if [[ "$mode" == "profile" ]]; then
        deploy_config="$RESULTS_DIR/deploy_${label}_${mode}.yaml"
        write_deploy_config "$deploy_config" "$PROFILE_BACKEND" "$label"
    fi
    local -a command=(
        env "CUDA_VISIBLE_DEVICES=0,1" "PYTHONPATH=$checkout" \
        "VLLM_OMNI_MAMMOTH_MODA2_PAYLOAD_STATS=$PAYLOAD_STATS" "$PYTHON_BIN"
        "$checkout/examples/offline_inference/text_to_image/text_to_image.py"
        --model "$MODEL"
        --deploy-config "$deploy_config"
        --prompt "$PROMPT"
        --width 512 --height 512
        --num-inference-steps 20
        --guidance-scale 4.0
        --seed 42
        --output "$output"
    )

    if [[ "$mode" == "profile" && "$PROFILE_BACKEND" == "torch" ]]; then
        local trace_dir="$RESULTS_DIR/torch_${label}"
        local profiler_json
        printf -v profiler_json "$TORCH_PROFILER_JSON_TEMPLATE" "$trace_dir"
        command+=(--profiler-config "$profiler_json")
    elif [[ "$mode" == "profile" && "$PROFILE_BACKEND" == "nsys" ]]; then
        # Existing vLLM CUDA-profiler support opens the worker capture range.
        command+=(--profiler-config '{"profiler":"cuda"}')
    fi

    echo "=== $label / $mode ===" | tee "$log"
    if [[ "$mode" == "profile" && "$PROFILE_BACKEND" == "nsys" ]]; then
        local -a nsys_command=(
            "$NSYS_BIN" profile --force-overwrite true --trace=cuda,nvtx,osrt
            --cuda-graph-trace=node --capture-range=cudaProfilerApi
            --capture-range-end=repeat --sample=none
            --output "$RESULTS_DIR/nsys_${label}"
        )
        if "$NSYS_BIN" profile --help 2>&1 | grep -q -- '--trace-fork-before-exec'; then
            nsys_command+=(--trace-fork-before-exec=true)
        fi
        env VLLM_OMNI_MAMMOTH_MODA2_NVTX=1 "${nsys_command[@]}" "${command[@]}" 2>&1 | tee -a "$log"
    else
        "${command[@]}" 2>&1 | tee -a "$log"
    fi

    test -s "$output"
}

if [[ "$PROFILE_BACKEND" == "nsys" ]] && command -v nsys >/dev/null 2>&1; then
    NSYS_BIN="$(command -v nsys)"
    "$NSYS_BIN" --version | tee "$RESULTS_DIR/nsys_version.txt"
    NSYS_ANALYZER="$REPO_ROOT/benchmarks/mammoth_moda2/analyze_nsys_transfer.py"
    if [[ ! -f "$NSYS_ANALYZER" ]]; then
        echo "Nsight analyzer is missing: $NSYS_ANALYZER" >&2
        echo "Restore benchmarks/mammoth_moda2/analyze_nsys_transfer.py before starting a costly run." >&2
        exit 1
    fi
else
    NSYS_BIN=""
    NSYS_ANALYZER=""
    if [[ "$PROFILE_BACKEND" == "nsys" ]]; then
        echo "PROFILE_BACKEND=nsys requires an nsys executable on PATH." >&2
        exit 1
    fi
    echo "PROFILE_BACKEND=torch; Nsight is intentionally disabled." | tee "$RESULTS_DIR/nsys_version.txt"
fi

git -C "$REPO_ROOT" rev-parse HEAD > "$RESULTS_DIR/optimized_commit.txt"
git -C "$BASE_WORKTREE" rev-parse HEAD > "$RESULTS_DIR/baseline_commit.txt"

# Warmups populate Triton caches. They are excluded from comparison artifacts.
run_case baseline "$BASE_WORKTREE" warmup
run_case optimized "$REPO_ROOT" warmup

run_case baseline "$BASE_WORKTREE" profile
run_case optimized "$REPO_ROOT" profile

"$PYTHON_BIN" - "$RESULTS_DIR/baseline_profile.png" "$RESULTS_DIR/optimized_profile.png" <<'PY'
from PIL import Image, ImageChops, ImageStat
import sys

baseline = Image.open(sys.argv[1]).convert("RGB")
optimized = Image.open(sys.argv[2]).convert("RGB")
if baseline.size != optimized.size:
    raise SystemExit(f"image size mismatch: {baseline.size} != {optimized.size}")
diff = ImageChops.difference(baseline, optimized)
stats = ImageStat.Stat(diff)
print(f"image_size={baseline.size}")
print(f"pixel_max_abs_diff={max(max(channel) for channel in diff.getextrema())}")
print(f"pixel_mean_abs_diff={sum(stats.mean) / len(stats.mean):.6f}")
PY

for label in baseline optimized; do
    grep -E 'Total generation time|stage_0_gen_ms|stage_1_gen_ms|hidden_(d2h|snapshot)|mammoth_moda2 payload stats' \
        "$RESULTS_DIR/${label}_profile.log" > "$RESULTS_DIR/${label}_summary.txt" || true
    if [[ "$PROFILE_BACKEND" == "nsys" && -f "$RESULTS_DIR/nsys_${label}.nsys-rep" ]]; then
        "$NSYS_BIN" export --type sqlite --force-overwrite true \
            --output "$RESULTS_DIR/nsys_${label}" \
            "$RESULTS_DIR/nsys_${label}.nsys-rep" \
            > "$RESULTS_DIR/${label}_nsys_export.txt" 2>&1
        for report in cuda_api_sum cuda_gpu_mem_time_sum cuda_gpu_mem_size_sum; do
            "$NSYS_BIN" stats --force-export true --report "$report" \
                "$RESULTS_DIR/nsys_${label}.nsys-rep" \
                > "$RESULTS_DIR/${label}_${report}.txt" 2>&1 || true
        done
    fi
done

if [[ "$PROFILE_BACKEND" == "torch" ]]; then
"$PYTHON_BIN" - "$RESULTS_DIR" <<'PY'
from pathlib import Path
import re
import sys

results_dir = Path(sys.argv[1])
launch_counts = {}
for label in ("baseline", "optimized"):
    traces = sorted((results_dir / f"torch_{label}").rglob("*.json"))
    traces.extend(sorted((results_dir / f"torch_{label}").rglob("*.json.gz")))
    stage0_tables = sorted((results_dir / f"torch_{label}").glob("*_stage0_rank0_*/profiler_out_0.txt"))
    to_calls = None
    if len(stage0_tables) == 1:
        for line in stage0_tables[0].read_text(errors="replace").splitlines():
            if line.strip().startswith("aten::to"):
                match = re.search(r"(\d+)\s*$", line)
                if match:
                    to_calls = int(match.group(1))
                break
    summary = results_dir / f"{label}_torch_trace_markers.txt"
    lines = [f"trace_files={len(traces)}"]
    lines.append(f"stage0_aten_to_calls={to_calls}")
    summary.write_text("\n".join(lines) + "\n")
    print(f"[{label}] " + ", ".join(lines))
PY
else
"$PYTHON_BIN" "$NSYS_ANALYZER" "$RESULTS_DIR"
fi

find "$RESULTS_DIR" -maxdepth 3 -type f -printf '%p\n' | sort
echo "Results directory: $RESULTS_DIR"
