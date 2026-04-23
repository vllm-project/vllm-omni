#!/bin/bash
# Offline OmniWeaving I2V launcher.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
METRICS_DOC="${REPO_ROOT}/examples/offline_inference/omniweaving/OMNIWEAVING_I2V_MI2V_PERF.md"
METRICS_LOG_DIR="${REPO_ROOT}/examples/offline_inference/omniweaving/perf_logs"
. "${REPO_ROOT}/examples/offline_inference/omniweaving/gpu_preflight.sh"

MODEL="${MODEL:-/root/autodl-tmp/HY-OmniWeaving}"
QWEN_PATH="${QWEN_PATH:-/root/autodl-tmp/hf_cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}"
SIGLIP_PATH="${SIGLIP_PATH:-/root/autodl-tmp/hf_cache/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3}"
T2V_PATH="${T2V_PATH:-/root/autodl-tmp/hf_cache/models--hunyuanvideo-community--HunyuanVideo-1.5-Diffusers-480p_t2v/snapshots/286be7ce72277246578a3e3cc2487e95ddae5bcf}"

IMAGE_PATH="${IMAGE_PATH:-/root/.cursor/projects/root-autodl-tmp/assets/c__Users_Z250911-3_AppData_Roaming_Cursor_User_workspaceStorage_6d70d8bc908833ca4a807cbaeeda4c91_images_naiwa_480p-085c9bb1-01b4-40a4-b3e2-b0c81ce1b928.png}"
PROMPT="${PROMPT:-The cartoon character in the picture suddenly bursts into laughter, clutching their stomach with exaggerated movements.}"

RESOLUTION="${RESOLUTION:-480p}"
PRECISION_LABEL="${PRECISION_LABEL:-BF16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
NUM_FRAMES="${NUM_FRAMES:-33}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-6.0}"
SEED="${SEED:-42}"
FLOW_SHIFT="${FLOW_SHIFT:-}"

case "${RESOLUTION}" in
    480p)
        RESOLUTION_DIMS="480x832"
        DEFAULT_FLOW_SHIFT="5.0"
        TEST_LABEL="I2V 480p"
        DEFAULT_OUTPUT="omniweaving_i2v_480p.mp4"
        ;;
    720p)
        RESOLUTION_DIMS="720x1280"
        DEFAULT_FLOW_SHIFT="7.0"
        TEST_LABEL="I2V 720p"
        DEFAULT_OUTPUT="omniweaving_i2v_720p.mp4"
        ;;
    *)
        echo "Unsupported RESOLUTION: ${RESOLUTION}. Expected 480p or 720p."
        exit 1
        ;;
esac

FLOW_SHIFT_LABEL="${FLOW_SHIFT_LABEL:-${FLOW_SHIFT:-${DEFAULT_FLOW_SHIFT}}}"
OUTPUT="${OUTPUT:-${DEFAULT_OUTPUT}}"

for path_var in MODEL QWEN_PATH SIGLIP_PATH T2V_PATH IMAGE_PATH; do
    if [ ! -e "${!path_var}" ]; then
        echo "Required path does not exist: ${path_var}=${!path_var}"
        exit 1
    fi
done

# A single OmniWeaving TP=2 job can peak near the full memory of both GPUs.
# Fail early with a clear message instead of crashing later during warmup.
ensure_omniweaving_gpus_idle "${TENSOR_PARALLEL_SIZE}"

echo "Running OmniWeaving I2V 480p..."
echo "Model: ${MODEL}"
echo "Image: ${IMAGE_PATH}"
echo "Resolution: ${RESOLUTION} (${RESOLUTION_DIMS})"
echo "Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "Output: ${OUTPUT}"

cd "${REPO_ROOT}"
mkdir -p "${METRICS_LOG_DIR}"

RUN_TIMESTAMP="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
LOG_STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
LOG_PATH="${METRICS_LOG_DIR}/i2v_${RESOLUTION}_${LOG_STAMP}.log"

CMD=(
    python examples/offline_inference/image_to_video/image_to_video_omniweaving.py
    --model "${MODEL}"
    --qwen-path "${QWEN_PATH}"
    --siglip-path "${SIGLIP_PATH}"
    --t2v-path "${T2V_PATH}"
    --image "${IMAGE_PATH}"
    --prompt "${PROMPT}"
    --resolution "${RESOLUTION}"
    --guidance-scale "${GUIDANCE_SCALE}"
    --num-frames "${NUM_FRAMES}"
    --num-inference-steps "${NUM_INFERENCE_STEPS}"
    --seed "${SEED}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --output "${OUTPUT}"
)

if [ -n "${FLOW_SHIFT}" ]; then
    CMD+=(--flow-shift "${FLOW_SHIFT}")
fi

set +e
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
VLLM_NO_DEEP_GEMM="${VLLM_NO_DEEP_GEMM:-1}" \
TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}" \
TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}" \
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}" \
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
"${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
CMD_STATUS=${PIPESTATUS[0]}
set -e

if [ "${CMD_STATUS}" -ne 0 ]; then
    echo "Run failed; metrics doc was not updated."
    echo "Log saved to ${LOG_PATH}"
    exit "${CMD_STATUS}"
fi

export METRICS_DOC LOG_PATH RUN_TIMESTAMP TEST_LABEL RESOLUTION_DIMS NUM_FRAMES NUM_INFERENCE_STEPS
export PRECISION_LABEL FLOW_SHIFT_LABEL OUTPUT REPO_ROOT
python - <<'PY'
from __future__ import annotations

import os
import re
from pathlib import Path


def extract_float(cell: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", cell)
    return float(match.group(1)) if match else None


def parse_table(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    header = lines[:2]
    rows = []
    for line in lines[2:]:
        if not line.startswith("|"):
            continue
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return header, rows


def format_table(header: list[str], rows: list[list[str]]) -> list[str]:
    return header + [f"| {' | '.join(row)} |" for row in rows]


def replace_section(text: str, title: str, new_lines: list[str], next_title: str | None) -> str:
    start = text.index(title)
    end = text.index(next_title, start) if next_title else len(text)
    prefix = text[:start]
    suffix = text[end:]
    body = title + "\n\n" + "\n".join(new_lines).rstrip() + "\n\n"
    return prefix + body + suffix.lstrip("\n")


doc_path = Path(os.environ["METRICS_DOC"])
log_path = Path(os.environ["LOG_PATH"])
text = doc_path.read_text(encoding="utf-8")
log_text = log_path.read_text(encoding="utf-8")

latency_match = re.search(r"Latency:\s*([0-9]+(?:\.[0-9]+)?)s", log_text)
peak_match = re.search(r"Peak VRAM:\s*([0-9]+(?:\.[0-9]+)?)\s*GiB", log_text)
if latency_match is None:
    raise SystemExit(f"Could not parse latency from log: {log_path}")

latency_value = float(latency_match.group(1))
peak_value = float(peak_match.group(1)) if peak_match else None

latency_start = text.index("## Latency")
peak_start = text.index("## Peak VRAM")
history_start = text.index("## Run History")

latency_header, latency_rows = parse_table(
    [line for line in text[latency_start:peak_start].splitlines() if line.startswith("|")]
)
peak_header, peak_rows = parse_table(
    [line for line in text[peak_start:history_start].splitlines() if line.startswith("|")]
)
history_header, history_rows = parse_table(
    [line for line in text[history_start:].splitlines() if line.startswith("|")]
)

test_label = os.environ["TEST_LABEL"]
resolution_dims = os.environ["RESOLUTION_DIMS"]
frames = os.environ["NUM_FRAMES"]
steps = os.environ["NUM_INFERENCE_STEPS"]
precision = os.environ["PRECISION_LABEL"]
flow_shift = os.environ["FLOW_SHIFT_LABEL"]
output = os.path.relpath(os.environ["OUTPUT"], os.environ["REPO_ROOT"])
log_rel = os.path.relpath(str(log_path), os.environ["REPO_ROOT"])

diffusers_latency = None
for row in latency_rows:
    if row[0] == test_label and row[1] == "diffusers":
        diffusers_latency = extract_float(row[7])
        break

speedup_cell = "**TBDx**"
if diffusers_latency is not None and latency_value > 0:
    speedup_cell = f"**{diffusers_latency / latency_value:.2f}x**"

for idx, row in enumerate(latency_rows):
    if row[0] == test_label and row[1] == "**vLLM-OMNI**":
        latency_rows[idx] = [
            test_label,
            "**vLLM-OMNI**",
            resolution_dims,
            frames,
            steps,
            precision,
            flow_shift,
            f"**{latency_value:.2f}**",
            speedup_cell,
        ]
        break
else:
    raise SystemExit(f"Latency row not found for {test_label}")

for idx, row in enumerate(peak_rows):
    if row[0] == test_label:
        peak_rows[idx] = [test_label, row[1], f"{peak_value:.2f}" if peak_value is not None else "TBD"]
        break
else:
    raise SystemExit(f"Peak VRAM row not found for {test_label}")

history_row = [
    os.environ["RUN_TIMESTAMP"],
    test_label,
    resolution_dims,
    frames,
    steps,
    precision,
    flow_shift,
    f"{latency_value:.2f}",
    f"{peak_value:.2f}" if peak_value is not None else "TBD",
    f"`{output}`",
    f"`{log_rel}`",
]

if history_row not in history_rows:
    history_rows.insert(0, history_row)

text = replace_section(text, "## Latency", format_table(latency_header, latency_rows), "## Peak VRAM")
text = replace_section(text, "## Peak VRAM", format_table(peak_header, peak_rows), "## Run History")
text = replace_section(text, "## Run History", format_table(history_header, history_rows), None)
doc_path.write_text(text, encoding="utf-8")
PY

echo "Updated metrics doc: ${METRICS_DOC}"
echo "Saved run log: ${LOG_PATH}"
