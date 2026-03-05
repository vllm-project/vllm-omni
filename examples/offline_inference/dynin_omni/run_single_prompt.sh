#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-snu-aidas/Dynin-Omni}"
STAGE_CONFIG_PATH="${STAGE_CONFIG_PATH:-${REPO_ROOT}/vllm_omni/model_executor/stage_configs/dynin_omni.yaml}"
DYNIN_CONFIG_PATH="${DYNIN_CONFIG_PATH:-${REPO_ROOT}/vllm_omni/model_executor/models/dynin_omni/models/configs/dynin_omni_demo.yaml}"

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") <task> [extra-args...]

Tasks:
  t2t  text -> text
  i2t  image -> text
  s2t  speech -> text
  t2i  text -> image
  t2s  text -> speech
  i2i  image -> image
  v2t  video -> text

Environment overrides:
  PYTHON_BIN, MODEL, STAGE_CONFIG_PATH, DYNIN_CONFIG_PATH

Task-specific env vars:
  t2t: QUESTION
  i2t: IMAGE_PATH, OUTPUT_DIR
  s2t: AUDIO_PATH, OUTPUT_DIR
  t2i: METADATA_PATH, OUTPUT_DIR
  t2s: TEXT, OUTPUT_DIR
  i2i: EDIT_JSON, ORIGIN_IMG_ROOT, OUTPUT_DIR
  v2t: VIDEO_PATH, OUTPUT_DIR

Examples:
  $(basename "$0") t2t
  $(basename "$0") i2t --i2t-max-new-tokens 64
  AUDIO_PATH=${SCRIPT_DIR}/results/t2s_from_vllm/cli-00000.wav $(basename "$0") s2t
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

TASK="$1"
shift

if [[ "$TASK" == "-h" || "$TASK" == "--help" ]]; then
  usage
  exit 0
fi

COMMON_ARGS=(
  --model "$MODEL"
  --stage-config-path "$STAGE_CONFIG_PATH"
  --dynin-config-path "$DYNIN_CONFIG_PATH"
)

case "$TASK" in
  t2t)
    QUESTION="${QUESTION:-What is 2 + 2? Answer in one short sentence.}"
    CMD=("$PYTHON_BIN" "$SCRIPT_DIR/t2t.py" "${COMMON_ARGS[@]}" --question "$QUESTION")
    ;;

  i2t)
    IMAGE_PATH="${IMAGE_PATH:-${SCRIPT_DIR}/data/image/dog.png}"
    OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results/i2t_from_vllm}"
    mkdir -p "$OUTPUT_DIR"
    CMD=("$PYTHON_BIN" "$SCRIPT_DIR/i2t.py" "${COMMON_ARGS[@]}" --image-path "$IMAGE_PATH" --output-dir "$OUTPUT_DIR")
    ;;

  s2t)
    AUDIO_PATH="${AUDIO_PATH:-${SCRIPT_DIR}/results/t2s_from_vllm/cli-00000.wav}"
    OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results/s2t_from_vllm}"
    if [[ ! -f "$AUDIO_PATH" ]]; then
      echo "[run_single_prompt] ERROR: audio file not found: $AUDIO_PATH" >&2
      echo "  Run t2s first or pass AUDIO_PATH=/path/to/file.wav" >&2
      exit 1
    fi
    mkdir -p "$OUTPUT_DIR"
    CMD=("$PYTHON_BIN" "$SCRIPT_DIR/s2t.py" "${COMMON_ARGS[@]}" --audio-path "$AUDIO_PATH" --output-dir "$OUTPUT_DIR")
    ;;

  t2i)
    METADATA_PATH="${METADATA_PATH:-${SCRIPT_DIR}/data/text/t2i_metadata.jsonl}"
    OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results/t2i_from_vllm}"
    mkdir -p "$OUTPUT_DIR"
    CMD=("$PYTHON_BIN" "$SCRIPT_DIR/t2i.py" "${COMMON_ARGS[@]}" --metadata-path "$METADATA_PATH" --output-dir "$OUTPUT_DIR")
    ;;

  t2s)
    TEXT="${TEXT:-A calm female voice says hello from Dynin Omni.}"
    OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results/t2s_from_vllm}"
    mkdir -p "$OUTPUT_DIR"
    CMD=("$PYTHON_BIN" "$SCRIPT_DIR/t2s.py" "${COMMON_ARGS[@]}" --text "$TEXT" --output-dir "$OUTPUT_DIR")
    ;;

  i2i)
    EDIT_JSON="${EDIT_JSON:-${SCRIPT_DIR}/data/text/i2i_edits.json}"
    ORIGIN_IMG_ROOT="${ORIGIN_IMG_ROOT:-${SCRIPT_DIR}/data/image}"
    OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results/i2i_from_vllm}"
    mkdir -p "$OUTPUT_DIR"
    CMD=("$PYTHON_BIN" "$SCRIPT_DIR/i2i.py" "${COMMON_ARGS[@]}" --edit-json "$EDIT_JSON" --origin-img-root "$ORIGIN_IMG_ROOT" --output-dir "$OUTPUT_DIR")
    ;;

  v2t)
    VIDEO_PATH="${VIDEO_PATH:-${SCRIPT_DIR}/data/video/baseball.mp4}"
    OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results/v2t_from_vllm}"
    mkdir -p "$OUTPUT_DIR"
    CMD=("$PYTHON_BIN" "$SCRIPT_DIR/v2t.py" "${COMMON_ARGS[@]}" --video-path "$VIDEO_PATH" --output-dir "$OUTPUT_DIR")
    ;;

  *)
    echo "[run_single_prompt] ERROR: unknown task '$TASK'" >&2
    usage
    exit 1
    ;;
esac

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

echo "[run_single_prompt] task=$TASK"
printf '[run_single_prompt] command:'
printf ' %q' "${CMD[@]}"
echo

exec "${CMD[@]}"
