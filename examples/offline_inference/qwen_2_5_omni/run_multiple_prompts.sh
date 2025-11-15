# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root (two levels up from examples/online_serving/)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
python end2end.py --model Qwen/Qwen2.5-Omni-7B \
                                 --voice-type "m02" \
                                 --dit-ckpt none \
                                 --bigvgan-ckpt none \
                                 --output-wav output_audio \
                                 --prompt_type text \
                                 --init-sleep-seconds 0 \
                                 --txt-prompts top100.txt
