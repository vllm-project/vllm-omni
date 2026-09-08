#!/usr/bin/env bash
# Runs the frontend, LiveKit server, and agent together; exits (and tears
# down the other two) the moment any one of them exits.
#
# Usage: run-livekit-stack.sh <VLLM_OMNI_HOST>
set -uo pipefail

if [ $# -ne 1 ]; then
	echo "Usage: $0 <VLLM_OMNI_HOST>" >&2
	exit 1
fi

export VLLM_OMNI_HOST="$1"
export LIVEKIT_URL=ws://localhost:7880
export LIVEKIT_API_KEY=devkey
export LIVEKIT_API_SECRET=secret

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/agent-starter-react"
LOG_DIR="/tmp/livekit"

mkdir -p "$LOG_DIR"

if [ ! -d "$FRONTEND_DIR" ]; then
	read -rp "agent-starter-react not found at $FRONTEND_DIR. Clone it now and install Node dependecies? [y/N] " reply
	if [[ "$reply" =~ ^[Yy]$ ]]; then
		git clone https://github.com/livekit-examples/agent-starter-react.git "$FRONTEND_DIR"
		(cd "$FRONTEND_DIR" && pnpm install)
	else
		echo "agent-starter-react is required; aborting." >&2
		exit 1
	fi
fi

if ! command -v livekit-server >/dev/null 2>&1; then
	read -rp "livekit-server not found. Install it now via the official install script? [y/N] " reply
	if [[ "$reply" =~ ^[Yy]$ ]]; then
		curl -sSL https://get.livekit.io | bash
	else
		echo "livekit-server is required; aborting." >&2
		exit 1
	fi
fi

if command -v uvx; then
	PYX_CMD=uv
elif command -v pipx; then
	PYX_CMD=pipx
else
	echo "either of uvx or pipx is required; aborting." >&2
fi

pkill -f 'livekit-server --dev' 2>/dev/null
while pgrep -f 'livekit-server --dev' >/dev/null; do sleep 0.2; done

pids=()
names=()
logs=()

cleanup() {
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null
    done
    wait 2>/dev/null
}
trap cleanup EXIT INT TERM

(cd "$FRONTEND_DIR" && pnpm dev) > "$LOG_DIR/frontend.log" 2>&1 &
pids+=("$!"); names+=("frontend"); logs+=("$LOG_DIR/frontend.log")

livekit-server --dev > "$LOG_DIR/server.log" 2>&1 &
pids+=("$!"); names+=("server"); logs+=("$LOG_DIR/server.log")

($PYX_CMD run agent.py start) > "$LOG_DIR/agent.log" 2>&1 &
pids+=("$!"); names+=("agent"); logs+=("$LOG_DIR/agent.log")

echo "frontend pid=${pids[0]} -> $LOG_DIR/frontend.log"
echo "server   pid=${pids[1]} -> $LOG_DIR/server.log"
echo "agent    pid=${pids[2]} -> $LOG_DIR/agent.log"

wait -n "${pids[@]}"
exit_code=$?
echo "A process exited (code $exit_code) -- shutting the rest down."

for i in "${!pids[@]}"; do
    if ! kill -0 "${pids[$i]}" 2>/dev/null; then
        wait "${pids[$i]}"
        status=$?
        if [ "$status" -ne 0 ]; then
            echo "--- ${names[$i]} exited with code $status; last 10 lines of ${logs[$i]} ---"
            tail -n 10 "${logs[$i]}"
        fi
    fi
done

exit "$exit_code"
