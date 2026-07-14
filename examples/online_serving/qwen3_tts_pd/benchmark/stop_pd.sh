#!/usr/bin/env bash
# Stop the Qwen3-TTS PD server and free ALL its resources.
#
# 干净地关闭 PD server：
#   * 杀掉 API server 进程 + 它派生的全部 stage/worker 子孙进程
#   * 释放 API 端口 (默认 8091) 与 mooncake bootstrap 端口 (25201-25206)
#   * 清理 /dev/shm 与 /tmp 里的 IPC/SHM 残段
#   * 等所有相关端口真正释放 (最多 30s)
#
# 用法:
#   bash stop_pd.sh                 # 关默认 PORT=8091，清理全部
#   bash stop_pd.sh 8091            # 指定 API 端口
#   PORT=8091 bash stop_pd.sh
#
# 幂等：没有任何 server 在跑时执行也安全 (纯 no-op)。
# 不用 set -e：这里大量 kill/fuser 返回非零是正常的。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"

PORT="${1:-${PORT:-8091}}"
BOOTSTRAP_PORTS=(25201 25202 25203 25204 25205 25206)

echo "==> [stop] 清理 PD server (api_port=$PORT, bootstrap=${BOOTSTRAP_PORTS[*]})"

# 递归收集某 pid 的所有后代并从叶子往根杀（spawn 出来的 worker cmdline 不含
# "vllm serve"，仅靠进程名 pkill 抓不全，必须按进程树 + 端口双管齐下）。
kill_process_tree() {
    local root_pid="$1"
    [[ -z "$root_pid" ]] && return
    local pids_to_kill=() stack=("$root_pid")
    while (( ${#stack[@]} > 0 )); do
        local pid="${stack[-1]}"; unset 'stack[-1]'
        pids_to_kill+=("$pid")
        local children; children="$(pgrep -P "$pid" 2>/dev/null || true)"
        for c in $children; do stack+=("$c"); done
    done
    local n=${#pids_to_kill[@]}
    for (( i=n-1; i>=0; i-- )); do
        kill -9 "${pids_to_kill[i]}" 2>/dev/null || true
    done
}

# 1) 通过 pid 文件精准杀 (results/ 新位置 + /tmp 旧位置都覆盖)
for pf in "$RESULTS_DIR"/server_*.pid /tmp/server_*.pid; do
    [[ -f "$pf" ]] || continue
    old_pid="$(cat "$pf" 2>/dev/null || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
        echo "    [stop] kill pid=$old_pid (from $pf) + 子孙"
        kill_process_tree "$old_pid"
    fi
    rm -f "$pf"
done

# 2) 通过 API 端口定位僵尸 APIServer (刚才假 health 的元凶) 并杀整棵树
for pid in $(lsof -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null); do
    echo "    [stop] kill pid=$pid (占用 API 端口 $PORT) + 子孙"
    kill_process_tree "$pid"
done

# 3) 兜底：按 cmdline 杀残留的 vllm / stage / spawn worker
pkill -9 -f "vllm serve"            2>/dev/null || true
pkill -9 -f "StageEngineCoreProc"   2>/dev/null || true
pkill -9 -f "from multiprocessing.spawn" 2>/dev/null || true

sleep 2

# 4) 强制释放端口 (API + bootstrap)
fuser -k -9 "${PORT}/tcp" 2>/dev/null || true
for p in "${BOOTSTRAP_PORTS[@]}"; do fuser -k -9 "${p}/tcp" 2>/dev/null || true; done

# 5) 清 /dev/shm + /tmp 的 IPC/SHM 残段
find /dev/shm -maxdepth 1 \
    \( -name 'psm_*' -o -name 'wnsm_*' -o -name 'vllm*' -o -name 'mooncake*' -o -name 'vllm_omni*' \) \
    -user "$(whoami)" -delete 2>/dev/null || true
find /tmp -maxdepth 1 -name 'zmq-*.sock' -user "$(whoami)" -delete 2>/dev/null || true
find /tmp -maxdepth 2 -name '*.ipc'      -user "$(whoami)" -delete 2>/dev/null || true

sleep 1

# 6) 等所有端口真正释放 (TIME_WAIT / 内核回收延迟)，最多 30s，期间反复补刀
port_in_use() {
    local p="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -ltn "( sport = :$p )" 2>/dev/null | grep -q ":$p"
    elif command -v lsof >/dev/null 2>&1; then
        lsof -iTCP:"$p" -sTCP:LISTEN -t >/dev/null 2>&1
    else
        fuser -n tcp "$p" >/dev/null 2>&1
    fi
}
for attempt in $(seq 1 30); do
    busy=""
    for p in "$PORT" "${BOOTSTRAP_PORTS[@]}"; do
        port_in_use "$p" && busy="$busy $p"
    done
    if [[ -z "$busy" ]]; then
        echo "    ✅ [stop] 所有端口已释放: $PORT ${BOOTSTRAP_PORTS[*]}"
        break
    fi
    echo "    [stop] 仍被占用:${busy} (${attempt}/30s)，继续补刀..."
    for p in $busy; do
        for hp in $(lsof -tiTCP:"$p" -sTCP:LISTEN 2>/dev/null); do kill_process_tree "$hp"; done
        fuser -k -9 "${p}/tcp" 2>/dev/null || true
    done
    sleep 1
done

echo "==> [stop] done"
