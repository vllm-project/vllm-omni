#!/usr/bin/env bash
# Usage:
#   bash start_pd.sh                       # 默认: PD 1P1D (3 卡)
#   bash start_pd.sh 1p1d                  # PD 1P1D (3 卡：prefill+decode+code2wav)
#   bash start_pd.sh 1p3d                  # PD 1P3D (1 prefill + 3 decode + code2wav)
#   bash start_pd.sh 1p6d                  # PD 1P6D (1 prefill + 6 decode + code2wav)
#   bash start_pd.sh 3p1d                  # PD 3P1D (3 prefill + 1 decode + code2wav)
#   bash start_pd.sh single                # 单机基线 (qwen3_tts.yaml)
#   bash start_pd.sh ar_3p1d               # AR-only 3P1D (无 code2wav，仅 talker)
#   bash start_pd.sh /abs/path/to/x.yaml   # 自定义 yaml 绝对路径
#
# Env override:
#   MODEL=/root/models/Qwen3-TTS-12Hz-0.6B-Base
#   PORT=8091
#   HOST=127.0.0.1
#
# Output:
#   server log -> $RESULTS_DIR/server_<topology>.log
#   server pid -> $RESULTS_DIR/server_<topology>.pid

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Walk up from this script to find the vllm-omni repo root (the directory that
# contains the vllm_omni/ package). This keeps the script location-independent
# so it works whether it lives at repo-root/examples/.../benchmark/ or anywhere
# else inside the checkout (e.g. when copied out to a personal scratch dir).
find_repo_root() {
    local d="$SCRIPT_DIR"
    while [[ "$d" != "/" ]]; do
        if [[ -d "$d/vllm_omni" ]]; then
            echo "$d"; return 0
        fi
        d="$(dirname "$d")"
    done
    return 1
}
DEFAULT_VLLM_OMNI_DIR="$(find_repo_root)"
if [[ -z "$DEFAULT_VLLM_OMNI_DIR" ]]; then
    echo "ERR: could not locate vllm-omni repo root (no vllm_omni/ package found upward from $SCRIPT_DIR)." >&2
    echo "     Set VLLM_OMNI_DIR=/path/to/vllm-omni explicitly." >&2
    exit 2
fi

# ---------- 参数解析 ----------
TOPO="${1:-1p1d}"
MODEL="${MODEL:-/root/models/Qwen3-TTS-12Hz-0.6B-Base}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8091}"
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-$DEFAULT_VLLM_OMNI_DIR}"
DEPLOY_DIR="$VLLM_OMNI_DIR/vllm_omni/deploy"

# 解析 topology -> yaml 路径
case "$TOPO" in
    1p1d)     YAML="$DEPLOY_DIR/qwen3_tts_pd_1p1d.yaml" ; TAG="pd_1p1d" ;;
    1p3d)     YAML="$DEPLOY_DIR/qwen3_tts_pd_1p3d.yaml" ; TAG="pd_1p3d" ;;
    1p6d)     YAML="$DEPLOY_DIR/qwen3_tts_pd_1p6d.yaml" ; TAG="pd_1p6d" ;;
    3p1d)     YAML="$DEPLOY_DIR/qwen3_tts_pd_3p1d.yaml" ; TAG="pd_3p1d" ;;
    2p2d)     YAML="$DEPLOY_DIR/qwen3_tts_pd_2p2d.yaml" ; TAG="pd_2p2d" ;;
    single)   YAML=""                                   ; TAG="single" ;;
    ar_3p1d)  YAML="$DEPLOY_DIR/qwen3_tts_pd_ar_only_3p1d.yaml" ; TAG="pd_ar_3p1d" ;;
    /*.yaml)  YAML="$TOPO"                              ; TAG="custom_$(basename "$TOPO" .yaml)" ;;
    *)        echo "ERR: unknown topology '$TOPO'. See script header for valid choices." >&2 ; exit 2 ;;
esac

if [[ ! -d "$VLLM_OMNI_DIR" ]]; then
    echo "ERR: vllm-omni dir not found: $VLLM_OMNI_DIR" >&2
    echo "     Set VLLM_OMNI_DIR=/path/to/vllm-omni explicitly." >&2
    exit 2
fi
if [[ -n "$YAML" && ! -f "$YAML" ]]; then
    echo "ERR: yaml not found: $YAML" >&2
    exit 2
fi
if ! command -v vllm >/dev/null 2>&1; then
    echo "ERR: vllm command not found. Activate the right Python/env first." >&2
    exit 2
fi

RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
mkdir -p "$RESULTS_DIR"
LOG="${RESULTS_DIR}/server_${TAG}.log"
PID_FILE="${RESULTS_DIR}/server_${TAG}.pid"

echo "==> topology   : $TOPO"
echo "==> vllm-omni  : $VLLM_OMNI_DIR"
echo "==> yaml       : ${YAML:-<single, no yaml>}"
echo "==> model      : $MODEL"
echo "==> listen     : $HOST:$PORT"
echo "==> log        : $LOG"

# ---------- 0) Mooncake 自检自愈（防止 mooncake/ 又被改名为 mooncake.disabled） ----------
ensure_mooncake_available() {
    if [[ "$TOPO" == "single" ]]; then
        echo "==> [0/4] skipping mooncake module check for single baseline..."
        return 0
    fi

    echo "==> [0/4] checking mooncake module..."
    SITE_PKG="${SITE_PKG:-$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')}"
    if [[ -d "${SITE_PKG}/mooncake.disabled" && ! -d "${SITE_PKG}/mooncake" ]]; then
        echo "    ⚠️  mooncake is currently disabled, restoring..."
        sudo -n mv "${SITE_PKG}/mooncake.disabled" "${SITE_PKG}/mooncake" 2>/dev/null \
            || mv "${SITE_PKG}/mooncake.disabled" "${SITE_PKG}/mooncake" 2>/dev/null \
            || echo "    !! failed to restore mooncake/, run with sudo or check perms"
    fi
    if [[ -d "${SITE_PKG}/mooncake_transfer_engine-0.3.8.post1.dist-info.disabled" \
          && ! -d "${SITE_PKG}/mooncake_transfer_engine-0.3.8.post1.dist-info" ]]; then
        sudo -n mv "${SITE_PKG}/mooncake_transfer_engine-0.3.8.post1.dist-info.disabled" \
                "${SITE_PKG}/mooncake_transfer_engine-0.3.8.post1.dist-info" 2>/dev/null \
            || mv "${SITE_PKG}/mooncake_transfer_engine-0.3.8.post1.dist-info.disabled" \
                  "${SITE_PKG}/mooncake_transfer_engine-0.3.8.post1.dist-info" 2>/dev/null || true
    fi
    # 验证 mooncake import 正常（如果还失败就提前退出，省得跑 60 秒等 worker 死）
    if ! python3 -c "from mooncake.engine import TransferEngine" 2>/dev/null; then
        echo "    ❌ mooncake import still failing. Manual fix required:"
        echo "       SITE_PKG=${SITE_PKG}"
        echo "       ls -ld ${SITE_PKG}/mooncake* ${SITE_PKG}/*mooncake*"
        echo "       python3 -c 'from mooncake.engine import TransferEngine; print(\"OK\")'"
        exit 1
    fi
    echo "    ✅ mooncake import OK"
}

ensure_mooncake_available

# ---------- 1) 先调用 stop 脚本，彻底关掉旧 server + 清残留 ----------
echo "==> [1/4] stopping any existing server..."
if [[ -f "$SCRIPT_DIR/stop_pd.sh" ]]; then
    bash "$SCRIPT_DIR/stop_pd.sh" "$PORT" || true
else
    echo "    stop_pd.sh not found; not killing port $PORT automatically to avoid affecting unrelated services"
fi

# Re-check after cleanup, so vLLM never imports MooncakeConnector while the
# mooncake package is still disabled/missing.
ensure_mooncake_available

# 1f) 显存检查（必须有空闲显存才能跑）
echo "==> GPU memory before start:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader 2>&1 | head -8
else
    echo "    (nvidia-smi not in PATH)"
fi

# ---------- 2) 清环境变量 ----------
echo "==> [2/4] cleaning proxy env..."
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="localhost,127.0.0.1,::1"
export VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"
export VLLM_USE_DEEP_GEMM=0
# MTP graph / KV-cache opt-in paths are OFF by default. Force-cleared here so a
# stale export in the parent shell can't accidentally enable them; to opt in,
# drop these three lines (KV_CACHE additionally needs FULL_GRAPH + INVERSE_CDF).
export VLLM_OMNI_MTP_FULL_GRAPH=0
export VLLM_OMNI_MTP_INVERSE_CDF=0
export VLLM_OMNI_MTP_KV_CACHE=0
# MTP multi-replica sub-batch path is DISABLED by default (NUM_REPLICAS=1).
# The >1 path pins each request to a deepcopied code_predictor replica by
# crc32(req_id); under PD the extra replica state was implicated in requests
# that never sample the codec-EOS (stop_token_ids=[2150]) and run to
# max_tokens ("停不下来"). Collapsing to a single code_predictor makes decode
# MTP behave exactly like the single-node baseline. Uses ${VAR:-default} so a
# command-line export can still opt back in (e.g. NUM_REPLICAS=2).
export VLLM_OMNI_MTP_NUM_REPLICAS="${VLLM_OMNI_MTP_NUM_REPLICAS:-1}"
export VLLM_OMNI_MTP_BATCH_PER_REPLICA="${VLLM_OMNI_MTP_BATCH_PER_REPLICA:-0}"
export PYTHONPATH="$VLLM_OMNI_DIR:${PYTHONPATH:-}"

echo "==> python vllm_omni module:"
python3 -c 'import vllm_omni, pathlib; print("   ", pathlib.Path(vllm_omni.__file__).resolve())'

# ---------- 3) 后台拉起 server ----------
echo "==> [3/4] starting server in background..."
cd "$VLLM_OMNI_DIR"

if [[ -n "$YAML" ]]; then
    nohup vllm serve "$MODEL" \
        --omni --host "$HOST" --port "$PORT" \
        --stage-configs-path "$YAML" \
        --trust-remote-code \
        > "$LOG" 2>&1 &
else
    nohup vllm serve "$MODEL" \
        --omni --host "$HOST" --port "$PORT" \
        --trust-remote-code \
        > "$LOG" 2>&1 &
fi
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"
echo "    server pid=$SERVER_PID"

# 安静启动：不再把 server 全过程日志刷屏（压测时不需要看启动过程），
# 全部写入 $LOG，只在下面按秒打点等待 /health；失败时才 dump 诊断信息。
echo "    ---- quiet start: 完整日志见 $LOG (tail -f 可自行查看) ----"

# ---------- 4) 等 /health ----------
# Multi-stage PD topologies (1p3d/1p6d/3p1d) initialize stages serially and
# may spend several minutes capturing CUDA/MTP graphs on cold cache. Keep the
# timeout configurable so larger topologies are not killed while still warming.
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-1800}"
echo "==> [4/4] waiting for /health (max ${STARTUP_TIMEOUT_S}s)..."
for i in $(seq 1 "$STARTUP_TIMEOUT_S"); do
    if curl -sS -m 2 --noproxy '*' "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
        echo "    ✅ /health OK after ${i}s. pid=$SERVER_PID"
        echo
        echo "    next: bash $SCRIPT_DIR/bench_pd.sh $TOPO sweep   # 扫 c=1,8,16,32,64"
        echo "    stop: kill -9 \$(cat $PID_FILE)   # 或按端口: kill -9 \$(lsof -t -iTCP:$PORT -sTCP:LISTEN)"
        exit 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "    !! server process died after ${i}s"
        echo "==== GPU state at death ===="
        nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader 2>&1 | head -8
        echo "==== kernel OOM / SIGKILL hints (dmesg) ===="
        dmesg 2>/dev/null | tail -50 | grep -iE "killed|oom|memory|cuda|nvidia|signal" | tail -20
        echo "==== last activity per worker/stage (last 60 lines per pid) ===="
        # 抓出每个子进程死前最后做的事 (StageEngineCoreProc + Worker)
        grep -E "Worker pid=|StageEngineCoreProc pid=" "$LOG" 2>/dev/null | tail -60
        echo "==== ERRORS / Tracebacks / signals across full log ===="
        grep -niE "killed|signal|fatal|abort|core dumped|cuda.*error|nccl.*error|FAILED|^[A-Z][a-zA-Z]*Error:|RuntimeError:|AssertionError:|exception|Traceback" "$LOG" 2>/dev/null | tail -40
        echo "==== last 250 log lines (raw tail) ===="
        tail -250 "$LOG"
        echo "==== full log saved at: $LOG (size: $(wc -l < "$LOG") lines) ===="
        exit 1
    fi
    if (( i % 30 == 0 )); then
        echo "    ...still waiting (${i}s)"
    fi
    sleep 1
done

echo "    !! /health timeout, last 80 log lines:"
tail -80 "$LOG"
exit 1
