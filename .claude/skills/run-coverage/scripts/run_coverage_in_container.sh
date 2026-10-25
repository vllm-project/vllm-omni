#!/usr/bin/env bash
# Run the vllm-omni coverage sequence inside a remote container.
#
# Usage: run_coverage_in_container.sh <ssh-alias> <container> [logfile] [base_dir]
#   ssh-alias   SSH config alias on this bridge (must already be authorized)
#   container   docker container name on that host (e.g. vllm_omni)
#   logfile     local log path (default /home/wy/coverage_runs/coverage-<UTC ts>.log)
#   base_dir    dir inside the container holding .venv/ and vllm-omni/
#               (default /rebase)
#
# The remote payload is piped via stdin (ssh -> docker exec -i bash -s) so the
# quoting stays flat regardless of the test markers inside run_coverage.sh.
set -uo pipefail

ALIAS=${1:?usage: run_coverage_in_container.sh <ssh-alias> <container> [logfile] [base_dir]}
CONTAINER=${2:?missing container name}
STAMP=$(date -u +%Y%m%d-%H%M%S)
LOG=${3:-/home/wy/coverage_runs/coverage-${STAMP}.log}
BASE=${4:-/rebase}

mkdir -p "$(dirname "$LOG")"

echo "[$(date -u '+%F %T')] ssh ${ALIAS} -> docker exec ${CONTAINER} (base ${BASE})" | tee "$LOG"

ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=20 \
    "$ALIAS" "docker exec -i -e BASE='${BASE}' ${CONTAINER} bash -s" <<'REMOTE' 2>&1 | tee -a "$LOG"
set -uo pipefail

# 1. source .venv/bin/activate
cd "$BASE" || { echo "BASE_DIR_MISSING: $BASE"; exit 11; }
if [ ! -f .venv/bin/activate ]; then echo "VENV_MISSING: $BASE/.venv"; exit 12; fi
source .venv/bin/activate

# 2. cd vllm-omni
cd vllm-omni || { echo "REPO_MISSING: $BASE/vllm-omni"; exit 13; }

# 3. git pull (retry: GitHub TLS from these hosts is flaky)
echo "=== git pull ==="
pull_ok=0
for i in 1 2 3; do
  if git -c http.lowSpeedLimit=500 -c http.lowSpeedTime=30 pull; then pull_ok=1; break; fi
  echo "git pull attempt $i failed; retrying in 5s..."
  sleep 5
done
if [ "$pull_ok" -ne 1 ]; then echo "GIT_PULL_FAILED"; exit 10; fi

# 4. bash run_coverage.sh
if [ ! -f run_coverage.sh ]; then echo "SCRIPT_MISSING: $BASE/vllm-omni/run_coverage.sh"; exit 14; fi
echo "=== run_coverage.sh (HEAD: $(git rev-parse --short HEAD)) ==="
bash run_coverage.sh
rc=$?

# Surface artifacts for the report step
echo "=== artifacts ==="
for f in coverage.xml .coverage htmlcov/index.html; do
  if [ -f "$f" ]; then echo "ARTIFACT: $f ($(du -h "$f" | cut -f1))"; else echo "ARTIFACT: $f (absent)"; fi
done
exit $rc
REMOTE

rc=${PIPESTATUS[0]}
echo "COVERAGE_EXIT=${rc}" | tee -a "$LOG"
exit $rc
