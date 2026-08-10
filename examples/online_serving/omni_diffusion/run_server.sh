#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

MODEL=${1:?"Usage: run_server.sh MODEL DEPLOY_CONFIG"}
DEPLOY_CONFIG=${2:?"Usage: run_server.sh MODEL DEPLOY_CONFIG"}
PORT=${PORT:-8091}

exec vllm serve "${MODEL}" \
  --omni \
  --trust-remote-code \
  --port "${PORT}" \
  --deploy-config "${DEPLOY_CONFIG}"
