#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# MODEL=/models/auk-omni bash examples/online_serving/text_to_speech/auk/run_server.sh
set -euo pipefail

: "${MODEL:?Set MODEL to the assembled AuK or AuK-Flash checkpoint directory}"
exec vllm serve "$MODEL" --omni --host "${HOST:-127.0.0.1}" --port "${PORT:-8091}" "$@"
