#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Server side for barge_in_client.py: start MiniCPM-o 4.5 in native
# full-duplex mode.
#
# MiniCPM-o 4.5 declares a duplex plugin, so vllm-omni serve runs it through
# DuplexOmni: the server is duplex-only (WS /v1/realtime?duplex=1, alias
# /v1/duplex, plus /v1/models and /health) and bounds the
# Thinker/Talker/Code2Wav stages to four live duplex sessions on one GPU. See
# vllm_omni/deploy/minicpmo_4_5.yaml (session_mode: duplex) for the session
# limits; every other duplex_session knob keeps its runtime default
# (idle TTL 300s, disconnect grace 30s, 16MiB pending input per session).
set -euo pipefail

MODEL="${MODEL:-openbmb/MiniCPM-o-4_5}"
PORT="${PORT:-8099}"

exec vllm-omni serve "$MODEL" \
    --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port "$PORT"
# duplex endpoint: ws://<host>:$PORT/v1/realtime?duplex=1 (alias /v1/duplex)
