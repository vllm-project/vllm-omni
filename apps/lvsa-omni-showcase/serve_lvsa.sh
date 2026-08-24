#!/usr/bin/env bash
# Start an LVSA-enabled vLLM-Omni server for Wan or HunyuanVideo.
#
# LVSA is selected as the attention backend via --diffusion-attention-config (see
# below); the `LVSA_*` env vars only tune it. The `lvsa-vllm-omni` plugin registers
# itself through vLLM-Omni's `general_plugins` entry points — nothing in vLLM-Omni
# core is modified. Watch the log for "[LVSA] Geometry detected" to confirm sparse
# attention is active for your request geometry.
#
#   MODEL=/path/to/Wan2.1-T2V-1.3B-Diffusers MODEL_FAMILY=wan     FRAMES=161 bash serve_lvsa.sh
#   MODEL=/path/to/HunyuanVideo-1.5-...       MODEL_FAMILY=hunyuan FRAMES=193 bash serve_lvsa.sh
set -euo pipefail

MODEL=${MODEL:?set MODEL=/path/to/Wan-or-HunyuanVideo-Diffusers}
MODEL_FAMILY=${MODEL_FAMILY:-wan}     # wan | hunyuan
PORT=${PORT:-8000}
FRAMES=${FRAMES:-161}                 # target horizon in frames
VAE_T=${VAE_T:-4}                     # VAE temporal compression (Wan/Hunyuan = 4)
BACKEND=${LVSA_BACKEND:-flashinfer}   # flashinfer (fused, fastest) | sdpa

T_LAT=$(( (FRAMES - 1) / VAE_T + 1 ))

# Per-family training reference horizon (Wan train ref = 21, Hunyuan = 33).
case "$MODEL_FAMILY" in
  wan)     REFERENCE=${REFERENCE:-21} ;;
  hunyuan) REFERENCE=${REFERENCE:-33} ;;
  *) echo "MODEL_FAMILY must be 'wan' or 'hunyuan'"; exit 1 ;;
esac
export LVSA_BACKEND="$BACKEND"
export LVSA_TOTAL_LATENT_FRAMES="$T_LAT"
export LVSA_REFERENCE_LATENT_FRAMES="$REFERENCE"
export LVSA_ROTATE_KEYFRAMES=${LVSA_ROTATE_KEYFRAMES:-1}

echo "[lvsa-app] serving $MODEL ($MODEL_FAMILY) on :$PORT"
echo "[lvsa-app]   LVSA backend=$BACKEND  T_lat=$T_LAT (frames=$FRAMES)  reference=$REFERENCE"
echo "[lvsa-app]   confirm activation by grep '\\[LVSA\\] Geometry detected' in this log"

# vLLM-Omni 0.22 selects the LVSA attention backend per-role via
# --diffusion-attention-config. The backend is what drives sparse attention
# here; it engages under tensor-parallel too, unlike the LVSA_*_HOOK monkey-patch
# path (which falls back to dense under sequence-parallel), so we deliberately do
# NOT set a hook.
exec vllm serve "$MODEL" --omni --host 0.0.0.0 --port "$PORT" \
     --diffusion-attention-config '{"per_role": {"self": {"backend": "LVSA"}}}' \
     --init-timeout 1800 ${EXTRA_ARGS:-}
