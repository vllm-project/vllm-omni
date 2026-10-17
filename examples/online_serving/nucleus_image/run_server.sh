#!/bin/bash
# Nucleus-Image online serving startup script

MODEL="${MODEL:-NucleusAI/Nucleus-Image}"
PORT="${PORT:-8091}"
TP="${TP:-1}"
USP="${USP:-1}"
QUANTIZATION="${QUANTIZATION:-}"

CMD=(vllm serve "$MODEL" --omni --port "$PORT")

if [[ "$TP" != "1" ]]; then
  CMD+=(--tensor-parallel-size "$TP")
fi
if [[ "$USP" != "1" ]]; then
  CMD+=(--usp "$USP")
fi
if [[ -n "$QUANTIZATION" ]]; then
  CMD+=(--quantization "$QUANTIZATION")
fi

echo "Starting Nucleus-Image server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "TP: $TP"
echo "USP: $USP"
if [[ -n "$QUANTIZATION" ]]; then
  echo "Quantization: $QUANTIZATION"
fi

"${CMD[@]}"
