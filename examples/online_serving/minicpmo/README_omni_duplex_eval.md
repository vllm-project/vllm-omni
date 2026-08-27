# Omni-DuplexEval

This runner generates official-shaped response artifacts through the native
MiniCPM-o duplex WebSocket. It does not change serving defaults or require the
upstream benchmark checkout at runtime.

```bash
python examples/online_serving/minicpmo/run_omni_duplex_eval.py generate \
  --url ws://127.0.0.1:8099/v1/realtime?duplex=1 \
  --model /models/MiniCPM-o-4_5 --ref-audio /data/ref.wav \
  --dataset Hothan/Omni-DuplexEval --family all \
  --response-root /data/duplex/responses

```

`--pace as-fast-as-possible` writes `clock=invalid`. Use `--limit 1` for a
smoke test. Evaluation and summarization are added by the follow-up scoring PR.
