# MiniCPM-o 4.5 Benchmarks

Benchmarks for the MiniCPM-o 4.5 three-stage TTS path:

- `benchmark_e2e_tts.py`: OpenAI-compatible `/v1/chat/completions` text + complete WAV E2E.
- `benchmark_token2wav.py`: isolated Stage-2 Token2Wav decode from pre-existing audio tokens.

Both scripts keep failed, timeout, and OOM rows in their outputs.

## E2E Text + Audio

Start a server first. The default deploy config auto-loads with `--omni`; pass
`--deploy-config` for a non-default GPU layout.

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5_2gpu.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
```

Run a short benchmark:

```bash
python benchmarks/minicpmo/benchmark_e2e_tts.py \
    --base-url http://127.0.0.1:8099/v1 \
    --model openbmb/MiniCPM-o-4_5 \
    --serve-config vllm_omni/deploy/minicpmo_4_5_2gpu.yaml \
    --requests 8 --concurrency 8 --warmup 1 \
    --output-dir ./results/minicpmo/e2e-c8
```

Outputs:

- `summary.json` / `summary.csv`: latency p50/p95, RTF, peak GPU memory, audio sanity, and null stage-timing fields when unavailable.
- `requests.jsonl`: one row per request, including failures.

## Token2Wav

```bash
python benchmarks/minicpmo/benchmark_token2wav.py \
    --token2wav-dir /path/to/MiniCPM-o-4_5/assets/token2wav \
    --prompt-wav /path/to/MiniCPM-o-4_5/assets/HT_ref_audio.wav \
    --lengths 1024 2048 4096 \
    --warmup 1 --iters 3 \
    --output-dir ./results/minicpmo/token2wav
```

Outputs are `token2wav_fp32.json` and `token2wav_fp32.csv` by default. Add
`--float16` only for opt-in comparison runs; PR1 does not change the default
Token2Wav precision.

Set `VLLM_OMNI_COMMIT=<sha>` when running from a copied source tree where
`git rev-parse HEAD` is unavailable.
