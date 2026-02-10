# Benchmarks Guide

This README explains how to (1) prepare benchmark datasets and (2) run the provided Qwen3-Omni benchmarks.

## 1) Prepare the dataset (SeedTTS top100)

```bash
cd benchmarks/build_dataset
pip install gdown

# Download SeedTTS test set from Google Drive
gdown --id 1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP

# Extract
tar -xf seedtts_testset.tar

# Copy metadata and extract top-100 prompts
cp seedtts_testset/en/meta.lst meta.lst
python extract_prompts.py -i meta.lst -o top100.txt -n 100

# (Optional) clean up to save space
rm -rf seedtts_testset seedtts_testset.tar meta.lst
```

Artifacts:
- `benchmarks/build_dataset/top100.txt` — 100 text prompts (one per line).

## 2) Run benchmarks

All commands assume repo root (`vllm-omni`).

### A. Transformers benchmark (offline, HF Transformers)

```
bash benchmarks/qwen3-omni/transformers/eval_qwen3_moe_omni_transformers.sh
```

What it does:
- Runs `qwen3_omni_moe_transformers.py` over `top100.txt` with `--num_prompts 100`.
- Outputs to `benchmarks/qwen3-omni/transformers/benchmark_results/`:
  - `perf_stats.json` — aggregated & per-prompt TPS/latency (thinker/talker/code2wav/overall).
  - `results.json` — per-prompt outputs and audio paths.
  - `audio/` — ~100 generated `.wav` files.

Key checks:
- `overall_tps` and `*_tps_avg` should be non-zero and reasonably stable.
- Investigate any 0/NaN or unusually low TPS / long-tail latency.

### B. vLLM Omni end-to-end benchmark (pipeline)

```
bash benchmarks/qwen3-omni/vllm_omni/eval_qwen3_moe_omni.sh
```

What it does:
- Runs `examples/offline_inference/qwen3_omni/end2end.py` with `--log-stats`.
- Uses `benchmarks/build_dataset/top100.txt` and writes to:
  - Logs: `benchmarks/qwen3-omni/vllm_omni/logs/`
    - `omni_llm_pipeline_text.orchestrator.stats.jsonl` — per-stage latency stats.
    - `omni_llm_pipeline_text.overall.stats.jsonl` — end-to-end latency/TPS.
    - `omni_llm_pipeline_text.stage{0,1,2}.log` — per-stage detailed logs/errors.
  - Outputs: `benchmarks/qwen3-omni/vllm_omni/outputs/` — ~100 text and `.wav` files.

Key checks:
- Overall stats: end-to-end latency/TPS should be reasonable.
- Orchestrator stats: per-stage latency should be stable; investigate long tails.
- Stage logs: ensure no errors and no unusually slow stages.


## Performance snapshot

The chart below summarizes our measured Qwen3-Omni MoE end-to-end benchmark, comparing vLLM-Omni against HF Transformers. It shows the overall throughput advantage for vLLM-Omni. These are actual experiment results—please refer to this performance when evaluating or reproducing the benchmark.

![vLLM-Omni vs HF](./vllm-omni-vs-hf.png)

## Directory layout
- `benchmarks/build_dataset/` — dataset prep utilities (e.g., SeedTTS top100).
- `benchmarks/<model>/vllm_omni/` — vLLM-Omni pipeline benchmarks, logs, outputs.
- Add new tasks under `benchmarks/<model>/...` with the same pattern: `transformers/`, `vllm_omni/`, task-specific README, and (optionally) dataset prep notes.
- `benchmarks/<model>/vllm-omni-vs-hf.png` — current performance snapshot (overall throughput comparison).
- `benchmarks/<model>/transformers/` — HF Transformers benchmarks (offline reference).

## Troubleshooting
- Make sure GPU/driver/FlashAttention2 requirements are met for the chosen model.
- If downloads fail, confirm network access to Google Drive (`gdown`) and Hugging Face.
- If audio files are missing, check for errors in stage logs or model generation.***
