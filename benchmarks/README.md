# Benchmarks Overview and Architecture

This document explains the benchmark architecture across all benchmark assets in this repo. It describes what we measure, and where to find or plug in new scenarios. Per-task details remain in subfolder READMEs (e.g., `benchmarks/<model>/README.md`).

## Scope and goals
- Establish repeatable latency/throughput measurements for multimodal LLM pipelines.
- Provide both HF Transformers (offline) and vLLM-Omni (multi-stage/pipeline) baselines.
- Make it easy to plug in new datasets and models with minimal changes to the runner scripts.

## Dataset and inputs
- Default example: SeedTTS top-100 prompts (`benchmarks/build_dataset/top100.txt`) via `benchmarks/build_dataset/`.
- Extensible: drop in new prompt files or modality-aligned payloads; keep the expected format for the consuming scripts (e.g., one prompt per line).
- If you add a new dataset, document it under `benchmarks/<model>/README.md` and point scripts to your data path.

## Directory layout
- `benchmarks/build_dataset/` — dataset prep utilities (e.g., SeedTTS top100).
- `benchmarks/<model>/vllm_omni/` — vLLM-Omni pipeline benchmarks, logs, outputs.
- Add new tasks under `benchmarks/<model>/...` with the same pattern: `transformers/`, `vllm_omni/`, task-specific README, and (optionally) dataset prep notes.

## Reference workflows
- **HF Transformers (offline, single process)**  
  Script (example): `benchmarks/<model>/transformers/eval_qwen3_moe_omni_transformers.sh`  
  Outputs: `benchmark_results/perf_stats.json`, `benchmark_results/results.json`, `benchmark_results/audio/` (if audio is produced).

- **vLLM-Omni end-to-end pipeline**  
  Script (example): `benchmarks/<model>/vllm_omni/eval_qwen3_moe_omni.sh`  
  Outputs: `vllm_omni/logs/*.stats.jsonl` (per-stage/overall latency & TPS), `vllm_omni/logs/stage*.log`, `vllm_omni/outputs/` (text/audio artifacts).

- **Adding a new task/model**  
  1) Create `benchmarks/<model>/transformers/` and/or `benchmarks/<model>/vllm_omni/` with scripts referencing your model and dataset.  
  2) Add a task README describing dataset, configs, and expected outputs.  
  3) Keep the output/log structure similar for easy comparison (perf_stats/results/audio or text outputs; stats.jsonl/logs for pipeline).

## Metrics to watch
- **Throughput**: `overall_tps`, `*_tps_avg` per stage.
- **Latency distribution**: look for long tails in `*.stats.jsonl`.
- **Quality/completeness**: missing outputs or errors in stage logs indicate pipeline failures or misconfigurations.

## Troubleshooting
- Verify GPU/driver/FlashAttention2 requirements for your chosen model/config.
- Ensure network access for dataset/model downloads (Google Drive, Hugging Face, etc.).
- If outputs are missing or slow, inspect per-stage logs and `*.stats.jsonl` for errors, stragglers, or contention.
