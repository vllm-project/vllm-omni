# 8-Instance Ingress DRR Strategy Package

This package is the server-side ingress scheduler branch handoff for teammate testing.

## What is included

- Strategy plugin code (server-side ingress dispatch):
  - `server_ingress_plugin/image_ingress_scheduler_single.py`
  - `server_ingress_plugin/patch_api_server_ingress_singlefile.py`
- Strategy startup script:
  - `t2i_8cards/start_tp1_8inst_ingress_strategy.sh`
- Strategy matrix runner:
  - `t2i_8cards/run_t2i_8inst_strategy_matrix.sh`
- Generic benchmark and summary tools:
  - `t2i_8cards/run_t2i_8inst_datasetc_detailed.py`
  - `t2i_8cards/summarize_t2i_8inst_matrix.py`
- Cleanup helper:
  - `t2i_8cards/cleanup_t2i_8inst_env.sh`

## Base image

Use upstream image name:

`quay.io/ascend/vllm-omni:v0.18.0`

Do not replace this with local renamed image tags in shared branch scripts.

## Default DRR budget policy

Budget can be configured in two ways:

1) Explicit override (highest priority):
- set `OMNI_INGRESS_DRR_QUEUE_BUDGET_OVERRIDES='{"512x512_20":8,...}'`

2) Auto default (when no explicit override is passed):
- controlled by `OMNI_INGRESS_DRR_DEFAULT_BUDGET_MODE`
  - `inv_cost`: budget ~ `1 / cost_i`
  - `weight_inv_cost`: budget ~ `weight_i / cost_i`
- `cost_i = width * height * steps` (normalized by the smallest ratio across active request types)
- `weight_i` source:
  - first from `OMNI_INGRESS_DRR_REQUEST_WEIGHTS` if provided
  - otherwise estimated online from observed request arrival mix

## Quick start

From repo root:

```bash
cd /docker/aixuan/vllm-omni-v0.18.0-test/t2i_8cards
bash cleanup_t2i_8inst_env.sh
nohup bash run_t2i_8inst_strategy_matrix.sh > "results/strategy_weight_inv_cost.nohup.log" 2>&1 &
```

Monitor:

```bash
tail -f "results/strategy_weight_inv_cost.nohup.log"
```

## Main configurable parameters

The following env vars are supported by `run_t2i_8inst_strategy_matrix.sh` and forwarded to `start_tp1_8inst_ingress_strategy.sh`.

### Deployment / workload

- `GPU_LIST` (default `0,1,2,3,4,5,6,7`)
- `BASE_PORT` (default `18291`)
- `PROXY_PORT` (default `28093`)
- `CONTAINER` (default `ax-vllm-qwen-8inst-ingress-strategy`)
- `TOTAL_REQUESTS` (default `500`)
- `RPS_LIST` (default `0.1,0.2,0.5,0.8,1,5,inf`)
- `MAX_CONCURRENCY` (default `inf`)
- `ARRIVAL_SEED` (default `42`)
- `RANDOM_REQUEST_SEED` (default `42`)
- `OUT_DIR` (default `t2i_8cards/results`)
- `TAG` (default `datasetc_t2i_8inst_strategy_weight_inv_cost_<timestamp>`)

### Ingress scheduler

- `OMNI_INGRESS_BATCH_DRR_ENABLE` (default `1`)
- `OMNI_INGRESS_BATCH_CAPS` (default `{"512x512_20":4,"768x768_20":4,"1024x1024_25":1,"1536x1536_35":1}`)
- `OMNI_INGRESS_DRR_MAX_WAIT_MS` (default `800`)
- `OMNI_INGRESS_DRR_STRICT_BATCHING` (default `0`)
- `OMNI_INGRESS_DRR_Q_BASE` (default `12`)
- `OMNI_INGRESS_DRR_AGE_THRESHOLD_MS` (default `2000`)
- `OMNI_INGRESS_DRR_AGE_BONUS_FACTOR` (default `1.0`)
- `OMNI_INGRESS_DRR_DEFAULT_BUDGET_MODE` (default `weight_inv_cost`)
- `OMNI_INGRESS_DRR_REQUEST_WEIGHTS` (default empty, JSON dict by request type)
- `OMNI_INGRESS_DRR_QUEUE_BUDGET_OVERRIDES` (default empty; set this to force fixed budgets)

## Output files

For each RPS point:

- `<TAG>_strategy_rps*_requests.csv`
- `<TAG>_strategy_rps*_by_type.csv`
- `<TAG>_strategy_rps*_summary.json`

Matrix summary:

- `<TAG>_strategy_matrix_summary.csv`
- `<TAG>_strategy_matrix_summary.md`
