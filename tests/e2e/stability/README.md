# Stability 资源监控在 CI 中的使用（GPU / 预留 CPU·NPU）

长稳结束后 CI 环境会被清理，网页和 CSV 会丢失。脚本在清理前**仅打包并生成 report.html**，不上传到 CI artifact；本地或 CI 运行结束后在输出目录中直接打开 `report.html` 查看。

**脚本与数据目录**：`tests/e2e/stability/`（`scripts/resource_monitor.sh`、`scripts/generate_report.py`、`scripts/test_benchmark_stability.py` 等）。脚本名为资源监控统一入口，当前仅实现 GPU；通过 `--backend gpu|cpu|npu` 预留 CPU/NPU 扩展。

### 长稳 Benchmark 用例

与 **perf** 一致使用 `vllm bench serve --omni`，支持 **`--request-rate`**（请求速率）或 **`--max-concurrency`**（并发数）发请求；长稳额外增加**指定时长**：超过 `duration_sec` 后脚本不再发送新的请求，等待现有请求结束。

- **`tests/stability_test.json`**：长稳用例配置（与 `tests/perf/tests/test.json` 格式一致），每个 `benchmark_params` 需包含 `duration_sec`，以及 `request_rate` 或 `max_concurrency` 之一。
- **`stage_configs/`**：本目录下的 stage 配置（如 `qwen3_omni.yaml`），长稳用例只读取此目录，不依赖 `tests/perf`。
- **`scripts/test_benchmark_stability.py`**：pytest 用例，内含长稳 benchmark 逻辑（在指定时长内按 request-rate 或 max-concurrency 调用 `vllm bench serve --omni`）；先起 OmniServer，再跑 benchmark，断言无失败请求。以下参数可通过环境变量覆盖，无需改源码：
  - `STABILITY_BENCHMARK_DURATION_SEC`：运行时长（秒），覆盖 JSON 中的 `duration_sec`，默认 300
  - `STABILITY_BENCHMARK_NUM_PROMPTS_PER_BATCH`：每批请求数，默认 20

**环境变量与可选参数一览**

| 作用对象 | 环境变量 / 参数 | 说明 | 默认值 |
|----------|-----------------|------|--------|
| 长稳用例 | `STABILITY_BENCHMARK_DURATION_SEC` | 运行时长（秒） | 300 |
| 长稳用例 | `STABILITY_BENCHMARK_NUM_PROMPTS_PER_BATCH` | 每批请求数 | 20 |
| 资源监控 | `RESOURCE_MONITOR_DATA_ROOT` / `GPU_MONITOR_DATA_ROOT` | 监控数据根目录 | `tests/e2e/stability/gpu_monitor_data` |
| 资源监控 | `RESOURCE_MONITOR_INTERVAL` / `GPU_MONITOR_INTERVAL` | 采样间隔（秒） | 5 |
| 资源监控 | `RESOURCE_MONITOR_LOG_INTERVAL` / `GPU_MONITOR_LOG_INTERVAL` | 日志中 `[GPU]` 行打印间隔（秒） | 15 |
| 资源监控 | `GPU_MONITOR_DEVICES` | 监控的 GPU ID，如 `0,1` 或 `all` | all |
| 资源监控 | `SKIP_DEPS_CHECK` | 非空则跳过 nvidia-smi 等依赖检查 | 未设置 |
| 资源监控脚本 | `--backend gpu\|cpu\|npu` / `-b` | 子命令 start/finalize/run 的后端 | gpu |

**示例：**

```bash

# 配合资源监控：长稳 10 分钟 + 采样 10s、日志每 30s、仅 GPU 0,1
export STABILITY_BENCHMARK_DURATION_SEC=600
export GPU_MONITOR_INTERVAL=10
export GPU_MONITOR_LOG_INTERVAL=30
export GPU_MONITOR_DEVICES=0,1
bash tests/e2e/stability/scripts/resource_monitor.sh run --backend gpu -- pytest -s -v tests/e2e/stability/scripts/test_benchmark_stability.py

# 自定义数据目录、跳过依赖检查
export RESOURCE_MONITOR_DATA_ROOT=/tmp/my_monitor_data
export SKIP_DEPS_CHECK=1
bash tests/e2e/stability/scripts/resource_monitor.sh run -- pytest -s -v -k "test_benchmark" tests/e2e/stability/scripts/test_benchmark_stability.py
```

## 如何查看？

运行结束后在输出目录（如 `gpu_monitor_data/gpu_monitor_bundle_<run_id>/`）用浏览器打开 `report.html` 即可查看完整显存曲线与统计。

## 目录分布

```
tests/e2e/stability/
├── README.md
├── scripts/
│   ├── __init__.py
│   ├── resource_monitor.sh
│   ├── generate_report.py
│   └── test_benchmark_stability.py
├── tests/
│   └── stability_test.json
├── stage_configs/
│   └── qwen3_omni.yaml
└── gpu_monitor_data/          # 运行时生成，默认数据根目录
    ├── run_<run_id>/          # 单次运行的 CSV
    │   └── gpu_metrics.csv
    └── gpu_monitor_bundle_<run_id>/   # finalize 打包目录
        ├── gpu_metrics.csv
        ├── report.html
        └── README.txt
```

### 文件与目录说明

| 路径 | 作用 |
|------|------|
| `README.md` | 本说明文档：资源监控与长稳 benchmark 的使用方式、环境变量、目录与流程。 |
| `scripts/__init__.py` | Python 包标识，便于 `scripts` 作为模块被引用。 |
| `scripts/resource_monitor.sh` | 资源监控统一入口。子命令：`start`（后台采集）、`finalize`（打包并生成 report.html）、`run`（start → 执行命令 → finalize）。支持 `--backend gpu\|cpu\|npu`，当前仅 gpu 已实现。 |
| `scripts/generate_report.py` | 由 `resource_monitor.sh finalize` 调用，读取监控 CSV，生成单文件 HTML 报告（统计表、时序图、简单异常标记）。 |
| `scripts/test_benchmark_stability.py` | 长稳 benchmark 的 pytest 用例：先起 OmniServer，再在指定时长内按 `request_rate` 或 `max_concurrency` 跑 `vllm bench serve --omni`，断言无失败请求。 |
| `tests/stability_test.json` | 长稳用例配置（与 perf 的 test.json 格式一致），定义 server_params、benchmark_params（含 `duration_sec`、`request_rate` 或 `max_concurrency`）。 |
| `stage_configs/qwen3_omni.yaml` | Stage 配置示例，长稳用例只读取本目录下的 yaml，不依赖 `tests/perf`。 |
| `gpu_monitor_data/` | 默认监控数据根目录（可由 `RESOURCE_MONITOR_DATA_ROOT` / `GPU_MONITOR_DATA_ROOT` 覆盖）。内含每次运行的 CSV 与 finalize 生成的 bundle（`gpu_metrics.csv`、`report.html`、`README.txt`）。 |
