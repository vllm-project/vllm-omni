# Stability 资源监控在 CI 中的使用（GPU / 预留 CPU·NPU）

长稳结束后 CI 环境会被清理，网页和 CSV 都会丢失。通过**在清理前打包并上传为 CI 产物**，流水线结束后仍可下载查看。

**脚本与数据目录**：`tests/e2e/stability/`（`resource_monitor.sh`、`generate_report.py`、`run_benchmark_duration.py` 等）。脚本名为资源监控统一入口，当前仅实现 GPU；通过 `--backend gpu|cpu|npu` 预留 CPU/NPU 扩展。

### 长稳 Benchmark 用例

与 **perf** 一致使用 `vllm bench serve --omni`，支持 **`--request-rate`**（请求速率）或 **`--max-concurrency`**（并发数）发请求；长稳额外增加**指定时长**：超过该时间后不再发送新请求，已发出的请求会等其完成。

- **`stability_test.json`**：长稳用例配置（参考 `tests/perf/tests/test.json`），每个 `benchmark_params` 需包含 `duration_sec`，以及 `request_rate` 或 `max_concurrency` 之一。
- **`run_benchmark_duration.py`**：在指定时长内按 request-rate 或 max-concurrency 调用 `vllm bench serve --omni`（通过环境变量 `VLLM_BENCH_MAX_DURATION_SEC` 限制时长）。可单独跑或由 pytest 调用。
- **`test_benchmark_stability.py`**：pytest 用例，先起 OmniServer，再在指定时长内跑上述脚本，断言无失败请求。时长优先取环境变量 `STABILITY_BENCHMARK_DURATION_SEC`，否则用配置中的 `duration_sec`（默认 300 秒）。

**示例：**

```bash
# 默认约 5 分钟（配置中的 duration_sec）
pytest -s -v tests/e2e/stability/test_benchmark_stability.py

# 配合资源监控跑 10 分钟
export STABILITY_BENCHMARK_DURATION_SEC=600
bash tests/e2e/stability/resource_monitor.sh run -- pytest -s -v tests/e2e/stability/test_benchmark_stability.py
```

**单独跑脚本（需先启动服务）：**

```bash
# 按 request-rate=1 跑 300 秒
python tests/e2e/stability/run_benchmark_duration.py --duration 300 --request-rate 1 --port 8000

# 按 max-concurrency=4 跑 600 秒
python tests/e2e/stability/run_benchmark_duration.py --duration 600 --max-concurrency 4 --port 8000
```

## 如何查看？

使用 **静态报告 `report.html`**。`resource_monitor.sh finalize` 会从 CSV 调用 `generate_report.py` 生成单文件 HTML（图表、统计、异常表均内嵌），**无需任何服务器**。把打包目录上传为 CI artifact，流水线结束后**下载 artifact，在本地用浏览器打开其中的 `report.html`** 即可查看完整显存曲线与统计，不依赖当时的环境与网址。

CI 中只做「监控 → 收尾打包 → 上传 artifact」；查看时从流水线下载 artifact，本地打开 `report.html` 即可。

## 单脚本子命令

所有功能通过一个脚本 `resource_monitor.sh` 提供（可选参数 `--backend gpu|cpu|npu`，默认 `gpu`，当前仅 `gpu` 已实现）：

| 子命令 | 说明 |
|--------|------|
| `resource_monitor.sh start [--backend gpu\|cpu\|npu] [gpu_ids] [interval]` | 后台采集（当前仅 gpu：显存） |
| `resource_monitor.sh finalize [--backend gpu\|cpu\|npu] [run_id]` | 打包当前 run，生成 report.html，输出 `GPU_MONITOR_BUNDLE_DIR=` / `RESOURCE_MONITOR_BUNDLE_DIR=` |
| `resource_monitor.sh run [--backend gpu\|cpu\|npu] -- <command>` | 一步完成：start → 执行命令 → finalize（CI 中自动上传 artifact） |

## 本地与 CI 统一：一步完成

同一条命令在**本地**和 **CI** 都能用，无需分步。

- **CI**：不设环境变量，直接执行。会启动监控、跑测试、收尾打包并上传 artifact；实时看日志里的 `[GPU]` 行，结束后在 Artifacts 下载 `report.html`。
- **本地**：同上。

示例（仓库根目录）：

```bash
# CI 或本地：日志 + 结束后 report.html
bash tests/e2e/stability/resource_monitor.sh run -- pytest -s -v tests/e2e/online_serving/test_qwen3_omni_full.py -k test_text_to_text_async_chunk_003 -v

# 改环境变量：先 export，再执行命令
export GPU_MONITOR_INTERVAL=60
bash tests/e2e/stability/resource_monitor.sh run -- pytest -s -v tests/e2e/online_serving/test_qwen3_omni_full.py -k test_sleep_001

# 多个环境变量：采样间隔 60s、只监控 GPU 0,1、日志每 30s 打一行
export GPU_MONITOR_INTERVAL=60
export GPU_MONITOR_DEVICES=0,1
export GPU_MONITOR_LOG_INTERVAL=30
bash tests/e2e/stability/resource_monitor.sh run -- pytest -s -v tests/e2e/online_serving/test_qwen3_omni_full.py -k test_sleep_001
```

- 打包目录：`tests/e2e/stability/gpu_monitor_data/gpu_monitor_bundle_<run_id>/`，内含 `gpu_metrics.csv`、`report.html`、`README.txt`。折线图在 `report.html` 中；日志结束时会打印路径（如 `Line chart: open in browser: .../report.html`）。

## 流程概览

1. **启动监控**：后台运行 `./resource_monitor.sh start`（在 stability 目录下），整个长稳期间持续写 CSV。
2. **跑长稳**：执行你的长稳用例（如 `test_qwen_edit.sh`）。
3. **收尾**：长稳结束后、环境清理前，执行 `./resource_monitor.sh finalize`（在 stability 目录下），生成报告并打包。
4. **归档**：把 `resource_monitor.sh finalize` 输出的目录上传为 CI artifact。

之后在流水线页面下载该 artifact，即可得到 `gpu_metrics.csv` 和 `report.html`（含图表与异常标记），无需再访问当时的环境。

## 步骤说明

### 1. 启动监控（后台）

```bash
cd tests/e2e/stability
./resource_monitor.sh start all 5 &
MONITOR_PID=$!
```

可选：把 `GPU_MONITOR_DATA_ROOT` 设到固定目录，便于与后续步骤一致。

### 2. 运行长稳测试

按你现有方式跑 24h/72h 长稳即可。

### 3. 收尾（必须放在「清理前」执行）

在 **finally** 或 **after script** 里执行（确保长稳失败也会跑）：

```bash
cd tests/e2e/stability
BUNDLE_LINE=$(./resource_monitor.sh finalize 2>/dev/null | grep '^GPU_MONITOR_BUNDLE_DIR=')
eval "$BUNDLE_LINE"
echo "归档目录: $GPU_MONITOR_BUNDLE_DIR"
```

### 4. 上传为 CI 产物

**GitHub Actions** 示例：

```yaml
- name: Finalize GPU monitor and upload
  if: always()
  run: |
    cd tests/e2e/stability
    BUNDLE_LINE=$(./resource_monitor.sh finalize 2>/dev/null | grep '^GPU_MONITOR_BUNDLE_DIR=') || true
    if [[ -n "$BUNDLE_LINE" ]]; then
      eval "$BUNDLE_LINE"
      echo "GPU_MONITOR_BUNDLE_DIR=$GPU_MONITOR_BUNDLE_DIR" >> $GITHUB_ENV
    fi
- name: Upload GPU monitor bundle
  if: env.GPU_MONITOR_BUNDLE_DIR != ''
  uses: actions/upload-artifact@v4
  with:
    name: gpu-monitor-${{ github.run_id }}
    path: ${{ env.GPU_MONITOR_BUNDLE_DIR }}
```

**Buildkite（推荐用 resource_monitor.sh run 一条龙）**

用单脚本一次完成「启动监控 + 跑测试 + 收尾 + 上传 artifact」：

```yaml
commands:
  - bash tests/e2e/stability/resource_monitor.sh run -- pytest -s -v tests/e2e/online_serving/test_foo.py ...
```

- **用到的脚本**：`tests/e2e/stability/resource_monitor.sh`（子命令 run 内部会 start、finalize，并调 `generate_report.py`）。
- **运行中「实时」看什么**：CI 没有单独的可访问网页。请打开 **Buildkite 该次构建的 Job 页面**，在**日志区域**里会每隔约 15 秒出现一行 `[GPU] ...`，即当前最新一次采样的显存数据。
- **结束后在哪里下载**：同一 Job 页面的 **Artifacts** 中可下载 `gpu_metrics.csv`、`report.html`、`README.txt`。本地用浏览器打开 `report.html` 即可。

## 产物内容

- `gpu_metrics.csv`：原始采样（时间戳、GPU 索引、显存占用、利用率）。
- `report.html`：单文件报告，含统计表、时序图、简单异常标记，浏览器打开即可。
- `README.txt`：简要说明。

长稳结束后在流水线里下载该 artifact，本地打开 `report.html` 即可查看，不依赖当时的环境与网址。
