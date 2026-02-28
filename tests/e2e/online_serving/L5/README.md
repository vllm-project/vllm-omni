# L5 GPU 监控在 CI 中的使用

长稳结束后 CI 环境会被清理，网页和 CSV 都会丢失。通过**在清理前打包并上传为 CI 产物**，流水线结束后仍可下载查看。

**脚本与数据目录**：`tests/e2e/online_serving/L5/`（`gpu_monitor.sh`、`generate_report.py`、`gpu_dashboard.html` 等）。

## CI 清理后如何查看？（实时仪表盘 vs 静态报告）

- **实时仪表盘**（`gpu_dashboard.html` + `gpu_monitor.sh serve`）依赖本机 HTTP 服务和 `latest.json`，适合**本地长稳时**边跑边看。CI 跑完环境被回收，**无法在 CI 里提供可访问的网页**。
- **解决方式**：不在 CI 里依赖实时网页，改用 **静态报告 `report.html`**。`gpu_monitor.sh finalize` 会从 CSV 调用 `generate_report.py` 生成单文件 HTML（图表、统计、异常表均内嵌），**无需任何服务器**。把打包目录上传为 CI artifact，流水线结束后**下载 artifact，在本地用浏览器打开其中的 `report.html`** 即可查看完整显存曲线与统计，不依赖当时的环境与网址。

因此：CI 中只做「监控 → 收尾打包 → 上传 artifact」；查看时从流水线下载 artifact，本地打开 `report.html` 即可。

## 单脚本子命令

所有功能通过一个脚本 `gpu_monitor.sh` 提供：

| 子命令 | 说明 |
|--------|------|
| `gpu_monitor.sh start [gpu_ids] [interval]` | 后台采集显存（原 moniter.sh） |
| `gpu_monitor.sh finalize [run_id]` | 打包当前 run，生成 report.html，输出 `GPU_MONITOR_BUNDLE_DIR=` |
| `gpu_monitor.sh serve [port]` | 启动实时仪表盘 HTTP 服务（默认 8765） |
| `gpu_monitor.sh run -- <command>` | 一步完成：start → 执行命令 → finalize（CI 中自动上传 artifact） |

## 本地与 CI 统一：一步完成

同一条命令在**本地**和 **CI** 都能用，无需分步。

- **CI**：不设环境变量，直接执行。会启动监控、跑测试、收尾打包并上传 artifact；实时看日志里的 `[GPU]` 行，结束后在 Artifacts 下载 `report.html`。
- **本地**：同上；若想**边跑边看网页仪表盘**，在命令前加 `GPU_MONITOR_SERVE_DASHBOARD=1`，脚本会同时拉起仪表盘服务，浏览器访问输出的 URL 即可（远程机器需 SSH 端口转发）。

示例（仓库根目录）：

```bash
# CI 或本地仅要日志 + 结束后 report.html
bash tests/e2e/online_serving/L5/gpu_monitor.sh run -- pytest -s -v tests/e2e/online_serving/test_qwen3_omni_full.py -k test_text_to_text_async_chunk_003 -v

# 改环境变量：先 export，再执行命令
export GPU_MONITOR_INTERVAL=60
bash tests/e2e/online_serving/L5/gpu_monitor.sh run -- pytest -s -v tests/e2e/online_serving/test_qwen3_omni_full.py -k test_sleep_001

# 多个环境变量示例：采样间隔 60s、只监控 GPU 0,1、日志每 30s 打一行、并开仪表盘
export GPU_MONITOR_INTERVAL=60
export GPU_MONITOR_DEVICES=0,1
export GPU_MONITOR_LOG_INTERVAL=30
export GPU_MONITOR_SERVE_DASHBOARD=1
bash tests/e2e/online_serving/L5/gpu_monitor.sh run -- pytest -s -v tests/e2e/online_serving/test_qwen3_omni_full.py -k test_sleep_001

# 本地想边跑边看仪表盘（一步，无需另开终端）
export GPU_MONITOR_SERVE_DASHBOARD=1
bash tests/e2e/online_serving/L5/gpu_monitor.sh run -- pytest -s -v tests/e2e/online_serving/test_qwen3_omni_full.py -k test_text_to_text_async_chunk_003 -v
```

- 仪表盘 URL 会打印在终端，默认 `http://127.0.0.1:8765/gpu_dashboard.html`。远程机器上在本机执行 `ssh -L 8765:127.0.0.1:8765 用户@主机` 后访问该 URL。
- 打包目录：`tests/e2e/online_serving/L5/gpu_monitor_data/gpu_monitor_bundle_<run_id>/`，内含 `gpu_metrics.csv`、`report.html`、`README.txt`。折线图在 `report.html` 中；日志结束时会打印路径（如 `Line chart: open in browser: .../report.html`）。

## 流程概览

1. **启动监控**：后台运行 `./gpu_monitor.sh start`（在 L5 目录下），整个长稳期间持续写 CSV。
2. **跑长稳**：执行你的长稳用例（如 `test_qwen_edit.sh`）。
3. **收尾**：长稳结束后、环境清理前，执行 `./gpu_monitor.sh finalize`（在 L5 目录下），生成报告并打包。
4. **归档**：把 `gpu_monitor.sh finalize` 输出的目录上传为 CI artifact。

之后在流水线页面下载该 artifact，即可得到 `gpu_metrics.csv` 和 `report.html`（含图表与异常标记），无需再访问当时的环境。

## 步骤说明

### 1. 启动监控（后台）

```bash
cd tests/e2e/online_serving/L5
./gpu_monitor.sh start all 5 &
MONITOR_PID=$!
```

可选：把 `GPU_MONITOR_DATA_ROOT` 设到固定目录，便于与后续步骤一致。

### 2. 运行长稳测试

按你现有方式跑 24h/72h 长稳即可。

### 3. 收尾（必须放在「清理前」执行）

在 **finally** 或 **after script** 里执行（确保长稳失败也会跑）：

```bash
cd tests/e2e/online_serving/L5
BUNDLE_LINE=$(./gpu_monitor.sh finalize 2>/dev/null | grep '^GPU_MONITOR_BUNDLE_DIR=')
eval "$BUNDLE_LINE"
echo "归档目录: $GPU_MONITOR_BUNDLE_DIR"
```

### 4. 上传为 CI 产物

**GitHub Actions** 示例：

```yaml
- name: Finalize GPU monitor and upload
  if: always()
  run: |
    cd tests/e2e/online_serving/L5
    BUNDLE_LINE=$(./gpu_monitor.sh finalize 2>/dev/null | grep '^GPU_MONITOR_BUNDLE_DIR=') || true
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

**Buildkite（推荐用 gpu_monitor.sh run 一条龙）**

用单脚本一次完成「启动监控 + 跑测试 + 收尾 + 上传 artifact」：

```yaml
commands:
  - bash tests/e2e/online_serving/L5/gpu_monitor.sh run -- pytest -s -v tests/e2e/online_serving/test_foo.py ...
```

- **用到的脚本**：`tests/e2e/online_serving/L5/gpu_monitor.sh`（子命令 run 内部会 start、finalize，并调 `generate_report.py`）。
- **运行中「实时」看什么**：CI 没有单独的可访问网页。请打开 **Buildkite 该次构建的 Job 页面**，在**日志区域**里会每隔约 15 秒出现一行 `[GPU] ...`，即当前最新一次采样的显存数据。
- **结束后在哪里下载**：同一 Job 页面的 **Artifacts** 中可下载 `gpu_metrics.csv`、`report.html`、`README.txt`。本地用浏览器打开 `report.html` 即可。

## 产物内容

- `gpu_metrics.csv`：原始采样（时间戳、GPU 索引、显存占用、利用率）。
- `report.html`：单文件报告，含统计表、时序图、简单异常标记，浏览器打开即可。
- `README.txt`：简要说明。

长稳结束后在流水线里下载该 artifact，本地打开 `report.html` 即可查看，不依赖当时的环境与网址。
