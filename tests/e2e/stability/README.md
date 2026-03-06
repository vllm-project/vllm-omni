# 资源监控脚本使用说明（GPU / 预留 CPU·NPU）

本目录下的 **`scripts/resource_monitor.sh`** 是资源监控的统一入口：在跑任意命令（如长稳测试、单测等）的同时采集 GPU 显存等指标，结束后打包并生成单文件 HTML 报告，便于在 CI 或本地查看。脚本**仅生成 report.html 与 CSV**，不上传 CI artifact；运行结束后在输出目录中打开 `report.html` 即可。

当前仅实现 **GPU** 后端；后端由子命令的 `--backend gpu|cpu|npu` 指定（默认 gpu），预留 CPU/NPU 扩展。

---

## 子命令

所有功能通过一个脚本完成，在仓库根目录执行：

| 子命令 | 说明 |
|--------|------|
| `scripts/resource_monitor.sh start [--backend gpu\|cpu\|npu] [gpu_ids] [interval]` | 后台采集（当前仅 gpu：nvidia-smi 写 CSV） |
| `scripts/resource_monitor.sh finalize [--backend gpu\|cpu\|npu] [run_id]` | 打包当前 run，生成 report.html，输出 `GPU_MONITOR_BUNDLE_DIR=` / `RESOURCE_MONITOR_BUNDLE_DIR=` |
| `scripts/resource_monitor.sh run [--backend gpu\|cpu\|npu] -- <command>` | 一步完成：start → 执行你指定的命令 → finalize |

`--backend` 为可选项，不传时默认使用 **gpu** 后端。

---

## 环境变量（仅监控脚本）

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `RESOURCE_MONITOR_DATA_ROOT` | 监控数据根目录 | `tests/e2e/stability/gpu_monitor_data` |
| `RESOURCE_MONITOR_INTERVAL` | 采样间隔（秒） | 5 |
| `RESOURCE_MONITOR_LOG_INTERVAL` | 日志打印间隔（秒） | 15 |
| `GPU_MONITOR_DEVICES` | [GPU 后端] 监控的 GPU 设备 ID，如 `0,1` 或 `all` | all |

---

## 推荐用法：`run` 一条龙

在 **finally** 或 **after script** 里执行（确保被测命令失败也会收尾）：

```bash
# 监控 + 执行任意命令，结束后自动打包并生成 report.html
# 不指定 --backend 时默认 gpu；使用其他后端时加上 --backend cpu 或 --backend npu
bash tests/e2e/stability/scripts/resource_monitor.sh run [--backend gpu|cpu|npu] -- <你的命令>
```

示例（仓库根目录）：

```bash
# 示例：跑某条 pytest（默认 gpu 后端，可不写 --backend）
bash tests/e2e/stability/scripts/resource_monitor.sh run -- pytest -s -v tests/e2e/online_serving/test_foo.py -k test_xxx

# 显式指定 gpu 后端、自定义采样间隔与 GPU 0,1、日志每 30s 打一行
export RESOURCE_MONITOR_INTERVAL=10
export GPU_MONITOR_DEVICES=0,1
export RESOURCE_MONITOR_LOG_INTERVAL=30
bash tests/e2e/stability/scripts/resource_monitor.sh run --backend gpu -- pytest -s -v tests/e2e/online_serving/test_foo.py
```

运行中可在日志里看每隔若干秒出现的 `[GPU] ...`；结束后日志会打印 bundle 路径，如：`Line chart: open in browser: .../report.html`。

---

## 分步用法：start → 你的命令 → finalize

若需要先起监控、再手动执行长时间任务，可分步调用：

```bash
# 1. 启动监控（在 scripts 目录下或指定 DATA_ROOT；不写 --backend 时默认 gpu）
cd tests/e2e/stability/scripts
./resource_monitor.sh start [--backend gpu] all 5 &
MONITOR_PID=$!

# 2. 执行你的长稳/测试命令（任意）
# ...

# 3. 收尾（必须放在环境清理前，建议放在 finally / after script；backend 需与 start 一致）
BUNDLE_LINE=$(./resource_monitor.sh finalize [--backend gpu] 2>/dev/null | grep '^GPU_MONITOR_BUNDLE_DIR=')
eval "$BUNDLE_LINE"
echo "报告目录: $GPU_MONITOR_BUNDLE_DIR"
```

---

## 目录与产物（仅监控相关）

- **脚本**：`tests/e2e/stability/scripts/resource_monitor.sh`（入口）、`scripts/generate_report.py`（由 finalize 调用，生成 HTML）。
- **数据目录**：默认 `tests/e2e/stability/gpu_monitor_data/`（可由 `RESOURCE_MONITOR_DATA_ROOT` 覆盖）。  
  - 每次运行会生成 `run_<run_id>/gpu_metrics.csv`；  
  - `finalize` 后得到 `gpu_monitor_bundle_<run_id>/`，内含 `gpu_metrics.csv`、`report.html`、`README.txt`。
- **查看报告**：在 bundle 目录下用浏览器打开 `report.html`，即可查看显存曲线与统计。

脚本仅生成 `report.html` 与 CSV，不上传 CI artifact；若需保留报告，请自行从工作目录归档或下载。
