# Resource Monitor Script Guide (GPU / Reserved CPU·NPU)

The **`scripts/resource_monitor.sh`** script in this directory is the unified entry point for resource monitoring: it collects GPU memory and related metrics while running any command (such as long-running stability tests or unit tests), then bundles the results and generates a single-file HTML report when the command finishes. The script generates **`report.html` and CSV** files; open `report.html` in the output directory after the run completes.

Currently only the **GPU** backend is implemented. The backend is selected with the subcommand option `--backend gpu|cpu|npu` (`gpu` by default), with CPU/NPU reserved for future extension.

---

## Subcommands

All functionality is provided by a single script and should be run from the repository root:

| Subcommand | Description |
|--------|------|
| `scripts/resource_monitor.sh start [--backend gpu\|cpu\|npu] [gpu_ids] [interval]` | Collect data in the background (currently only `gpu`: `nvidia-smi` writes CSV) |
| `scripts/resource_monitor.sh finalize [--backend gpu\|cpu\|npu] [run_id]` | Bundle the current run, generate `report.html`, and print `GPU_MONITOR_BUNDLE_DIR=` / `RESOURCE_MONITOR_BUNDLE_DIR=` |
| `scripts/resource_monitor.sh run [--backend gpu\|cpu\|npu] -- <command>` | Complete everything in one step: `start` -> run your command -> `finalize` |

`--backend` is optional. If omitted, the **gpu** backend is used by default.

---

## Environment Variables (monitoring script only)

| Environment Variable | Description | Default |
|----------|------|--------|
| `RESOURCE_MONITOR_DATA_ROOT` | Root directory for monitoring data | `tests/dfx/stability/gpu_monitor_data` |
| `RESOURCE_MONITOR_INTERVAL` | Sampling interval (seconds) | 5 |
| `RESOURCE_MONITOR_LOG_INTERVAL` | Log print interval (seconds) | 15 |
| `GPU_MONITOR_DEVICES` | [GPU backend] GPU device IDs to monitor, such as `0,1` or `all` | all |

---

## Recommended Usage: one-step `run`

Run this in **finally** or **after script** blocks so cleanup still happens even if the tested command fails:

```bash
# Monitor + run any command, then automatically bundle results and generate report.html
# If --backend is omitted, gpu is used by default; use --backend cpu or --backend npu for other backends
bash tests/dfx/stability/scripts/resource_monitor.sh run [--backend gpu|cpu|npu] -- <your-command>
```

Examples (from the repository root):

```bash
# Example: run a pytest case (gpu backend by default, so --backend can be omitted)
bash tests/dfx/stability/scripts/resource_monitor.sh run -- pytest -s -v tests/e2e/online_serving/test_foo.py -k test_xxx

# Explicitly select the gpu backend, customize the sampling interval and GPUs 0,1, and print one log line every 30s
export RESOURCE_MONITOR_INTERVAL=10
export GPU_MONITOR_DEVICES=0,1
export RESOURCE_MONITOR_LOG_INTERVAL=30
bash tests/dfx/stability/scripts/resource_monitor.sh run --backend gpu -- pytest -s -v tests/e2e/online_serving/test_foo.py
```

During execution, the log will show `[GPU] ...` every few seconds. When the run ends, the log prints the bundle path, for example: `Line chart: open in browser: .../report.html`.

---

## Step-by-step Usage: `start` -> your command -> `finalize`

If you need to start monitoring first and then manually run a long task, you can call the steps separately:

```bash
# 1. Start monitoring (run from the `scripts` directory or set `DATA_ROOT`; gpu is the default if `--backend` is omitted)
cd tests/dfx/stability/scripts
./resource_monitor.sh start [--backend gpu] all 5 &
MONITOR_PID=$!

# 2. Run your long-running/stability test command (any command)
# ...

# 3. Finalize (must run before environment cleanup; putting it in `finally` / `after script` is recommended; the backend must match `start`)
BUNDLE_LINE=$(./resource_monitor.sh finalize [--backend gpu] 2>/dev/null | grep '^GPU_MONITOR_BUNDLE_DIR=')
eval "$BUNDLE_LINE"
echo "Report directory: $GPU_MONITOR_BUNDLE_DIR"
```

---

## Directories and Outputs (monitoring only)

- **Scripts**: `tests/dfx/stability/scripts/resource_monitor.sh` (entry point) and `scripts/generate_report.py` (called by `finalize` to generate HTML).
- **Data directory**: `tests/dfx/stability/gpu_monitor_data/` by default (can be overridden with `RESOURCE_MONITOR_DATA_ROOT`).  
  - Each run generates `run_<run_id>/gpu_metrics.csv`.  
  - After `finalize`, you get `gpu_monitor_bundle_<run_id>/` containing `gpu_metrics.csv`, `report.html`, and `README.txt`.
- **View the report**: open `report.html` in the bundle directory with a browser to inspect memory usage curves and statistics.

The script only generates `report.html` and CSV files. If you need to keep the report, archive or download it from the working directory yourself.
