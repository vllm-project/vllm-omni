# Resource Monitor Script Guide (GPU / Reserved CPU·NPU)

The **`scripts/resource_monitor.sh`** script in this directory is the unified entry point for resource monitoring: it collects GPU memory and related metrics while running any command (such as long-running stability tests or unit tests), then bundles the results and generates a single-file HTML report when the command finishes. The script generates **`report.html` and CSV** files; open `report.html` in the output directory after the run completes.

Currently only the **GPU** backend is implemented. The backend is selected with the subcommand option `--backend gpu|cpu|npu` (`gpu` by default), with CPU/NPU reserved for future extension.

---

## Subcommands

All functionality is provided by a single script and should be run from the repository root:

| Subcommand | Description |
| -------- | ------ |
| `scripts/resource_monitor.sh start [--backend gpu\|cpu\|npu] [gpu_ids] [interval]` | Collect data in the background (currently only `gpu`: `nvidia-smi` writes CSV) |
| `scripts/resource_monitor.sh finalize [--backend gpu\|cpu\|npu] [run_id]` | Bundle the current run, generate `report.html`, and print `GPU_MONITOR_BUNDLE_DIR=` / `RESOURCE_MONITOR_BUNDLE_DIR=` |
| `scripts/resource_monitor.sh run [--backend gpu\|cpu\|npu] -- <command>` | Complete everything in one step: `start` -> run your command -> `finalize` |

`--backend` is optional. If omitted, the **gpu** backend is used by default.

---

## Environment Variables (monitoring script only)

| Environment Variable | Description | Default |
| ---------- | ------ | -------- |
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

## LingBot CUDA lifecycle checks

The existing CUDA case in `tests/test_lingbot_world.json` enables
`lifecycle_checks` after each completed rollout. It retains its 24-hour duration,
40-chunk sessions and eight-H800 topology. The NPU case is unchanged.
The probes stop a paced session, close an active WebSocket, and abort an active
transport. Each interruption must permit a complete, ordered fresh rollout with
a distinct request ID on the same server. Per-device memory after each recovery
must stay within `lifecycle_memory_growth_budget_bytes` (256 MiB) of the warmed
server immediately before the probes. This bounds recovery growth within a batch;
the existing resource monitor records the complete sustained-run memory timeline.

Each batch keeps a unique `lingbot-world-batch-*` directory under the runner's
result directory, containing `benchmark.json`, the shared
`benchmark-serve-result.json` (including failed raw sessions), client stdout/stderr,
and `lifecycle.json` when probes run. Lifecycle reports retain passed, failed and
unstarted cases plus baseline/recovery memory values and the failure cause.
Errors fail the stability case; the shared `omni_server` fixture owns final
server teardown. Archive these directories with the resource-monitor outputs.
Probes and memory observations are stability diagnostics, not performance samples.

CPU regression checks need development dependencies, no checkpoint or GPU:

```bash
python -m pytest -q tests/benchmarks/test_lingbot_world_lifecycle.py -m 'core_model and cpu' --run-level core_model
```

On the configured eight-H800 host with model access, run the existing case:

```bash
python -m pytest -sv tests/dfx/stability/scripts/run_stability_benchmark.py --test-config-file tests/dfx/stability/tests/test_lingbot_world.json -k test_lingbot_world_realtime_stability -m cuda
```

CPU and scripted WebSocket tests validate the harness only. Real-checkpoint
sustained generation, GPU reclamation and the configured device topology require
separate hardware execution; no new CI lane or hardware qualification is implied.
