"""
Performance benchmark CI runner for diffusion models.

This runner separates two concepts:

1. ``server_type``: how the serving process is started.
   Currently only ``vllm-omni`` is supported here.
2. ``benchmark_backend``: which serving API the benchmark client calls.
   Examples: ``vllm-omni`` for ``/v1/chat/completions`` and ``v1/videos``
   for async video jobs.

A config JSON file is REQUIRED via --test-config-file:
  pytest run_diffusion_benchmark.py --test-config-file tests/dfx/perf/tests/test_qwen_image_vllm_omni.json

All benchmark results for a session are consolidated into a single JSON file under
BENCHMARK_RESULT_DIR (override via the DIFFUSION_BENCHMARK_DIR environment variable).
Each entry in the file contains the test metadata (test_name, backend, benchmark_params,
timestamp) together with the raw metrics returned by the benchmark script.
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import psutil
import pytest

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"
os.environ.setdefault("DIFFUSION_ATTENTION_BACKEND", "FLASH_ATTN")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DEFAULT_RESULT_DIR = Path(__file__).parent.parent / "results"
BENCHMARK_RESULT_DIR = Path(os.environ.get("DIFFUSION_BENCHMARK_DIR", str(_DEFAULT_RESULT_DIR)))

BENCHMARK_SCRIPT = str(
    Path(__file__).parent.parent.parent.parent.parent / "benchmarks" / "diffusion" / "diffusion_benchmark_serving.py"
)

# Single aggregated result file for the entire benchmark session.
# Populated lazily after CONFIG_FILE_PATH is resolved.
_SESSION_TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
_RESULT_LOCK = threading.Lock()
_BRANCHPOINT_COMMIT_SHA: str | None = None
DIFFUSION_RESULT_TEMPLATE_PATH = Path(__file__).parent / "diffusion_result_template.json"


def _get_config_file_from_argv() -> str | None:
    """Read --test-config-file from sys.argv at import time so pytest parametrize can use it.

    pytest_addoption (below) registers the same flag so pytest does not reject it.
    Supports both ``--test-config-file path`` and ``--test-config-file=path`` forms.
    Returns None if the flag is not present; callers must handle the missing case.
    """
    for i, arg in enumerate(sys.argv):
        if arg == "--test-config-file" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--test-config-file="):
            return arg.split("=", 1)[1]
    return None


CONFIG_FILE_PATH = _get_config_file_from_argv()
if CONFIG_FILE_PATH is None:
    print("No config file provided, using default config file: tests/dfx/perf/tests/test_qwen_image_vllm_omni.json")
    CONFIG_FILE_PATH = "tests/dfx/perf/tests/test_qwen_image_vllm_omni.json"

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _resolve_refs(configs: list[dict[str, Any]], config_dir: Path) -> list[dict[str, Any]]:
    """Resolve {"$ref": "filename.json"} in benchmark_params fields."""
    for cfg in configs:
        bp = cfg.get("benchmark_params")
        if isinstance(bp, dict) and "$ref" in bp:
            ref_path = config_dir / bp["$ref"]
            try:
                with open(ref_path, encoding="utf-8") as f:
                    cfg["benchmark_params"] = json.load(f)
            except FileNotFoundError:
                raise ValueError(f"benchmark_params $ref not found: {ref_path}")
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parsing error in {ref_path}: {e}")
    return configs


def load_configs(config_path: str) -> list[dict[str, Any]]:
    try:
        abs_path = Path(config_path).resolve()
        with open(abs_path, encoding="utf-8") as f:
            configs = json.load(f)
        return _resolve_refs(configs, abs_path.parent)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing error: {str(e)}")
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found: {config_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration file: {str(e)}")


BENCHMARK_CONFIGS = load_configs(CONFIG_FILE_PATH)

_config_stem = Path(CONFIG_FILE_PATH).stem  # e.g. "test_qwen_image_vllm_omni"
AGGREGATED_RESULT_FILE = BENCHMARK_RESULT_DIR / f"diffusion_result_{_config_stem}_{_SESSION_TIMESTAMP}.json"


def _append_to_aggregated_file(record: dict[str, Any]) -> None:
    """Thread-safe append of *record* to the session-level aggregated JSON file.

    The file contains a JSON array; each call loads the existing array (or
    starts a new one), appends the record, and writes the file back atomically.
    """
    with _RESULT_LOCK:
        BENCHMARK_RESULT_DIR.mkdir(parents=True, exist_ok=True)
        if AGGREGATED_RESULT_FILE.exists():
            with open(AGGREGATED_RESULT_FILE, encoding="utf-8") as f:
                records: list[dict] = json.load(f)
        else:
            records = []
        records.append(record)
        with open(AGGREGATED_RESULT_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)


_server_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_open_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout: int = 1200) -> None:
    """Block until the given host:port accepts connections or timeout expires."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                if s.connect_ex((host, port)) == 0:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"Server did not start on {host}:{port} within {timeout}s")


def _kill_process_tree(pid: int) -> None:
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        all_pids = [pid] + [c.pid for c in children]

        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass

        gone, alive = psutil.wait_procs(children, timeout=10)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass

        try:
            parent.terminate()
            parent.wait(timeout=10)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            try:
                parent.kill()
            except psutil.NoSuchProcess:
                pass

        time.sleep(1)
        still_alive = [p for p in all_pids if psutil.pid_exists(p)]
        if still_alive:
            print(f"Warning: processes still alive after shutdown: {still_alive}")
            for p in still_alive:
                try:
                    subprocess.run(["kill", "-9", str(p)], timeout=2)
                except Exception:
                    pass
    except psutil.NoSuchProcess:
        pass


# ---------------------------------------------------------------------------
# Server classes
# ---------------------------------------------------------------------------


class DiffusionServer:
    """Start a vLLM-Omni diffusion model server as a subprocess.

    Launched via vllm_omni.entrypoints.cli.main with the diffusion-specific
    parallelism flags (--usp, --ring, --cfg-parallel-size, etc.) passed directly
    on the CLI.  Minimum hardware: 4× NVIDIA H100 80 GB.
    """

    server_type = "vllm-omni"

    def __init__(
        self,
        server_cfg: dict[str, Any],
        *,
        port: int | None = None,
    ) -> None:
        self.server_cfg: dict[str, Any] = server_cfg
        self.model = server_cfg["model"]
        self.serve_args = server_cfg["serve_args"]
        self.host = "127.0.0.1"
        self.port = port if port is not None else _get_open_port()
        self.proc: subprocess.Popen | None = None
        self.test_name: str = ""

    def _start_server(self) -> None:
        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        cmd = [
            sys.executable,
            "-m",
            "vllm_omni.entrypoints.cli.main",
            "serve",
            self.model,
            "--omni",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ] + self.serve_args

        print(f"Launching DiffusionServer: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(Path(__file__).parent.parent.parent.parent),
        )
        _wait_for_port(self.host, self.port)
        print(f"DiffusionServer ready on {self.host}:{self.port}")

    def __enter__(self):
        self._start_server()
        return self

    def __exit__(self, *_):
        if self.proc:
            _kill_process_tree(self.proc.pid)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _build_serve_args(serve_args_dict: dict[str, Any]) -> list[str]:
    """Convert a serve_args dict from test.json into a flat CLI argument list."""
    args: list[str] = []
    for key, value in serve_args_dict.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
        elif isinstance(value, dict):
            args.extend([flag, json.dumps(value, separators=(",", ":"))])
        else:
            args.extend([flag, str(value)])
    return args


def _get_branchpoint_commit_sha() -> str:
    """Return the branch-point commit SHA against main.

    Uses git command: ``git merge-base HEAD origin/main``.
    """
    global _BRANCHPOINT_COMMIT_SHA
    if _BRANCHPOINT_COMMIT_SHA is not None:
        return _BRANCHPOINT_COMMIT_SHA

    repo_root = Path(__file__).parent.parent.parent.parent
    try:
        sha = (
            subprocess.check_output(
                ["git", "merge-base", "HEAD", "origin/main"],
                cwd=str(repo_root),
                stderr=subprocess.STDOUT,
                text=True,
            )
            .strip()
            .splitlines()[0]
        )
        _BRANCHPOINT_COMMIT_SHA = sha
    except Exception as e:
        print(f"Warning: failed to get branch-point commit SHA: {e}")
        _BRANCHPOINT_COMMIT_SHA = ""
    return _BRANCHPOINT_COMMIT_SHA


def _to_resolution_string(params: dict[str, Any]) -> str:
    width = params.get("width", "unknown width")
    height = params.get("height", "unknown height")
    return f"{width}x{height}"


def _to_parallelism_string(framework: str, serve_args_dict: dict[str, Any]) -> str:
    parts: list[str] = []
    if framework == "vllm-omni":
        keys = [
            "num-gpus",
            "usp",
            "ulysses-degree",
            "ring",
            "ring-degree",
            "cfg-parallel-size",
            "vae-patch-parallel-size",
            "vae-use-tiling",
            "tensor-parallel-size",
        ]
        for key in keys:
            if key in serve_args_dict:
                parts.append(f"{key}={serve_args_dict[key]}")
    return ",".join(parts) if parts else "none"


def _to_cache_string(framework: str, serve_args_dict: dict[str, Any]) -> str:
    if framework == "vllm-omni":
        if "cache-backend" in serve_args_dict:
            return str(serve_args_dict["cache-backend"])
    return "disabled"


def _to_offload_string(framework: str, serve_args_dict: dict[str, Any]) -> str:
    selected: list[str] = []
    if framework == "vllm-omni":
        offload_keys = [
            "enable-cpu-offload",
            "enable-layerwise-offload",
        ]
        for key in offload_keys:
            if key in serve_args_dict:
                selected.append(key)
    return f"enabled({';'.join(selected)})" if selected else "disabled"


def _to_compile_value(framework: str, serve_args_dict: dict[str, Any]) -> str:
    if framework == "vllm-omni":
        if "enforce-eager" in serve_args_dict:
            return "disabled"
        return "enabled"
    return "disabled"


def _to_quantization_value(framework: str, serve_args_dict: dict[str, Any]) -> str:
    if framework == "vllm-omni":
        quant = serve_args_dict.get("quantization")
        return str(quant) if quant else "disabled"
    return "disabled"


def _unique_server_params(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one server-config dict per unique test_name."""
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for cfg in configs:
        test_name = cfg["test_name"]
        if test_name in seen:
            continue
        seen.add(test_name)
        server_type = cfg.get("server_type", "vllm-omni")
        if server_type != "vllm-omni":
            raise ValueError(f"Unsupported server_type in config: {server_type}")
        serve_args_dict = cfg["server_params"].get("serve_args", {})
        result.append(
            {
                "test_name": test_name,
                "server_type": server_type,
                "model": cfg["server_params"]["model"],
                "serve_args_dict": serve_args_dict,
                "serve_args": _build_serve_args(serve_args_dict),
                "benchmark_backend": cfg.get("benchmark_backend"),
                "server_params": cfg["server_params"],
            }
        )
    return result


def _test_param_mapping(configs: list[dict[str, Any]]) -> dict[str, list[dict]]:
    mapping: dict[str, list[dict]] = {}
    for cfg in configs:
        name = cfg["test_name"]
        mapping.setdefault(name, [])
        mapping[name].extend(cfg["benchmark_params"])
    return mapping


def _make_server(server_cfg: dict[str, Any]) -> DiffusionServer:
    """Factory: return a vLLM-Omni diffusion server instance for the config."""
    return DiffusionServer(server_cfg=server_cfg)


# ---------------------------------------------------------------------------
# Parametrize data
# ---------------------------------------------------------------------------

server_params = _unique_server_params(BENCHMARK_CONFIGS)
test_param_map = _test_param_mapping(BENCHMARK_CONFIGS)

benchmark_indices: list[int] = list(range(max(len(v) for v in test_param_map.values())))

# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def diffusion_server(request):
    """Start one vLLM-Omni server per unique test configuration."""
    with _server_lock:
        server_cfg: dict[str, Any] = request.param
        test_name = server_cfg["test_name"]
        server_type = server_cfg["server_type"]

        print(f"\nStarting {server_type} server for test: {test_name}")
        with _make_server(server_cfg) as server:
            server.test_name = test_name
            print(f"{server_type} server started successfully")
            yield server
            print(f"{server_type} server stopping…")

    print(f"{server_type} server stopped")


@pytest.fixture
def benchmark_params(request, diffusion_server):
    """Yield the benchmark params dict for the current (server, index) pair."""
    param_index: int = request.param
    test_name = diffusion_server.test_name

    params_list = test_param_map.get(test_name, [])
    if not params_list:
        raise ValueError(f"No benchmark params for test: {test_name}")
    if param_index >= len(params_list):
        pytest.skip(f"Param index {param_index} out of range for {test_name} (has {len(params_list)} params)")

    current = param_index + 1
    total = len(params_list)
    print(f"\n  Running benchmark {current}/{total} for {test_name}")
    return {"test_name": test_name, "params": params_list[param_index]}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    host: str,
    port: int,
    model: str,
    params: dict[str, Any],
    test_name: str,
    backend: str = "vllm-omni",
    server_cfg: dict[str, Any] | None = None,
    source_file: str = "",
) -> dict[str, Any]:
    """Run diffusion_benchmark_serving.py as a subprocess and return parsed metrics.

    The raw metrics are written to a temporary file by the subprocess.  After
    the run completes the metrics are merged with full metadata (test_name,
    backend, benchmark_params, timestamp, flat reporting fields) and appended
    to the session-wide aggregated JSON file (AGGREGATED_RESULT_FILE).  The
    temporary file is removed afterwards.  Subprocess stdout/stderr are tee'd
    to a .log file under BENCHMARK_RESULT_DIR/logs/; its path is stored in
    the record.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    log_dir = BENCHMARK_RESULT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    backend_label = backend.replace("/", "_")
    log_file = log_dir / f"{test_name}_{backend_label}_{timestamp}.log"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="diffusion_bench_tmp_", delete=False) as tmp:
        tmp_result_file = Path(tmp.name)

    exclude_keys = {"baseline", "dataset", "task", "name", "skip-performance-assertion"}

    cmd = [
        sys.executable,
        BENCHMARK_SCRIPT,
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        model,
        "--backend",
        backend,
        "--dataset",
        params.get("dataset", "random"),
        "--task",
        params.get("task", "t2i"),
        "--output-file",
        str(tmp_result_file),
    ]

    for key, value in params.items():
        if key in exclude_keys or value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        elif isinstance(value, (dict, list)):
            cmd.extend([flag, json.dumps(value, separators=(",", ":"))])
        else:
            cmd.extend([flag, str(value)])

    # Insert -u so the subprocess runs with unbuffered stdout/stderr, ensuring
    # all print() output is flushed to the pipe immediately instead of being
    # held in Python's internal block-buffer until process exit (which can
    # cause truncated or out-of-order log output when stdout is piped).
    cmd = [cmd[0], "-u"] + cmd[1:]

    print(f"\nRunning benchmark (backend={backend}): {' '.join(cmd)}")
    print(f"  Log file: {log_file}")

    # Redirect stdout + stderr directly to the log file at the OS level
    # (equivalent to `cmd > log 2>&1`), so no output is ever lost regardless
    # of how the subprocess exits.  The log is echoed to the terminal afterwards.
    with open(log_file, "w", encoding="utf-8") as log_fh:
        log_fh.write(f"cmd: {' '.join(cmd)}\n\n")
        log_fh.flush()

        process = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=log_fh,
            cwd=str(Path(__file__).parent.parent.parent.parent),
        )
        process.wait()

    with open(log_file, encoding="utf-8") as log_fh:
        print(log_fh.read(), end="")

    if process.returncode != 0:
        tmp_result_file.unlink(missing_ok=True)
        print(f"ERROR:Benchmark script exited with code {process.returncode}")

    if not tmp_result_file.exists():
        with open(DIFFUSION_RESULT_TEMPLATE_PATH, encoding="utf-8") as f:
            template_payload = json.load(f)
        # Template schema is fixed and owned by this repo:
        # ``diffusion_result_template.json`` is a one-item list and metrics live at [0]["result"].
        template_metrics: dict[str, Any] = template_payload[0]["result"]
        with open(tmp_result_file, "w", encoding="utf-8") as f:
            json.dump(template_metrics, f, ensure_ascii=False, indent=2)
        print(f"Benchmark result file not generated, fallback to template: {tmp_result_file}")

    try:
        with open(tmp_result_file, encoding="utf-8") as f:
            metrics: dict[str, Any] = json.load(f)
    finally:
        tmp_result_file.unlink(missing_ok=True)

    server_cfg = server_cfg or {}
    server_type = cast(str, server_cfg.get("server_type", "vllm-omni"))
    serve_args_dict = server_cfg.get("serve_args_dict", {})
    if not isinstance(serve_args_dict, dict):
        serve_args_dict = {}

    completed = metrics.get("completed_requests", metrics.get("completed", 0))
    failed = metrics.get("failed_requests", metrics.get("failed", 0))

    record: dict[str, Any] = {
        "test_name": test_name,
        "backend": backend,
        "timestamp": timestamp,
        "server_params": server_cfg.get("server_params"),
        "benchmark_params": params,
        "result": metrics,
        "log_file": str(log_file),
        "Model": model,
        "Framework": server_type,
        "API Backend": backend,
        "Hardware": "",
        "Deployment": "",
        "Task": params.get("task", "t2i"),
        "Dataset": params.get("dataset", "random"),
        "resolution": _to_resolution_string(params),
        "Parallelism": _to_parallelism_string(server_type, serve_args_dict),
        "max_concurrency": params.get("max-concurrency", ""),
        "Cache": _to_cache_string(server_type, serve_args_dict),
        "Quantization": _to_quantization_value(server_type, serve_args_dict),
        "offload": _to_offload_string(server_type, serve_args_dict),
        "compile": _to_compile_value(server_type, serve_args_dict),
        "Attn_backend": os.environ.get("DIFFUSION_ATTENTION_BACKEND", ""),
        "num_inference_steps": params.get("num-inference-steps", ""),
        "completed": completed,
        "failed": failed,
        "throughput_qps": metrics.get("throughput_qps"),
        "latency_mean": metrics.get("latency_mean"),
        "latency_median": metrics.get("latency_median"),
        "latency_p99": metrics.get("latency_p99"),
        "latency_p95": metrics.get("latency_p95"),
        "latency_p50": metrics.get("latency_p50"),
        "peak_memory_mb_max": metrics.get("peak_memory_mb_max"),
        "peak_memory_mb_mean": metrics.get("peak_memory_mb_mean"),
        "peak_memory_mb_median": metrics.get("peak_memory_mb_median"),
        "stage_durations_mean": metrics.get("stage_durations_mean"),
        "stage_durations_p50": metrics.get("stage_durations_p50"),
        "stage_durations_p99": metrics.get("stage_durations_p99"),
        "commit_sha": _get_branchpoint_commit_sha(),
        "build_id": os.environ.get("BUILDKITE_BUILD_ID", ""),
        "build_url": os.environ.get("BUILDKITE_BUILD_URL", ""),
        "source_file": source_file,
    }
    _append_to_aggregated_file(record)
    print(f"\n  Result appended to: {AGGREGATED_RESULT_FILE}")
    print(f"  Log saved to:       {log_file}")

    return metrics


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def assert_result(result: dict[str, Any], params: dict[str, Any]) -> None:
    """Assert that benchmark metrics satisfy the configured baselines."""
    num_prompts = params.get("num-prompts", 10)
    completed = result.get("completed_requests", result.get("completed", 0))
    assert completed == num_prompts, f"Expected {num_prompts} completed requests, got {completed}"

    if params.get("skip-performance-assertion", False):
        print("Skipping performance assertions.")
        return

    for metric, threshold in params.get("baseline", {}).items():
        current = result.get(metric)
        assert current is not None, f"Metric '{metric}' not found in result: {list(result.keys())}"
        if "throughput" in metric:
            assert current >= threshold, f"{metric}: {current:.4f} < baseline {threshold}"
        else:
            assert current <= threshold, f"{metric}: {current:.4f} > baseline {threshold}"


def _default_benchmark_backend_for_task(task: str) -> str:
    """Return the default client-side benchmark backend for a diffusion task."""
    if task in {"t2v", "i2v", "ti2v"}:
        return "v1/videos"
    if task in {"t2i", "i2i", "ti2i"}:
        return "vllm-omni"
    raise ValueError(f"Unsupported task for benchmark backend resolution: {task}")


def _resolve_benchmark_backend(server_cfg: dict[str, Any], params: dict[str, Any]) -> str:
    """Resolve which serving API the benchmark client should call."""
    configured = server_cfg.get("benchmark_backend")
    if configured:
        return cast(str, configured)
    return _default_benchmark_backend_for_task(cast(str, params.get("task", "t2i")))


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------
@pytest.mark.benchmark
@pytest.mark.parametrize(
    "diffusion_server",
    server_params,
    ids=[p["test_name"] for p in server_params],
    indirect=True,
)
@pytest.mark.parametrize("benchmark_params", benchmark_indices, indirect=True)
def test_diffusion_performance_benchmark(diffusion_server, benchmark_params):
    """Run the diffusion performance benchmark and assert against baselines.

    One server is started per unique parallel configuration (module scope).
    For each server, all benchmark parameter sets defined in the config JSON
    are executed sequentially; results are asserted against the baselines.

    Tracked metrics:
        - throughput_qps          (higher is better)
        - latency_p50, latency_p99 (lower is better)
    """
    test_name = benchmark_params["test_name"]
    params = benchmark_params["params"]
    server_cfg = getattr(diffusion_server, "server_cfg", {})
    backend = _resolve_benchmark_backend(server_cfg, params)

    result = run_benchmark(
        host=diffusion_server.host,
        port=diffusion_server.port,
        model=diffusion_server.model,
        params=params,
        test_name=test_name,
        backend=backend,
        server_cfg=server_cfg,
        source_file=cast(str, CONFIG_FILE_PATH),
    )

    print(f"\n{'=' * 60}")
    print(f"Results for {test_name} (server={diffusion_server.server_type}, backend={backend}):")
    for key in (
        "throughput_qps",
        "latency_mean",
        "latency_median",
        "latency_p50",
        "latency_p99",
        "peak_memory_mb_max",
        "peak_memory_mb_mean",
        "peak_memory_mb_median",
    ):
        if key in result:
            print(f"  {key}: {result[key]:.4f}")

    print(f"\n  Aggregated results: {AGGREGATED_RESULT_FILE}")
    print("=" * 60)

    assert_result(result, params)
