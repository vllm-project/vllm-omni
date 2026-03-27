"""
Performance benchmark CI runner for diffusion models.

Supports vLLM-Omni server backend:
  - vllm-omni (default): starts DiffusionServer via vllm_omni.entrypoints.cli.main,
    benchmarks with diffusion_benchmark_serving.py --backend vllm-omni

A config JSON file is REQUIRED via --config-file:
  pytest run_diffusion_benchmark.py --config-file tests/dfx/perf/tests/test_qwen_image_vllm_omni.json

JSON config entries use a "server_type" field, and this runner executes
the vllm-omni path.

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
from typing import Any

import psutil
import pytest

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

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


def _get_config_file_from_argv() -> str | None:
    """Read --config-file from sys.argv at import time so pytest parametrize can use it.

    pytest_addoption (below) registers the same flag so pytest does not reject it.
    Supports both ``--config-file path`` and ``--config-file=path`` forms.
    Returns None if the flag is not present; callers must handle the missing case.
    """
    for i, arg in enumerate(sys.argv):
        if arg == "--config-file" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--config-file="):
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
AGGREGATED_RESULT_FILE = BENCHMARK_RESULT_DIR / f"benchmark_results_{_config_stem}_{_SESSION_TIMESTAMP}.json"


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


# Register --config-file with pytest so it does not reject the argument.
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--config-file",
        action="store",
        default=None,
        help=(
            "Path to the benchmark config JSON file (required). "
            "Example: --config-file tests/dfx/perf/tests/test_qwen_image_vllm_omni.json"
        ),
    )


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
        model: str,
        serve_args: list[str],
        *,
        port: int | None = None,
    ) -> None:
        self.model = model
        self.serve_args = serve_args
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


def _unique_server_params(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one server-config dict per unique test_name."""
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for cfg in configs:
        test_name = cfg["test_name"]
        if test_name in seen:
            continue
        seen.add(test_name)
        if cfg.get("server_type", "vllm-omni") != "vllm-omni":
            raise ValueError(f"Unsupported server_type in config: {cfg.get('server_type')}")
        result.append(
            {
                "test_name": test_name,
                "server_type": "vllm-omni",
                "model": cfg["server_params"]["model"],
                "serve_args": _build_serve_args(cfg["server_params"].get("serve_args", {})),
                "benchmark_backend": "vllm-omni",
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
    model = server_cfg["model"]
    serve_args = server_cfg["serve_args"]
    return DiffusionServer(model=model, serve_args=serve_args)


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
            server.server_params = server_cfg["server_params"]
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
    server_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run diffusion_benchmark_serving.py as a subprocess and return parsed metrics.

    The raw metrics are written to a temporary file by the subprocess.  After
    the run completes the metrics are merged with full metadata (test_name,
    backend, benchmark_params, timestamp) and appended to the session-wide
    aggregated JSON file (AGGREGATED_RESULT_FILE).  The temporary file is
    removed afterwards.  Subprocess stdout/stderr are tee'd to a .log file
    under BENCHMARK_RESULT_DIR/logs/; its path is stored in the record.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    log_dir = BENCHMARK_RESULT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{test_name}_{backend}_{timestamp}.log"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="diffusion_bench_tmp_", delete=False) as tmp:
        tmp_result_file = Path(tmp.name)

    exclude_keys = {"baseline", "dataset", "task", "name"}

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
        raise RuntimeError(f"Benchmark script exited with code {process.returncode}")

    if not tmp_result_file.exists():
        raise FileNotFoundError(f"Benchmark result file not found: {tmp_result_file}")

    try:
        with open(tmp_result_file, encoding="utf-8") as f:
            metrics: dict[str, Any] = json.load(f)
    finally:
        tmp_result_file.unlink(missing_ok=True)

    record: dict[str, Any] = {
        "test_name": test_name,
        "backend": backend,
        "timestamp": timestamp,
        "server_params": server_params,
        "benchmark_params": params,
        "result": metrics,
        "log_file": str(log_file),
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

    for metric, threshold in params.get("baseline", {}).items():
        current = result.get(metric)
        assert current is not None, f"Metric '{metric}' not found in result: {list(result.keys())}"
        if "throughput" in metric:
            assert current >= threshold, f"{metric}: {current:.4f} < baseline {threshold}"
        else:
            assert current <= threshold, f"{metric}: {current:.4f} > baseline {threshold}"


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------


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
    backend = diffusion_server.server_type  # "vllm-omni"

    result = run_benchmark(
        host=diffusion_server.host,
        port=diffusion_server.port,
        model=diffusion_server.model,
        params=params,
        test_name=test_name,
        backend=backend,
        server_params=diffusion_server.server_params,
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
