"""Run B0 nonstreaming and F0 streaming, all LIBERO10 tasks once each."""

import argparse
import contextlib
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument("--client-python", type=Path, default=ROOT / ".venv-client/bin/python")
    parser.add_argument("--worker-gpu", type=int, default=6)
    parser.add_argument("--render-gpu", type=int, default=7)
    parser.add_argument("--worker-port", type=int, default=21680)
    parser.add_argument("--policy-port", type=int, default=21681)
    parser.add_argument("--cases", nargs="+", choices=("b0", "f0"), default=["b0", "f0"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    state = {"status": "preflight", "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}
    children = []
    results = {}

    def save(**updates):
        state.update(updates, updated_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
        write_json(run / "state.json", state)

    def spawn(command, name, env):
        write_json(run / f"{name}_command.json", {"command": command, "cwd": str(ROOT), "env": env})
        with (run / f"{name}.log").open("wb") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env={**os.environ, **env},
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        children.append(process)
        return process

    def stop(process):
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)

    def ready(port, endpoint, services, expected_model=None):
        deadline = time.monotonic() + 1800
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while time.monotonic() < deadline:
            if any(p.poll() is not None for p in services):
                raise RuntimeError(f"Service failed while waiting for {port}{endpoint}; see logs")
            try:
                with opener.open(f"http://127.0.0.1:{port}{endpoint}", timeout=5) as response:
                    info = json.load(response)
                    if expected_model is None or [item["id"] for item in info.get("data", [])] == [expected_model]:
                        return info
            except (OSError, ValueError):
                time.sleep(2)
        raise TimeoutError(f"Waiting for {port}{endpoint}")

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for port in (args.worker_port, args.policy_port):
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))
        for interpreter in (args.server_python, args.client_python):
            if not interpreter.is_file():
                raise FileNotFoundError(interpreter)
        sources = [ROOT / "vllm_omni/entrypoints/cosmos3/libero_remote_policy.py", Path(__file__)]
        sources += list((ROOT / "plugins/cosmos3_client/vllm_omni_cosmos3_client").glob("*.py"))
        write_json(
            run / "code_sha256.json",
            {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        )
        base_env = {
            "VLLM_PYTHON": str(args.server_python),
            "PYTHONPATH": str(ROOT),
            "LD_LIBRARY_PATH": str(args.server_python.parent.parent / "lib"),
            "VLLM_PLUGINS": "vllm_omni_register_models",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "HF_HOME": str(run / "cache/huggingface"),
            "HF_HUB_OFFLINE": "1",
            "TORCHINDUCTOR_CACHE_DIR": str(run / "cache/inductor"),
        }
        monitor = spawn(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,memory.used,utilization.gpu",
                "--format=csv",
                "--loop=10",
            ],
            "gpu_usage",
            {},
        )
        for label, model_name, schedule, streaming in (
            ("b0", "cosmos3-edge-libero10-baseline-18000", "constant_euler", False),
            ("f0", "cosmos3-edge-libero10-faster-18000", "has_euler", True),
        ):
            if label not in args.cases:
                continue
            case = run / label
            case.mkdir()
            model = ROOT / "models" / model_name
            identity = json.loads((model / "IMPORT_COMPLETE.json").read_text())
            assert identity["algorithm"] == ("FASTER/F0" if streaming else "baseline/B0")
            write_json(case / "model_identity.json", identity)
            save(status="loading", active_case=label)
            worker = spawn(
                ["bash", str(ROOT / "examples/cosmos3/run_worker.sh")],
                f"{label}_worker",
                {
                    **base_env,
                    "CUDA_VISIBLE_DEVICES": str(args.worker_gpu),
                    "COSMOS3_MODEL": str(model / "diffusers"),
                    "COSMOS3_WORKER_PORT": str(args.worker_port),
                    "COSMOS3_LOG_DIR": str(case / "server"),
                },
            )
            ready(args.worker_port, "/v1/models", [worker], expected_model=str(model / "diffusers"))
            command = [
                str(args.server_python),
                "-m",
                "vllm_omni.entrypoints.cosmos3.policy_server",
                "--vllm-base-url",
                f"http://127.0.0.1:{args.worker_port}",
                "--port",
                str(args.policy_port),
                "--model",
                str(model / "diffusers"),
                "--action-stats-path",
                str(model / "action_stats.json"),
                "--seed",
                "0",
                "--flow-shift",
                "1.0",
                "--action-time-schedule",
                schedule,
            ]
            if not streaming:
                command.append("--disable-streaming")
            policy = spawn(command, f"{label}_policy", {**base_env, "CUDA_VISIBLE_DEVICES": ""})
            info = ready(args.policy_port, "/info", [worker, policy])
            write_json(case / "server_info.json", info)
            assert info["model"] == str(model / "diffusers")
            assert info["early_stop_actions"] is None
            command = [
                "bash",
                str(ROOT / "examples/cosmos3/run_client.sh"),
                "--server_url",
                f"http://127.0.0.1:{args.policy_port}",
                "--num_trials_per_task",
                "1",
                "--task_ids",
                "",
                "--max_steps",
                "520",
                "--synthetic_warmup_requests",
                "2",
                "--timing_warmup_requests",
                "2",
            ]
            if streaming:
                command.append("--streaming")
            client = spawn(
                command,
                f"{label}_client",
                {
                    "COSMOS3_CLIENT_PYTHON": str(args.client_python),
                    "CUDA_VISIBLE_DEVICES": "",
                    "MUJOCO_EGL_DEVICE_ID": str(args.render_gpu),
                    "EVAL_OUT": str(case / "eval"),
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )
            save(status="evaluating", active_case=label, client_pid=client.pid)
            deadline = time.monotonic() + 7200
            while client.poll() is None:
                if worker.poll() is not None or policy.poll() is not None:
                    raise RuntimeError(f"{label} service stopped during evaluation")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{label} evaluation exceeded two hours")
                lines = (run / f"{label}_client.log").read_text(errors="replace").splitlines()
                progress = [line for line in lines if "| Episode " in line and "overall SR" in line]
                save(completed_episodes=len(progress), latest_episode=progress[-1] if progress else None)
                time.sleep(10)
            if client.returncode != 0:
                raise RuntimeError(f"{label} client exited {client.returncode}; see {label}_client.log")
            summary = json.loads((case / "eval/summary.json").read_text())
            assert summary["total_episodes"] == 10
            assert summary["accuracy"]["overall"]["error_episodes"] == 0
            assert summary["accuracy"]["overall"]["skipped_episodes"] == 0
            tasks = summary["task_results"]
            assert sorted(task["task_id"] for task in tasks) == list(range(10))
            assert all(len(task["episode_results"]) == 1 for task in tasks)
            results[label] = {
                "successes": summary["total_successes"],
                "episodes": 10,
                "streaming": streaming,
                "tasks": [
                    {"task_id": t["task_id"], "description": t["task_description"], **t["episode_results"][0]}
                    for t in tasks
                ],
            }
            write_json(run / "results.json", results)
            save(completed_cases=list(results))
            stop(policy)
            stop(worker)
        stop(monitor)
        save(status="complete", active_case=None)
    except BaseException as exc:
        save(status="failed", error=repr(exc))
        raise
    finally:
        for process in reversed(children):
            stop(process)


if __name__ == "__main__":
    main()
