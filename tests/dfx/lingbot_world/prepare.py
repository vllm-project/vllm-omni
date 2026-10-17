# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Provision pinned reference code and an isolated video evaluator environment."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

from benchmarks.lingbot_world.regression import RegressionError, read_json, sha256, verify_environment, write_json

ROOT = Path(__file__).resolve().parents[3]


def profile_for_case_shard(settings: dict, selection: str) -> dict:
    match = re.fullmatch(r"(\d+)/(\d+)", selection)
    if match is None:
        raise RegressionError("Case shard must be INDEX/COUNT")
    index, count = map(int, match.groups())
    cases = settings["cases"]
    if count < 1 or index >= count or count > len(cases):
        raise RegressionError("Case shard would be empty or outside the configured population")
    return {**settings, "cases": cases[index::count], "case_shard": {"index": index, "count": count}}


def select_cuda_devices(available: int, mask: str | None, *, dies: int, reserved_cards: int) -> str:
    """Use a fixed subset without releasing the remaining scheduler reservation."""
    for value in (available, dies, reserved_cards):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RegressionError("CUDA allocation counts must be positive integers")
    if available != reserved_cards or dies > reserved_cards:
        raise RegressionError(f"Expected {reserved_cards} reserved GPUs, found {available}; requested {dies} dies")
    identifiers = mask.split(",") if mask is not None else [str(index) for index in range(available)]
    identifiers = [identifier.strip() for identifier in identifiers]
    if len(identifiers) != available or len(set(identifiers)) != available or not all(identifiers):
        raise RegressionError("CUDA visibility does not match the scheduler allocation")
    return ",".join(identifiers[:dies])


def prepare_cuda_allocation(settings: dict, output: Path) -> str:
    if settings["platform"] != "cuda":
        raise RegressionError("CUDA allocation preparation requires a CUDA profile")
    import torch

    reserved = settings.get("reserved_cards", settings["cards"])
    visible = select_cuda_devices(
        torch.accelerator.device_count(),
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        dies=settings["dies"],
        reserved_cards=reserved,
    )
    write_json(
        output,
        {
            "reserved_cards": reserved,
            "visible_dies": settings["dies"],
            "visible_devices": visible,
            "devices": [
                {"name": torch.cuda.get_device_name(index), "uuid": str(torch.cuda.get_device_properties(index).uuid)}
                for index in range(reserved)
            ],
        },
    )
    return visible


def select_npu_devices(available: int, mask: str | None, *, dies: int, reserved_cards: int) -> str:
    """Preserve CANN runtime ordinals while keeping the scheduler reservation."""
    for value in (available, dies, reserved_cards):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RegressionError("NPU allocation counts must be positive integers")
    if available != reserved_cards or dies > reserved_cards:
        raise RegressionError(f"Expected {reserved_cards} reserved NPUs, found {available}; requested {dies} dies")
    identifiers = mask.split(",") if mask is not None else [str(index) for index in range(available)]
    identifiers = [identifier.strip() for identifier in identifiers]
    if (
        len(identifiers) != available
        or any(re.fullmatch(r"0|[1-9][0-9]*", identifier) is None for identifier in identifiers)
        or len(set(identifiers)) != available
    ):
        raise RegressionError("NPU visibility does not match the scheduler allocation")
    return ",".join(identifiers[:dies])


def prepare_npu_allocation(settings: dict, output: Path) -> str:
    if settings["platform"] != "npu":
        raise RegressionError("NPU allocation preparation requires an NPU profile")
    from vllm_omni.platforms import current_omni_platform as platform

    if platform.device_type != "npu":
        raise RegressionError("NPU allocation preparation requires the Ascend runtime")
    reserved = settings.get("reserved_cards", settings["cards"])
    available = platform.get_device_count()
    original = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    visible = select_npu_devices(available, original, dies=settings["dies"], reserved_cards=reserved)
    runtime_ids = original.split(",") if original is not None else [str(i) for i in range(available)]
    write_json(
        output,
        {
            "platform": "npu",
            "reserved_cards": reserved,
            "visible_dies": settings["dies"],
            "visible_devices": visible,
            # These are CANN runtime identifiers, not physical UUIDs.
            "devices": [
                {
                    "runtime_id": identifier.strip(),
                    "name": platform.get_device_name(index),
                    "total_bytes": platform.get_device_total_memory(index),
                }
                for index, identifier in enumerate(runtime_ids)
            ],
        },
    )
    return visible


def evaluator_cache_key(root: Path = ROOT) -> str:
    files = (root / "tests/dfx/lingbot_world/evaluators.lock.json", root / "requirements/lingbot-quality.txt")
    return hashlib.sha256(b"\0".join(path.read_bytes() for path in files)).hexdigest()[:16]


def checkout(path: Path, url: str, revision: str, *, submodules: bool = False) -> None:
    if not (path / ".git").exists():
        path.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "remote", "add", "origin", url], check=True)
        subprocess.run(["git", "-C", str(path), "fetch", "--depth=1", "origin", revision], check=True, timeout=600)
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", "FETCH_HEAD"], check=True)
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise RegressionError(f"Existing checkout is at {actual}, expected {revision}: {path}")
    if submodules:
        subprocess.run(
            ["git", "-C", str(path), "submodule", "update", "--init", "--recursive"], check=True, timeout=1200
        )


def _prepare_evaluators(root: Path) -> Path:
    key = evaluator_cache_key(ROOT)
    ready = root / "environment-ready"
    if ready.exists() and ready.read_text().strip() != key:
        raise RegressionError(
            "Evaluator locks changed; use a fresh evaluator root instead of mutating a live environment"
        )
    lock = read_json(ROOT / "tests/dfx/lingbot_world/evaluators.lock.json")
    for name, repository in lock["repositories"].items():
        checkout(root / name, repository["url"], repository["revision"], submodules=name in {"wbench", "vbench"})
    cached_weights = os.environ.get("LINGBOT_EVALUATOR_CACHE")
    for relative, digest in lock["files"].items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            cached = Path(cached_weights) / relative if cached_weights else None
            if cached is not None and cached.is_file():
                shutil.copyfile(cached, destination)
            else:
                # Write atomically; an interrupted download must not poison a
                # shared cache or pass a later preflight.
                partial = destination.with_name(destination.name + ".partial")
                with (
                    urllib.request.urlopen(lock["urls"][relative], timeout=300) as response,
                    partial.open("wb") as handle,
                ):
                    shutil.copyfileobj(response, handle)
                if sha256(partial) != digest:
                    partial.unlink()
                    raise RegressionError(f"Downloaded asset checksum mismatch: {relative}")
                partial.replace(destination)
        if sha256(destination) != digest:
            raise RegressionError(f"Evaluator cache checksum mismatch: {relative}")
    venv = root / "venv"
    if not ready.exists():
        # Only incomplete provisioning can be repaired. Once returned to a
        # scorer, this interpreter is immutable even when jobs share a cache.
        if not (venv / "bin/python").exists():
            subprocess.run(["uv", "venv", "--python", "3.12", str(venv)], check=True)
        subprocess.run(
            [
                "uv",
                "pip",
                "sync",
                "--python",
                str(venv / "bin/python"),
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu124",
                "--index-strategy",
                "unsafe-best-match",
                str(ROOT / "requirements/lingbot-quality.txt"),
            ],
            check=True,
            timeout=1800,
        )
    verify_environment(venv / "bin/python", ROOT / "requirements/lingbot-quality.txt")
    # Import the actual scoring stack before taking a GPU evaluator slot.
    # HPSv3 imports training helpers even during inference, so its dependencies
    # are not discovered by importing WBench's lazy evaluator alone.
    subprocess.run(
        [
            str(venv / "bin/python"),
            "-c",
            "import sys; sys.path[:0] = sys.argv[1:]; "
            "import clip, pyiqa; "
            "from vbench import VBench; "
            "from src.metrics.video_quality.evaluator import VideoQualityEvaluator; "
            "from hpsv3.inference import HPSv3RewardInferencer",
            str(root / "vbench"),
            str(root / "wbench"),
            str(root / "wbench/third_party/HPSv3"),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        check=True,
        timeout=180,
    )
    ready.write_text(key + "\n")
    return venv / "bin/python"


def prepare_evaluators(root: Path) -> Path:
    from filelock import FileLock

    root.parent.mkdir(parents=True, exist_ok=True)
    # CUDA and NPU evaluation jobs can share an HF cache on the same host.
    with FileLock(str(root) + ".lock", timeout=7200):
        return _prepare_evaluators(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("reference", "evaluators", "devices", "shard"))
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env-output", type=Path)
    parser.add_argument("--case-shard")
    args = parser.parse_args()
    if args.stage == "evaluators":
        print(prepare_evaluators(args.root.resolve()))
    elif args.stage == "shard":
        if args.profile is None or args.case_shard is None:
            parser.error("shard needs --profile and --case-shard")
        settings = profile_for_case_shard(read_json(args.profile), args.case_shard)
        settings["case_shard"]["source_sha256"] = sha256(args.profile)
        output = args.root.resolve() / "profile.json"
        write_json(output, settings)
        print(output)
    elif args.stage == "devices":
        if args.profile is None:
            parser.error("devices needs --profile")
        settings = read_json(args.profile)
        prepare = prepare_npu_allocation if settings["platform"] == "npu" else prepare_cuda_allocation
        print(prepare(settings, args.root.resolve() / "allocation.json"))
    else:
        if args.profile is None:
            parser.error("reference needs --profile")
        settings = read_json(args.profile)
        checkout(args.root.resolve(), "https://github.com/vllm-project/vllm-omni.git", settings["reference_revision"])
        model = os.environ.get("LINGBOT_MODEL_PATH")
        if model is None:
            lock = read_json(ROOT / "tests/dfx/lingbot_world/assets.lock.json")
            # Use the CLI so runtime hub access remains owned by the existing
            # model-loading stack, not another Python Hub API wrapper.
            model = (
                subprocess.check_output(
                    ["hf", "download", lock["model_id"], "--revision", lock["model_revision"], "--quiet"],
                    text=True,
                    timeout=7200,
                    # NPU CI defaults to offline inference. This explicit,
                    # pinned provisioning subprocess needs Hub access.
                    env={**os.environ, "HF_HUB_OFFLINE": "0"},
                )
                .strip()
                .splitlines()[-1]
            )
        if args.env_output:
            from tests.dfx.lingbot_world.resources import collect_resources

            model_path = Path(model).resolve()
            lock = read_json(ROOT / "tests/dfx/lingbot_world/assets.lock.json")
            weight = next(
                path
                for path in lock["model_files"]
                if path.startswith("transformer/") and path.endswith(".safetensors")
            )
            write_json(
                args.env_output, {"model_path": str(model_path), "resources": collect_resources(model_path / weight)}
            )


if __name__ == "__main__":
    main()
