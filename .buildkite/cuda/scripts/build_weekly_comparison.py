# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Freeze one upstream main snapshot, build it on CPU, then upload weekly GPU jobs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

REPOSITORY = "https://github.com/sgl-project/sglang-omni.git"
IMAGE_REPOSITORY = "public.ecr.aws/q9t5s3a7/vllm-ci-test-repo"
RESOLUTION_KEY = "comparison-upstream-revision"
SOURCE_KEY = "comparison-source-lock"
ENVIRONMENT_KEY = "comparison-environment-lock"
LOCK_PATH = Path("tests/dfx/perf/results/comparison-environment.json")
SOURCE_PATH = Path("tests/dfx/perf/results/comparison-source.json")


def output(*command: str) -> str:
    return subprocess.check_output(command, text=True).strip()


def run(*command: str) -> None:
    subprocess.run(command, check=True)


def metadata(key: str) -> dict | None:
    result = subprocess.run(["buildkite-agent", "meta-data", "exists", key], check=False)
    if result.returncode == 100:
        return None
    result.check_returncode()
    return json.loads(output("buildkite-agent", "meta-data", "get", key))


def save_metadata(key: str, value: dict) -> None:
    run("buildkite-agent", "meta-data", "set", key, json.dumps(value, sort_keys=True))


def write_artifact(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    run("buildkite-agent", "artifact", "upload", str(path))


def snapshot_file(revision: str, path: str) -> str:
    url = f"https://raw.githubusercontent.com/sgl-project/sglang-omni/{revision}/{path}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode()


def resolve_main() -> str:
    revision, ref = output("git", "ls-remote", REPOSITORY, "refs/heads/main").split()
    if ref != "refs/heads/main" or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Upstream main did not resolve to a full commit SHA")
    return revision


def source_lock(build_id: str, runner_commit: str, revision: str) -> dict:
    pyproject = snapshot_file(revision, "pyproject.toml")
    project = tomllib.loads(pyproject)
    dockerfile = snapshot_file(revision, "docker/Dockerfile")
    arguments = dict(
        line.removeprefix("ARG ").split("=", 1)
        for line in dockerfile.splitlines()
        if line.startswith("ARG ") and "=" in line
    )
    base = arguments["SGLANG_IMAGE"]
    if "@sha256:" not in base or len(base.rsplit("@sha256:", 1)[1]) != 64:
        raise ValueError("Upstream Dockerfile must declare an immutable SGLang base image")
    dependencies = project["project"]["dependencies"]
    (sglang_requirement,) = [requirement for requirement in dependencies if requirement.startswith("sglang==")]
    return {
        "schema_version": 1,
        "build_id": build_id,
        "source_identity": {
            "pipeline_slug": os.environ["BUILDKITE_PIPELINE_SLUG"],
            "build_number": os.environ["BUILDKITE_BUILD_NUMBER"],
        },
        "upstream_pyproject_sha256": hashlib.sha256(pyproject.encode()).hexdigest(),
        "upstream_dockerfile_sha256": hashlib.sha256(dockerfile.encode()).hexdigest(),
        "runner_commit": runner_commit,
        "sglang_omni_revision": revision,
        "upstream_dependencies": dependencies,
        "sglang_requirement": sglang_requirement,
        "sglang_base_image": base,
        "ucx_commit": arguments["UCX_COMMIT"],
        "upstream_uv_overrides": project.get("tool", {}).get("uv", {}).get("override-dependencies", []),
        "image_tag": f"{IMAGE_REPOSITORY}:weekly-{build_id}-{runner_commit[:12]}-{revision[:12]}",
    }


def build_environment(source: dict) -> dict:
    tag = source["image_tag"]
    run(
        "docker",
        "build",
        "--progress=plain",
        "--file",
        "docker/Dockerfile.other",
        "--build-arg",
        f"SGLANG_OMNI_COMMIT={source['sglang_omni_revision']}",
        "--build-arg",
        f"SGLANG_BASE_IMAGE={source['sglang_base_image']}",
        "--build-arg",
        f"UCX_COMMIT={source['ucx_commit']}",
        "--tag",
        tag,
        ".",
    )
    packages = {}
    for name, interpreter in {
        "sglang": "/opt/sglang/bin/python3",
        "benchmark": "/opt/vllm-omni-benchmark/bin/python",
    }.items():
        packages[name] = json.loads(
            output(
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                interpreter,
                tag,
                "-c",
                "import importlib.metadata as m,json; "
                "print(json.dumps({d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()}))",
            )
        )
    required = {"sglang", "sglang-omni", "torch", "diffusers"}
    if not required <= packages["sglang"].keys():
        raise ValueError(f"Backend package inventory lacks {sorted(required - packages['sglang'].keys())}")
    declared_sglang = source["sglang_requirement"].split(";", 1)[0].removeprefix("sglang==").strip()
    if packages["sglang"]["sglang"] != declared_sglang:
        raise ValueError("Installed SGLang version does not match the upstream dependency declaration")
    installation = json.loads(
        output(
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "/opt/sglang/bin/python3",
            tag,
            "-c",
            "import importlib.metadata as m; print(m.distribution('sglang-omni').read_text('direct_url.json'))",
        )
    )
    if installation.get("vcs_info", {}).get("commit_id") != source["sglang_omni_revision"]:
        raise ValueError("Installed SGLang-Omni commit does not match the frozen upstream snapshot")
    sglang_installation = json.loads(
        output(
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "/opt/sglang/bin/python3",
            tag,
            "-c",
            """import importlib.metadata as metadata
import importlib.util
import json
import subprocess
from pathlib import Path
source = Path(importlib.util.find_spec('sglang').origin).resolve()
tracked = subprocess.run(
    ['git', '-C', str(source.parent), 'ls-files', '--error-unmatch', source.name],
    capture_output=True, text=True,
)
commit = None
if tracked.returncode == 0:
    commit = subprocess.check_output(
        ['git', '-C', str(source.parent), 'rev-parse', 'HEAD'], text=True,
    ).strip()
raw = metadata.distribution('sglang').read_text('direct_url.json')
print(json.dumps({'direct_url': json.loads(raw) if raw else None, 'source_commit': commit}))
""",
        )
    )
    runtime_sources = json.loads(
        output(
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "/opt/sglang/bin/python3",
            tag,
            "-c",
            "import hashlib,importlib.util,json; from pathlib import Path; "
            "files=[('sglang','multimodal_gen/runtime/pipelines/diffusers_pipeline.py'),"
            "('diffusers','pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5.py')]; "
            "print(json.dumps([{'package':package,'path':path,'sha256':hashlib.sha256("
            "(Path(importlib.util.find_spec(package).origin).parent/path).read_bytes()).hexdigest()} "
            "for package,path in files]))",
        )
    )
    patches = json.loads(
        output(
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "cat",
            tag,
            "/opt/sglang-omni-comparison-patches.json",
        )
    )
    run("docker", "push", tag)
    digests = json.loads(output("docker", "image", "inspect", "--format", "{{json .RepoDigests}}", tag))
    (reference,) = [digest for digest in digests if digest.startswith(IMAGE_REPOSITORY + "@sha256:")]
    return {
        **source,
        "image_reference": reference,
        "image_digest": reference.split("@", 1)[1],
        "installed_packages": packages,
        "sglang_omni_installation": installation,
        "sglang_installation": sglang_installation,
        "runtime_source_files": runtime_sources,
        "compatibility_patches": patches,
    }


def main() -> None:
    build_id = os.environ["BUILDKITE_BUILD_ID"]
    runner_commit = os.environ["BACKEND_COMPARISON_COMMIT"]
    if output("git", "rev-parse", "HEAD") != runner_commit:
        raise ValueError("Checkout does not match the comparison runner commit")
    source = metadata(SOURCE_KEY)
    if source is None:
        resolution = metadata(RESOLUTION_KEY)
        if resolution is None:
            resolution = {
                "build_id": build_id,
                "runner_commit": runner_commit,
                "sglang_omni_revision": resolve_main(),
            }
            # Freeze the successful main lookup before HTTP reads: a transient
            # snapshot download failure must not select a newer SHA on retry.
            save_metadata(RESOLUTION_KEY, resolution)
        if (resolution["build_id"], resolution["runner_commit"]) != (build_id, runner_commit):
            raise ValueError("Resolved upstream revision does not belong to this build and checkout")
        source = source_lock(build_id, runner_commit, resolution["sglang_omni_revision"])
        save_metadata(SOURCE_KEY, source)
    if (source["build_id"], source["runner_commit"]) != (build_id, runner_commit):
        raise ValueError("Source lock does not belong to this build and checkout")
    write_artifact(SOURCE_PATH, source)
    environment = metadata(ENVIRONMENT_KEY)
    if environment is None:
        environment = build_environment(source)
        save_metadata(ENVIRONMENT_KEY, environment)
    if any(environment[key] != value for key, value in source.items()):
        raise ValueError("Environment lock does not match the frozen source")
    write_artifact(LOCK_PATH, environment)
    pipeline = output(
        "python3",
        ".buildkite/cuda/scripts/comparison_smoke_pipeline.py",
        "--weekly",
        "--environment-lock",
        str(LOCK_PATH),
    )
    subprocess.run(["buildkite-agent", "pipeline", "upload"], input=pipeline, text=True, check=True)


if __name__ == "__main__":
    main()
