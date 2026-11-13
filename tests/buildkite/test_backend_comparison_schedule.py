# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise PR-owned comparison routing with local stubs; never submit jobs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / ".buildkite/cuda/pipeline.yml"
COMMIT = "a" * 40


@pytest.fixture
def comparison_runner(tmp_path: Path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls = tmp_path / "calls"
    stub = bindir / "buildkite-agent"
    stub.write_text(
        "#!/bin/bash\n"
        'if [[ "$*" == *"--dry-run"* ]]; then cat "$TEST_TEMPLATE"; exit 0; fi\n'
        'printf "%s %s\\n" "${0##*/}" "$*" >> "$TEST_CALLS"\n'
        'if [[ "$*" == "pipeline upload" ]]; then cat > "$TEST_CALLS.json"; fi\n'
    )
    stub.chmod(0o755)
    template = tmp_path / "template.json"
    pipeline = yaml.safe_load((ROOT / ".buildkite/cuda/test-sglang-omni.yml").read_text())
    for step in pipeline["steps"]:
        for plugin in step.get("plugins", []):
            if "kubernetes" in plugin:
                plugin["github.com/buildkite-plugins/kubernetes-buildkite-plugin"] = plugin.pop("kubernetes")
    template.write_text(json.dumps(pipeline))
    git = bindir / "git"
    git.write_text('#!/bin/bash\nprintf "%s\\n" "$TEST_CHECKOUT"\n')
    git.chmod(0o755)
    # Set up the checked-out renderer path used by the real upload command.
    (bindir / "python3").symlink_to(sys.executable)
    (tmp_path / ".buildkite").symlink_to(ROOT / ".buildkite", target_is_directory=True)
    entry = yaml.safe_load(ENTRY.read_text())
    command = next(step for step in entry["steps"] if step["key"] == "upload-backend-comparison")["commands"][0]
    # Every dollar in the routing command is interpreted at job time, including the regex end anchor.
    assert "$" not in command.replace("$$", ""), "Unescaped dollar in Buildkite upload command"
    command = command.replace("$$", "$")
    env = {
        "PATH": f"{bindir}:/usr/bin:/bin",
        "TEST_CALLS": str(calls),
        "TEST_TEMPLATE": str(template),
        "TEST_CHECKOUT": COMMIT,
        "BACKEND_COMPARISON_ONLY": "1",
        "BUILDKITE_SOURCE": "schedule",
        "BUILDKITE_BRANCH": "refs/pull/1994/head",
        "BUILDKITE_PIPELINE_SLUG": "vllm-omni",
        "BUILDKITE_REPO": "https://github.com/vllm-project/vllm-omni.git",
        "BUILDKITE_COMMIT": COMMIT,
    }

    def run(overrides: dict[str, str], environment: dict | None = None):
        selected_command = command
        if environment is not None:
            lock = tmp_path / "environment.json"
            lock.write_text(json.dumps(environment))
            selected_command = (
                "set -euo pipefail; python3 .buildkite/cuda/scripts/comparison_smoke_pipeline.py --weekly "
                f"--environment-lock {lock} | buildkite-agent pipeline upload"
            )
        result = subprocess.run(
            ["/bin/bash", "-c", selected_command],
            cwd=tmp_path,
            env={**env, **overrides},
            capture_output=True,
            text=True,
        )
        return result, calls.read_text().splitlines() if calls.exists() else []

    return run


def test_schedule_first_uploads_only_cpu_image_builder(comparison_runner, tmp_path):
    result, calls = comparison_runner({})
    assert result.returncode == 0, result.stderr
    assert calls == ["buildkite-agent pipeline upload"]
    (job,) = json.loads((tmp_path / "calls.json").read_text())["steps"]
    assert job["key"] == "image-build-sglang-omni"
    assert job["agents"] == {"queue": "medium_cpu_queue_premerge"}
    assert job["commands"][-1] == "python3 .buildkite/cuda/scripts/build_weekly_comparison.py"


@pytest.mark.parametrize("toml_available", [False, True])
def test_weekly_cpu_installs_tomli_only_when_no_toml_parser_exists(comparison_runner, tmp_path, toml_available):
    result, _ = comparison_runner({})
    assert result.returncode == 0, result.stderr
    (job,) = json.loads((tmp_path / "calls.json").read_text())["steps"]
    bootstrap = job["commands"][-2]
    bindir = tmp_path / "bootstrap-bin"
    bindir.mkdir()
    stub = bindir / "python3"
    stub.write_text(
        '#!/bin/bash\nif [[ "$1" == "-c" ]]; then exit "$TEST_IMPORT_EXIT"; fi\n'
        'printf "%s\\n" "$*" > "$TEST_PIP_CALL"\n'
    )
    stub.chmod(0o755)
    pip_call = tmp_path / "pip-call"
    result = subprocess.run(
        ["/bin/bash", "-c", bootstrap],
        env={
            "PATH": f"{bindir}:/usr/bin:/bin",
            "TEST_IMPORT_EXIT": "0" if toml_available else "1",
            "TEST_PIP_CALL": str(pip_call),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert pip_call.exists() is not toml_available
    if not toml_available:
        assert pip_call.read_text().strip() == "-m pip install tomli==2.4.1"


def test_built_weekly_image_uploads_only_validated_sglang_workloads(comparison_runner, tmp_path):
    digest = "sha256:" + "c" * 64
    environment = {
        "image_reference": "public.ecr.aws/q9t5s3a7/vllm-ci-test-repo@" + digest,
        "image_digest": digest,
        "sglang_omni_revision": "b" * 40,
        "installed_packages": {"sglang": {"sglang": "0.5.23"}},
    }
    result, calls = comparison_runner({}, environment)
    assert result.returncode == 0, result.stderr
    assert calls == ["buildkite-agent pipeline upload"]
    pipeline = json.loads((tmp_path / "calls.json").read_text())
    assert pipeline["env"] == {"BACKEND_COMPARISON_COMMIT": "${BUILDKITE_COMMIT}"}
    jobs = {step["key"]: step for step in pipeline["steps"]}
    assert set(jobs) == {
        "nightly-sglang-omni-qwen3-omni",
        "nightly-sglang-omni-qwen3-tts",
        "perf-sglang-qwen-image-21",
        "perf-sglang-minicpm-o",
        "perf-sglang-cosmos3-t2i",
        "perf-sglang-hunyuanvideo15",
        "perf-sglang-minimax-h3",
        "perf-sglang-wan22-i2v",
    }
    for key, cards in (
        ("nightly-sglang-omni-qwen3-omni", 2),
        ("nightly-sglang-omni-qwen3-tts", 2),
        ("perf-sglang-qwen-image-21", 1),
        ("perf-sglang-minicpm-o", 1),
        ("perf-sglang-cosmos3-t2i", 2),
        ("perf-sglang-hunyuanvideo15", 1),
        ("perf-sglang-minimax-h3", 4),
        ("perf-sglang-wan22-i2v", 1),
    ):
        job = jobs[key]
        assert job["agents"] == {"queue": "mithril-h100-pool"}
        assert job["concurrency"] == 2
        assert job["concurrency_group"] == "vllm-omni/pr1994/sglang-smoke"
        container = next(iter(job["plugins"][0].values()))["podSpec"]["containers"][0]
        assert container["resources"]["limits"]["nvidia.com/gpu"] == cards
        if key in {
            "perf-sglang-qwen-image-21",
            "perf-sglang-cosmos3-t2i",
            "perf-sglang-hunyuanvideo15",
            "perf-sglang-minimax-h3",
            "perf-sglang-wan22-i2v",
        }:
            assert '.server_revision == "0.5.23"' in job["command"]
            assert "git -C /tmp apply" not in job["command"]
            prompts = 3 if key in {"perf-sglang-cosmos3-t2i", "perf-sglang-minimax-h3"} else 10
            assert f".completed == {prompts}" in job["command"]
        assert container["image"] == environment["image_reference"]
        assert job["depends_on"] == "image-build-sglang-omni"
        assert "commands" not in job
        assert "artifact download" in job["command"]
        assert job["env"]["COMPARISON_SGLANG_OMNI_REVISION"] == "b" * 40
        assert job["env"]["COMPARISON_SGLANG_REVISION"] == "0.5.23"
    assert ".completed == 32" in jobs["perf-sglang-minicpm-o"]["command"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"BACKEND_COMPARISON_ONLY": ""},
        {"BACKEND_COMPARISON_ONLY": "0"},
        {"BUILDKITE_SOURCE": "webhook"},
        {"BUILDKITE_SOURCE": "ui"},
        {"BUILDKITE_SOURCE": "api"},
        {"BUILDKITE_BRANCH": "main"},
        {"BUILDKITE_BRANCH": "congw729:ci/separate-image"},
        {"BUILDKITE_PIPELINE_SLUG": "vllm-omni-npu-ci"},
        {"BUILDKITE_REPO": "https://github.com/congw729/vllm-omni.git"},
        {"BUILDKITE_COMMIT": "HEAD"},
        {"TEST_CHECKOUT": "b" * 40},
        {"NIGHTLY": "1"},
        {"WEEKLY": "1"},
        {"NON_CRITICAL": "1"},
        {"MIRROR_HW": "B200"},
    ],
)
def test_invalid_comparison_stops_without_uploading_any_pipeline(comparison_runner, overrides):
    result, calls = comparison_runner(overrides)
    assert result.returncode != 0
    assert calls == []


@pytest.mark.parametrize(
    ("mode", "source", "cases", "success"),
    [
        ("smoke", "api", "qwen-image,cosmos3-nano", True),
        ("smoke", "webhook", "qwen-image", False),
        ("smoke", "schedule", "qwen-image", False),
        ("smoke", "api", "npu", False),
        ("smoke", "api", "", False),
    ],
)
def test_manual_smokes_require_explicit_models_and_api_source(
    comparison_runner, tmp_path, mode, source, cases, success
):
    result, calls = comparison_runner(
        {"BACKEND_COMPARISON_ONLY": mode, "BUILDKITE_SOURCE": source, "BACKEND_COMPARISON_CASES": cases}
    )
    assert (result.returncode == 0) == success, result.stderr
    assert calls == (["buildkite-agent pipeline upload"] if success else [])
    if success:
        jobs = json.loads((tmp_path / "calls.json").read_text())["steps"]
        for job in jobs[1:]:
            assert 'export PYTHONPATH="$$PWD$${PYTHONPATH:+:$$PYTHONPATH}"' in job["command"]
        assert [job["key"] for job in jobs] == [
            "image-build-sglang-omni",
            "smoke-sglang-qwen-image",
            "smoke-sglang-cosmos3-nano",
        ]


def test_minicpm_performance_runs_after_audio_smoke_on_one_gpu(comparison_runner, tmp_path):
    result, calls = comparison_runner(
        {"BACKEND_COMPARISON_ONLY": "smoke", "BUILDKITE_SOURCE": "api", "BACKEND_COMPARISON_CASES": "minicpm-o-perf"}
    )
    assert result.returncode == 0, result.stderr
    assert calls == ["buildkite-agent pipeline upload"]
    image, job = json.loads((tmp_path / "calls.json").read_text())["steps"]
    assert image["key"] == "image-build-sglang-omni"
    assert job["key"] == "perf-sglang-minicpm-o"
    plugin = job["plugins"][0]["github.com/buildkite-plugins/kubernetes-buildkite-plugin"]
    assert plugin["podSpec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == 1
    commands = job["command"]
    assert commands.startswith("set -eu")
    assert commands.index(".completed == 1") < commands.index("--test-config-file tests/dfx/perf/tests/")
    assert commands.index("--test-config-file tests/dfx/perf/tests/") < commands.index(".completed == 32")
    assert job["artifact_paths"] == [
        "tests/dfx/perf/results/minicpm-smoke/*.json",
        "tests/dfx/perf/results/result_test_sglang_minicpmo_4_5_*.json",
    ]


def test_existing_label_selected_comparison_still_uploads(comparison_runner):
    result, calls = comparison_runner(
        {"BACKEND_COMPARISON_ONLY": "", "BUILDKITE_BRANCH": "congw729:ci/separate-image", "BUILDKITE_SOURCE": "webhook"}
    )
    assert result.returncode == 0, result.stderr
    assert calls == ["buildkite-agent pipeline upload .buildkite/cuda/test-sglang-omni.yml"]


def test_shared_bootstrap_reaches_the_pr_owned_cuda_entry():
    # The installed init selects v2 only when this file is present. Recheck
    # this contract when rebasing: a new generator needs its own routing.
    assert not (ROOT / ".buildkite/.pipeline_gen_v2").exists()
    assert ENTRY.is_file()


@pytest.mark.parametrize("branch", ["main", "refs/pull/1994/head", "congw729:ci/separate-image"])
@pytest.mark.parametrize("flag", [None, "", "0", "1"])
@pytest.mark.parametrize("comparison_label", [False, True])
def test_only_one_entry_route_is_selected(branch: str, flag: str | None, comparison_label: bool):
    selected = []
    for step in yaml.safe_load(ENTRY.read_text())["steps"]:
        # Evaluate the boolean subset used by these actual YAML expressions;
        # values are fixed test inputs, and no builtins are available.
        expression = step["if"].replace('build.env("BACKEND_COMPARISON_ONLY")', repr(flag))
        expression = expression.replace(
            'build.pull_request.labels includes "backend-comparison-test"', repr(comparison_label)
        )
        expression = expression.replace("build.branch", repr(branch)).replace("null", "None")
        expression = expression.replace("&&", "and").replace("||", "or").replace("!(", "not (")
        if eval(expression, {"__builtins__": {}}):
            selected.append(step["key"])
    if flag or branch == "refs/pull/1994/head" or (branch != "main" and comparison_label):
        assert selected == ["upload-backend-comparison"]
    else:
        assert selected == ["upload-ci-pipeline"]


@pytest.mark.parametrize("checkout_matches", [False, True])
def test_child_jobs_verify_the_uploader_commit_before_work(tmp_path: Path, checkout_matches: bool):
    pipeline = yaml.safe_load((ROOT / ".buildkite/cuda/test-sglang-omni.yml").read_text())
    assert pipeline["env"] == {"BACKEND_COMPARISON_COMMIT": "${BUILDKITE_COMMIT}"}
    stub = tmp_path / "git"
    stub.write_text('#!/bin/bash\nprintf "%s\\n" "$TEST_CHECKOUT"\n')
    stub.chmod(0o755)
    for step in pipeline["steps"]:
        result = subprocess.run(
            ["/bin/bash", "-c", step["commands"][0].replace("$$", "$")],
            env={
                "PATH": f"{tmp_path}{os.pathsep}/usr/bin:/bin",
                "TEST_CHECKOUT": COMMIT if checkout_matches else "b" * 40,
                "BACKEND_COMPARISON_COMMIT": COMMIT,
            },
            capture_output=True,
            text=True,
        )
        assert (result.returncode == 0) == checkout_matches


def test_qwen_image_performance_uses_only_the_verified_image(comparison_runner, tmp_path):
    result, calls = comparison_runner(
        {
            "BACKEND_COMPARISON_ONLY": "smoke",
            "BUILDKITE_SOURCE": "api",
            "BACKEND_COMPARISON_CASES": "qwen-image-21-perf",
        }
    )
    assert result.returncode == 0, result.stderr
    assert calls == ["buildkite-agent pipeline upload"]
    (job,) = json.loads((tmp_path / "calls.json").read_text())["steps"]
    assert job["key"] == "perf-sglang-qwen-image-21"
    assert "depends_on" not in job
    (container,) = next(iter(job["plugins"][0].values()))["podSpec"]["containers"]
    assert container["image"].endswith("@" + job["env"]["COMPARISON_IMAGE_DIGEST"])
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 1
    (config,) = json.loads((ROOT / job["env"]["COMPARISON_CONFIG"]).read_text())
    reference = json.loads((ROOT / "tests/dfx/perf/tests/test_qwen_image_21_vllm_omni.json").read_text())[0]
    assert config["server_type"] == "sglang-diffusion"
    assert config["server_params"]["model"] == reference["server_params"]["model"]
    (workload,) = config["benchmark_params"]
    daily = reference["benchmark_params"][0]
    # Defaults resolved by the common client on build 18773.
    assert workload == {
        **daily,
        "seed": 0,
        "num_warmups": 2,
        "extra_body": {**daily["extra_body"], "return_stage_metrics": True},
    }


def test_wan_performance_runs_only_the_selected_video_workload(comparison_runner, tmp_path):
    result, calls = comparison_runner(
        {"BACKEND_COMPARISON_ONLY": "smoke", "BUILDKITE_SOURCE": "api", "BACKEND_COMPARISON_CASES": "wan22-i2v-perf"}
    )
    assert result.returncode == 0, result.stderr
    assert calls == ["buildkite-agent pipeline upload"]
    (job,) = json.loads((tmp_path / "calls.json").read_text())["steps"]
    assert job["key"] == "perf-sglang-wan22-i2v"
    assert job["concurrency"] == 2
    assert job["concurrency_group"] == "vllm-omni/pr1994/sglang-smoke"
    assert "depends_on" not in job
    (container,) = next(iter(job["plugins"][0].values()))["podSpec"]["containers"]
    assert container["image"].endswith("@" + job["env"]["COMPARISON_IMAGE_DIGEST"])
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert "validate_sglang_videos.py" in job["command"]
    assert "sglang-video-image-reference.patch" in job["command"]
    (config,) = json.loads((ROOT / job["env"]["COMPARISON_CONFIG"]).read_text())
    reference = json.loads((ROOT / "tests/dfx/perf/tests/test_wan22_i2v_vllm_omni.json").read_text())[0]
    daily = {key: value for key, value in reference["benchmark_params"][0].items() if key != "baseline"}
    assert config["benchmark_params"] == [
        {
            **daily,
            "seed": 0,
            "num_warmups": 2,
            "extra_body": {
                **daily["extra_body"],
                # Resolved from nightly #18955's Wan I2V defaults.
                "guidance_scale": 5.0,
                "guidance_scale_2": 5.0,
                "boundary_ratio": 0.875,
                "flow_shift": 5.0,
                "max_sequence_length": 512,
            },
        }
    ]
    assert config["server_params"]["model"] == reference["server_params"]["model"]


def test_cosmos_performance_only_runs_its_two_gpu_workload(comparison_runner, tmp_path):
    result, _ = comparison_runner(
        {"BACKEND_COMPARISON_ONLY": "smoke", "BUILDKITE_SOURCE": "api", "BACKEND_COMPARISON_CASES": "cosmos3-t2i-perf"}
    )
    assert result.returncode == 0, result.stderr
    (job,) = json.loads((tmp_path / "calls.json").read_text())["steps"]
    assert job["key"] == "perf-sglang-cosmos3-t2i"
    assert "depends_on" not in job
    (container,) = next(iter(job["plugins"][0].values()))["podSpec"]["containers"]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 2
    assert container["image"].endswith("@" + job["env"]["COMPARISON_IMAGE_DIGEST"])
    assert job["command"].index(".completed == 1") < job["command"].index(".completed == 3")


def test_video_diagnostics_reuse_image_and_apply_loader_fix_before_gpu(comparison_runner, tmp_path):
    result, _ = comparison_runner(
        {
            "BACKEND_COMPARISON_ONLY": "smoke",
            "BUILDKITE_SOURCE": "api",
            "BACKEND_COMPARISON_CASES": "hunyuanvideo15,minimax-h3",
        }
    )
    assert result.returncode == 0, result.stderr
    jobs = json.loads((tmp_path / "calls.json").read_text())["steps"]
    assert [j["key"] for j in jobs] == ["smoke-sglang-hunyuanvideo15", "smoke-sglang-minimax-h3"]
    for job, cards in zip(jobs, [1, 2]):
        assert "depends_on" not in job
        container = next(iter(job["plugins"][0].values()))["podSpec"]["containers"][0]
        assert container["resources"]["limits"]["nvidia.com/gpu"] == cards
        assert container["image"].endswith("@" + job["env"]["COMPARISON_IMAGE_DIGEST"])
        assert job["command"].startswith("set -eu")
        assert job["concurrency"] == 2
    assert jobs[0]["command"].index("test_sglang_diffusion_loader.py") < jobs[0]["command"].index(
        "run_sglang_diffusion_smoke.py"
    )


@pytest.mark.parametrize(("case", "cards", "prompts"), [("hunyuanvideo15", 1, 10), ("minimax-h3", 4, 3)])
def test_formal_video_jobs_validate_smoke_before_measuring(comparison_runner, tmp_path, case, cards, prompts):
    result, _ = comparison_runner(
        {"BACKEND_COMPARISON_ONLY": "smoke", "BUILDKITE_SOURCE": "api", "BACKEND_COMPARISON_CASES": f"{case}-perf"}
    )
    assert result.returncode == 0, result.stderr
    (job,) = json.loads((tmp_path / "calls.json").read_text())["steps"]
    assert job["key"] == f"perf-sglang-{case}"
    assert "depends_on" not in job
    (container,) = next(iter(job["plugins"][0].values()))["podSpec"]["containers"]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == cards
    assert job["concurrency"] == 2
    command = job["command"]
    assert command.startswith("set -eu")
    assert command.index(".completed == 1") < command.index("validate_sglang_videos.py")
    assert command.index("validate_sglang_videos.py") < command.index(
        f".completed == {prompts}", command.index("validate_sglang_videos.py")
    )
    if case == "hunyuanvideo15":
        assert command.index("validate_sglang_videos.py") < command.index("audit_diffusers_snapshot.py")
        assert command.index("audit_diffusers_snapshot.py") < command.index(
            ".completed == 10", command.index("audit_diffusers_snapshot.py")
        )
        assert "sglang-diffusers-revision.patch" in command
        assert "test_sglang_diffusers_revision.py" in command
        assert "COMPARISON_DIFFUSERS_PATCHES=1" in command
        assert "tests/dfx/perf/results/validated_hunyuanvideo15_config.json" in job["artifact_paths"]
    else:
        # Native MiniMax reports neither frame count nor fps in VideoResponse.
        # Validate the saved media, retaining only reported performance metrics.
        assert ".total_video_frames" not in command
        assert ".total_video_duration_s" not in command
        (config,) = json.loads((ROOT / job["env"]["COMPARISON_CONFIG"]).read_text())
        assert config["media_expectations"] == {
            "width": 1344,
            "height": 768,
            "frames": 209,
            "fps": 24,
            "require_audio": True,
            "audio_sample_rate": 32000,
            "audio_channels": 2,
            "audio_video_tolerance_s": 0.25,
        }
        assert "num_frames" not in config["benchmark_params"][0]["extra_body"]
