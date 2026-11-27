# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import json

import pytest

from vllm_omni_cosmos3_client.acceptance import SUITES, compare, main


def create_run(path, duration, *, suite="libero_10", trials=1):
    path.mkdir(parents=True)
    config = dict(
        task_suite=suite,
        task_ids=list(range(10)),
        num_trials_per_task=trials,
        num_envs=1,
        seed=0,
        max_steps=520,
        action_horizon=16,
        warmup_steps=10,
        exclude_dummy_time=True,
        timing_warmup_requests=0,
        camera=["agentview", "wrist"],
        action_space="frame_wise_relative",
        rotation_space="6d",
        action_dim=10,
        initial_states_path="DEFAULT",
        debug_output_enabled=False,
        render_mode="remote",
        image_size=256,
        env_image_size=256,
        numba_disable_jit="1",
        server_info=dict(
            num_steps=10,
            guidance=1,
            flow_shift=1,
            seed=0,
            action_chunk_size=16,
            sampler_dtype="float32",
            action_normalization="quantile_rot",
        ),
    )
    tasks = [
        dict(task_id=i, episode_results=[dict(episode=j, success=True, executed_actions=16) for j in range(trials)])
        for i in range(10)
    ]
    summary = dict(
        run_config=config, total_episodes=10 * trials, total_successes=10 * trials, task_suite=suite, task_results=tasks
    )
    config["streaming"] = path.name == "faster"
    config["server_info"]["action_time_schedule"] = "has_euler" if config["streaming"] else "constant_euler"
    (path / "summary.json").write_text(json.dumps(summary))
    records, events = [], []
    for task, episode in [(i, j) for i in range(10) for j in range(trials)]:
        trace = f"{task}:{episode}"
        records.append(
            dict(trace_id=trace, clock_domain="boot", execution_window=16, batch_size=1, task_id=task, episode=episode)
        )
        events.extend(
            [
                dict(trace_id=trace, clock_domain="boot", stage="client.render_observation", start_s=0, end_s=1),
                dict(trace_id=trace, clock_domain="boot", stage="client.request", start_s=1, end_s=1 + duration / 2),
            ]
        )
        for _ in range(16):
            events.append(
                dict(
                    trace_id=trace,
                    clock_domain="boot",
                    stage="client.execute_action",
                    start_s=1 + duration / 2,
                    end_s=1 + duration,
                )
            )
    for name, rows in [("request_timings.jsonl", records), ("client_timeline.jsonl", events)]:
        (path / name).write_text("\n".join(json.dumps(r) for r in rows))


def test_target_is_a_real_gate_and_boundary_passes(tmp_path):
    base, faster = tmp_path / "base", tmp_path / "faster"
    create_run(base, 10)
    create_run(faster, 9)
    result = compare([base], [faster], minimum_windows=10, expected_suites=["libero_10"])
    assert result["passed"]
    assert result["aggregate"]["faster_p95_ms"] == 9000
    result = compare([base], [faster], minimum_windows=11, expected_suites=["libero_10"])
    assert not result["passed"]
    with pytest.raises(ValueError, match="baseline synchronous"):
        compare([faster], [base], minimum_windows=10, expected_suites=["libero_10"])


def test_config_mismatch_and_missing_timing_refuse_acceptance(tmp_path):
    base, faster = tmp_path / "base", tmp_path / "faster"
    create_run(base, 10)
    create_run(faster, 8)
    path = faster / "summary.json"
    summary = json.loads(path.read_text())
    summary["run_config"]["seed"] = 1
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="setting.*seed"):
        compare([base], [faster], minimum_windows=10, expected_suites=["libero_10"])
    (faster / "client_timeline.jsonl").write_text("")
    with pytest.raises(ValueError, match="Missing E2E"):
        compare([base], [faster], minimum_windows=10, expected_suites=["libero_10"])


def test_real_p95_regression_cannot_pass(tmp_path):
    base, faster = tmp_path / "base", tmp_path / "faster"
    create_run(base, 10)
    create_run(faster, 9.5)
    result = compare([base], [faster], minimum_windows=10, expected_suites=["libero_10"])
    assert not result["passed"]
    assert not result["checks"][-1]["passed"]
    assert result["aggregate"]["reduction_percent"] == pytest.approx(5)


def fail_episodes(path, number):
    target = path / "summary.json"
    value = json.loads(target.read_text())
    episodes = [ep for task in value["task_results"] for ep in task["episode_results"]]
    for episode in episodes[:number]:
        episode["success"] = False
    value["total_successes"] -= number
    target.write_text(json.dumps(value))


def test_a_normal_task_failure_is_retained_and_can_pass_the_tolerance(tmp_path):
    base, faster = tmp_path / "base", tmp_path / "faster"
    create_run(base, 10, trials=10)
    create_run(faster, 9, trials=10)
    fail_episodes(faster, 1)  # 1/100 = 1 pp: inclusive overall boundary
    result = compare(
        [base], [faster], minimum_windows=10, expected_suites=["libero_10"], success_bootstrap_samples=1000
    )
    assert result["passed"]
    assert result["quality"]["overall"]["observed"]["success_rate_change_pp"] == -1
    assert result["aggregate"]["faster_windows"] == 100
    assert result["suites"][0]["faster_successes"] == 99
    strict = compare(
        [base],
        [faster],
        minimum_windows=10,
        expected_suites=["libero_10"],
        overall_success_drop_pp=0,
        suite_success_drop_pp=0,
        success_bootstrap_samples=1000,
    )
    assert not strict["passed"]


@pytest.mark.parametrize(
    "losses, failed_check",
    [
        ([3, 0, 0, 0], "libero_spatial/success_rate_tolerance"),
        ([2, 1, 1, 1], "overall/success_rate_tolerance"),
    ],
)
def test_overall_and_suite_limits_are_independent(tmp_path, losses, failed_check):
    bases, faster = [], []
    for suite, failures in zip(SUITES, losses):
        b, f = tmp_path / suite / "base", tmp_path / suite / "faster"
        create_run(b, 10, suite=suite, trials=10)
        create_run(f, 9, suite=suite, trials=10)
        fail_episodes(f, failures)
        bases.append(b)
        faster.append(f)
    result = compare(bases, faster, minimum_windows=10, success_bootstrap_samples=1000)
    assert not result["passed"] and failed_check in result["failed_checks"]
    if losses[0] == 3:
        assert result["quality"]["overall"]["observed"]["passed"]
    else:
        assert all(s["observed"]["passed"] for s in result["quality"]["suites"])


def test_confidence_mode_requires_evidence_not_just_equal_observed_rates(tmp_path):
    base, faster = tmp_path / "base", tmp_path / "faster"
    create_run(base, 10)
    create_run(faster, 9)
    result = compare(
        [base],
        [faster],
        minimum_windows=10,
        expected_suites=["libero_10"],
        success_policy="confidence",
        success_bootstrap_samples=1000,
    )
    assert result["quality"]["observed_tolerance_passed"]
    assert not result["quality"]["statistical_noninferiority_supported"]
    assert not result["passed"]
    assert "overall/success_rate_confidence" in result["failed_checks"]


def test_cli_prints_quality_failures_and_preserves_exit_code(tmp_path, monkeypatch, capsys):
    argv = ["acceptance", "--minimum-windows", "10", "--success-bootstrap-samples", "1000"]
    for suite in SUITES:
        b, f = tmp_path / suite / "base", tmp_path / suite / "faster"
        create_run(b, 10, suite=suite)
        create_run(f, 9, suite=suite)
        if suite == "libero_spatial":
            fail_episodes(f, 1)
        argv.extend(["--baseline", str(b), "--faster", str(f)])
    target = tmp_path / "acceptance.json"
    argv.extend(["--output", str(target)])
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit) as ex:
        main()
    assert ex.value.code == 2
    terminal = json.loads(capsys.readouterr().out)
    stored = json.loads(target.read_text())
    assert not terminal["passed"] and terminal["failed_checks"] == stored["failed_checks"]
    assert "libero_spatial/success_rate_tolerance" in terminal["failed_checks"]


@pytest.mark.parametrize("field", ["error", "skipped"])
def test_runtime_errors_and_skips_are_not_success_tolerance(tmp_path, field):
    base, faster = tmp_path / "base", tmp_path / "faster"
    create_run(base, 10)
    create_run(faster, 9)
    path = faster / "summary.json"
    value = json.loads(path.read_text())
    value["task_results"][0]["episode_results"][0][field] = "failed" if field == "error" else True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Failed/skipped"):
        compare([base], [faster], minimum_windows=10, expected_suites=["libero_10"])
