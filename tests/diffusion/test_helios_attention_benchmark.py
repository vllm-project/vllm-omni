# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from benchmarks.diffusion.benchmark_helios_attention import normalize_stage_durations, one_worker_result, summarize_runs

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_summary_excludes_cold_warmup_and_keeps_frame_counts_separate():
    records: list[dict[str, bool | int | float]] = [
        {"warmup": True, "num_frames": 33, "wall_ms": 90000, "worker_peak_reserved_mib": 999},
        {"warmup": False, "num_frames": 33, "wall_ms": 100, "worker_peak_reserved_mib": 40},
        {"warmup": False, "num_frames": 33, "wall_ms": 200, "worker_peak_reserved_mib": 50},
        {"warmup": False, "num_frames": 66, "wall_ms": 500, "worker_peak_reserved_mib": 80},
    ]
    for row in records:
        row["transformer_gpu_ms"] = row["wall_ms"] / 2
        row["mean_transformer_forward_ms"] = row["wall_ms"] / 24
    summary = summarize_runs(records)
    assert summary[33]["median_wall_ms"] == 150
    assert summary[33]["count"] == 2
    assert summary[33]["median_transformer_gpu_ms"] == 75
    assert summary[33]["peak_worker_reserved_mib"] == 50
    assert summary[66]["median_wall_ms"] == 500
    assert summary[66]["count"] == 1


def test_warmup_only_run_does_not_report_a_benchmark():
    assert summarize_runs([{"warmup": True, "num_frames": 33, "wall_ms": 90000, "worker_peak_reserved_mib": 999}]) == {}


def test_rpc_unwrap_rejects_multiple_workers():
    assert one_worker_result([[{"steps": [1, 2]}]]) == {"steps": [1, 2]}
    with pytest.raises(ValueError, match="one worker"):
        one_worker_result([[{"steps": [1]}, {"steps": [2]}]])


def test_stage_durations_preserve_existing_milliseconds():
    assert normalize_stage_durations({"vae.decode": 1.5, "queue_wait_ms": 0.25, "stage_0_gen_ms": 23000.0}) == {
        "vae.decode": 1500.0,
        "queue_wait_ms": 0.25,
        "stage_0_gen_ms": 23000.0,
    }


def test_flash_bindings_describe_resolved_functions():
    from types import SimpleNamespace

    from benchmarks.diffusion.benchmark_helios_attention import describe_flash_bindings

    def forward():
        pass

    forward.__module__ = "fa3_fwd_interface"
    result = describe_flash_bindings(SimpleNamespace(flash_attn_func=forward, flash_attn_varlen_func=None))
    assert result["flash_attn_func"] == {"module": "fa3_fwd_interface", "qualname": forward.__qualname__}
    assert result["flash_attn_varlen_func"] is None


@pytest.mark.parametrize(
    "frames,steps,amplify,expected",
    [(33, [2, 2, 2], True, 12), (66, [2, 2, 2], True, 18), (66, [1, 2, 3], False, 12), (33, [1, 1, 1], True, 6)],
)
def test_expected_forwards_follow_sampling_config(frames, steps, amplify, expected):
    import runpy
    from pathlib import Path

    from benchmarks.diffusion.benchmark_helios_attention import GUIDANCE_SCALE, expected_transformer_forwards

    assert GUIDANCE_SCALE == 1.0  # This Distilled harness intentionally disables true CFG.
    extra = {"pyramid_num_inference_steps_list": steps, "is_amplify_first_chunk": amplify}
    assert expected_transformer_forwards(frames, extra, guidance_scale=GUIDANCE_SCALE) == expected
    # Load the real scheduler without importing the GPU pipeline from helios/__init__.py.
    path = Path(__file__).resolve().parents[2] / "vllm_omni/diffusion/models/helios/scheduling_helios.py"
    scheduler = runpy.run_path(str(path))["HeliosScheduler"](scheduler_type="dmd", stages=3)
    total = 0
    for chunk in range(frames // 33):
        for stage, num_steps in enumerate(steps):
            scheduler.set_timesteps(num_steps, stage_index=stage, is_amplify_first_chunk=bool(amplify and chunk == 0))
            total += len(scheduler.timesteps)
    assert total == expected


def test_forward_contract_rejects_cfg():
    from benchmarks.diffusion.benchmark_helios_attention import expected_transformer_forwards

    with pytest.raises(ValueError, match="guidance_scale=1"):
        expected_transformer_forwards(
            33, {"pyramid_num_inference_steps_list": [2, 2, 2], "is_amplify_first_chunk": True}, guidance_scale=2.0
        )


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--repeats", "0"],
        ["--warmup", "0"],
        ["--frames", "34"],
        ["--frames", "33", "33"],
        ["--seeds", "42", "42"],
        ["--backend", "INVALID"],
        ["--model", "/nonexistent/helios-checkpoint"],
    ],
)
def test_cli_contract(tmp_path, monkeypatch, extra):
    from benchmarks.diffusion.benchmark_helios_attention import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark",
            "--model",
            str(tmp_path),
            "--backend",
            "TORCH_SDPA",
            "--output-dir",
            str(tmp_path / "output"),
            *extra,
        ],
    )
    if extra:
        with pytest.raises(SystemExit) as exc:
            parse_args()
        assert exc.value.code == 2
    else:
        args = parse_args()
        assert args.frames == [33, 66]
        assert args.repeats == 3


def test_cli_preserves_existing_evidence(tmp_path, monkeypatch):
    from benchmarks.diffusion.benchmark_helios_attention import parse_args

    evidence = tmp_path / "results.json"
    evidence.write_text("original")
    monkeypatch.setattr(
        "sys.argv", ["benchmark", "--model", str(tmp_path), "--backend", "TORCH_SDPA", "--output-dir", str(tmp_path)]
    )
    with pytest.raises(SystemExit):
        parse_args()
    assert evidence.read_text() == "original"
