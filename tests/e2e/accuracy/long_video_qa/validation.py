# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Accuracy artifact and coverage checks for the DFX visual-session replay."""

from pathlib import Path


def assert_accuracy_result(result: dict, params: dict, expected: int) -> None:
    score = result["long_video_qa"]
    assert not score.get("artifact_error"), score.get("artifact_error")
    assert score["protocol"] == "continuous-visual-mcq-v1"
    assert score["samples"] == score["completed"] == expected
    assert score["valid_answers"] == expected, "Missing or invalid multiple-choice answers"
    items = score["items"]
    assert len(items) == expected
    if selected := params.get("long_video_ids"):
        assert {item["id"] for item in items} == set(selected)
    for item in items:
        assert item["input_frames"] > 0 and item["input_duration_s"] > 0
        assert item["video_sha256"] and item["annotation_sha256"]
        assert item["fps"] == params.get("long_video_fps", 1.0)
        assert item["max_edge"] == params.get("long_video_max_edge", 336)
    assert score["accuracy"] == sum(item["correct"] for item in items) / expected
    # The initial fixed samples are regression diagnostics. Calibrate a score
    # floor on an expanded, held-out selection before enabling a quality gate.


def write_accuracy_result(
    result: dict, *, args: list[str], num_prompts: int, returncode: int, test_name: str, name: str | None, when
) -> Path:
    from tests.e2e.accuracy.helpers import _acc_result_filename, dfx_accuracy_result_dir, write_dfx_accuracy_result

    def argument(key, default=None):
        return args[args.index(key) + 1] if key in args else default

    model = result.get("model_id") or argument("--model")
    dataset = argument("--dataset-name")
    path = dfx_accuracy_result_dir(Path(__file__).parent / "results") / _acc_result_filename(model, dataset, when=when)
    write_dfx_accuracy_result(
        path,
        test_name=test_name,
        name=name,
        model_id=model,
        dataset=dataset,
        dataset_path=argument("--dataset-path"),
        date=str(result.get("date") or when.isoformat()),
        hardware=str(result["Hardware"]),
        baseline={},
        branch=str(result["branch"]),
        num_prompts=num_prompts,
        max_concurrency=int(argument("--max-concurrency", "1")),
        benchmark_exit_code=returncode,
        generation={"completed": result.get("completed", 0), "failed": result.get("failed", 0)},
        accuracy=result.get("long_video_qa", {"status": "failed", "error": "No visual-session QA result"}),
    )
    return path
