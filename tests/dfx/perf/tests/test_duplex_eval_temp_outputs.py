# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Lifetime and isolation of intermediate files in DFX accuracy runs."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def benchmark_context(monkeypatch, tmp_path):
    from tests.dfx.perf.scripts import run_benchmark as runner

    monkeypatch.chdir(tmp_path)
    params = {
        "dataset_name": "omni-duplex-eval",
        "duplex_eval_accuracy_config": "accuracy.json",
        "duplex_eval_response_root": "results/responses",
    }
    monkeypatch.setattr(runner, "server_to_benchmark_mapping", {"test_duplex": {"benchmark_params": [params]}})

    @dataclass
    class Request:
        param: tuple[str, int] = ("test_duplex", 0)

    def context():
        return contextmanager(runner.benchmark_params.__wrapped__)(Request())

    return context, params


@pytest.mark.parametrize("failure", [None, AssertionError, KeyboardInterrupt])
def test_temporary_outputs_survive_until_evaluation_finishes(benchmark_context, tmp_path, failure):
    context, params = benchmark_context
    root = tmp_path / params["duplex_eval_response_root"]
    previous = root / "previous-run" / "response.json"
    previous.parent.mkdir(parents=True)
    previous.write_text("previous response")
    summary = root.parent / "performance.json"
    summary.write_text("summary")

    with pytest.raises(failure) if failure else nullcontext():
        with context() as benchmark:
            temporary = Path(benchmark["params"]["duplex_eval_response_root"])
            assert temporary.is_absolute() and temporary.parent == root
            assert temporary != root
            response = temporary / "run-1" / "response.json"
            response.parent.mkdir()
            response.write_text("judge input")
            assert response.read_text() == "judge input"
            assert params["duplex_eval_response_root"] == "results/responses"
            if failure:
                raise failure("evaluation failed")

    assert not temporary.exists()
    assert previous.read_text() == "previous response"
    assert summary.read_text() == "summary"


def test_temporary_output_directories_are_isolated(benchmark_context):
    context, _ = benchmark_context
    with context() as first:
        first_root = Path(first["params"]["duplex_eval_response_root"])
        with context() as second:
            second_root = Path(second["params"]["duplex_eval_response_root"])
            assert first_root != second_root
            assert first_root.is_dir() and second_root.is_dir()
        assert first_root.is_dir() and not second_root.exists()
    assert not first_root.exists()


def test_benchmark_without_accuracy_guard_keeps_output_path(benchmark_context, tmp_path):
    context, params = benchmark_context
    del params["duplex_eval_accuracy_config"]
    with context() as benchmark:
        assert benchmark["params"]["duplex_eval_response_root"] == params["duplex_eval_response_root"]
        output = tmp_path / params["duplex_eval_response_root"] / "response.json"
        output.parent.mkdir(parents=True)
        output.write_text("retained output")
    assert output.read_text() == "retained output"
