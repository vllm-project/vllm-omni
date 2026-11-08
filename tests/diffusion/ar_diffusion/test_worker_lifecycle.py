# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeARDiffusionRunner:
    def __init__(self) -> None:
        self.resets: list[str] = []
        self.closes: list[str] = []

    def reset_session(self, session_id: str) -> None:
        self.resets.append(session_id)

    def close_session(self, session_id: str) -> None:
        self.closes.append(session_id)


class FakeRegularRunner:
    pass


def make_worker(runner: object) -> DiffusionWorker:
    worker = object.__new__(DiffusionWorker)
    worker.model_runner = runner
    return worker


def test_worker_delegates_ar_session_lifecycle_to_runner() -> None:
    runner = FakeARDiffusionRunner()
    worker = make_worker(runner)

    assert worker.reset_ar_diffusion_session("world-1") is True
    assert worker.close_ar_diffusion_session("world-1") is True
    assert runner.resets == ["world-1"]
    assert runner.closes == ["world-1"]


def test_worker_reports_unsupported_runner_without_model_assumptions() -> None:
    worker = make_worker(FakeRegularRunner())

    assert worker.reset_ar_diffusion_session("world-1") is False
    assert worker.close_ar_diffusion_session("world-1") is False


def test_worker_rejects_empty_session_id() -> None:
    worker = make_worker(FakeARDiffusionRunner())

    with pytest.raises(ValueError, match="session_id"):
        worker.close_ar_diffusion_session("")
