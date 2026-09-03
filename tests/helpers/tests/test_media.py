# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from concurrent.futures.process import BrokenProcessPool
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from tests.helpers import media

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _stub_transcribe(monkeypatch) -> dict:
    """Capture the kwargs the helper hands to whisper, with no model and no device."""
    captured: dict = {}

    class FakeModel:
        def transcribe(self, path, **kwargs):
            captured.update(kwargs)
            return {"text": "London"}

    monkeypatch.setitem(
        sys.modules,
        "whisper",
        SimpleNamespace(load_model=lambda size, device=None: FakeModel()),
    )
    # Keep the unit test off the accelerator probe so it stays hermetic.
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.platforms",
        SimpleNamespace(current_omni_platform=SimpleNamespace(is_available=lambda: False)),
    )
    return captured


def test_transcribe_forwards_requested_language(monkeypatch):
    captured = _stub_transcribe(monkeypatch)

    media._whisper_transcribe_in_current_process("/tmp/does-not-matter.wav", "small", language="en")

    assert captured["language"] == "en"


def test_transcribe_defaults_to_auto_language(monkeypatch):
    # Auto-detect must stay the default: forcing a language globally would break
    # the non-English audio tests (e.g. the Chinese Qwen3-Omni prompts).
    captured = _stub_transcribe(monkeypatch)

    media._whisper_transcribe_in_current_process("/tmp/does-not-matter.wav", "small")

    assert captured.get("language") is None


class _FakeExecutor:
    """Stand-in for ProcessPoolExecutor recording what was submitted and shut down."""

    def __init__(self, outcomes=()):
        self._outcomes = list(outcomes)
        self.submitted: list[tuple] = []
        self.shutdown_calls = 0

    def submit(self, fn, *args):
        self.submitted.append((fn, args))
        outcome = self._outcomes.pop(0) if self._outcomes else "London"

        def _result():
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        return SimpleNamespace(result=_result)

    def shutdown(self, wait=True):
        self.shutdown_calls += 1


def _patch_executors(monkeypatch, outcomes_per_executor=()) -> list[_FakeExecutor]:
    """Replace the pool constructor and hand back the fakes it produces, in creation order."""
    created: list[_FakeExecutor] = []

    def factory(*args, **kwargs):
        index = len(created)
        outcomes = outcomes_per_executor[index] if index < len(outcomes_per_executor) else ()
        executor = _FakeExecutor(outcomes)
        executor.init_kwargs = kwargs
        created.append(executor)
        return executor

    monkeypatch.setattr(media.concurrent.futures, "ProcessPoolExecutor", factory)
    return created


@pytest.fixture(autouse=True)
def _reset_transcriber_singletons():
    """The worker and its model cache are module-level state; do not leak them across tests."""
    media.release_audio_transcriber()
    media._WHISPER_MODELS.clear()
    yield
    media.release_audio_transcriber()
    media._WHISPER_MODELS.clear()


def test_bytes_entrypoint_forwards_language_to_subprocess(monkeypatch, tmp_path):
    """Cover the two hops the tests above skip: bytes -> file -> executor.submit.

    The real call crosses a spawn ProcessPoolExecutor, so capture what gets
    submitted rather than what the worker eventually does.
    """
    created = _patch_executors(monkeypatch)

    wav = tmp_path / "clip.wav"
    sf.write(wav, np.zeros(2400, dtype=np.float32), 24000)

    assert media.convert_audio_bytes_to_text(wav.read_bytes(), "small", "en") == "London"

    fn, args = created[0].submitted[0]
    assert fn is media._whisper_transcribe_in_current_process
    assert args[1:] == ("small", "en")


def test_parent_reuses_one_spawned_worker_across_calls(monkeypatch):
    """Two calls share one spawned worker.

    Pins the parent-side contract: the pool is constructed once, with
    ``spawn`` (CUDA is not fork-safe) and ``max_workers=1`` (one resident
    model). Reuse of the worker's *loaded model* is pinned by the worker-side
    cache test below, which is where the startup cost actually lives.
    """
    created = _patch_executors(monkeypatch)

    media.convert_audio_file_to_text("/tmp/a.wav")
    media.convert_audio_file_to_text("/tmp/b.wav")

    assert len(created) == 1
    assert len(created[0].submitted) == 2
    assert created[0].init_kwargs["max_workers"] == 1
    assert created[0].init_kwargs["mp_context"].get_start_method() == "spawn"


def test_release_forces_a_new_process_pool(monkeypatch):
    created = _patch_executors(monkeypatch)

    media.convert_audio_file_to_text("/tmp/a.wav")
    media.release_audio_transcriber()
    media.convert_audio_file_to_text("/tmp/b.wav")

    assert len(created) == 2
    assert created[0].shutdown_calls == 1


def test_worker_loads_each_model_size_once(monkeypatch):
    """The worker-side cache is what removes the repeated whisper.load_model cost."""
    loaded: list[str] = []

    class FakeModel:
        def transcribe(self, path, **kwargs):
            return {"text": "London"}

    def load_model(size, device=None):
        loaded.append(size)
        return FakeModel()

    monkeypatch.setitem(sys.modules, "whisper", SimpleNamespace(load_model=load_model))
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.platforms",
        SimpleNamespace(current_omni_platform=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setattr(media, "_serialize_whisper_model_download", lambda model_size: nullcontext())

    media._whisper_transcribe_in_current_process("/tmp/a.wav", "small")
    media._whisper_transcribe_in_current_process("/tmp/b.wav", "small")

    assert loaded == ["small"]

    # The ASR escalation path asks the same worker for a stronger model.
    media._whisper_transcribe_in_current_process("/tmp/c.wav", "large-v3")

    assert loaded == ["small", "large-v3"]


@pytest.mark.parametrize(
    ("second_outcome", "recovers"),
    [(["London"], True), ([BrokenProcessPool("second")], False)],
)
def test_broken_pool_is_discarded_and_retried_once(monkeypatch, second_outcome, recovers):
    """A dead worker is discarded and the call retried exactly once.

    The retry either succeeds, or -- if the replacement also dies -- fails with
    both pools discarded and no executor left installed.
    """
    created = _patch_executors(
        monkeypatch,
        outcomes_per_executor=([BrokenProcessPool("first")], second_outcome),
    )

    if recovers:
        assert media.convert_audio_file_to_text("/tmp/a.wav") == "London"
    else:
        with pytest.raises(BrokenProcessPool):
            media.convert_audio_file_to_text("/tmp/a.wav")
        assert created[1].shutdown_calls == 1
        assert media._TRANSCRIBER is None

    assert len(created) == 2  # discarded the first, retried on a fresh one
    assert created[0].shutdown_calls == 1


@pytest.mark.parametrize(
    "error",
    [RuntimeError("CUDA out of memory"), KeyboardInterrupt()],
    ids=["Exception", "BaseException"],
)
def test_transcription_failure_propagates_and_discards_the_worker(monkeypatch, error):
    """Any failure -- an Exception or a BaseException like KeyboardInterrupt --
    propagates AND tears the worker down.

    It arrives as the task's own exception, not ``BrokenProcessPool``, but the
    worker -- and its resident model -- is still discarded, restoring the old
    one-process-per-call isolation. It is not retried; the next call builds a
    fresh worker.
    """
    created = _patch_executors(monkeypatch, outcomes_per_executor=([error],))

    with pytest.raises(type(error)):
        media.convert_audio_file_to_text("/tmp/a.wav")

    assert len(created) == 1  # not retried
    assert created[0].shutdown_calls == 1
    assert media._TRANSCRIBER is None

    assert media.convert_audio_file_to_text("/tmp/b.wav") == "London"
    assert len(created) == 2  # a fresh worker for the next call
