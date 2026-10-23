# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Asynchronous reference preparation for the MiniCPM-o 4.5 Code2Wav stage.

With ``token2wav_async_ref_prepare``, ``on_requests_added`` hands a
placeholder's reference to a worker thread that runs the host half of the
prompt (normalize, WAV, feature extraction) and touches no device. The next
forward or idle step queues the device half and pins the prompt to its
request; chunk 0 then finds the prompt prepared and must stay bit-exact with
the inline path. Without CUDA the device half runs on the current stream, so
these tests run on CPU with the platform check patched.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
import torch

from tests.model_executor.models.minicpmo_4_5.test_code2wav_batching import (
    _config,
    _enable_fake_ragged_kernel,
    _FakeToken2Wav,
    _forward,
    _info,
    _runtime_ref_info,
)
from vllm_omni.core.sched.output import OmniRequestPrewarm
from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import BatchedToken2Wav
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import MiniCPMO45Code2Wav
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_SR = 16000
_REF_A = torch.tensor([0.0, 0.25, -0.25, 0.0])
_REF_B = torch.tensor([0.0, 0.2])


class _SplitFakeToken2Wav(_FakeToken2Wav):
    """Fake Token2wav with the host/device split; records where each half ran."""

    def __init__(self):
        super().__init__()
        self.host_threads: list[int] = []
        self.device_calls = 0
        self.fail_host = False

    def _prepare_prompt_host(self, prompt_wav):
        self.host_threads.append(threading.get_ident())
        if self.fail_host:
            raise RuntimeError("injected host failure")
        assert Path(prompt_wav).is_file()
        return ("host", prompt_wav)

    def _prepare_prompt_device(self, host, *, non_blocking=False):
        assert host[0] == "host" and non_blocking
        self.device_calls += 1
        return self._prepare_prompt(host[1])


@pytest.fixture
def cuda_platform(monkeypatch):
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: True)


def _model(*, async_prepare: bool = True, capacity: int = 4) -> tuple[MiniCPMO45Code2Wav, _SplitFakeToken2Wav]:
    config = _config(runtime_prompt_cache_size=capacity)
    extra = config.model_config.stage_connector_config["extra"]
    extra["token2wav_async_ref_prepare"] = async_prepare
    token2wav = _SplitFakeToken2Wav()
    backend = BatchedToken2Wav(token2wav, setup_cache_size=1)
    _enable_fake_ragged_kernel(backend)
    model = MiniCPMO45Code2Wav(vllm_config=config)
    model.backend = backend
    return model, token2wav


def _prewarm(request_id: str, reference: torch.Tensor = _REF_A) -> OmniRequestPrewarm:
    return OmniRequestPrewarm(request_id=request_id, payload={"ref_audio": reference.clone(), "ref_audio_sr": _SR})


def _placeholder(request_id: str) -> dict:
    return {"request_id": request_id, "meta": {"request_id": request_id}}


def _finish_jobs(model: MiniCPMO45Code2Wav) -> None:
    for job in list(model._ref_jobs.values()):
        job.exception(timeout=10)


def _chunk0(model: MiniCPMO45Code2Wav, reference: torch.Tensor = _REF_A, request_id: str = "req-a"):
    return _forward(model, [_runtime_ref_info("external-id", reference)], request_ids=[request_id])


def _assert_no_pins(model: MiniCPMO45Code2Wav) -> None:
    assert not model._prefetch_pins
    assert not any(entry.pins for entry in model._runtime_prompts.values())


def test_reference_is_prepared_off_thread_and_chunk0_is_bit_exact(cuda_platform):
    model, token2wav = _model()
    baseline, _ = _model(async_prepare=False)
    assert model._async_ref_active()

    model.on_requests_added([_prewarm("req-a")])
    assert set(model._ref_jobs) == {"req-a"}
    assert not model._prefetch_queue  # the idle prefetch is replaced, not added to
    _finish_jobs(model)
    assert token2wav.host_threads and threading.get_ident() not in token2wav.host_threads
    assert token2wav.device_calls == 0

    # A placeholder step queues the device half and pins the prompt to the request.
    _forward(model, [_placeholder("req-a")], request_ids=["req-a"])
    assert token2wav.device_calls == 1
    key = model._prefetch_pins["req-a"]
    entry = model._runtime_prompts[key]
    assert (entry.owners, entry.pins) == (set(), {"req-a"})
    assert model.backend.has_prompt_features(entry.cache_id, entry.path)

    got = _chunk0(model).multimodal_outputs["model_outputs"]
    want = _chunk0(baseline).multimodal_outputs["model_outputs"]
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    # Chunk 0 hit the prepared prompt: no second extraction.
    assert token2wav.device_calls == 1
    assert model._ref_stats["chunk0_prepared_async"] == 1
    assert model._ref_stats["chunk0_inline"] == 0
    assert baseline._ref_stats["chunk0_inline"] == 1
    assert model._request_prompt_keys == {"req-a": key}
    _assert_no_pins(model)

    follow = _info("external-id", 1, [12, 13, 14])
    got = _forward(model, [follow], request_ids=["req-a"]).multimodal_outputs["model_outputs"]
    want = _forward(baseline, [follow], request_ids=["req-a"]).multimodal_outputs["model_outputs"]
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_idle_step_also_queues_the_device_half(cuda_platform):
    model, token2wav = _model()
    model.on_requests_added([_prewarm("req-a")])
    _finish_jobs(model)
    assert model.run_idle_prefetch() is True
    assert token2wav.device_calls == 1
    assert model.run_idle_prefetch() is False


def test_chunk0_before_the_job_finishes_runs_inline_and_the_late_result_is_dropped(cuda_platform, monkeypatch):
    model, token2wav = _model()
    release = threading.Event()
    original = token2wav._prepare_prompt_host

    def blocked_host(prompt_wav):
        release.wait(timeout=10)
        return original(prompt_wav)

    monkeypatch.setattr(token2wav, "_prepare_prompt_host", blocked_host)
    model.on_requests_added([_prewarm("req-a")])
    _chunk0(model)
    assert model._ref_stats["chunk0_inline"] == 1
    release.set()
    _finish_jobs(model)
    _forward(model, [_info("external-id", 1, [12, 13])], request_ids=["req-a"])
    assert model._ref_stats["async_late"] == 1
    assert not model._ref_jobs
    _assert_no_pins(model)


def test_failed_preparation_falls_back_to_inline(cuda_platform):
    model, token2wav = _model()
    baseline, _ = _model(async_prepare=False)
    token2wav.fail_host = True
    model.on_requests_added([_prewarm("req-a")])
    _finish_jobs(model)
    token2wav.fail_host = False
    got = _chunk0(model).multimodal_outputs["model_outputs"]
    want = _chunk0(baseline).multimodal_outputs["model_outputs"]
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert model._ref_stats["async_failed"] == 1
    assert model._ref_stats["chunk0_inline"] == 1
    _assert_no_pins(model)


def test_finishing_before_chunk0_cancels_the_job_and_releases_the_pin(cuda_platform):
    model, _ = _model()
    model.on_requests_added([_prewarm("req-a")])
    _finish_jobs(model)
    model.run_idle_prefetch()
    assert model._prefetch_pins
    model.on_requests_finished(["req-a"])
    assert not model._ref_jobs
    _assert_no_pins(model)


def test_shared_reference_installs_once_and_pins_per_request(cuda_platform):
    model, token2wav = _model()
    model.on_requests_added([_prewarm("req-a"), _prewarm("req-b")])
    _finish_jobs(model)
    model.run_idle_prefetch()
    assert token2wav.device_calls == 1
    assert set(model._prefetch_pins) == {"req-a", "req-b"}
    (entry,) = model._runtime_prompts.values()
    assert entry.pins == {"req-a", "req-b"}


def test_known_requests_and_invalid_payloads_are_not_submitted(cuda_platform, mocker):
    model, _ = _model()
    warn = mocker.patch("vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav.logger.warning")
    bad = OmniRequestPrewarm(request_id="req-bad", payload={"ref_audio": _REF_A.clone(), "ref_audio_sr": 0})
    model.on_requests_added([_prewarm("req-a"), _prewarm("req-a"), bad])
    assert set(model._ref_jobs) == {"req-a"}
    assert (model._ref_stats["prewarm_accepted"], model._ref_stats["prewarm_rejected"]) == (1, 1)
    warn.assert_called_once()


@pytest.mark.parametrize("case", ["knob-off", "not-cuda", "non-fp32-default"])
def test_inactive_cases_keep_the_idle_prefetch(case, monkeypatch):
    model, _ = _model(async_prepare=case != "knob-off")
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: case != "not-cuda")
    if case == "non-fp32-default":
        monkeypatch.setattr(torch, "get_default_dtype", lambda: torch.bfloat16)
    assert not model._async_ref_active()
    model.on_requests_added([_prewarm("req-a")])
    assert not model._ref_jobs
    assert set(model._prefetch_queue) == {"req-a"}
    assert model._ref_stats["async_unavailable"] == (0 if case == "knob-off" else 1)


def test_install_prompt_matches_prepare_prompt_and_is_cached():
    token2wav = _SplitFakeToken2Wav()
    backend = BatchedToken2Wav(token2wav, setup_cache_size=1)
    reference = BatchedToken2Wav(_FakeToken2Wav(), setup_cache_size=1)
    installed = backend.install_prompt("id", "/fake/prompt.wav", ("host", "/fake/prompt.wav"))
    want = reference.prepare_prompt("id", "/fake/prompt.wav")
    for name in ("speech_tokens", "speaker_embedding", "mels"):
        torch.testing.assert_close(getattr(installed, name), getattr(want, name), rtol=0, atol=0)
    assert backend.has_prompt_features("id", "/fake/prompt.wav")
    assert backend.prepare_prompt("id", "/fake/prompt.wav") is installed
    assert backend.install_prompt("id", "/fake/prompt.wav", ("host", "/fake/prompt.wav")) is installed
    assert token2wav.device_calls == 1
