# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from __future__ import annotations

import itertools
import logging
import threading
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from vllm_omni.metrics import duplex_frame_timing
from vllm_omni.metrics.duplex_frame_timing import (
    DuplexTickPacer,
    duplex_frame_timing_enabled,
    get_tick_pacer,
    log_frame_timing,
    pop_chunk_put_age_ms,
    record_chunk_put,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MODULE_LOGGER = "vllm_omni.metrics.duplex_frame_timing"


@contextmanager
def _capture_module_logs(caplog: pytest.LogCaptureFixture, level: int = logging.INFO):
    target = logging.getLogger(_MODULE_LOGGER)
    target.addHandler(caplog.handler)
    previous_level = target.level
    previous_propagate = target.propagate
    target.setLevel(level)
    # caplog also listens at the root; without this the same record lands
    # in caplog twice via propagation.
    target.propagate = False
    try:
        yield
    finally:
        target.removeHandler(caplog.handler)
        target.setLevel(previous_level)
        target.propagate = previous_propagate


def _lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records]


@pytest.fixture
def timing_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_OMNI_DUPLEX_FRAME_TIMING", "1")
    monkeypatch.setattr(duplex_frame_timing, "_put_stamps_ns", OrderedDict())
    monkeypatch.setattr(duplex_frame_timing, "_log_sequence", itertools.count(start=1))


def test_disabled_by_default_emits_nothing(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("VLLM_OMNI_DUPLEX_FRAME_TIMING", raising=False)

    assert not duplex_frame_timing_enabled()
    with _capture_module_logs(caplog):
        log_frame_timing("append", session="s1", jitter_ms=1.0)

    assert _lines(caplog) == []


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE"])
def test_flag_accepts_truthy_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("VLLM_OMNI_DUPLEX_FRAME_TIMING", value)
    assert duplex_frame_timing_enabled()


@pytest.mark.parametrize("value", ["", "0", "false", "off", "no"])
def test_flag_rejects_falsy_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("VLLM_OMNI_DUPLEX_FRAME_TIMING", value)
    assert not duplex_frame_timing_enabled()


def test_log_line_is_greppable_and_joinable(
    timing_enabled: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _capture_module_logs(caplog):
        log_frame_timing(
            "append",
            session="s1",
            epoch=3,
            bytes=7680,
            jitter_ms=1.25,
            drift_ms=None,
        )

    (line,) = _lines(caplog)
    assert line.startswith("DUPLEX_FRAME_TIMING event=append t_ns=")
    stamp = int(line.split("t_ns=")[1].split()[0])
    assert stamp > 0
    assert "session=s1" in line
    assert "epoch=3" in line
    assert "bytes=7680" in line
    assert "jitter_ms=1.250" in line
    assert "drift_ms=na" in line


def test_log_every_throttles_per_event_lines(
    monkeypatch: pytest.MonkeyPatch,
    timing_enabled: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("VLLM_OMNI_DUPLEX_FRAME_TIMING_LOG_EVERY", "3")

    with _capture_module_logs(caplog):
        for _ in range(7):
            log_frame_timing("audio_emit", session="s1")

    emitted = _lines(caplog)
    assert len(emitted) == 3  # events 1, 4 and 7 survive the 1-in-3 throttle


def test_tick_pacer_first_observation_has_no_jitter() -> None:
    pacer = DuplexTickPacer(0.08)

    jitter_s, drift_s = pacer.observe(now_ns=1_000_000_000)

    assert jitter_s is None
    assert drift_s == pytest.approx(0.0)


def test_tick_pacer_reports_jitter_and_drift() -> None:
    pacer = DuplexTickPacer(0.08)
    pacer.observe(now_ns=1_000_000_000)

    # 85 ms after the anchor: 5 ms late for an 80 ms tick.
    jitter_s, drift_s = pacer.observe(now_ns=1_000_000_000 + 85_000_000)

    assert jitter_s == pytest.approx(0.005)
    assert drift_s == pytest.approx(0.005)


def test_tick_pacer_scales_expected_interval_with_frame_count() -> None:
    pacer = DuplexTickPacer(0.08)
    pacer.observe(now_ns=1_000_000_000)

    # A 5-frame batch is on time if it lands 400 ms after the previous emit.
    jitter_s, drift_s = pacer.observe(now_ns=1_000_000_000 + 400_000_000, ticks=5)

    assert jitter_s == pytest.approx(0.0)
    assert drift_s == pytest.approx(0.0)


def test_tick_pacer_reports_stream_running_ahead_of_realtime() -> None:
    pacer = DuplexTickPacer(0.08)
    pacer.observe(now_ns=1_000_000_000)

    # 6 frames of audio delivered over 400 ms: one frame ahead of realtime,
    # reported as negative drift (positive would mean the stream is late).
    _, drift_s = pacer.observe(now_ns=1_000_000_000 + 400_000_000, ticks=6)

    assert drift_s == pytest.approx(-0.08)


def test_tick_pacer_reanchors_after_a_long_pause() -> None:
    pacer = DuplexTickPacer(0.08, reanchor_s=2.0)
    pacer.observe(now_ns=1_000_000_000)
    pacer.observe(now_ns=1_000_000_000 + 80_000_000)

    # A 4 s pause is a stream restart, not 50 dropped ticks: the anchor
    # re-bases so the following measurements stay meaningful.
    _, drift_s = pacer.observe(now_ns=5_000_000_000)
    assert drift_s == pytest.approx(0.0)

    jitter_s, drift_s = pacer.observe(now_ns=5_000_000_000 + 80_000_000)
    assert jitter_s == pytest.approx(0.0)
    assert drift_s == pytest.approx(0.0)


def test_get_tick_pacer_reuses_stream_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(duplex_frame_timing, "_tick_pacers", OrderedDict())

    first = get_tick_pacer("append", "s1", 0.08)
    assert get_tick_pacer("append", "s1", 0.08) is first
    assert get_tick_pacer("emit", "s1", 0.08) is not first
    # A changed tick period invalidates the cached cadence state.
    assert get_tick_pacer("append", "s1", 0.1) is not first


def test_get_tick_pacer_evicts_oldest_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(duplex_frame_timing, "_tick_pacers", OrderedDict())
    monkeypatch.setattr(duplex_frame_timing, "_MAX_TRACKED_STREAMS", 2)

    get_tick_pacer("append", "s1", 0.08)
    get_tick_pacer("append", "s2", 0.08)
    get_tick_pacer("append", "s3", 0.08)

    assert ("append", "s1") not in duplex_frame_timing._tick_pacers
    assert ("append", "s3") in duplex_frame_timing._tick_pacers


def test_chunk_handoff_age_uses_put_stamp(timing_enabled: None) -> None:
    record_chunk_put("r1_0_0", t_ns=1_000_000_000)

    age_ms = pop_chunk_put_age_ms("r1_0_0", now_ns=1_000_000_000 + 1_500_000)

    assert age_ms == pytest.approx(1.5)
    # The stamp is consumed: a second pop cannot double-report.
    assert pop_chunk_put_age_ms("r1_0_0") is None


def test_chunk_handoff_age_is_none_without_a_same_process_put(
    timing_enabled: None,
) -> None:
    assert pop_chunk_put_age_ms("never-put") is None


def test_record_chunk_put_requires_the_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_OMNI_DUPLEX_FRAME_TIMING", raising=False)
    monkeypatch.setattr(duplex_frame_timing, "_put_stamps_ns", OrderedDict())

    record_chunk_put("r1_0_0")

    assert pop_chunk_put_age_ms("r1_0_0") is None


def test_connector_put_site_emits_and_stamps(
    timing_enabled: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from vllm_omni.data_entry_keys import OmniPayloadStruct
    from vllm_omni.distributed.omni_connectors.transfer_adapter import chunk_transfer_adapter

    adapter = SimpleNamespace(
        connector=SimpleNamespace(stage_id=1, put=lambda **_kwargs: (True, 128, {})),
        custom_process_next_stage_input_func=lambda **_kwargs: OmniPayloadStruct(),
        _accepts_new_token_ids=lambda _processor: False,
        put_req_chunk=defaultdict(int),
        ramp_chunk_count=defaultdict(int),
        _sender_state_lock=threading.Lock(),
        _sender_tokens={},
        record_send_failure=lambda *_args, **_kwargs: None,
    )
    task = {
        "multimodal_output": None,
        "request": SimpleNamespace(request_id="internal-1", external_req_id="r1"),
        "is_finished": False,
        "is_segment_finished": False,
    }

    with _capture_module_logs(caplog):
        chunk_transfer_adapter.OmniChunkTransferAdapter._send_single_request_for_generation(adapter, task)

    (line,) = _lines(caplog)
    assert line.startswith("DUPLEX_FRAME_TIMING event=connector_put t_ns=")
    assert "key=r1_1_0" in line
    assert "ok=true" in line
    assert "bytes=128" in line
    assert "wrap_ms=" in line
    # The put side stamped the chunk for the same-process handoff age.
    assert pop_chunk_put_age_ms("r1_1_0") is not None


def test_stage1_decode_site_reports_frame_count(
    timing_enabled: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import torch
    from torch import nn

    from vllm_omni.model_executor.models.personaplex.personaplex_code2wav import PersonaPlexCode2Wav

    class _FakeStreamingMimi(nn.Module):
        def streaming_init(self, batch_size: int) -> None:
            pass

        def decode_frame(self, codes: torch.Tensor) -> torch.Tensor:
            return torch.zeros(codes.shape[-1] * 4, dtype=torch.float32)

        def reset_streaming(self) -> None:
            pass

    mimi_config = SimpleNamespace(num_codebooks=2, sample_rate=24000, samples_per_frame=4, mimi_name=None)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="/unused",
            hf_config=SimpleNamespace(mimi_config=mimi_config, mimi_name=None),
            duplex_max_sessions=1,
        ),
        device_config=SimpleNamespace(device="cpu"),
    )
    model = PersonaPlexCode2Wav(vllm_config=vllm_config)
    model.mimi = _FakeStreamingMimi()
    model._mimi_device = torch.device("cpu")

    # 2 codebooks x 3 frames, flattened codebook-major.
    codes = torch.arange(3, dtype=torch.long).repeat(2)

    with _capture_module_logs(caplog):
        model(input_ids=codes, request_ids=["req-1"])

    (line,) = _lines(caplog)
    assert line.startswith("DUPLEX_FRAME_TIMING event=stage1_decode t_ns=")
    assert "request_id=req-1" in line
    assert "frames=3" in line
    assert "decode_ms=" in line
