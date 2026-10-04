# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from vllm_omni.metrics import duplex_frame_timing
from vllm_omni.metrics.duplex_frame_timing import (
    DuplexTickPacer,
    duplex_frame_timing_enabled,
    frame_timing_clock,
    frame_timing_synchronize,
    get_tick_pacer,
    log_append_event,
    log_audio_emit_event,
    log_connector_get_event,
    log_frame_timing,
    log_stage1_decode_event,
    stamp_chunk_put,
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
    monkeypatch.setattr(duplex_frame_timing, "_DUPLEX_FRAME_TIMING_ENABLED", True)
    monkeypatch.setattr(duplex_frame_timing, "_event_counts", {})


@pytest.fixture
def timing_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(duplex_frame_timing, "_DUPLEX_FRAME_TIMING_ENABLED", False)


def test_disabled_by_default_emits_nothing(
    timing_disabled: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert not duplex_frame_timing_enabled()
    with _capture_module_logs(caplog):
        log_frame_timing("append", session="s1", jitter_ms=1.0)

    assert _lines(caplog) == []


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE"])
def test_flag_accepts_truthy_values(value: str) -> None:
    assert duplex_frame_timing._parse_flag(value)


@pytest.mark.parametrize("value", ["", "0", "false", "off", "no", None])
def test_flag_rejects_falsy_values(value: str | None) -> None:
    assert not duplex_frame_timing._parse_flag(value)


@pytest.mark.parametrize("raw,expected", [("1", 1), ("5", 5), ("0", 1), ("-3", 1), ("", 1), (None, 1), ("n/a", 1)])
def test_log_every_parsing_clamps_to_every_event(raw: str | None, expected: int) -> None:
    assert duplex_frame_timing._parse_log_every(raw) == expected


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
    monkeypatch.setattr(duplex_frame_timing, "_LOG_EVERY", 3)

    with _capture_module_logs(caplog):
        for _ in range(7):
            log_frame_timing("audio_emit", session="s1")

    emitted = _lines(caplog)
    assert len(emitted) == 3  # events 1, 4 and 7 survive the 1-in-3 throttle


def test_log_every_counts_per_event_name(
    monkeypatch: pytest.MonkeyPatch,
    timing_enabled: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(duplex_frame_timing, "_LOG_EVERY", 2)

    # Alternating sites must not starve each other: a single global
    # sequence would let one event type monopolize the surviving phase.
    with _capture_module_logs(caplog):
        for _ in range(4):
            log_frame_timing("connector_put", key="r1_1_0")
            log_frame_timing("connector_get", key="r1_1_0")

    emitted = _lines(caplog)
    assert len(emitted) == 4
    assert sum("event=connector_put" in line for line in emitted) == 2
    assert sum("event=connector_get" in line for line in emitted) == 2


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


def test_stamp_chunk_put_stamps_meta_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_omni.data_entry_keys import MetaStruct

    meta = MetaStruct()
    monkeypatch.setattr(duplex_frame_timing, "_DUPLEX_FRAME_TIMING_ENABLED", False)
    stamp_chunk_put(meta)
    # Unstamped meta keeps omit_defaults true, so the field stays off the
    # wire for consumers running any code version.
    assert meta.put_t_ns is None

    monkeypatch.setattr(duplex_frame_timing, "_DUPLEX_FRAME_TIMING_ENABLED", True)
    before_ns = time.monotonic_ns()
    stamp_chunk_put(meta)
    assert meta.put_t_ns is not None
    assert before_ns <= meta.put_t_ns <= time.monotonic_ns()


def test_append_event_reports_tick_period_with_default_fallback(
    monkeypatch: pytest.MonkeyPatch,
    timing_enabled: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(duplex_frame_timing, "_tick_pacers", OrderedDict())

    with _capture_module_logs(caplog):
        # Capabilities advertise 80 ms; a missing value falls back to the
        # duplex tick default rather than a per-site constant.
        log_append_event("s1", 2, 7680, 80)
        log_append_event("s1", 2, 7680, None)

    first, second = _lines(caplog)
    assert first.startswith("DUPLEX_FRAME_TIMING event=append t_ns=")
    assert "bytes=7680" in first
    assert "tick_period_ms=80.000" in first
    assert "tick_period_ms=80.000" in second


def test_append_event_skips_empty_reservations_and_the_disabled_flag(
    monkeypatch: pytest.MonkeyPatch,
    timing_enabled: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _capture_module_logs(caplog):
        log_append_event("s1", 2, 0, 80)

    assert _lines(caplog) == []

    monkeypatch.setattr(duplex_frame_timing, "_DUPLEX_FRAME_TIMING_ENABLED", False)
    with _capture_module_logs(caplog):
        log_append_event("s1", 2, 7680, 80)

    assert _lines(caplog) == []


def test_audio_emit_event_derives_frames_from_duration(
    monkeypatch: pytest.MonkeyPatch,
    timing_enabled: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(duplex_frame_timing, "_tick_pacers", OrderedDict())

    with _capture_module_logs(caplog):
        log_audio_emit_event("s1", 1, "req-1", {"audio_duration_ms": 240}, 80, sent_ms=160)

    (line,) = _lines(caplog)
    assert line.startswith("DUPLEX_FRAME_TIMING event=audio_emit t_ns=")
    assert "request_id=req-1" in line
    assert "frames=3" in line
    assert "audio_duration_ms=240.000" in line
    # Pre-emit playback watermark: with the stream's first emitted t_ns the
    # underrun margin is sent_ms - elapsed, joinable offline.
    assert "sent_ms=160" in line


def test_audio_emit_event_reports_na_watermark_without_a_playback_ledger(
    monkeypatch: pytest.MonkeyPatch,
    timing_enabled: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(duplex_frame_timing, "_tick_pacers", OrderedDict())

    with _capture_module_logs(caplog):
        log_audio_emit_event("s1", 1, "req-1", {"audio_duration_ms": 240}, 80)

    (line,) = _lines(caplog)
    assert "sent_ms=na" in line


@pytest.mark.parametrize(
    "audio_result",
    [None, "text-delta", {}, {"audio_duration_ms": 0}, {"audio_duration_ms": "n/a"}],
)
def test_audio_emit_event_skips_non_cadence_results(
    timing_enabled: None,
    caplog: pytest.LogCaptureFixture,
    audio_result: object,
) -> None:
    with _capture_module_logs(caplog):
        log_audio_emit_event("s1", 1, "req-1", audio_result, 80)

    assert _lines(caplog) == []


def test_connector_get_event_reports_chunk_age_from_put_stamp(
    timing_enabled: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _capture_module_logs(caplog):
        log_connector_get_event("r1_0_0", 1, 128, frame_timing_clock(), put_t_ns=time.monotonic_ns() - 1_500_000)

    (line,) = _lines(caplog)
    assert line.startswith("DUPLEX_FRAME_TIMING event=connector_get t_ns=")
    assert "bytes=128" in line
    assert "wrap_ms=" in line
    age_ms = float(line.split("chunk_age_ms=")[1].split()[0])
    # The stamp is a host-clock monotonic stamp, so the age crosses process
    # boundaries; here it is the 1.5 ms since the synthetic put.
    assert age_ms == pytest.approx(1.5, abs=0.5)


def test_connector_get_event_reports_na_without_a_put_stamp(
    timing_enabled: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _capture_module_logs(caplog):
        log_connector_get_event("r1_0_0", 1, 128, frame_timing_clock())

    (line,) = _lines(caplog)
    # Producer ran with the flag off: no stamp rode the chunk.
    assert "chunk_age_ms=na" in line


def test_frame_timing_synchronize_only_syncs_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_omni.platforms as platforms_module

    sync_calls: list[str] = []
    monkeypatch.setattr(
        platforms_module,
        "current_omni_platform",
        SimpleNamespace(synchronize=lambda: sync_calls.append("sync")),
    )

    monkeypatch.setattr(duplex_frame_timing, "_DUPLEX_FRAME_TIMING_ENABLED", False)
    frame_timing_synchronize()
    assert sync_calls == []

    monkeypatch.setattr(duplex_frame_timing, "_DUPLEX_FRAME_TIMING_ENABLED", True)
    frame_timing_synchronize()
    assert sync_calls == ["sync"]


def test_site_hooks_are_free_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(duplex_frame_timing, "_DUPLEX_FRAME_TIMING_ENABLED", False)

    assert frame_timing_clock() == 0.0
    with _capture_module_logs(caplog):
        log_stage1_decode_event("req-1", 5, frame_timing_clock(), 1)

    assert _lines(caplog) == []
    # No perf_counter stamp is taken while disabled, so wrap timings stay
    # free of clock reads on the disabled path.
    assert duplex_frame_timing_enabled() is False


def test_stage1_decode_event_maps_missing_request_id(
    timing_enabled: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _capture_module_logs(caplog):
        log_stage1_decode_event(None, 5, frame_timing_clock(), 3)

    (line,) = _lines(caplog)
    assert line.startswith("DUPLEX_FRAME_TIMING event=stage1_decode t_ns=")
    assert "request_id=unknown" in line
    assert "frames=5" in line
    assert "num_req=3" in line
    assert "decode_ms=" in line


def test_connector_put_site_emits_and_stamps(
    timing_enabled: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from vllm_omni.data_entry_keys import OmniPayloadStruct
    from vllm_omni.distributed.omni_connectors.transfer_adapter import chunk_transfer_adapter

    put_payloads: list[OmniPayloadStruct] = []

    def _fake_put(**kwargs: object) -> tuple[bool, int, dict]:
        put_payloads.append(kwargs["data"])
        return True, 128, {}

    adapter = SimpleNamespace(
        connector=SimpleNamespace(stage_id=1, put=_fake_put),
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
    # The outgoing chunk itself carries the put stamp the receiving process
    # will report as its chunk_age_ms.
    (payload,) = put_payloads
    assert payload.meta.put_t_ns is not None


def test_stage1_decode_site_reports_frame_count(
    monkeypatch: pytest.MonkeyPatch,
    timing_enabled: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import torch
    from torch import nn

    import vllm_omni.platforms as platforms_module
    from vllm_omni.model_executor.models.personaplex.personaplex_code2wav import PersonaPlexCode2Wav

    # The site closes its span with a platform host sync; stub it so the
    # test stays on CPU and records that the sync ran.
    sync_calls: list[str] = []
    monkeypatch.setattr(
        platforms_module,
        "current_omni_platform",
        SimpleNamespace(synchronize=lambda: sync_calls.append("sync")),
    )

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
    assert "num_req=1" in line
    assert "decode_ms=" in line
    assert sync_calls == ["sync"]
