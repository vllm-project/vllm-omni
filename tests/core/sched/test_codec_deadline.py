# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Deadline batching policy for the codec stage (``additional_config.codec_deadline``), pure Python."""

from __future__ import annotations

import pytest
import torch
from vllm_omni.core.sched.codec_deadline import (
    ChunkMeta,
    CodecDeadlineConfig,
    CodecDeadlinePolicy,
    LateBreaker,
    ReadyChunk,
    StepTimeModel,
    StreamLedger,
    build_codec_deadline_policy,
    graph_step_tiers,
    seed_step_s,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

NOW = 100.0


@pytest.fixture(autouse=True)
def _default_graph_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_CFM_GRAPH_GRID", raising=False)


def _meta(chunk_seq: int = 3, *, frames: int = 25, last: bool = False, turn_end: bool = False, epoch=0, cache=0):
    return ChunkMeta(
        chunk_seq=chunk_seq,
        cache_epoch=cache,
        last_chunk=last,
        turn_end=turn_end,
        codec_chunk_frames=frames,
        duplex_epoch=epoch,
    )


def _policy(**config: object) -> CodecDeadlinePolicy:
    return CodecDeadlinePolicy(CodecDeadlineConfig(enabled=True, **config))


def _stream(policy: CodecDeadlinePolicy, rid: str, *, play_end: float, anchor: float = 90.0, audio_s: float = 20.0):
    policy.ledgers[rid] = StreamLedger(key=(0, 0), anchor=anchor, play_end=play_end, audio_s=audio_s, d1=1.0)
    return ReadyChunk(rid, _meta())


def test_config_defaults_validation_and_parsing() -> None:
    config = CodecDeadlineConfig()
    assert config.enabled is False
    assert (config.margin_s, config.max_hold_s, config.max_rows, config.eager_rows) == (0.1, 0.35, 8, 4)
    assert CodecDeadlineConfig.from_additional_config({}) is None
    assert CodecDeadlineConfig.from_additional_config(None) is None
    parsed = CodecDeadlineConfig.from_additional_config({"codec_deadline": {"enabled": True, "max_rows": 16}})
    assert parsed is not None and parsed.enabled and parsed.max_rows == 16
    with pytest.raises(ValueError, match="unknown keys"):
        CodecDeadlineConfig.from_additional_config({"codec_deadline": {"enabld": True}})
    # The client buffer and the graph tiers are not configured here any more.
    for removed in ("prebuffer_s", "step_tiers", "step_seed_s", "trim_split_batches"):
        with pytest.raises(ValueError, match="unknown keys"):
            CodecDeadlineConfig.from_additional_config({"codec_deadline": {removed: 1}})
    with pytest.raises(ValueError, match="must be a boolean"):
        CodecDeadlineConfig(enabled=1)  # type: ignore[arg-type]


def test_chunk_meta_reads_tensor_fields_and_ignores_tts_is_last_chunk() -> None:
    info = {
        "meta": {
            "chunk_seq": torch.tensor(4),
            "cache_epoch": torch.tensor([1]),
            "last_chunk": torch.tensor(False),
            "turn_end": False,
            "codec_chunk_frames": 28,
            "duplex_epoch": 2,
            # True on every native duplex unit (flush_pending): not a last chunk.
            "tts_is_last_chunk": True,
        }
    }
    meta = ChunkMeta.from_info(info)
    assert meta == ChunkMeta(4, 1, False, False, 28, 2)
    assert ChunkMeta.from_info({}) is None
    assert ChunkMeta.from_info({"meta": {"request_id": "x"}}) is None


@pytest.mark.parametrize(
    "meta",
    [
        _meta(0),
        _meta(last=True),
        _meta(turn_end=True),
        _meta(frames=0),
        _meta(epoch=None),
        _meta(cache=1),  # another turn than the ledger
        None,
    ],
)
def test_eager_chunks(meta) -> None:
    policy = _policy()
    _stream(policy, "a", play_end=NOW + 10)
    assert policy._eager(ReadyChunk("a", meta), NOW) is True


def test_a_continuation_with_a_ledger_is_paced_but_not_without_one() -> None:
    policy = _policy()
    chunk = _stream(policy, "a", play_end=NOW + 10)
    assert policy._eager(chunk, NOW) is False
    assert policy._eager(ReadyChunk("b", _meta()), NOW) is True
    # The client ran dry long ago: a new segment, go now.
    policy.ledgers["a"].play_end = NOW - 2.0
    assert policy._eager(chunk, NOW) is True


def test_ledger_anchors_buffers_once_and_reanchors_late_chunks() -> None:
    policy = _policy()
    policy.on_scheduled("a", _meta(0), NOW)
    policy.on_output("a", 0.12, NOW)
    ledger = policy.ledgers["a"]
    assert ledger.play_end == pytest.approx(NOW + 0.5 + 0.12)
    assert (ledger.anchor, ledger.d1) == (NOW, 0.12)
    # Arrives 0.3 s after the client ran out: re-anchor at arrival.
    policy.on_scheduled("a", _meta(1), NOW + 0.9)
    policy.on_output("a", 1.0, NOW + 0.92)
    assert ledger.late_s == pytest.approx(0.3)
    assert ledger.play_end == pytest.approx(NOW + 1.92)
    assert ledger.audio_s == pytest.approx(1.12)
    # The turn ends: the next turn re-anchors without the buffer.
    policy.on_scheduled("a", _meta(2, last=True), NOW + 1.5)
    policy.on_output("a", 0.5, NOW + 1.5)
    assert "a" not in policy.ledgers
    policy.on_scheduled("a", _meta(0, cache=1), NOW + 5)
    policy.on_output("a", 1.0, NOW + 5)
    assert policy.ledgers["a"].play_end == pytest.approx(NOW + 6.0)
    policy.forget("a")
    assert "a" not in policy.ledgers and "a" not in policy.buffered


def test_rate_deadline_is_tighter_after_a_long_first_chunk() -> None:
    policy = _policy(max_hold_s=10.0)
    policy.ledgers["a"] = StreamLedger(key=(0, 0), anchor=NOW, play_end=NOW + 1.34, audio_s=0.84, d1=0.84)
    chunk = ReadyChunk("a", _meta(1, frames=25))
    # rate: NOW + 1.1 * (0.84 + 1.0 - 0.84) = NOW + 1.1 < zero-stall NOW + 1.34;
    # minus the margin, its own one-row step and an eager step that may run first.
    release = policy.release_by(chunk, NOW, 1)
    assert release == pytest.approx(NOW + 1.1 - 0.1 - seed_step_s(1) - seed_step_s(4))


def test_eager_step_takes_only_rows_due_within_a_step() -> None:
    policy = _policy()
    ready = [
        ReadyChunk("e", _meta(0)),
        _stream(policy, "soon", play_end=NOW + 0.5),  # release 100.1 <= now + T(1)
        _stream(policy, "slack", play_end=NOW + 10),  # release by max hold: 100.35
    ]
    plan = policy.plan(ready, NOW)
    assert plan.trigger == "eager"
    assert plan.released == ["e", "soon"]
    assert plan.held == {"slack"}
    assert policy.next_release_in(NOW) == pytest.approx(0.35)


def _two_due_three_slack(policy: CodecDeadlinePolicy) -> list[ReadyChunk]:
    ready = [_stream(policy, f"due{i}", play_end=NOW + 0.3) for i in range(2)]
    ready += [_stream(policy, f"slack{i}", play_end=NOW + 10 + i) for i in range(3)]
    return ready


def test_deadline_step_fills_the_graph_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_CFM_GRAPH_GRID", "1,4,8,16,20")
    policy = _policy()
    plan = policy.plan(_two_due_three_slack(policy), NOW)
    assert plan.trigger == "deadline"
    assert len(plan.released) == 4  # 2 due -> tier 4
    assert plan.released[:2] == ["due0", "due1"]
    assert len(plan.held) == 1


def test_deadline_step_does_not_fill_while_a_first_chunk_is_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_CFM_GRAPH_GRID", "1,4,8,16,20")
    policy = _policy()
    plan = policy.plan(_two_due_three_slack(policy), NOW, onset_pending=True)
    assert plan.trigger == "deadline"
    assert plan.released == ["due0", "due1"]
    assert len(plan.held) == 3


def test_graph_step_tiers_follow_the_code2wav_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    assert graph_step_tiers(8) == [1, 2, 4, 8]  # the Code2Wav default: powers of two
    monkeypatch.setenv("VLLM_OMNI_CFM_GRAPH_GRID", "1,4,8,16,20")
    assert graph_step_tiers(8) == [1, 4, 8]
    assert graph_step_tiers(16) == [1, 4, 8, 16]
    monkeypatch.setenv("VLLM_OMNI_CFM_GRAPH_GRID", "3,5")
    assert graph_step_tiers(8) == [3, 5, 8]


def test_holds_everything_until_a_release_time() -> None:
    policy = _policy()
    ready = [_stream(policy, "a", play_end=NOW + 10), _stream(policy, "b", play_end=NOW + 10)]
    plan = policy.plan(ready, NOW)
    assert plan.trigger == "hold"
    assert plan.released == [] and plan.held == {"a", "b"}
    assert policy.next_release_in(NOW) == pytest.approx(0.35)
    # The max hold expires: both go.
    plan = policy.plan(ready, NOW + 0.36)
    assert plan.trigger == "deadline" and set(plan.released) == {"a", "b"}


def test_max_rows_caps_a_rows_trigger() -> None:
    policy = _policy(max_rows=8)
    ready = [_stream(policy, f"r{i}", play_end=NOW + 10 + i) for i in range(10)]
    plan = policy.plan(ready, NOW)
    assert plan.trigger == "rows" and len(plan.released) == 8
    assert len(plan.held) == 2


def test_step_time_model_ewma() -> None:
    model = StepTimeModel((1, 4, 8), 0.5)
    assert model.mean == pytest.approx([seed_step_s(1), seed_step_s(4), seed_step_s(8)])
    model.mean = [0.1, 0.2, 0.3]
    assert model.tier_capacity(2) == 4 and model.tier_capacity(30) == 8
    assert model.estimate(1) == pytest.approx(0.1)
    model.observe(1, 0.2)
    assert model.mean[0] == pytest.approx(0.15)
    assert model.mad[0] == pytest.approx(0.05)
    assert model.estimate(1) == pytest.approx(0.15 + 1.5 * 0.05)
    model.observe(0, 1.0)
    assert model.mean[0] == pytest.approx(0.15)


def test_breaker_opens_after_late_held_chunks_and_cools_off() -> None:
    breaker = LateBreaker(window=10, late_frac=0.1, cooloff_s=5.0)
    breaker.record(True, NOW)
    assert not breaker.is_open(NOW)
    breaker.record(True, NOW)
    assert breaker.is_open(NOW + 1) and breaker.opens == 1
    assert not breaker.is_open(NOW + 5.1)

    policy = _policy(breaker_window=2, breaker_late_frac=0.4)
    for step in range(2):
        rid = "a"
        _stream(policy, rid, play_end=NOW - 0.5)
        policy.held_once.add(rid)
        policy.on_scheduled(rid, _meta(3 + step), NOW)
        policy.on_output(rid, 1.0, NOW)
    assert policy.breaker.is_open(NOW)
    assert policy._eager(_stream(policy, "b", play_end=NOW + 10), NOW) is True


def test_late_held_chunks_are_counted_by_the_trigger_that_released_them() -> None:
    policy = _policy(stats=True)
    for trigger in ("eager", "deadline", "deadline"):
        _stream(policy, "a", play_end=NOW - 0.5)
        policy.held_once.add("a")
        policy._last_trigger = trigger
        policy.on_scheduled("a", _meta(3), NOW)
        policy.on_output("a", 1.0, NOW)
    assert policy._stats.late == {"eager": 1, "deadline": 2}


@pytest.mark.parametrize(
    ("kwargs", "enabled"),
    [
        ({"async_scheduling": False, "native_data_plane": False, "idle_wait_s": 0.05}, True),
        ({"async_scheduling": True, "native_data_plane": False, "idle_wait_s": 0.05}, False),
        ({"async_scheduling": False, "native_data_plane": True, "idle_wait_s": 0.05}, False),
        ({"async_scheduling": False, "native_data_plane": False, "idle_wait_s": 0.0}, False),
    ],
)
def test_build_requires_sync_scheduling_the_chunk_adapter_and_the_idle_park(kwargs, enabled) -> None:
    policy = build_codec_deadline_policy({"codec_deadline": {"enabled": True}}, **kwargs)
    assert (policy is not None) is enabled
    assert build_codec_deadline_policy({"codec_deadline": {"enabled": False}}, **kwargs) is None
    assert build_codec_deadline_policy({}, **kwargs) is None
