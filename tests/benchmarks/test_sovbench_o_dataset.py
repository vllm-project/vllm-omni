# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SOVBench-O dataset: session parsing and per-turn prompt construction (issue #10954).

Covers the multi-group session semantics the issue asks to test: group mode
resets dialogue history while earlier media stays available, future clips are
absent from the model input, and the gold answers never enter the prompt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vllm_omni.benchmarks.data_modules.sovbench_o_dataset import (
    SOVBENCH_O_JSONL_PINNED_URL,
    SOVBenchOFrameCountOverreportedError,
    SOVBenchOMediaResolver,
    SOVBenchOSampleRequest,
    _parse_session,
    build_sovbench_o_turn_messages,
    sovbench_o_history_text,
    sovbench_o_prefix_frame_plan,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


def _user_content(question: str) -> str:
    if question:
        return (
            f"<video>{question}\n\nChoices:\n(A). first option\n(B). second option\n"
            "(C). third option\n(D). fourth option"
        )
    return "<video>\n\nChoices:\n(A). first option\n(B). second option\n(C). third option\n(D). fourth option"


def _session_item() -> dict[str, Any]:
    """Five turns in two groups: [0, 1, 2] and [3, 4].

    Gold answers: C, A, B / D, A. Turn 1 and turn 4 are continuation turns
    (empty stem); turn 2 opens group 1 but still carries a stem.
    """
    return {
        "messages": [
            {"role": "user", "content": _user_content("What is in the opening scene?")},
            {"role": "assistant", "content": "C", "category": "Real-time"},
            {"role": "user", "content": _user_content("")},
            {"role": "assistant", "content": "A", "category": "Proactive"},
            {"role": "user", "content": _user_content("What happens after that?")},
            {"role": "assistant", "content": "B", "category": "Recall"},
            {"role": "user", "content": _user_content("What color is the cup?")},
            {"role": "assistant", "content": "D", "category": "Real-time"},
            {"role": "user", "content": _user_content("")},
            {"role": "assistant", "content": "A", "category": "Proactive"},
        ],
        "videos": [
            "benchvideos/s0/s0_0.00_7.00.mp4",
            "benchvideos/s0/s0_7.00_14.00.mp4",
            "benchvideos/s0/s0_14.00_21.00.mp4",
            "benchvideos/s0/s0_21.00_28.00.mp4",
            "benchvideos/s0/s0_28.00_35.00.mp4",
        ],
        "entry_metadata": [
            {"correct_indices": [2, 0, 1], "p2_choice_mode": "stepwise_choices"},
            {"correct_indices": [3, 0], "p2_choice_mode": "stepwise_choices"},
        ],
    }


@pytest.fixture()
def session() -> Any:
    parsed = _parse_session(_session_item(), session_id=0)
    assert parsed is not None
    return parsed


class _FakeMedia:
    """Records requested clip prefixes; one image part per clip, no decoding."""

    def __init__(self) -> None:
        self.requested: list[tuple[str, ...]] = []

    def prefix_frame_image_parts(self, clip_paths: Any) -> list[dict[str, Any]]:
        clips = tuple(clip_paths)
        self.requested.append(clips)
        return [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA=="}} for _ in clips]

    def audio_part(self, clip_relpath: str) -> dict[str, Any] | None:
        return None


def _text_parts(messages: list[dict[str, Any]]) -> str:
    return "\n".join(part["text"] for part in messages[1]["content"] if part["type"] == "text")


def test_parse_session_groups_turns_and_answers(session: Any) -> None:
    assert session.num_turns == 5
    turn = session.turns[3]
    assert (turn.group_id, turn.group_start_turn_id, turn.step_id, turn.group_num_turns) == (1, 3, 0, 2)
    assert [t.answer for t in session.turns] == ["C", "A", "B", "D", "A"]
    assert [t.turn_type for t in session.turns] == ["Real-time", "Proactive", "Recall", "Real-time", "Proactive"]
    # Each turn's media grows by one clip, in session order.
    assert session.turns[0].clip_paths == ["benchvideos/s0/s0_0.00_7.00.mp4"]
    assert len(session.turns[4].clip_paths) == 5


def test_parse_session_carries_question_stem_forward(session: Any) -> None:
    # Turn 1 has no stem of its own; it inherits turn 0's.
    assert session.turns[1].raw_question == ""
    assert session.turns[1].resolved_question == "What is in the opening scene?"
    # Turn 4 is a continuation inside group 1 but still inherits the group's stem.
    assert session.turns[4].resolved_question == "What color is the cup?"
    assert session.turns[3].raw_question == "What color is the cup?"


def test_parse_session_skips_answer_mismatch() -> None:
    item = _session_item()
    # Assistant says "B" (index 1) for turn 0 while entry_metadata says index 2 ("C").
    item["messages"][1]["content"] = "B"
    assert _parse_session(item, session_id=0) is None


def test_group_mode_resets_history_while_media_accumulates(session: Any) -> None:
    turn_index = 3  # first turn of group 1
    media = _FakeMedia()

    group_messages = build_sovbench_o_turn_messages(
        session, turn_index, ["C", "A", "B"], reference_mode="group_level_multi_turn", media=media
    )
    body = _text_parts(group_messages)
    assert "Dialogue history for relevant previous turns:" not in body
    assert "Turn 1" not in body and "Turn 3" not in body

    sample_messages = build_sovbench_o_turn_messages(
        session, turn_index, ["C", "A", "B"], reference_mode="sample_level_multi_turn", media=media
    )
    body = _text_parts(sample_messages)
    assert "Turn 1\nUser question: What is in the opening scene?" in body
    assert "Turn 3\nUser question: What happens after that?" in body
    assert "entire session history" in body
    assert "current group history" in _text_parts(group_messages)

    # Both modes replay the media accumulated through the current turn —
    # the group reset never rewinds or truncates the visual context.
    assert media.requested == [
        tuple(session.turns[3].clip_paths),
        tuple(session.turns[3].clip_paths),
    ]
    build_sovbench_o_turn_messages(
        session, 4, ["C", "A", "B", "D"], reference_mode="group_level_multi_turn", media=media
    )
    assert media.requested[-1] == tuple(session.turns[4].clip_paths)


def test_future_clips_and_answer_keys_absent_from_model_input(session: Any) -> None:
    media = _FakeMedia()
    # Turn 0's prediction is "B" — distinct from every gold answer that must
    # stay out of this turn's prompt (turn 0 gold C, turn 1 gold A).
    messages = build_sovbench_o_turn_messages(session, 1, ["B"], reference_mode="sample_level_multi_turn", media=media)
    # Only the clips through turn 1 were touched, as one prefix grid.
    assert media.requested == [tuple(session.turns[1].clip_paths)]
    body = _text_parts(messages)
    assert "The accumulated context contains 2 chronological clip(s)." in body

    # History shows only the model's own prediction, never a gold answer.
    assert "Assistant answer: B" in body
    assert "Assistant answer: C" not in body
    assert "Assistant answer: A" not in body
    assert "Assistant answer: D" not in body

    # The current turn renders its own question (carried stem) and its own choices.
    assert "Question: What is in the opening scene?" in body
    assert "Current turn type: Proactive" in body
    assert "This step continues the current streaming question." in body


def test_history_omits_predictions_that_are_not_yet_available(session: Any) -> None:
    text = sovbench_o_history_text(session, session.turns[3], ["C", "A"], "sample_level_multi_turn")
    assert "Turn 1\n" in text and "Turn 2\n" in text
    assert "Turn 4\n" not in text
    # Turn 3 (the last history turn, 1-based) has no prediction yet: its block
    # must not carry an assistant answer line.
    turn3_block = text.split("Turn 3\n")[1]
    assert "Assistant answer:" not in turn3_block


def test_full_session_backfill_through_all_turns(session: Any) -> None:
    """The whole session can be chained: each turn sees all prior predictions."""
    predictions = []
    for turn_index in range(session.num_turns):
        messages = build_sovbench_o_turn_messages(
            session, turn_index, predictions, reference_mode="sample_level_multi_turn"
        )
        body = _text_parts(messages)
        for prediction in predictions:
            assert f"Assistant answer: {prediction}" in body
        predictions.append(f"pred-{turn_index}")
    assert len(predictions) == session.num_turns


def test_media_none_leaves_text_only_parts(session: Any) -> None:
    messages = build_sovbench_o_turn_messages(session, 2, ["C", "A"], reference_mode="group_level_multi_turn")
    assert all(part["type"] == "text" for part in messages[1]["content"])
    assert messages[0]["role"] == "system"


def test_sample_request_construction() -> None:
    # A minimal direct construction path used by the eval-parity test file.
    parsed = _parse_session(_session_item(), session_id=7)
    assert parsed is not None
    request = SOVBenchOSampleRequest(
        prompt="SOVBench-O session 7",
        prompt_len=0,
        expected_output_len=64,
        sovbench_session=parsed,
    )
    assert request.sovbench_session.num_turns == 5
    assert SOVBENCH_O_JSONL_PINNED_URL.startswith("https://raw.githubusercontent.com/")


def _make_small_benchvideos_tar(tar_path: Path) -> None:
    """A two-session benchvideos tar with marker bytes per clip."""
    import io
    import tarfile

    with tarfile.open(tar_path, "w") as tf:
        for name in ("benchvideos/s0/s0_0.00_7.00.mp4", "benchvideos/s1/s1_0.00_7.00.mp4"):
            data = b"clip:" + name.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def _concurrent_extract_worker(tar_path: str, queue: Any) -> None:
    from vllm_omni.benchmarks.data_modules.sovbench_o_dataset import _extract_sovbench_o_videos_tar

    try:
        root = _extract_sovbench_o_videos_tar(Path(tar_path), "sovbench-o-concurrency-test")
        files = sorted(str(p.relative_to(root)) for p in root.rglob("*.mp4"))
        queue.put(("ok", str(root), files))
    except Exception as exc:  # noqa: BLE001 - the worker reports its own failure
        queue.put(("err", str(exc), None))


def test_concurrent_cold_cache_extraction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two processes racing a cold, shared HF cache both publish and read the same extract."""
    import multiprocessing

    tar_path = tmp_path / "benchvideos.tar"
    _make_small_benchvideos_tar(tar_path)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    procs = [context.Process(target=_concurrent_extract_worker, args=(str(tar_path), queue)) for _ in range(2)]
    for proc in procs:
        proc.start()
    results = [queue.get(timeout=120) for _ in procs]
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0

    assert [status for status, _, _ in results] == ["ok", "ok"], results
    roots = {root for _, root, _ in results}
    assert len(roots) == 1
    expected = ["s0/s0_0.00_7.00.mp4", "s1/s1_0.00_7.00.mp4"]
    for _, _, files in results:
        assert files == expected

    # A later caller takes the cached fast path inside the lock.
    from vllm_omni.benchmarks.data_modules.sovbench_o_dataset import _extract_sovbench_o_videos_tar

    again = _extract_sovbench_o_videos_tar(tar_path, "sovbench-o-concurrency-test")
    assert str(again) == next(iter(roots))


def test_prefix_frame_plan_matches_reference_grid() -> None:
    """The reviewer's example: three 1.5 s clips at a constant 10 FPS.

    The reference samples the concatenated prefix at 1 FPS and sends frames at
    0, 1, 2, 3 s — four frames — while per-clip sampling would send three
    frames at 0, 1.5, 3 s. The plan must match the reference.
    """
    probes = [(15, 10.0), (15, 10.0), (15, 10.0)]
    assert sovbench_o_prefix_frame_plan(probes, 1.0) == [(0, 0), (0, 10), (1, 5), (2, 0)]


def test_prefix_frame_plan_fractional_and_boundary_cases() -> None:
    # The frame count floors once over the whole prefix: 3.99 s at 1 FPS -> 3 grid points.
    assert sovbench_o_prefix_frame_plan([(399, 100.0)], 1.0) == [(0, 0), (0, 100), (0, 200)]
    # Exactly 4 s -> 4 grid points; a grid point on a clip boundary belongs to the next clip.
    assert sovbench_o_prefix_frame_plan([(8, 4.0), (8, 4.0)], 1.0) == [(0, 0), (0, 4), (1, 0), (1, 4)]
    # Mixed clip frame rates: the grid is session-relative, local indices use each clip's own FPS.
    assert sovbench_o_prefix_frame_plan([(10, 10.0), (20, 20.0)], 1.0) == [(0, 0), (1, 0)]
    # A sub-second prefix still yields one frame.
    assert sovbench_o_prefix_frame_plan([(1, 10.0)], 1.0) == [(0, 0)]
    # Empty prefix -> no frames.
    assert sovbench_o_prefix_frame_plan([], 1.0) == []


def test_clip_declared_duration_from_file_name(tmp_path: Path) -> None:
    clip = tmp_path / "C90dVxnA8vM" / "C90dVxnA8vM_5.00_8.50.mp4"
    clip.parent.mkdir()
    clip.write_bytes(b"")
    from vllm_omni.benchmarks.data_modules.sovbench_o_dataset import _clip_declared_duration

    assert _clip_declared_duration(clip) == 3.5
    assert _clip_declared_duration(tmp_path / "C90dVxnA8vM" / "weird.mp4") is None


def test_reconcile_frame_count_prefers_declared_window_when_metadata_drifts() -> None:
    from vllm_omni.benchmarks.data_modules.sovbench_o_dataset import _reconcile_frame_count

    # 5 s at 50 FPS declares 250 frames; a container claiming 315 drifts >15% and is corrected.
    assert _reconcile_frame_count(315, 50.0, 5.0) == 250
    # Small drift (<=15%) keeps the container count: 260 vs 250 is within tolerance.
    assert _reconcile_frame_count(260, 50.0, 5.0) == 260
    # No declared window or unusable values keep the container count untouched.
    assert _reconcile_frame_count(315, 50.0, None) == 315
    assert _reconcile_frame_count(315, 50.0, 0.0) == 315
    assert _reconcile_frame_count(315, 0.0, 5.0) == 315


def test_overreported_frame_count_rebuilds_plan_and_persists_real_count(tmp_path, monkeypatch):
    """All requested indices out of range: rebuild from the real count, persist, no stand-ins."""
    import json as _json

    daily = "vllm_omni.benchmarks.data_modules.daily_omni_dataset"
    media_root = tmp_path / "benchvideos"
    (media_root / "s0").mkdir(parents=True)
    (media_root / "s0" / "s0_0.00_4.00.mp4").write_bytes(b"fake")
    clip = "benchvideos/s0/s0_0.00_4.00.mp4"

    def fake_validated(path, wanted):
        if max(wanted) >= 15:
            raise SOVBenchOFrameCountOverreportedError(f"{path}: up to {max(wanted)}, only 15 decode")
        return {index: f"img{index}" for index in wanted}

    def fake_jpeg(image, quality=85):
        return (f"jpeg:{image}").encode()

    monkeypatch.setattr(daily + "._probe_video_frames_and_fps", lambda path: (40, 10.0))
    monkeypatch.setattr(SOVBenchOMediaResolver, "_decode_indices_validated", staticmethod(fake_validated))
    monkeypatch.setattr(daily + "._pil_to_jpeg_bytes", fake_jpeg)
    monkeypatch.setattr(SOVBenchOMediaResolver, "_count_real_frames", lambda self, rel: 15)

    resolver = SOVBenchOMediaResolver(media_root)
    parts = resolver.prefix_frame_image_parts([clip])
    assert len(parts) == 1  # 15 frames at 10 FPS = 1.5 s -> one grid point

    probe_files = list((media_root / ".sovbench_o_cache" / "probe").glob("*.json"))
    assert len(probe_files) == 1
    assert _json.loads(probe_files[0].read_text(encoding="utf-8"))["num_frames"] == 15

    frame_files = sorted(p.name for p in (media_root / ".sovbench_o_cache" / "frames").rglob("idx-*.jpg"))
    assert frame_files == ["idx-000000.jpg"]

    def _no_count(self, rel):
        raise AssertionError("the persisted probe should have been used")

    monkeypatch.setattr(SOVBenchOMediaResolver, "_count_real_frames", _no_count)
    resolver2 = SOVBenchOMediaResolver(media_root)
    assert len(resolver2.prefix_frame_image_parts([clip])) == 1


def test_partially_overreported_frame_count_rebuilds_plan(tmp_path, monkeypatch):
    """Reviewer's example: 100 real frames at 10 FPS, container claims 110.

    A partial overflow (the plan asks for index 100) must raise at the decode
    boundary, correct the probe to 100 frames, rebuild the grid to 10 s
    (indices 0..90), and never write an idx-000100.jpg stand-in. A fresh
    resolver reuses the persisted probe without recounting.
    """
    import json as _json

    daily = "vllm_omni.benchmarks.data_modules.daily_omni_dataset"
    media_root = tmp_path / "benchvideos"
    (media_root / "s0").mkdir(parents=True)
    (media_root / "s0" / "s0_0.00_11.00.mp4").write_bytes(b"fake")
    clip = "benchvideos/s0/s0_0.00_11.00.mp4"
    real_frames = 100

    def fake_validated(path, wanted):
        if max(wanted) >= real_frames:
            raise SOVBenchOFrameCountOverreportedError(f"{path}: up to {max(wanted)}, only {real_frames} decode")
        return {index: f"img{index}" for index in wanted}

    def fake_jpeg(image, quality=85):
        return (f"jpeg:{image}").encode()

    monkeypatch.setattr(daily + "._probe_video_frames_and_fps", lambda path: (110, 10.0))
    monkeypatch.setattr(SOVBenchOMediaResolver, "_decode_indices_validated", staticmethod(fake_validated))
    monkeypatch.setattr(daily + "._pil_to_jpeg_bytes", fake_jpeg)
    monkeypatch.setattr(SOVBenchOMediaResolver, "_count_real_frames", lambda self, rel: real_frames)

    resolver = SOVBenchOMediaResolver(media_root)
    parts = resolver.prefix_frame_image_parts([clip])
    assert len(parts) == 10  # the real 100 frames rebuild the grid to 10 s

    probe_files = list((media_root / ".sovbench_o_cache" / "probe").glob("*.json"))
    assert len(probe_files) == 1
    assert _json.loads(probe_files[0].read_text(encoding="utf-8"))["num_frames"] == real_frames

    frame_files = sorted(p.name for p in (media_root / ".sovbench_o_cache" / "frames").rglob("idx-*.jpg"))
    assert frame_files == [f"idx-{i:06d}.jpg" for i in range(0, 91, 10)]
    assert "idx-000100.jpg" not in frame_files

    def _no_count(self, rel):
        raise AssertionError("the persisted probe should have been used")

    monkeypatch.setattr(SOVBenchOMediaResolver, "_count_real_frames", _no_count)
    resolver2 = SOVBenchOMediaResolver(media_root)
    assert len(resolver2.prefix_frame_image_parts([clip])) == 10


def test_session_request_aggregates_per_turn_prompt_len(monkeypatch):
    """The session aggregate sums the per-turn server prompt token counts (P2 review)."""
    import asyncio

    from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput

    import vllm_omni.benchmarks.patch.patch as patch_mod

    session = _parse_session(_session_item(), session_id=0)
    assert session is not None
    request = SOVBenchOSampleRequest(
        prompt="x",
        prompt_len=7,  # the stale placeholder: the first session's first stem
        expected_output_len=8,
        sovbench_session=session,
        sovbench_output_len=8,
    )

    async def fake_chat_omni(rfi, client_session, pbar=None, mm_position="last"):
        output = patch_mod.RequestFuncOutput()
        output.success = True
        output.prompt_len = 100 + int(rfi.output_len)  # distinct per-turn server usage
        output.output_tokens = int(rfi.output_len)
        output.generated_text = "A"
        return output

    monkeypatch.setattr(patch_mod, "async_request_openai_chat_omni_completions", fake_chat_omni)

    rfi = RequestFuncInput(
        prompt="x",
        api_url="http://127.0.0.1:1/v1/chat/completions",
        prompt_len=7,
        output_len=8,
        model="m",
    )
    patch_mod._attach_sovbench_to_request_func_input(request, rfi)
    output = asyncio.run(patch_mod.async_request_sovbench_o_completions(rfi, session=None))

    turns = session.num_turns
    assert output.success is True
    assert output.prompt_len == turns * (100 + 8), "per-turn server usage must be summed"
    assert output.output_tokens == turns * 8


def test_attach_carries_max_frames_to_request_input():
    """A non-None sample sovbench_max_frames reaches the turn-building path (regression)."""
    from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput

    import vllm_omni.benchmarks.patch.patch as patch_mod

    session = _parse_session(_session_item(), session_id=0)
    assert session is not None
    for cap in (None, 24):
        request = SOVBenchOSampleRequest(
            prompt="x",
            prompt_len=0,
            expected_output_len=8,
            sovbench_session=session,
            sovbench_max_frames=cap,
        )
        rfi = RequestFuncInput(
            prompt="x", api_url="http://127.0.0.1:1/v1/chat/completions", prompt_len=0, output_len=8, model="m"
        )
        patch_mod._attach_sovbench_to_request_func_input(request, rfi)
        assert getattr(rfi, "sovbench_max_frames", "missing") == cap
        assert getattr(rfi, "sovbench_media", "missing") is None


def test_length_limited_turn_stays_out_of_replay_history() -> None:
    """A length-limited non-answer is omitted from the next prompt's history.

    The raw response stays available for scoring (turn_predictions), while the
    replay history for the next turn keeps the question block but drops the
    truncated `Assistant answer`. A length-limited response that did produce a
    complete option is kept, and so is any response with an unknown reason.
    """
    session = _parse_session(_session_item(), session_id=0)
    assert session is not None
    partial = "Answer: Answer: Answer:"  # repeated markers, no option letter
    complete = "The answer is B"

    messages = build_sovbench_o_turn_messages(
        session,
        1,
        [partial],
        reference_mode="sample_level_multi_turn",
        prior_finish_reasons=["length"],
    )
    body = _text_parts(messages)
    assert "Assistant answer:" not in body, "truncated non-answer must not be replayed"
    assert "Turn 1\nUser question: What is in the opening scene?" in body

    # The raw partial stays available for the scorer.
    session.turn_predictions = [partial]
    from vllm_omni.benchmarks.data_modules.sovbench_o_eval import (
        compute_sovbench_o_accuracy_metrics,
        extract_sovbench_choice_letter,
    )

    assert extract_sovbench_choice_letter(partial) == ""
    request = SOVBenchOSampleRequest(prompt="x", prompt_len=0, expected_output_len=8, sovbench_session=session)
    metrics = compute_sovbench_o_accuracy_metrics([request], [type("O", (), {})])
    assert metrics["sovbench_o_by_turn_type"]["Real-time"]["total"] == 2

    # A length-limited response with a complete option is replayed.
    messages = build_sovbench_o_turn_messages(
        session, 1, ["B"], reference_mode="sample_level_multi_turn", prior_finish_reasons=["length"]
    )
    assert "Assistant answer: B" in _text_parts(messages)

    # Unknown/stop reasons keep the answer line as before.
    messages = build_sovbench_o_turn_messages(
        session, 1, [complete], reference_mode="sample_level_multi_turn", prior_finish_reasons=[]
    )
    assert f"Assistant answer: {complete}" in _text_parts(messages)


def test_length_limited_question_echo_stays_out_of_replay_history() -> None:
    """A length-limited question echo is omitted even though the loose scorer
    fallback reads a letter out of it (y-null's session-76, 256-token run).

    The truncated turn began with ``Turn 22``/``User question:`` and repeated
    the question until the token limit without ever naming an option; the
    any-letter fallback still extracted ``E`` from ``USER``. The replay history
    requires an explicit option for length-limited turns, and the two-turn
    replay keeps the question block but drops the answer line.
    """
    from vllm_omni.benchmarks.data_modules.sovbench_o_eval import extract_sovbench_choice_letter

    session = _parse_session(_session_item(), session_id=0)
    assert session is not None
    echo = "Turn 2\nUser question: What does the user set up first?\nUser question: What does the user set up first?"
    # The loose fallback reads a letter out of the echoed question; the
    # explicit mode (answer tag / JSON / boxed / phrase-stripped letter) does not.
    assert extract_sovbench_choice_letter(echo) != ""
    assert extract_sovbench_choice_letter(echo, allow_any_letter_fallback=False) == ""

    messages = build_sovbench_o_turn_messages(
        session, 1, [echo], reference_mode="sample_level_multi_turn", prior_finish_reasons=["length"]
    )
    body = _text_parts(messages)
    assert "Assistant answer:" not in body, "truncated echo must not be replayed"
    assert "Turn 1\nUser question: What is in the opening scene?" in body

    # The same echo under an ordinary finish reason is replayed as-is.
    messages = build_sovbench_o_turn_messages(
        session, 1, [echo], reference_mode="sample_level_multi_turn", prior_finish_reasons=["stop"]
    )
    assert f"Assistant answer: {echo}" in _text_parts(messages)

    # A length-limited response that names an option explicitly is kept.
    boxed = "Still thinking through the clips. <answer>D</answer>"
    assert extract_sovbench_choice_letter(boxed, allow_any_letter_fallback=False) == "D"
    messages = build_sovbench_o_turn_messages(
        session, 1, [boxed], reference_mode="sample_level_multi_turn", prior_finish_reasons=["length"]
    )
    assert f"Assistant answer: {boxed}" in _text_parts(messages)
