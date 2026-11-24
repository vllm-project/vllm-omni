# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SOVBench-O streaming omni benchmark dataset for ``vllm bench serve --omni``.

SOVBench-O (issue #10954) evaluates multiple-choice questions over an
audio-video stream that keeps advancing as a session progresses: every turn
reveals one new clip, the model sees every clip from the session start through
the current turn, and the dialogue history is built from the model's own
previous predictions (never the gold answers). The protocol, prompt text, and
scorer are ported from the reference VLMEvalKit integration in the StreamOV
repository (``streamov_bench/streaming_omni.py`` at commit 3decd08a).

Differences from the reference, both documented for eval reproducibility:

- Each turn replays the accumulated clips as one chat-completions request
  (the reference's replay protocol); it does not advance a persistent duplex
  session, so evicted history is never re-exposed to the model.
- Frames follow the reference's session-relative sampling grid: the grid is
  computed once over the concatenated clip prefix and mapped back to
  clip-local frame indices (``sovbench_o_prefix_frame_plan``). No merge step
  runs; decoded frames are cached per (clip, frame index) on disk so later
  turns of the same session reuse them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pybase64 as base64
import regex as re

try:
    from vllm.benchmarks.datasets import BenchmarkDataset, SampleRequest
except ImportError:  # pragma: no cover - older vllm layouts
    from vllm.benchmarks.datasets import HuggingFaceDataset as BenchmarkDataset
    from vllm.benchmarks.datasets import SampleRequest
from vllm.tokenizers import TokenizerLike

logger = logging.getLogger(__name__)

SOVBENCH_O_DEFAULT_MEDIA_REPO = "PatrickyMing/StreamOV"
SOVBENCH_O_MEDIA_TAR = "benchvideos.tar"
SOVBENCH_O_MEDIA_DIRNAME = "benchvideos"
SOVBENCH_O_JSONL_PINNED_URL = (
    "https://raw.githubusercontent.com/PatrickXM/StreamOV/"
    "3decd08ac36e4e9f1e5c918a4dd28d9a2cd63455/streamov_bench/streamov_bench.jsonl"
)
SOVBENCH_O_AUDIO_SAMPLE_RATE = 16000

SOVBENCH_O_REFERENCE_MODES = ("sample_level_multi_turn", "group_level_multi_turn")

SYS = (
    "You are a helpful assistant for streaming video understanding. The media clips are ordered "
    "chronologically from earlier to later in the same session. Use the accumulated visual/audio "
    "context and the dialogue history to answer the current multiple-choice question."
)

QUESTION_TMPL = (
    "Current turn type: {turn_type}\n"
    "{question_prefix}"
    "Question: {question}\n"
    "Choices:\n"
    "{choices}\n"
    "Respond with only the option letter (A-E) on the final line. "
    "Do not include explanation, markdown emphasis, or LaTeX boxes."
)

HISTORY_TMPL = "Dialogue history for relevant previous turns:\n{history}\nEnd of history."

_CHOICE_LINE_RE = re.compile(r"^\(?[A-Z]\)?[.)]")
_VIDEO_TAG_RE = re.compile(r"^\s*<video>\s*", re.IGNORECASE)


def _safe_str(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _choice_index_to_letter(idx: object) -> str:
    try:
        index = int(idx)
    except (TypeError, ValueError):
        return ""
    if index < 0 or index >= 26:
        return ""
    return chr(ord("A") + index)


def _entry_meta_turn_count(meta: dict[str, Any]) -> int:
    if "correct_indices" in meta:
        return len(meta["correct_indices"])
    if "correct_index" in meta:
        return 1
    return 0


def _parse_user_content(content: object) -> tuple[str, list[str]]:
    """Split a user turn into (question stem, choice lines), dropping the ``<video>`` tag."""
    text = _safe_str(content).strip()
    text = _VIDEO_TAG_RE.sub("", text)
    head, sep, tail = text.partition("Choices:")
    question = head.strip()
    choices_text = tail if sep else ""
    choices = []
    for line in choices_text.splitlines():
        line = line.strip()
        if _CHOICE_LINE_RE.match(line):
            choices.append(line)
    return question, choices


@dataclass
class SOVBenchOTurn:
    """One question turn inside a session, with its gold answer held aside for scoring."""

    session_turn_id: int
    session_num_turns: int
    group_id: int
    group_start_turn_id: int
    step_id: int
    group_num_turns: int
    turn_type: str
    resolved_question: str
    choices: list[str]
    current_user_content: str
    answer: str
    clip_paths: list[str]
    raw_question: str


@dataclass
class SOVBenchOSessionData:
    """Per-session turn table shared between the request function and the scorer.

    ``turn_predictions`` is filled in by the turn-chained request function as
    the session runs; the accuracy hook reads it back from these same objects.
    """

    session_id: int
    user_contents: list[str]
    resolved_questions: list[str]
    turns: list[SOVBenchOTurn] = field(default_factory=list)
    turn_predictions: list[str] = field(default_factory=list)
    turn_finish_reasons: list[str] = field(default_factory=list)

    @property
    def num_turns(self) -> int:
        return len(self.turns)


@dataclass
class SOVBenchOSampleRequest(SampleRequest):
    """One benchmark request = one SOVBench-O session (turns are chained in order)."""

    sovbench_session: SOVBenchOSessionData | None = None
    sovbench_reference_mode: str = "sample_level_multi_turn"
    sovbench_use_audio: bool = False
    sovbench_max_frames: int | None = None
    sovbench_media: SOVBenchOMediaResolver | None = None
    sovbench_output_len: int = 64


def _parse_session(item: dict[str, Any], session_id: int) -> SOVBenchOSessionData | None:
    """Expand one JSONL session line into turn rows, or return ``None`` with a warning.

    Faithful port of the reference ``prepare_dataset`` session loop: strict
    user/assistant alternation, group coverage by ``entry_metadata``, and the
    gold answer cross-check between the assistant letter and
    ``correct_indices``.
    """
    messages = item.get("messages", [])
    videos = item.get("videos", [])
    entry_metadata = item.get("entry_metadata", [])

    try:
        if len(messages) % 2 != 0:
            raise ValueError(f"Session {session_id} has odd number of messages.")

        user_contents: list[str] = []
        assistant_contents: list[str] = []
        turn_types: list[str] = []
        for i in range(0, len(messages), 2):
            user_msg = messages[i]
            assistant_msg = messages[i + 1]
            if user_msg.get("role") != "user" or assistant_msg.get("role") != "assistant":
                raise ValueError(f"Session {session_id} has non user/assistant alternating messages.")
            user_contents.append(_safe_str(user_msg.get("content")))
            assistant_contents.append(_safe_str(assistant_msg.get("content")))
            turn_types.append(_safe_str(assistant_msg.get("category")))

        if len(videos) > len(user_contents):
            raise ValueError(f"Session {session_id} has {len(videos)} videos but only {len(user_contents)} user turns.")

        # Carry the question stem forward over continuation turns ("<video>\n\nChoices: ...").
        resolved_questions: list[str] = []
        current_resolved_question = ""
        for user_content in user_contents:
            question, _ = _parse_user_content(user_content)
            if question:
                current_resolved_question = question
            resolved_questions.append(current_resolved_question)

        # Group boundaries follow entry_metadata; groups must cover the session exactly.
        turn_to_group: dict[int, dict[str, Any]] = {}
        group_sizes: dict[int, int] = {}
        turn_ptr = 0
        for group_id, meta in enumerate(entry_metadata):
            if not isinstance(meta, dict):
                raise ValueError(f"Session {session_id} group {group_id} entry_metadata is not a mapping.")
            n_turns = _entry_meta_turn_count(meta)
            if n_turns <= 0:
                raise ValueError(
                    f"Session {session_id} group {group_id} entry_metadata missing correct_index/correct_indices."
                )
            if turn_ptr + n_turns > len(user_contents):
                raise ValueError(
                    f"Session {session_id} group {group_id} exceeds session length: "
                    f"need turns up to {turn_ptr + n_turns - 1}, have {len(user_contents)}."
                )
            if "correct_indices" in meta:
                expected = list(meta["correct_indices"])
            elif "correct_index" in meta:
                expected = [meta["correct_index"]]
            else:
                raise ValueError(f"Session {session_id} group {group_id} missing answer metadata.")
            if len(expected) != n_turns:
                raise ValueError(
                    f"Session {session_id} group {group_id} has {n_turns} turns "
                    f"but metadata has {len(expected)} answers."
                )
            group_sizes[group_id] = n_turns
            for step_id, (turn_idx, answer_idx) in enumerate(zip(range(turn_ptr, turn_ptr + n_turns), expected)):
                turn_to_group[turn_idx] = {
                    "group_id": group_id,
                    "group_start_turn_id": turn_ptr,
                    "step_id": step_id,
                    "answer_by_meta": _choice_index_to_letter(answer_idx),
                }
            turn_ptr += n_turns
        if turn_ptr != len(user_contents):
            raise ValueError(
                f"Session {session_id}: entry_metadata covers {turn_ptr} turns but "
                f"session has {len(user_contents)} user turns."
            )
    except ValueError as exc:
        logger.warning("SOVBench-O JSONL session %d skipped: %s", session_id, exc)
        return None

    from vllm_omni.benchmarks.data_modules.sovbench_o_eval import extract_sovbench_choice_letter

    session = SOVBenchOSessionData(
        session_id=session_id,
        user_contents=user_contents,
        resolved_questions=resolved_questions,
    )
    for turn_idx in range(len(user_contents)):
        group_info = turn_to_group[turn_idx]
        answer_from_assistant = extract_sovbench_choice_letter(assistant_contents[turn_idx])
        answer_from_meta = group_info["answer_by_meta"]
        if answer_from_assistant and answer_from_meta and answer_from_assistant != answer_from_meta:
            logger.warning(
                "SOVBench-O JSONL session %d skipped: turn %d answer mismatch: %s vs %s",
                session_id,
                turn_idx,
                answer_from_assistant,
                answer_from_meta,
            )
            return None

        raw_question, choices = _parse_user_content(user_contents[turn_idx])
        session.turns.append(
            SOVBenchOTurn(
                session_turn_id=turn_idx,
                session_num_turns=len(user_contents),
                group_id=group_info["group_id"],
                group_start_turn_id=group_info["group_start_turn_id"],
                step_id=group_info["step_id"],
                group_num_turns=group_sizes[group_info["group_id"]],
                turn_type=turn_types[turn_idx] or "Unknown",
                resolved_question=resolved_questions[turn_idx],
                choices=choices,
                current_user_content=user_contents[turn_idx],
                answer=answer_from_assistant or answer_from_meta,
                clip_paths=[_safe_str(v) for v in videos[: turn_idx + 1]],
                raw_question=raw_question,
            )
        )
    return session


def _history_turn_indices(turn: SOVBenchOTurn, reference_mode: str) -> list[int]:
    """History range by reference mode: whole session, or the current group only."""
    if reference_mode == "sample_level_multi_turn":
        start_turn = 0
    elif reference_mode == "group_level_multi_turn":
        start_turn = turn.group_start_turn_id
    else:
        raise ValueError(f"Unknown SOVBench-O reference mode: {reference_mode}")
    return list(range(start_turn, turn.session_turn_id))


def sovbench_o_history_text(
    session: SOVBenchOSessionData,
    turn: SOVBenchOTurn,
    prior_predictions: Sequence[str],
    reference_mode: str,
    prior_finish_reasons: Sequence[str] = (),
) -> str:
    """Render the dialogue-history block, using the model's own previous predictions.

    A length-limited response is omitted from the history unless it names an
    option explicitly (answer tag, JSON, boxed/bold, or a phrase-stripped
    letter); the any-letter fallback does not count, so a truncated question
    echo cannot masquerade as an answer (its raw text stays in
    ``turn_predictions`` for scoring, where the loose fallback is fine):
    replaying a truncated non-answer as an ``Assistant answer`` teaches the
    next turns to repeat the truncation.
    """
    from vllm_omni.benchmarks.data_modules.sovbench_o_eval import extract_sovbench_choice_letter

    history_lines = []
    for turn_idx in _history_turn_indices(turn, reference_mode):
        _, choices = _parse_user_content(session.user_contents[turn_idx])
        question = session.resolved_questions[turn_idx]
        block = f"Turn {turn_idx + 1}\nUser question: {question}\nChoices:\n" + "\n".join(choices)
        pred = _safe_str(prior_predictions[turn_idx]).strip() if turn_idx < len(prior_predictions) else ""
        finish = prior_finish_reasons[turn_idx] if turn_idx < len(prior_finish_reasons) else ""
        explicit = extract_sovbench_choice_letter(pred, allow_any_letter_fallback=False) if finish == "length" else True
        if pred and explicit:
            block += f"\nAssistant answer: {pred}"
        history_lines.append(block)
    return "\n\n".join(history_lines)


def _subsample_parts(parts: list[dict[str, Any]], max_frames: int) -> list[dict[str, Any]]:
    """Uniformly thin the image parts down to ``max_frames``, preserving order."""
    if max_frames <= 0 or len(parts) <= max_frames:
        return parts
    step = len(parts) / max_frames
    picked = {min(len(parts) - 1, int(i * step)) for i in range(max_frames)}
    return [part for idx, part in enumerate(parts) if idx in picked]


def build_sovbench_o_turn_messages(
    session: SOVBenchOSessionData,
    turn_index: int,
    prior_predictions: Sequence[str],
    *,
    reference_mode: str,
    media: SOVBenchOMediaResolver | None = None,
    use_audio: bool = False,
    max_frames: int | None = None,
    prior_finish_reasons: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Build the chat messages for one turn.

    Media parts are attached only when ``media`` is provided, so tests can
    assert on the text parts without decoding anything. The gold answer never
    enters the prompt: history renders model predictions, and the current turn
    carries only the question, its choices, and the revealed clips.
    """
    turn = session.turns[turn_index]
    mode_label = "entire session history" if reference_mode == "sample_level_multi_turn" else "current group history"

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "The following media clips are ordered chronologically for one streaming session. "
                f"Use all clips as context. Dialogue accumulation mode: {mode_label}."
            ),
        }
    ]

    frame_parts: list[dict[str, Any]] = []
    audio_parts: dict[str, dict[str, Any]] = {}
    if media is not None and turn.clip_paths:
        frame_parts = media.prefix_frame_image_parts(turn.clip_paths)
        if use_audio:
            for clip_relpath in turn.clip_paths:
                audio_part = media.audio_part(clip_relpath)
                if audio_part is not None:
                    audio_parts[clip_relpath] = audio_part
        if max_frames is not None and len(frame_parts) > max_frames:
            logger.warning(
                "SOVBench-O session %d turn %d: %d frames exceed --sovbench-max-frames=%d; "
                "uniformly subsampling (the run is no longer a faithful full-context replay)",
                session.session_id,
                turn.session_turn_id,
                len(frame_parts),
                max_frames,
            )
            frame_parts = _subsample_parts(frame_parts, max_frames)
        content.extend(frame_parts)
        content.extend(audio_parts[clip] for clip in turn.clip_paths if clip in audio_parts)
    content.insert(
        1, {"type": "text", "text": f"The accumulated context contains {len(turn.clip_paths)} chronological clip(s)."}
    )

    history = sovbench_o_history_text(session, turn, prior_predictions, reference_mode, prior_finish_reasons)
    if history:
        content.append({"type": "text", "text": HISTORY_TMPL.format(history=history)})

    question_prefix = "" if turn.raw_question else "This step continues the current streaming question.\n"
    content.append(
        {
            "type": "text",
            "text": QUESTION_TMPL.format(
                turn_type=turn.turn_type,
                question_prefix=question_prefix,
                question=turn.resolved_question.strip(),
                choices="\n".join(_safe_str(c) for c in turn.choices),
            ),
        }
    )

    return [
        {"role": "system", "content": [{"type": "text", "text": SYS}]},
        {"role": "user", "content": content},
    ]


def _clip_media_key(clip_path: Path) -> str:
    stem = clip_path.stem
    short = hashlib.md5(str(clip_path).encode("utf-8")).hexdigest()[:10]
    return f"{stem}-{short}"


def _clip_declared_duration(path: Path) -> float | None:
    """Clip duration declared by the SOVBench-O file name, in seconds.

    Clips are named ``<session>_<start>_<end>.mp4`` with the window in seconds,
    which is the reliable source when container metadata over-reports frames.
    """
    stem = path.stem
    session = path.parent.name
    if stem.startswith(session + "_"):
        stem = stem[len(session) + 1 :]
    match = re.fullmatch(r"(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)", stem)
    if not match:
        return None
    return float(match.group(2)) - float(match.group(1))


def _reconcile_frame_count(num_frames: int, avg_fps: float, declared_duration: float | None) -> int:
    """Prefer the file-name window when the container frame count is off by >15%.

    Some benchvideos containers over-report the frame count (and hence the
    duration the sampling grid is derived from); the clip names carry the true
    window in seconds, so the declared window wins when they disagree enough.
    """
    if declared_duration is None or declared_duration <= 0 or avg_fps <= 0:
        return num_frames
    expected = declared_duration * avg_fps
    if abs(num_frames - expected) > 0.15 * expected:
        return int(expected)
    return num_frames


class SOVBenchOFrameCountOverreportedError(ValueError):
    """A decode reveals fewer real frames than the probe claimed.

    Carries no correction by itself: the caller counts the real frames,
    persists the corrected probe, and rebuilds the sampling plan. Raised for
    partially out-of-range requests too, where the daily-omni decode helper
    would silently fill the missing indices with the last decoded frame.
    """


def sovbench_o_prefix_frame_plan(probe_results: Sequence[tuple[int, float]], fps: float) -> list[tuple[int, int]]:
    """Session-relative sampling grid over the concatenated clip prefix.

    Mirrors the reference's merged-video sampling: one grid point per ``1 / fps``
    seconds of the whole prefix (frame count floored, at least one), each grid
    point mapped to the clip covering that session time and to that clip's
    local frame index (floor of the local time times the clip's average FPS,
    clamped to the clip's last frame). ``probe_results`` is one
    ``(num_frames, avg_fps)`` per revealed clip, in chronological order.
    """
    if not probe_results:
        return []
    durations = [n / max(f, 1e-6) for n, f in probe_results]
    starts: list[float] = []
    acc = 0.0
    for duration in durations:
        starts.append(acc)
        acc += duration
    total_duration = acc
    required = max(int(total_duration * fps), 1)

    plan: list[tuple[int, int]] = []
    for i in range(required):
        t = i / fps
        clip_idx = 0
        for j in range(len(durations)):
            if starts[j] <= t:
                clip_idx = j
        num_frames, avg_fps = probe_results[clip_idx]
        local = min(int((t - starts[clip_idx]) * avg_fps), max(num_frames - 1, 0))
        plan.append((clip_idx, local))
    return plan


class SOVBenchOMediaResolver:
    """Decode and cache JPEG frames on the session-relative grid, plus 16 kHz mono WAV audio.

    Frames are addressed per (clip, frame index) and cached under
    ``<benchvideos root>/.sovbench_o_cache`` so a session's later turns reuse
    the earlier turns' decoded frames without re-decoding or merging.

    ``media_root`` is the ``benchvideos`` directory itself (the parent of the
    per-session clip folders).
    """

    def __init__(self, media_root: Path, *, fps: float = 1.0) -> None:
        self.media_root = Path(media_root)
        self.fps = float(fps) if fps and float(fps) > 0 else 1.0
        self._cache_root = self.media_root / ".sovbench_o_cache"
        self._probe_memo: dict[str, tuple[int, float]] = {}
        self._frame_memo: dict[str, dict[int, bytes]] = {}
        self._audio_memo: dict[str, bytes] = {}

    def clip_path(self, clip_relpath: str) -> Path:
        rel = clip_relpath.strip()
        prefix = SOVBENCH_O_MEDIA_DIRNAME + "/"
        if rel.startswith(prefix):
            rel = rel[len(prefix) :]
        return (self.media_root / rel).resolve()

    def prefix_frame_image_parts(self, clip_paths: Sequence[str]) -> list[dict[str, Any]]:
        """Image parts for the session-relative grid over the whole revealed prefix."""
        probes = [self._probe_clip(clip_relpath) for clip_relpath in clip_paths]
        try:
            plan, per_clip = self._collect_plan_frames(probes, clip_paths)
        except SOVBenchOFrameCountOverreportedError:
            # A container over-reports its frame count (fully or partially), so
            # a grid point maps beyond the decodable range. Count the real
            # frames, persist the corrected probe, and rebuild the current
            # turn's plan from the truth instead of emitting a too-long grid
            # with stand-in frames.
            self._correct_probes_with_real_counts(probes, clip_paths)
            plan, per_clip = self._collect_plan_frames(probes, clip_paths)

        parts = []
        for clip_idx, local_idx in plan:
            b64 = base64.b64encode(per_clip[clip_idx][local_idx]).decode("ascii")
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        return parts

    def _collect_plan_frames(
        self, probes: list[tuple[int, float]], clip_paths: Sequence[str]
    ) -> tuple[list[tuple[int, int]], dict[int, dict[int, bytes]]]:
        plan = sovbench_o_prefix_frame_plan(probes, self.fps)
        needed: dict[int, list[int]] = {}
        for clip_idx, local_idx in plan:
            if local_idx not in needed.setdefault(clip_idx, []):
                needed[clip_idx].append(local_idx)
        per_clip = {
            clip_idx: self._frames_by_index(clip_paths[clip_idx], indices) for clip_idx, indices in needed.items()
        }
        return plan, per_clip

    def audio_part(self, clip_relpath: str) -> dict[str, Any] | None:
        wav = self.audio_wav(clip_relpath)
        if wav is None:
            return None
        b64 = base64.b64encode(wav).decode("ascii")
        return {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{b64}"}}

    def _probe_clip(self, clip_relpath: str) -> tuple[int, float]:
        key = _clip_media_key(self.clip_path(clip_relpath))
        if key not in self._probe_memo:
            from vllm_omni.benchmarks.data_modules.daily_omni_dataset import _probe_video_frames_and_fps

            path = self.clip_path(clip_relpath)
            # A persisted real-frame correction (from an earlier decode that
            # caught an over-reporting container) wins over fresh metadata.
            corrected = self._read_corrected_probe(key)
            if corrected is not None:
                num_frames, avg_fps = corrected
            else:
                num_frames, avg_fps = _probe_video_frames_and_fps(path)
                num_frames = _reconcile_frame_count(num_frames, avg_fps, _clip_declared_duration(path))
            self._probe_memo[key] = (num_frames, avg_fps)
        return self._probe_memo[key]

    def _read_corrected_probe(self, key: str) -> tuple[int, float] | None:
        cache_path = self._cache_root / "probe" / f"{key}.json"
        if not cache_path.is_file():
            return None
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return int(data["num_frames"]), float(data["avg_fps"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _write_corrected_probe(self, key: str, num_frames: int, avg_fps: float) -> None:
        cache_path = self._cache_root / "probe" / f"{key}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"num_frames": num_frames, "avg_fps": avg_fps}), encoding="utf-8")

    def _correct_probes_with_real_counts(self, probes: list[tuple[int, float]], clip_paths: Sequence[str]) -> None:
        """Replace every clip's probe with its real decodable frame count and persist it."""
        for clip_idx, clip_relpath in enumerate(clip_paths):
            key = _clip_media_key(self.clip_path(clip_relpath))
            real_frames = self._count_real_frames(clip_relpath)
            avg_fps = probes[clip_idx][1]
            probes[clip_idx] = (real_frames, avg_fps)
            self._probe_memo[key] = (real_frames, avg_fps)
            self._write_corrected_probe(key, real_frames, avg_fps)

    def _count_real_frames(self, clip_relpath: str) -> int:
        """Count the frames a clip actually decodes, in presentation order."""
        import av

        count = 0
        with av.open(str(self.clip_path(clip_relpath))) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for _ in container.decode(stream):
                count += 1
        if count <= 0:
            raise ValueError(f"Decoded no frames from {self.clip_path(clip_relpath)}")
        return count

    def _frames_by_index(self, clip_relpath: str, indices: Sequence[int]) -> dict[int, bytes]:
        """Decode the requested frame indices of one clip, with disk and memory caching.

        Holds a per-clip file lock for the read-decode-write sequence so two
        processes decoding the same missing index cannot interleave JPEG
        writes. Raises ``ValueError`` when a requested index is beyond the
        decodable range — the caller rebuilds the plan from the real count.
        """
        from filelock import FileLock

        key = _clip_media_key(self.clip_path(clip_relpath))
        frame_dir = self._cache_root / "frames" / key
        frame_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(frame_dir / ".lock"), timeout=3600):
            memo = self._frame_memo.setdefault(key, {})
            still_missing: list[int] = []
            for index in indices:
                if index not in memo:
                    cached = frame_dir / f"idx-{index:06d}.jpg"
                    if cached.is_file():
                        memo[index] = cached.read_bytes()
                    else:
                        still_missing.append(index)
            if still_missing:
                wanted = sorted(still_missing)
                decoded = self._decode_indices_validated(self.clip_path(clip_relpath), wanted)
                for index in wanted:
                    jpeg = self._encode_frame(decoded[index])
                    memo[index] = jpeg
                    (frame_dir / f"idx-{index:06d}.jpg").write_bytes(jpeg)
        return {index: memo[index] for index in indices}

    @staticmethod
    def _decode_indices_validated(path: Path, wanted: list[int]) -> dict[int, Any]:
        """Decode exactly the wanted frame indices; raise when any is out of range.

        Unlike the daily-omni helper (which fills out-of-range indices with the
        last decoded frame), a partial overflow raises so the caller can count
        the real frames, persist the corrected probe, and rebuild the plan
        instead of caching substitute frames under out-of-range indices.
        """
        import av

        wanted_set = set(wanted)
        max_wanted = max(wanted_set)
        decoded: dict[int, Any] = {}
        position = 0
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in container.decode(stream):
                if position in wanted_set:
                    decoded[position] = frame.to_image()
                if position >= max_wanted:
                    break
                position += 1
        if len(decoded) != len(wanted_set):
            raise SOVBenchOFrameCountOverreportedError(
                f"{path}: requested indices up to {max_wanted} but only {position + 1} frames decode"
            )
        return decoded

    @staticmethod
    def _encode_frame(image: Any) -> bytes:
        from vllm_omni.benchmarks.data_modules.daily_omni_dataset import _pil_to_jpeg_bytes

        return _pil_to_jpeg_bytes(image)

    def audio_wav(self, clip_relpath: str) -> bytes | None:
        key = _clip_media_key(self.clip_path(clip_relpath))
        if key in self._audio_memo:
            return self._audio_memo[key]

        cache_path = self._cache_root / "audio" / f"{key}.wav"
        if cache_path.is_file():
            data = cache_path.read_bytes()
            self._audio_memo[key] = data
            return data

        pcm = self._decode_audio_pcm16(clip_relpath)
        if pcm is None:
            return None
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = _pcm16_to_wav_bytes(pcm)
        cache_path.write_bytes(data)
        self._audio_memo[key] = data
        return data

    def _decode_audio_pcm16(self, clip_relpath: str) -> bytes | None:
        """Decode the clip's audio as 16 kHz mono s16 PCM, reusing the duplex-eval reader."""
        from vllm_omni.benchmarks.duplex.omni_duplex_eval_media import read_audio_pcm16

        try:
            return read_audio_pcm16(self.clip_path(clip_relpath))
        except Exception as exc:  # noqa: BLE001 - a clip without a usable audio track is skipped
            logger.warning("SOVBench-O: no usable audio in %s (%s)", clip_relpath, exc)
            return None


def _pcm16_to_wav_bytes(pcm: bytes) -> bytes:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SOVBENCH_O_AUDIO_SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()


def _sovbench_o_hf_cache_root() -> Path:
    root = os.environ.get("HF_HOME", None)
    if root:
        return Path(root).expanduser().resolve()
    return Path.home() / ".cache" / "huggingface"


def _sovbench_o_tar_fingerprint(tar_path: Path) -> str:
    st = tar_path.stat()
    return f"v1:{st.st_size}:{int(st.st_mtime_ns)}"


def _extract_sovbench_o_videos_tar(tar_path: Path, cache_key: str) -> Path:
    """Extract ``benchvideos.tar`` once into the shared media cache; return the ``benchvideos`` root.

    The cache check, the extraction, and the publication are one critical section
    guarded by a per-dataset file lock, so two benchmark processes sharing the
    same HF cache cannot race on ``_extract_work`` or on replacing the final
    directory when both start from a cold cache.
    """
    from filelock import FileLock

    safe = cache_key.replace("/", "__").replace("\\", "_")
    staging_root = _sovbench_o_hf_cache_root() / "vllm_omni" / "sovbench_o_media" / safe
    videos_dir = staging_root / SOVBENCH_O_MEDIA_DIRNAME
    marker = staging_root / ".videos_extracted"
    fp = _sovbench_o_tar_fingerprint(tar_path)

    staging_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(staging_root / ".videos_extract.lock"), timeout=3600):
        if marker.is_file() and videos_dir.is_dir():
            try:
                if marker.read_text(encoding="utf-8").strip() == fp:
                    next(videos_dir.iterdir())
                    logger.info("Reusing cached SOVBench-O benchvideos extract at %s", videos_dir)
                    return videos_dir
            except (OSError, StopIteration):
                shutil.rmtree(videos_dir, ignore_errors=True)
                marker.unlink(missing_ok=True)

        work = staging_root / "_extract_work"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        try:
            logger.info("Extracting SOVBench-O %s from %s", SOVBENCH_O_MEDIA_TAR, tar_path)
            with tarfile.open(tar_path, "r:*") as tf:
                tf.extractall(path=work, filter="data")
            found = work / SOVBENCH_O_MEDIA_DIRNAME
            if not found.is_dir():
                raise RuntimeError(
                    f"Unrecognized layout after extracting {SOVBENCH_O_MEDIA_TAR} under {work} "
                    f"(expected a top-level {SOVBENCH_O_MEDIA_DIRNAME!r} directory)."
                )
            if videos_dir.exists():
                shutil.rmtree(videos_dir, ignore_errors=True)
            shutil.move(str(found), str(videos_dir))
        finally:
            shutil.rmtree(work, ignore_errors=True)

        marker.write_text(fp, encoding="utf-8")
        logger.info("SOVBench-O media ready at %s", videos_dir)
        return videos_dir


def ensure_sovbench_o_videos_dir(repo_id: str | None = None) -> Path:
    """Download ``benchvideos.tar`` from the Hub dataset repo and return the extracted root."""
    rid = (repo_id or SOVBENCH_O_DEFAULT_MEDIA_REPO).strip()
    try:
        from vllm_omni.transformers_utils.repo_utils import hf_api
    except ImportError as e:
        raise ImportError(
            "SOVBench-O Hub media download requires huggingface_hub. Install it (e.g. with "
            "vLLM) or provide --sovbench-video-dir with a local benchvideos extract."
        ) from e

    tar_path = Path(hf_api().hf_hub_download(repo_id=rid, filename=SOVBENCH_O_MEDIA_TAR, repo_type="dataset"))
    return _extract_sovbench_o_videos_tar(tar_path, rid)


class SOVBenchODataset(BenchmarkDataset):
    """SOVBench-O multi-choice streaming sessions (one sample request per session)."""

    SUPPORTED_DATASET_PATHS: set[str] = {SOVBENCH_O_DEFAULT_MEDIA_REPO, "sovbench-o"}
    IS_MULTIMODAL = True
    DEFAULT_OUTPUT_LEN = 64

    def __init__(
        self,
        dataset_path: str | None = None,
        random_seed: int = 0,
        *,
        jsonl_path: str | None = None,
        reference_mode: str = "sample_level_multi_turn",
        use_audio: bool = False,
        fps: float = 1.0,
        media_dir: str | None = None,
        max_frames: int | None = None,
        **kwargs,
    ) -> None:
        if reference_mode not in SOVBENCH_O_REFERENCE_MODES:
            raise ValueError(f"reference_mode must be one of {SOVBENCH_O_REFERENCE_MODES}, got {reference_mode!r}")
        if fps is not None and float(fps) <= 0:
            raise ValueError(f"fps must be positive, got {fps!r}")
        self.jsonl_path = Path(jsonl_path).expanduser() if jsonl_path else None
        self.reference_mode = reference_mode
        self.use_audio = bool(use_audio)
        self.fps = float(fps) if fps is not None else 1.0
        self._media_dir_arg = Path(media_dir).expanduser() if media_dir else None
        self.max_frames = int(max_frames) if max_frames else None
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive when set")

        super().__init__(dataset_path=dataset_path, random_seed=random_seed, **kwargs)
        self.load_data()

    # ------------------------------------------------------------------ loading

    def _resolve_jsonl(self) -> Path:
        candidates: list[Path] = []
        if self.jsonl_path is not None:
            candidates.append(self.jsonl_path)
        env_path = os.environ.get("VLLM_SOVBENCH_O_JSONL", "").strip()
        if env_path:
            candidates.append(Path(env_path).expanduser())
        if self.dataset_path:
            path = Path(self.dataset_path).expanduser()
            if path.suffix == ".jsonl":
                candidates.append(path)
            else:
                candidates.append(path / "streamov_bench.jsonl")
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            "SOVBench-O annotations not found. Pass --sovbench-jsonl (or set VLLM_SOVBENCH_O_JSONL) "
            f"pointing at streamov_bench.jsonl, downloadable from {SOVBENCH_O_JSONL_PINNED_URL}"
        )

    def _resolve_media_root(self, jsonl_path: Path) -> Path:
        """Return the ``benchvideos`` directory: the flag value, the JSONL sibling, or a Hub extract."""
        if self._media_dir_arg is not None:
            if self._media_dir_arg.name == SOVBENCH_O_MEDIA_DIRNAME and self._media_dir_arg.is_dir():
                return self._media_dir_arg
            nested = self._media_dir_arg / SOVBENCH_O_MEDIA_DIRNAME
            if nested.is_dir():
                return nested
        sibling = jsonl_path.parent / SOVBENCH_O_MEDIA_DIRNAME
        if sibling.is_dir():
            return sibling
        return ensure_sovbench_o_videos_dir()

    def load_data(self) -> None:
        jsonl_path = self._resolve_jsonl()
        media_root = self._resolve_media_root(jsonl_path)
        self.media = SOVBenchOMediaResolver(media_root, fps=self.fps)

        self.sessions: list[SOVBenchOSessionData] = []
        skipped = 0
        with open(jsonl_path, encoding="utf-8") as f:
            for session_id, raw_line in enumerate(f):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                session = _parse_session(json.loads(raw_line), session_id)
                if session is None:
                    skipped += 1
                else:
                    self.sessions.append(session)
        if skipped:
            logger.warning("SOVBench-O: skipped %d invalid session(s) from %s", skipped, jsonl_path)
        if not self.sessions:
            raise ValueError(f"No valid SOVBench-O sessions loaded from {jsonl_path}")
        total_turns = sum(s.num_turns for s in self.sessions)
        logger.info(
            "Loaded SOVBench-O: %d session(s), %d turn(s), mode=%s, fps=%s, use_audio=%s, media=%s",
            len(self.sessions),
            total_turns,
            self.reference_mode,
            self.fps,
            self.use_audio,
            media_root,
        )

    # ------------------------------------------------------------------ sampling

    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        output_len: int | None = None,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list[SampleRequest]:
        if output_len is None:
            output_len = self.DEFAULT_OUTPUT_LEN

        # 0 selects all sessions; sessions are never oversampled because
        # duplicating one would double-count its turns in the accuracy.
        wanted = len(self.sessions) if num_requests <= 0 else min(num_requests, len(self.sessions))
        if num_requests > len(self.sessions):
            logger.warning(
                "SOVBench-O: requested %d sessions but only %d available; running the available set "
                "(sessions are not oversampled to keep accuracy well-defined)",
                num_requests,
                len(self.sessions),
            )

        prompt_len = 0
        try:
            if tokenizer is not None:
                prompt_len = len(tokenizer.encode(self.sessions[0].turns[0].resolved_question))
        except (TypeError, ValueError):
            prompt_len = 0

        sampled: list[SampleRequest] = []
        for session in self.sessions[:wanted]:
            sampled.append(
                SOVBenchOSampleRequest(
                    prompt=f"SOVBench-O session {session.session_id} ({session.num_turns} turns)",
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=None,
                    request_id=f"{request_id_prefix}{len(sampled)}",
                    sovbench_session=session,
                    sovbench_reference_mode=self.reference_mode,
                    sovbench_use_audio=self.use_audio,
                    sovbench_max_frames=self.max_frames,
                    sovbench_media=self.media,
                    sovbench_output_len=output_len,
                )
            )
        return sampled
