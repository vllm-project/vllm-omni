# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""LingBot World camera moves: authoring, and compilation to a camera script.

A move holds one camera lane at one value for a span of a shot::

    CameraMove(lane="move", value="forward", start_frame=0, frames=45)

which means "press W for the shot's first 45 frames". Moves are authored on a
frame clock relative to their shot and are compiled into the per-chunk key
states the server takes: one action list per latent frame, three per chunk::

    [[["w"], ["w"], ["w"]], [["w", "l"], ["w", "l"], ["w", "l"]], ...]

The key vocabulary is the server's, and so are the rates: the engine integrates
a held key once per latent frame (0.05 world units of travel, 4 degrees of
pitch, 6 degrees of yaw). There is no speed parameter to set, which is why a
move's length is what decides how far the camera travels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .lingbot_geometry import DEFAULT_GEOMETRY, LingBotGeometry

#: Action keys, in the order the server normalizes them to.
KEY_ORDER = ("w", "a", "s", "d", "i", "j", "k", "l")

#: Lane -> value -> the key the server integrates while it is held.
LANES: dict[str, dict[str, str]] = {
    "move": {"forward": "w", "back": "s"},
    "strafe": {"left": "a", "right": "d"},
    "look_v": {"up": "i", "down": "k"},
    "look_h": {"left": "j", "right": "l"},
}

#: Spellings a moves text may use for the four lanes.
LANE_ALIASES: dict[str, str] = {
    "move": "move",
    "movement": "move",
    "strafe": "strafe",
    "lateral": "strafe",
    "look_h": "look_h",
    "look_horizontal": "look_h",
    "pan": "look_h",
    "look_v": "look_v",
    "look_vertical": "look_v",
    "tilt": "look_v",
}

#: Values that name their own lane, so the lane word may be left out.
UNAMBIGUOUS_VALUES: dict[str, str] = {"forward": "move", "back": "move", "up": "look_v", "down": "look_v"}


@dataclass(frozen=True)
class CameraMove:
    """One camera lane held at one value, in frames relative to its shot."""

    lane: str
    value: str
    start_frame: int
    frames: int

    def __post_init__(self) -> None:
        if self.lane not in LANES:
            raise ValueError(f"Unknown camera lane {self.lane!r}; use one of {', '.join(LANES)}.")
        if self.value not in LANES[self.lane]:
            allowed = ", ".join(LANES[self.lane])
            raise ValueError(f"{self.value!r} is not a {self.lane} move; use one of {allowed}.")
        if self.start_frame < 0:
            raise ValueError("A move cannot start before its shot.")
        if self.frames <= 0:
            raise ValueError("A move must last at least one frame.")

    @property
    def key(self) -> str:
        """The action key this move holds."""
        return LANES[self.lane][self.value]

    def to_json(self) -> dict[str, Any]:
        return {"lane": self.lane, "value": self.value, "start_frame": self.start_frame, "frames": self.frames}


@dataclass(frozen=True)
class Shot:
    """One prompt held for ``frames`` pixel frames, with its own camera moves."""

    prompt: str
    frames: int
    moves: tuple[CameraMove, ...] = ()

    def __post_init__(self) -> None:
        if self.frames <= 0:
            raise ValueError("A shot must last at least one frame.")

    def to_json(self, start_frame: int) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "frames": self.frames,
            "start_frame": start_frame,
            "moves": [move.to_json() for move in self.moves],
        }


@dataclass(frozen=True)
class LingBotSequence:
    """Shots played one after another inside a single LingBot session."""

    shots: tuple[Shot, ...]

    def __post_init__(self) -> None:
        if not self.shots:
            raise ValueError("A sequence needs at least one shot.")


@dataclass(frozen=True)
class CompiledSequence:
    """A sequence compiled onto the chunk grid."""

    script: tuple[tuple[tuple[str, ...], ...], ...]
    #: ``(chunk index, prompt)`` for every shot after the first, in play order.
    prompt_switches: tuple[tuple[int, str], ...]
    shot_start_chunks: tuple[int, ...]
    total_chunks: int
    total_frames: int
    warnings: tuple[str, ...] = field(default=())

    def camera_script(self) -> list[list[list[str]]]:
        """The ``camera_action_script`` value the server accepts."""
        return [[list(frame) for frame in chunk] for chunk in self.script]


def _parse_frame(token: str, *, line: int) -> int:
    text = token.strip().lower()
    if text.endswith("f"):
        text = text[:-1]
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"Moves line {line}: {token!r} is not a frame count.") from None


def parse_moves(text: str, *, shot_frames: int | None = None) -> tuple[CameraMove, ...]:
    """Read a moves text into moves relative to one shot.

    One move per line, ``lane value start end``; the lane word may be dropped
    for values that name their own lane (``forward``, ``back``, ``up``,
    ``down``). ``end`` is exclusive and may be the word ``end`` to mean the end
    of the shot. ``#`` starts a comment. A text that begins with ``[`` is read
    as the older per-chunk camera JSON instead.
    """
    stripped = text.strip()
    if stripped.startswith("["):
        return moves_from_camera_script(stripped, shot_frames=shot_frames)

    moves: list[CameraMove] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) == 4:
            lane_token, value, start_token, end_token = tokens
            lane = LANE_ALIASES.get(lane_token.lower())
            if lane is None:
                raise ValueError(
                    f"Moves line {line_number}: unknown lane {lane_token!r}; "
                    f"use one of {', '.join(sorted(set(LANE_ALIASES.values())))}."
                )
        elif len(tokens) == 3:
            value, start_token, end_token = tokens
            lane = UNAMBIGUOUS_VALUES.get(value.lower())
            if lane is None:
                raise ValueError(
                    f"Moves line {line_number}: {value!r} can be a strafe or a look move, "
                    "so write the lane too, such as 'strafe left 0 24'."
                )
        else:
            raise ValueError(
                f"Moves line {line_number}: expected 'lane value start end', 'value start end' or a comment; "
                f"got {raw_line.strip()!r}."
            )

        start = _parse_frame(start_token, line=line_number)
        if end_token.lower() == "end":
            if shot_frames is None:
                raise ValueError(f"Moves line {line_number}: 'end' needs the shot length.")
            end = shot_frames
        else:
            end = _parse_frame(end_token, line=line_number)
        if shot_frames is not None and start >= shot_frames:
            raise ValueError(
                f"Moves line {line_number}: this move starts at frame {start}, after the shot's "
                f"{shot_frames} frames. Use 'end' to reach the last frame."
            )
        moves.append(CameraMove(lane=lane, value=value.lower(), start_frame=start, frames=max(end - start, 1)))

    _check_lane_overlaps(moves)
    return tuple(moves)


def moves_from_camera_script(text: str, *, shot_frames: int | None = None) -> tuple[CameraMove, ...]:
    """Turn the older per-chunk camera JSON into moves.

    Each chunk's key set becomes a move covering that chunk; runs of chunks
    holding the same key become one move. Lanes are rebuilt from the keys, and
    keys outside the vocabulary are refused.
    """
    try:
        script = json.loads(text)
    except ValueError as exc:
        raise ValueError("Camera JSON must be a JSON array of per-chunk action lists.") from exc
    if not isinstance(script, list) or not script:
        raise ValueError("Camera JSON must be a non-empty array of per-chunk action lists.")

    frames: list[tuple[str, ...]] = []
    for index, chunk in enumerate(script):
        if not isinstance(chunk, list) or not chunk:
            raise ValueError(f"Camera JSON chunk {index} must be a non-empty list of action lists.")
        frames.extend(tuple(str(key).lower() for key in actions) for actions in chunk)
    unknown = sorted({key for actions in frames for key in actions} - set(KEY_ORDER))
    if unknown:
        raise ValueError(f"Camera JSON uses unknown action keys: {', '.join(unknown)}.")

    # One direction per lane per frame: the server would integrate both.
    per_lane: dict[str, list[tuple[int, str]]] = {}
    for offset, actions in enumerate(frames):
        seen: dict[str, str] = {}
        for key in actions:
            lane, value = _lane_of_key(key)
            seen.setdefault(lane, value)
        for lane, value in seen.items():
            per_lane.setdefault(lane, []).append((offset, value))

    moves: list[CameraMove] = []
    for lane, entries in per_lane.items():
        start, value = entries[0]
        for index in range(1, len(entries) + 1):
            continues = (
                index < len(entries) and entries[index][1] == value and entries[index][0] == entries[index - 1][0] + 1
            )
            if continues:
                continue
            end = entries[index - 1][0] + 1
            moves.append(CameraMove(lane=lane, value=value, start_frame=start, frames=end - start))
            if index < len(entries):
                start, value = entries[index]
    moves.sort(key=lambda move: (move.start_frame, move.lane))
    if shot_frames is not None:
        moves = [
            move
            if move.start_frame + move.frames <= shot_frames
            else CameraMove(move.lane, move.value, move.start_frame, max(shot_frames - move.start_frame, 0))
            for move in moves
        ]
    return tuple(move for move in moves if move.frames > 0)


def _lane_of_key(key: str) -> tuple[str, str]:
    for lane, values in LANES.items():
        for value, candidate in values.items():
            if candidate == key:
                return lane, value
    raise ValueError(f"Unknown camera action key {key!r}.")


def _check_lane_overlaps(moves: tuple[CameraMove, ...] | list[CameraMove]) -> None:
    by_lane: dict[str, list[CameraMove]] = {}
    for move in moves:
        by_lane.setdefault(move.lane, []).append(move)
    for lane, lane_moves in by_lane.items():
        ordered = sorted(lane_moves, key=lambda move: move.start_frame)
        for previous, current in zip(ordered, ordered[1:]):
            if current.start_frame < previous.start_frame + previous.frames:
                raise ValueError(
                    f"Two {lane} moves overlap: one runs {previous.start_frame}-"
                    f"{previous.start_frame + previous.frames} and the next starts at {current.start_frame}."
                )


def compile_sequence(
    sequence: LingBotSequence,
    geometry: LingBotGeometry = DEFAULT_GEOMETRY,
    *,
    verbose: bool = False,
) -> CompiledSequence:
    """Compile a sequence into per-chunk key states and prompt switches.

    Moves are read on their shot's clock, moved onto the sequence's clock, cut
    at their shot's end, and sampled once per latent frame. A move too short to
    cover a latent frame is widened to one and reported in ``warnings``.
    """
    del verbose
    warnings: list[str] = []
    absolute_moves: list[CameraMove] = []
    shot_start_chunks: list[int] = []
    prompt_switches: list[tuple[int, str]] = []
    offset = 0
    chunk_cursor = 0

    for index, shot in enumerate(sequence.shots):
        chunks = geometry.shot_chunks(index, shot.frames)
        shot_start_chunks.append(chunk_cursor)
        if index:
            prompt_switches.append((chunk_cursor, shot.prompt))
        _check_lane_overlaps(shot.moves)
        for move in shot.moves:
            if move.start_frame >= shot.frames:
                raise ValueError(
                    f"Shot {index}: the {move.lane} move starts at frame {move.start_frame}, "
                    f"after the shot's {shot.frames} frames."
                )
            frames = min(move.frames, shot.frames - move.start_frame)
            if frames != move.frames:
                warnings.append(f"Shot {index}: the {move.lane} move is cut to {frames} frames at the shot's end.")
            if frames < geometry.temporal_scale:
                frames = geometry.temporal_scale
                warnings.append(
                    f"Shot {index}: the {move.lane} move is widened to one latent frame "
                    f"({geometry.temporal_scale} pixel frames)."
                )
            absolute_moves.append(CameraMove(move.lane, move.value, offset + move.start_frame, frames))
        offset += shot.frames
        chunk_cursor += chunks

    total_chunks = chunk_cursor
    total_frames = geometry.frames_for(total_chunks)
    if offset != total_frames:
        raise ValueError(
            f"The sequence covers {offset} frames but its shots span {total_chunks} chunks ({total_frames} frames)."
        )

    script: list[list[tuple[str, ...]]] = []
    for chunk in range(total_chunks):
        actions: list[tuple[str, ...]] = []
        for latent in range(geometry.latent_frames_per_chunk):
            frame = geometry.sample_frame(chunk, latent)
            keys = {move.key for move in absolute_moves if move.start_frame <= frame < move.start_frame + move.frames}
            actions.append(tuple(key for key in KEY_ORDER if key in keys))
        script.append(actions)

    return CompiledSequence(
        script=tuple(tuple(chunk) for chunk in script),
        prompt_switches=tuple(prompt_switches),
        shot_start_chunks=tuple(shot_start_chunks),
        total_chunks=total_chunks,
        total_frames=total_frames,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def sequence_payload(
    sequence: LingBotSequence,
    compiled: CompiledSequence,
    geometry: LingBotGeometry = DEFAULT_GEOMETRY,
    *,
    fps: float | None = None,
) -> dict[str, Any]:
    """What the visualizer draws: shots, moves and the chunk grid."""
    shots = []
    start_frame = 0
    for index, (shot, start_chunk) in enumerate(zip(sequence.shots, compiled.shot_start_chunks)):
        entry = shot.to_json(start_frame)
        entry["start_chunk"] = start_chunk
        entry["chunks"] = geometry.shot_chunks(index, shot.frames)
        shots.append(entry)
        start_frame += shot.frames
    return {
        "fps": fps,
        "geometry": {
            "first_chunk_frames": geometry.first_chunk_frames,
            "chunk_frames": geometry.chunk_frames,
            "latent_frames_per_chunk": geometry.latent_frames_per_chunk,
            "temporal_scale": geometry.temporal_scale,
        },
        "total_frames": compiled.total_frames,
        "total_chunks": compiled.total_chunks,
        "shots": shots,
        "prompt_switches": [[chunk, prompt] for chunk, prompt in compiled.prompt_switches],
        "warnings": list(compiled.warnings),
    }
