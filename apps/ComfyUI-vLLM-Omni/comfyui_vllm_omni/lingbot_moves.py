# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""LingBot World camera moves, from the lane editor to the server's key states.

A move holds one camera lane at one value for a span of a shot; the compiler
turns a sequence of shots into one action list per latent frame (three per
chunk), which is what ``extra_params.camera_action_script`` takes. The engine
integrates a held key once per latent frame -- 0.05 world units, 4 degrees of
pitch, 6 degrees of yaw -- so a move's length is what decides how far the
camera travels.
"""

from __future__ import annotations

from dataclasses import dataclass

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

#: Values that name their own lane, so the lane word may be left out.


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


@dataclass(frozen=True)
class Shot:
    """One prompt held for ``frames`` pixel frames, with its own camera moves."""

    prompt: str
    frames: int
    moves: tuple[CameraMove, ...] = ()

    def __post_init__(self) -> None:
        if self.frames <= 0:
            raise ValueError("A shot must last at least one frame.")


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

    def camera_script(self) -> list[list[list[str]]]:
        """The ``camera_action_script`` value the server accepts."""
        return [[list(frame) for frame in chunk] for chunk in self.script]


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


def moves_from_editor(value: object, *, shot_frames: int) -> tuple[CameraMove, ...]:
    """Read the in-node timeline editor's value, ``{"moves": [...]}``.

    The editor stores moves on the shot's own frame clock, exactly as a moves
    text does, so both authoring surfaces produce the same moves. A value that
    is missing or empty means the shot has no camera movement.
    """
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise ValueError("The timeline editor expects an object with a 'moves' list.")
    entries = value.get("moves") or []
    if not isinstance(entries, list):
        raise ValueError("The timeline editor's 'moves' must be a list.")
    moves: list[CameraMove] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Timeline move {index} must be an object with lane, value, start_frame and frames.")
        try:
            lane = entry["lane"]
            move_value = entry["value"]
            start = int(entry["start_frame"])
            frames = int(entry["frames"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                f"Timeline move {index} needs lane, value, start_frame and frames; got {entry!r}."
            ) from None
        if start >= shot_frames:
            raise ValueError(f"Timeline move {index} starts at frame {start}, after the shot's {shot_frames} frames.")
        moves.append(CameraMove(lane, move_value, start, max(1, min(frames, shot_frames - start))))
    _check_lane_overlaps(moves)
    return tuple(moves)


def compile_sequence(
    sequence: LingBotSequence,
    geometry: LingBotGeometry = DEFAULT_GEOMETRY,
    *,
    verbose: bool = False,
) -> CompiledSequence:
    """Compile a sequence into per-chunk key states and prompt switches.

    Moves are read on their shot's clock, moved onto the sequence's clock, cut
    at their shot's end, and sampled once per latent frame. A move too short to
    cover a latent frame is widened to one.
    """
    del verbose
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
            frames = max(frames, geometry.temporal_scale)
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
    )
