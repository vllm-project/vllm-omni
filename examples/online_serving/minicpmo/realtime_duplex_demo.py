# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Minimal single-input MiniCPM-o 4.5 Realtime duplex demo.

Run this after starting the duplex server. Strict lifecycle, overlap, and
multi-session validation lives under ``tests/e2e/online_serving``.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni.clients.duplex import (  # noqa: E402
    DEFINITIVE_REFUSAL_CODES,
    PCM16_BYTES_PER_SAMPLE,
    PCM16_SAMPLE_RATE,
    AudioFormat,
    DuplexClient,
    DuplexClientBase,
    DuplexClientError,
    DuplexConnectionError,
    DuplexSessionClosedError,
    ErrorEvent,
    EventCollector,
    acknowledge_collected_playback,
    read_pcm16_wav,
    wait_for_condition,
    write_pcm16_wav,
)
from vllm_omni.clients.duplex import chunk_period_ms as _chunk_period_ms  # noqa: E402
from vllm_omni.clients.duplex import (  # noqa: E402
    has_residual_model_unit as _has_residual_model_unit,
)
from vllm_omni.clients.duplex import (  # noqa: E402
    reference_audio_data_url as _ref_audio_data_url,
)
from vllm_omni.clients.minicpmo_4_5 import (  # noqa: E402
    create_duplex_session_config,
)
from vllm_omni.experimental.fullduplex.video_stacking import (  # noqa: E402
    concat_frames_b64,
    unit_subframe_offsets,
)


class _StreamingOutputWriter:
    """Persist and report output deltas as the client receives them."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.audio_chunk_dir = output_dir / "audio_chunks"
        self.audio_chunk_paths: list[Path] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_chunk_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "output.pcm").write_bytes(b"")

    def handle(self, event: dict[str, object]) -> None:
        event_type = event.get("type")
        if event_type in {
            "response.output_audio_transcript.delta",
            "response.output_text.delta",
        }:
            delta = event.get("delta")
            if isinstance(delta, str) and delta:
                print(delta, end="", file=sys.stderr, flush=True)
            return
        if event_type != "response.output_audio.delta":
            return

        delta = event.get("delta") or event.get("audio")
        if not isinstance(delta, str) or not delta:
            return
        try:
            pcm16 = base64.b64decode(delta)
        except ValueError:
            return
        if not pcm16:
            return

        chunk_index = len(self.audio_chunk_paths) + 1
        chunk_path = self.audio_chunk_dir / f"chunk_{chunk_index:04d}.wav"
        sample_rate_hz = event.get("sample_rate_hz")
        if not isinstance(sample_rate_hz, int) or sample_rate_hz <= 0:
            sample_rate_hz = 24_000
        with (self.output_dir / "output.pcm").open("ab") as output_pcm:
            output_pcm.write(pcm16)
        write_pcm16_wav(chunk_path, pcm16, sample_rate_hz=sample_rate_hz)
        self.audio_chunk_paths.append(chunk_path)
        print(
            f"\n[audio chunk {chunk_index}: {len(pcm16)} bytes -> {chunk_path}]",
            file=sys.stderr,
            flush=True,
        )


class _TimedInputClient(DuplexClient):
    """Record the transmitted media clock without retaining audio payloads."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_timeline: list[dict[str, float]] = []

    async def send(self, event: dict[str, object]) -> str:
        event_id = await super().send(event)
        if event.get("type") == "input_audio_buffer.append" and "audio_end_ms" in event:
            self.input_timeline.append(
                {
                    "sent_at_s": time.monotonic(),
                    "audio_end_s": float(event["audio_end_ms"]) / 1000.0,
                }
            )
        return event_id


class _StreamingEventCollector(EventCollector):
    def __init__(self, writer: _StreamingOutputWriter) -> None:
        super().__init__()
        self._writer = writer

    def add(self, event: dict[str, object], *, received_at_s: float | None = None) -> None:
        super().add(event, received_at_s=received_at_s)
        self._writer.handle(self.events[-1])

    async def consume(self, client: DuplexClientBase) -> None:
        """Collect events and, like the web client, ack each reply once it has fully played.

        This offline client plays everything it receives, so a finished reply's
        whole audio counts as played.
        """
        try:
            async for event in client.events():
                self.add(event.raw)
                response_id = self.response_id(event.raw)
                if event.raw.get("type") != "response.done" or not response_id:
                    continue
                pcm16 = self.audio_bytes(response_id)
                if not pcm16:
                    continue
                played_ms = AudioFormat("pcm16", self.output_sample_rate_hz).duration_ms(len(pcm16))
                with suppress(DuplexClientError):
                    await client.ack_playback(played_ms, response_id=response_id, item_id=f"item_{response_id}")
        except DuplexSessionClosedError:
            pass


def _input_committed_index(
    events: list[dict[str, object]],
    after_index: int,
) -> int | None:
    for index, event in enumerate(events[max(after_index, 0) :], start=max(after_index, 0)):
        if event.get("type") == "input_audio_buffer.committed":
            return index
    return None


def _post_commit_model_decision(
    events: list[dict[str, object]],
    committed_index: int | None,
) -> str | None:
    if committed_index is None:
        return None
    for event in events[committed_index + 1 :]:
        event_type = event.get("type")
        if event_type == "response.listen":
            return "listen"
        if event_type == "response.done":
            response = event.get("response")
            if not isinstance(response, dict) or response.get("status") != "cancelled":
                return "speak"
    return None


def _post_commit_response_created(
    events: list[dict[str, object]],
    committed_index: int | None,
) -> bool:
    if committed_index is None:
        return False
    return any(event.get("type") == "response.created" for event in events[committed_index + 1 :])


def _response_event_id(event: dict[str, object]) -> str | None:
    """Response id of a response.* event, tolerating both wire shapes.

    The server projects ``response.created``/``response.done`` with a nested
    ``response.id``; a few legacy events only carry a top-level ``response_id``.
    """
    response = event.get("response")
    if isinstance(response, dict) and isinstance(response.get("id"), str):
        return response["id"]
    response_id = event.get("response_id")
    return response_id if isinstance(response_id, str) else None


def _response_done_is_terminal(event: dict[str, object]) -> bool:
    """Whether a ``response.done`` counts as a drained answer.

    A cancelled generation is a terminal state, but it produced no answer, so
    it must not satisfy the drain on its own.
    """
    response = event.get("response")
    return not isinstance(response, dict) or response.get("status") != "cancelled"


def _post_commit_definitive_refusal(
    events: list[dict[str, object]],
    committed_index: int | None,
) -> bool:
    """Whether the server definitively refused the forced post-commit create.

    ``DEFINITIVE_REFUSAL_CODES`` are terminal answers to ``response.create``:
    the committed turn holds nothing the model can answer. Only the fallback
    re-ask can trigger them (the demo sends no other create, and a server that
    already answered commits with a precreated response never refuses), so a
    post-commit event with one of these codes means the mandatory final
    answer, if any, was delivered by an earlier response and the recognition
    judging -- not this timeout -- is what decides whether that answer covers
    the input.
    """
    return bool(_post_commit_refusal_indexes(events, committed_index))


def _post_commit_refusal_indexes(
    events: list[dict[str, object]],
    committed_index: int | None,
) -> frozenset[int]:
    """Positions of the refusal errors that answered the forced create.

    Scoped to the post-commit window on purpose: an earlier refusal came from
    some other ask and must keep failing the run. Positions rather than object
    identities keep the filter correct even if a collector hands back copies.
    """
    if committed_index is None:
        return frozenset()
    return frozenset(
        index
        for index in range(committed_index + 1, len(events))
        if events[index].get("type") == "error" and ErrorEvent(events[index]).code in DEFINITIVE_REFUSAL_CODES
    )


def _post_commit_forced_response_settled(
    events: list[dict[str, object]],
    committed_index: int | None,
) -> bool:
    """Whether every response created after the commit reached a terminal state.

    A forced commit must be drained before the session closes: closing while
    the requested response is still generating drops its transcript and audio,
    which is exactly the silent failure a mandatory final answer must not have.

    The forced answer is always a response created after the commit -- the
    server precreates it on an idle commit, or promotes the deferred create
    once an in-flight response settles. So the drain matches ``response.created``
    against ``response.done`` by response id: it settles when every post-commit
    creation is closed by a done of the same id and at least one of those
    terminals is a non-cancelled one. This is what keeps a pre-commit response
    that settles after the commit, or a spontaneous one, from posing as the
    answer while the promoted create is still queued; and it lets a cancelled
    spontaneous response close its own creation without either counting as the
    answer or hanging the drain behind the completed forced one.
    """
    if committed_index is None:
        return False
    created: set[str] = set()
    # Any done closes its creation, cancelled or not, so an interrupted
    # response cannot block the drain forever.
    done: set[str] = set()
    # Only a non-cancelled terminal counts as a drained answer.
    answered: set[str] = set()
    anonymous_created = 0
    anonymous_done = 0
    anonymous_answered = 0
    for event in events[committed_index + 1 :]:
        event_type = event.get("type")
        if event_type == "response.created":
            response_id = _response_event_id(event)
            if response_id is None:
                anonymous_created += 1
            else:
                created.add(response_id)
        elif event_type == "response.done":
            response_id = _response_event_id(event)
            answered_done = _response_done_is_terminal(event)
            if response_id is None:
                # Events without an id cannot pair; count them so an
                # anonymous terminal still closes the matching anonymous
                # creation instead of hanging the drain.
                anonymous_done += 1
                if answered_done:
                    anonymous_answered += 1
            else:
                done.add(response_id)
                if answered_done:
                    answered.add(response_id)
    # The drained answer must be one of the post-commit creations, so a
    # pre-commit response settling after the commit still cannot pose as it;
    # every post-commit creation is closed; at least one closed with a
    # non-cancelled terminal. The anonymous branch is best-effort: without
    # ids nothing pairs, so an id-less pre-commit settle can still pass for
    # the answer and only the quiescence guard holds the window open. The
    # server always sends ids on the wire, so this only matters for the
    # synthetic/anonymous event shapes.
    post_commit_answered = bool(answered & created) or (anonymous_created > 0 and anonymous_answered > 0)
    return post_commit_answered and created <= done and anonymous_done >= anonymous_created


def _latest_model_decision(
    events: list[dict[str, object]],
    after_index: int,
) -> str | None:
    decision: str | None = None
    for event in events[max(after_index, 0) :]:
        event_type = event.get("type")
        if event_type == "response.listen":
            decision = "listen"
        elif event_type == "response.done":
            response = event.get("response")
            if not isinstance(response, dict) or response.get("status") != "cancelled":
                decision = "speak"
    return decision


def _response_in_progress(events: list[dict[str, object]]) -> bool:
    return sum(event.get("type") == "response.created" for event in events) > sum(
        event.get("type") == "response.done" for event in events
    )


async def _commit_input_after_playback_checkpoint(
    client: DuplexClient,
    collector: EventCollector,
    *,
    create_response: bool = False,
) -> float:
    """Commit input only after preserving already-played assistant history.

    Native duplex can finish several responses while the user is still
    streaming.  Under the ``ack_only`` playback policy, acknowledging those
    responses after the final input commit is intentionally rejected: it could
    append an old assistant message after the newly committed user message.
    Checkpoint playback first so the server commits each assistant item at its
    original history position.  A second acknowledgement after the commit can
    then safely advance response-local playback that drained in the meantime.

    ``create_response`` sets the commit's ``response_create`` flag. Native
    duplex sessions auto-respond: the model decides per unit whether to speak
    or keep listening, so a session can end with the last user turn
    unanswered. Forwarding this flag makes the commit ask for one response
    built from the committed input: when the commit retains answerable audio,
    the server opens the forced response immediately (``precreate_response``);
    when the input was already consumed unit by unit, the server answers the
    commit without scheduling anything and the drain's ``response.create``
    re-ask settles the session on a definitive refusal. Either way the final
    turn reaches a definitive state instead of ending on model luck.
    """
    await acknowledge_collected_playback(client, collector)
    commit_sent_at_s = time.monotonic()
    await client.commit(create_response=True if create_response else None)
    return commit_sent_at_s


def _event_count_after(
    events: list[dict[str, object]],
    event_type: str,
    index: int | None,
) -> int:
    if index is None:
        return 0
    return sum(event.get("type") == event_type for event in events[index + 1 :])


#: How long a forced final commit may take to open its response before the
#: client re-asks with an explicit ``response.create``. A native duplex
#: precreated response usually arrives well under a second after the commit
#: acknowledgement, so this only absorbs a slow stage-0 schedule.
_FORCED_RESPONSE_CREATE_TIMEOUT_S = 15.0

#: Once every post-commit response looks settled, keep waiting this long for
#: one more ``response.created`` before the drain returns. After a spontaneous
#: (or pre-commit) response settles, there is a short window before the
#: promoted forced create opens -- an id-paired predicate alone cannot see a
#: promotion that is still queued, so the drain also requires a brief
#: quiescence over ``response.created``/``response.done``/``response.listen``.
_FORCED_RESPONSE_SETTLE_QUIET_S = 1.5

#: How long the fallback ``response.create`` keeps retrying through transient
#: transport failures. A send can land inside a resume/teardown window and
#: raise ``DuplexConnectionError`` even though the session is still alive; the
#: re-ask itself is idempotent server-side (a duplicate create while a response
#: is active is refused with ``response_already_active`` rather than corrupting
#: the session), so retrying within this window is the difference between a
#: drained forced answer and a silent 90 s hang.
_FORCED_RESPONSE_FALLBACK_RETRY_WINDOW_S = 15.0

#: Interval between fallback ``response.create`` retries.
_FORCED_RESPONSE_FALLBACK_RETRY_INTERVAL_S = 1.0

#: Response lifecycle signals that can precede a promoted forced create.
_RESPONSE_SIGNAL_TYPES = frozenset(
    {
        "response.created",
        "response.done",
        "response.listen",
    }
)


def _last_post_commit_response_signal_at_s(
    events: list[dict[str, object]],
    committed_index: int | None,
) -> float | None:
    """Monotonic receive time of the last response signal after the commit.

    ``EventCollector.add`` stamps every stored event with
    ``_client_received_at_s`` on the monotonic clock, so the quiescence guard
    measures real event times instead of the polling phase.
    """
    if committed_index is None:
        return None
    last: float | None = None
    for event in events[committed_index + 1 :]:
        if event.get("type") in _RESPONSE_SIGNAL_TYPES:
            received_at_s = event.get("_client_received_at_s")
            if isinstance(received_at_s, int | float):
                last = float(received_at_s)
    return last


async def _drain_forced_post_commit_response(
    client: DuplexClient,
    collector: EventCollector,
    *,
    committed_index: int | None,
    timeout_s: float,
    create_timeout_s: float = _FORCED_RESPONSE_CREATE_TIMEOUT_S,
    settle_quiet_s: float = _FORCED_RESPONSE_SETTLE_QUIET_S,
    fallback_retry_window_s: float = _FORCED_RESPONSE_FALLBACK_RETRY_WINDOW_S,
    fallback_retry_interval_s: float = _FORCED_RESPONSE_FALLBACK_RETRY_INTERVAL_S,
    commit_may_defer_create: bool = True,
    on_fallback_sent: Callable[[], None] | None = None,
) -> bool:
    """Drain the commit-requested response, re-asking if it never opens.

    The commit carries ``response_create``, so the server answers its retained
    audio directly. The drain settles only when at least one response created
    after the commit has reached a terminal state, every post-commit creation
    is done -- matched by response id, so a pre-commit response settling after
    the commit cannot pose as the answer -- and no further response signal has
    arrived for ``settle_quiet_s`` seconds, which spans the window between a
    spontaneous settle and the promoted forced create.

    If no response opened, ``response.create`` is the protocol's second,
    explicit way to ask for the same answer. Whether a post-commit
    ``response.done`` may keep the re-ask armed depends on how the server can
    treat the commit (``commit_may_defer_create``): a session whose model does
    not auto-respond can hold the commit behind an active response and promote
    it with its forced create once that response settles, so a post-commit
    ``response.done`` -- even a cancelled one, or the settle of a pre-commit
    response -- means the promotion may still be in flight and the drain keeps
    waiting instead of re-asking into ``response_already_active``. A native
    auto-response session does not defer an audio commit taken while it is
    idle: it submits the retained input at once or acknowledges with nothing
    scheduled, so a post-commit settle is final -- the server is idle, and
    suppressing the re-ask there would strand an already-answered session in
    ``TimeoutError``; the re-ask stays armed and settles it on the forced
    answer or on the definitive refusal. (The deferrals such a session still
    has, and the caller keys ``commit_may_defer_create`` on all of them: a
    video-only commit -- no audio in the buffer while a response runs, on a
    model that accepts video without audio and has no concurrent-turn
    capability, and only when the session ever appended video frames; and a
    commit that landed while a response was already running -- there the
    runner rides the in-flight answer instead of precreating, so that settle
    belongs to the old response and the answer's own ``response.created``
    follows. A settle is treated as final only when none of the conditions
    above applies; a session whose model does not auto-respond always takes
    the park-and-promote deferral, so its commits are always treated as
    deferring ``(not auto_response)``.) Errors reported after the commit
    suppress the re-ask
    either way, since the server is refusing rather than slow. The flip side
    for CI triage on a deferring session: if the server drops the deferred
    create after such activity, the drain can only end in ``TimeoutError`` --
    no re-ask will have fired -- so a client timeout in that situation points
    at a lost promotion on the server side, not at a missing fallback.

    A re-ask can itself fail client-side: a send landing inside a transport
    resume window raises ``DuplexConnectionError`` while the session is still
    alive. Connection-level failures are therefore retried for up to
    ``fallback_retry_window_s`` (bounded by ``timeout_s``); a closed session
    or any other client error ends the re-ask immediately, and so does a
    response that finally opens on its own. The re-ask is idempotent
    server-side -- a duplicate create while a response is active is refused
    with ``response_already_active`` -- so a retry that races a promotion is
    harmless.

    A delivered re-ask can also be *refused*: when the commit retained no
    answerable audio (the model already answered every scene while the input
    was still streaming, so the final buffer is trailing silence), the server
    answers ``response.create`` with ``response_create_without_input`` or
    ``text_only_turn_unsupported`` instead of a response. Those refusals are
    definitive -- no amount of waiting produces the missing response -- so
    the drain settles on them and the caller forgives exactly those two
    error codes; whether the earlier answers actually covered the input is
    what the recognition judging decides.

    Returns whether the explicit fallback was sent. Raises ``TimeoutError``
    when the drain does not finish, so the caller records a client timeout
    instead of closing on a half-generated answer.
    """
    fallback_sent = False
    try:
        await wait_for_condition(
            lambda: _post_commit_response_created(collector.events, committed_index),
            timeout_s=create_timeout_s,
            label="post-commit forced response.created",
        )
    except TimeoutError:
        if (
            _event_count_after(collector.events, "error", committed_index) == 0
            and (
                not commit_may_defer_create
                or _event_count_after(collector.events, "response.done", committed_index) == 0
            )
            and not _response_in_progress(collector.events)
        ):
            retry_deadline_s = time.monotonic() + min(fallback_retry_window_s, timeout_s)
            attempt = 0
            while not fallback_sent:
                if _post_commit_response_created(collector.events, committed_index):
                    break
                if (
                    _event_count_after(collector.events, "error", committed_index)
                    or (
                        commit_may_defer_create
                        and _event_count_after(collector.events, "response.done", committed_index)
                    )
                    or _response_in_progress(collector.events)
                ):
                    break
                attempt += 1
                try:
                    await client.send({"type": "response.create"})
                    fallback_sent = True
                    if on_fallback_sent is not None:
                        # The caller reports the fallback even when this drain
                        # goes on to time out: a timeout must not read as "the
                        # re-ask was never sent".
                        on_fallback_sent()
                    if attempt > 1:
                        # Confirms the retry rescued the re-ask; a first-try
                        # success needs no extra noise.
                        print(
                            f"[forced-response] response.create fallback succeeded on attempt {attempt}",
                            flush=True,
                        )
                except (DuplexClientError, DuplexSessionClosedError) as exc:
                    # Surface which exception delayed the re-ask: the drain
                    # timeout alone cannot say which side failed.
                    print(
                        f"[forced-response] response.create fallback attempt {attempt} failed: {exc!r}",
                        flush=True,
                    )
                    if not isinstance(exc, DuplexConnectionError):
                        break
                    remaining_s = retry_deadline_s - time.monotonic()
                    if remaining_s <= 0:
                        break
                    await asyncio.sleep(min(fallback_retry_interval_s, remaining_s))

    def _drained() -> bool:
        if _post_commit_definitive_refusal(collector.events, committed_index):
            # The server refused to force an answer from the committed turn:
            # nothing more can arrive for this drain. Whether the input was
            # in fact already answered is the recognition report's judgement.
            return True
        if not (
            _post_commit_forced_response_settled(collector.events, committed_index)
            and not _response_in_progress(collector.events)
        ):
            return False
        if settle_quiet_s <= 0:
            return True
        last_signal_at_s = _last_post_commit_response_signal_at_s(collector.events, committed_index)
        if last_signal_at_s is None:
            return True
        return time.monotonic() - last_signal_at_s >= settle_quiet_s

    await wait_for_condition(
        _drained,
        timeout_s=timeout_s,
        label="post-commit forced response drain",
    )
    return fallback_sent


def _video_has_audio(video_path: Path) -> bool:
    import av

    with av.open(str(video_path)) as container:
        return bool(container.streams.audio)


def _probe_video_frames_and_fps(video_path: Path) -> tuple[int, float]:
    """Frame count and average FPS via PyAV (same approach as Daily-Omni)."""
    import av

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        avg_fps = float(stream.average_rate) if stream.average_rate else 0.0
        num_frames = stream.frames or 0
        if num_frames <= 0:
            duration: float | None = None
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = container.duration / av.time_base
            if duration and avg_fps:
                num_frames = int(math.floor(duration * avg_fps))
        if num_frames <= 0:
            stream.thread_type = "AUTO"
            num_frames = sum(1 for _ in container.decode(stream))
    if num_frames <= 0:
        raise ValueError(f"No decodable video frames in {video_path}")
    return num_frames, avg_fps or 1.0


def _upright(image, rotation: int):
    """Apply a container display-matrix rotation, in degrees, to ``image``.

    PyAV hands back the stored raster and leaves the display matrix for the
    caller, unlike the ffmpeg CLI which autorotates. Phone clips routinely
    carry +/-90, so skipping this feeds the model a sideways frame: arrows and
    digits stop being readable and the model mirrors the motion it reports.
    """
    from PIL import Image

    quarter_turns = {
        90: Image.Transpose.ROTATE_90,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_270,
    }
    turn = quarter_turns.get(int(rotation) % 360)
    return image.transpose(turn) if turn is not None else image


def _decode_video_frames_rgb(video_path: Path, frame_idx: list[int]) -> list:
    """Decode presentation-order ``frame_idx`` into upright RGB ``PIL.Image``s."""
    import av

    wanted = sorted({int(i) for i in frame_idx})
    if not wanted:
        return []

    decoded: dict[int, object] = {}
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        cursor = 0
        for position, frame in enumerate(container.decode(stream)):
            if wanted[cursor] > position:
                continue
            image = _upright(frame.to_image(), getattr(frame, "rotation", 0) or 0)
            while cursor < len(wanted) and wanted[cursor] <= position:
                decoded[wanted[cursor]] = image
                cursor += 1
            if cursor >= len(wanted):
                break

    if not decoded:
        raise ValueError(f"Decoded no frames from {video_path} for indices {wanted[:8]}")
    last_image = decoded[max(decoded)]
    return [decoded.get(int(i), last_image) for i in frame_idx]


def _resize_max_side(image, max_side: int):
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side or max_side <= 0:
        return image
    scale = max_side / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size)


def _jpeg_bytes(image, *, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _to_pcm16_bytes(audio_np) -> bytes:
    import numpy as np

    arr = np.asarray(audio_np).reshape(-1)
    if arr.dtype != np.int16:
        arr = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
    return arr.tobytes()


def _pcm16_wav_duration_s(path: Path) -> float:
    pcm16 = read_pcm16_wav(path)
    return len(pcm16) / float(PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE)


def _extract_video_frames(
    video_path: Path,
    work_dir: Path,
    *,
    fps: float,
    max_side: int,
    duration_s: float,
    stack_frames: int = 1,
) -> tuple[list[str], list[str | None]]:
    """Demux JPEG frames only. Frame count follows ``duration_s``, not soundtrack.

    Returns the base frames (one per model unit) and the parallel stacked track.
    With ``stack_frames=N`` each unit also samples its ``N-1`` interior
    sub-frames and tiles them into one composite, which is how official duplex
    raises the visual refresh rate while keeping one second of audio per unit.
    A unit whose interior falls past the end of the clip stacks nothing.
    """
    num_frames, avg_fps = _probe_video_frames_and_fps(video_path)
    sample_fps = max(float(fps), 1e-6)
    unit_s = 1.0 / sample_fps
    timeline_s = max(float(duration_s), unit_s)
    num_samples = max(1, int(math.ceil(timeline_s * sample_fps - 1e-9)))
    offsets = unit_subframe_offsets(stack_frames, unit_s=unit_s)

    def frame_at(seconds: float) -> int:
        # Official get_video_frame_audio_segments uses truncating int(), not
        # round(), so 29.97 fps lands on the same presentation index.
        return min(int(seconds * avg_fps), num_frames - 1)

    base_idx = [frame_at(unit * unit_s) for unit in range(num_samples)]
    sub_times = [
        [unit * unit_s + offset for offset in offsets if unit * unit_s + offset < timeline_s]
        for unit in range(num_samples)
    ]
    sub_idx = [frame_at(seconds) for times in sub_times for seconds in times]
    images = _decode_video_frames_rgb(video_path, base_idx + sub_idx)
    sub_images = iter(images[len(base_idx) :])

    frame_dir = work_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for stale in frame_dir.glob("*.jpg"):
        stale.unlink()
    frames_b64: list[str] = []
    stacked_b64: list[str | None] = []
    for index, (image, times) in enumerate(zip(images[: len(base_idx)], sub_times), start=1):
        jpeg = _jpeg_bytes(_resize_max_side(image, max_side))
        (frame_dir / f"frame_{index:05d}.jpg").write_bytes(jpeg)
        frames_b64.append(base64.b64encode(jpeg).decode("ascii"))
        interior = [_resize_max_side(next(sub_images), max_side) for _ in times]
        if not interior:
            stacked_b64.append(None)
            continue
        composite = concat_frames_b64(interior)
        (frame_dir / f"stack_{index:05d}.jpg").write_bytes(base64.b64decode(composite))
        stacked_b64.append(composite)
    return frames_b64, stacked_b64


def _extract_video_soundtrack(video_path: Path, work_dir: Path) -> Path:
    """Write a 16 kHz mono WAV from the video soundtrack."""
    try:
        from vllm.multimodal.media.audio import load_audio
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("--input-video needs vLLM audio helpers (load_audio)") from exc

    audio_np, sr = load_audio(str(video_path), sr=PCM16_SAMPLE_RATE, mono=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    audio_path = work_dir / "input_16k.wav"
    write_pcm16_wav(audio_path, _to_pcm16_bytes(audio_np), sample_rate_hz=int(sr))
    return audio_path


def _extract_video_input(
    video_path: Path,
    work_dir: Path,
    *,
    fps: float,
    max_side: int,
    stack_frames: int,
) -> tuple[Path, list[str], list[str | None]]:
    """Demux a video into duplex inputs: 16 kHz mono WAV plus JPEG frames.

    Use this only when the video soundtrack is the audio source. When the
    caller already has ``--input-wav``, extract frames against that WAV
    duration instead so a silent clip cannot fail ``load_audio``.
    """
    num_frames, frame_rate = _probe_video_frames_and_fps(video_path)
    video_duration_s = num_frames / frame_rate
    audio_path = _extract_video_soundtrack(video_path, work_dir)
    # Match official get_audio_segments(adjust_length=True): the video
    # timeline determines the number of one-second audio units. Decoders
    # can retain AAC padding (e.g. 180.032 s for a 180 s video); treating it
    # as real input creates an extra, frame-less unit on the final commit.
    unit_count = max(1, int(math.ceil(video_duration_s - 1e-9)))
    target_bytes = unit_count * PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE
    pcm = read_pcm16_wav(audio_path)
    aligned_pcm = pcm[:target_bytes].ljust(target_bytes, b"\0")
    if aligned_pcm != pcm:
        write_pcm16_wav(audio_path, aligned_pcm, sample_rate_hz=PCM16_SAMPLE_RATE)
    frames_b64, stacked_b64 = _extract_video_frames(
        video_path,
        work_dir,
        fps=fps,
        max_side=max_side,
        duration_s=video_duration_s,
        stack_frames=stack_frames,
    )
    return audio_path, frames_b64, stacked_b64


def _resolve_duplex_av_inputs(
    *,
    input_wav: str | None,
    input_video: str | None,
    work_dir: Path,
    fps: float,
    max_side: int,
    stack_frames: int = 1,
) -> tuple[str | None, list[str], list[str | None]]:
    """Return ``(wav_path, frames_b64, stacked_b64)``. External WAV wins over video audio."""
    if input_video:
        video_path = Path(input_video)
        if input_wav:
            frames, stacked = _extract_video_frames(
                video_path,
                work_dir,
                fps=fps,
                max_side=max_side,
                duration_s=_pcm16_wav_duration_s(Path(input_wav)),
                stack_frames=stack_frames,
            )
            return input_wav, frames, stacked
        if not _video_has_audio(video_path):
            frame_count, frame_rate = _probe_video_frames_and_fps(video_path)
            frames, stacked = _extract_video_frames(
                video_path,
                work_dir,
                fps=fps,
                max_side=max_side,
                stack_frames=stack_frames,
                duration_s=frame_count / frame_rate,
            )
            return None, frames, stacked
        extracted_wav, frames, stacked = _extract_video_input(
            video_path,
            work_dir,
            fps=fps,
            max_side=max_side,
            stack_frames=stack_frames,
        )
        return str(extracted_wav), frames, stacked
    if not input_wav:
        raise SystemExit("provide --input-wav, --input-video, or both")
    return input_wav, [], []


async def run_demo(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    stream_writer = _StreamingOutputWriter(output_dir)

    input_wav, video_frames, stacked_video_frames = _resolve_duplex_av_inputs(
        input_wav=args.input_wav,
        input_video=args.input_video,
        work_dir=output_dir / "video_input",
        fps=args.video_fps,
        max_side=args.frame_max_side,
        stack_frames=args.stack_frames,
    )

    input_pcm16 = read_pcm16_wav(Path(input_wav)) if input_wav is not None else b""
    if input_wav is not None and not input_pcm16:
        raise ValueError("input WAV has no audio")

    collector = _StreamingEventCollector(stream_writer)
    client = _TimedInputClient(
        args.url,
        model=args.model,
        config=create_duplex_session_config(
            ref_audio=_ref_audio_data_url(args.ref_audio),
            temperature=args.temperature,
            instructions=args.instructions,
            extra_body={
                "sliding_window_mode": args.sliding_window_mode,
                "basic_window_high_tokens": args.basic_window_high_tokens,
                "basic_window_low_tokens": args.basic_window_low_tokens,
                "context_max_units": args.context_max_units,
                "context_previous_max_tokens": args.context_previous_max_tokens,
            },
        ),
        reconnect=None,
        heartbeat_interval_s=None,
        handshake_timeout_s=args.timeout_s,
    )
    async with client:
        consume_task = asyncio.create_task(collector.consume(client))
        # The handshake consumed session.created before the collector
        # subscribed; seed it so counts and capability lookups still work.
        collector.add({"type": "session.created", "session": client.session_info})
        stream_event_cursor = len(collector.events)
        input_stream_started_at_s = time.monotonic()
        if input_wav is None:
            frames_sent = 0
            frame_period_s = 1.0 / max(args.video_fps, 1e-6)
            for i, frame in enumerate(video_frames):
                frames = [frame]
                if i < len(stacked_video_frames) and stacked_video_frames[i] is not None:
                    frames.append(stacked_video_frames[i])
                await client.send({"type": "input_audio_buffer.append", "video_frames": frames})
                frames_sent += 1
                if not args.no_realtime_pacing:
                    # Same media clock as stream_pcm: a sleep of one frame period
                    # after each send would accumulate send overhead.
                    delay_s = input_stream_started_at_s + frames_sent * frame_period_s - time.monotonic()
                    await asyncio.sleep(max(0.0, delay_s))
        else:
            frames_sent = await client.stream_pcm(
                input_pcm16,
                chunk_ms=args.chunk_ms,
                realtime=not args.no_realtime_pacing,
                video_frames=video_frames,
                stacked_video_frames=stacked_video_frames,
            )
        # Let the model answer the final utterance while holding the last frame.
        # A commit alone does not supply more model time to native duplex.
        tail_frames_sent = 0
        if args.tail_silence_s > 0:
            tail_pcm = bytes(round(args.tail_silence_s * PCM16_SAMPLE_RATE) * PCM16_BYTES_PER_SAMPLE)
            # PCM still buffered on the server keeps its media clock: a fresh
            # stream would attach the held frame one second into the tail,
            # after the unit that contains the end of the utterance closed.
            # A clip with no audio has no buffered PCM, so the tail holds the
            # last frame on its own clock.
            tail_timeline = (
                {
                    "video_frames": video_frames,
                    "stacked_video_frames": stacked_video_frames,
                    "media_origin_ms": client.config.input_audio.duration_ms(len(input_pcm16)),
                    "unit_index": frames_sent,
                }
                if input_pcm16
                else {
                    "video_frames": video_frames[-1:],
                    "stacked_video_frames": stacked_video_frames[-1:],
                }
            )
            tail_frames_sent = await client.stream_pcm(
                tail_pcm,
                chunk_ms=args.chunk_ms,
                realtime=not args.no_realtime_pacing,
                **tail_timeline,
            )
            input_pcm16 += tail_pcm
        commit_event_cursor = len(collector.events)
        stream_decision = _latest_model_decision(collector.events, stream_event_cursor)
        input_has_residual_model_unit = _has_residual_model_unit(
            input_pcm16,
            chunk_period_ms=_chunk_period_ms(collector.events),
        )
        # ``getattr`` keeps every Namespace built before this flag existed
        # working unchanged: only callers that ask for a mandatory final
        # answer opt in.
        final_create_response = bool(getattr(args, "final_create_response", False))
        forced_response_fallback_sent = False
        # Whether this session ever appended video frames: only such a session
        # can take the runner's video-only commit deferral (see the drain call
        # below). The recognition driver never appends video, so CI runs
        # record False here by construction.
        video_frames_appended = bool(video_frames)
        # Whether a response was still running when the final commit was
        # issued: the runner then rides the in-flight answer instead of
        # precreating, so a settle right after the commit belongs to the old
        # response rather than to an idle server.
        response_in_progress_at_commit = False
        # A drain timeout must not erase the fact that the re-ask reached the
        # server: record it as soon as the send succeeds, then fold it back in
        # after the wait ends.
        fallback_progress = {"sent": False}
        wait_for_post_commit_decision = False
        commit_sent_at_s = await _commit_input_after_playback_checkpoint(
            client,
            collector,
            create_response=final_create_response,
        )
        wait_error: str | None = None
        committed_index: int | None = None
        post_commit_decision: str | None = None
        try:
            await wait_for_condition(
                lambda: _input_committed_index(collector.events, commit_event_cursor) is not None,
                timeout_s=args.timeout_s,
                label="input_audio_buffer.committed",
            )
            committed_index = _input_committed_index(collector.events, commit_event_cursor)
            stream_decision = _latest_model_decision(collector.events[: committed_index + 1], stream_event_cursor)
            wait_for_post_commit_decision = (
                final_create_response
                or input_has_residual_model_unit
                or _response_in_progress(collector.events[: committed_index + 1])
            )
            if wait_for_post_commit_decision:
                if final_create_response:
                    response_in_progress_at_commit = _response_in_progress(collector.events[: committed_index + 1])
                    forced_response_fallback_sent = await _drain_forced_post_commit_response(
                        client,
                        collector,
                        committed_index=committed_index,
                        timeout_s=args.timeout_s,
                        create_timeout_s=min(_FORCED_RESPONSE_CREATE_TIMEOUT_S, float(args.timeout_s)),
                        # When may a settle that lands after the commit be the
                        # preamble of more server work, so the re-ask must
                        # stay armed instead of firing?
                        # - A session whose model does not auto-respond can
                        #   park the forced create behind an active response
                        #   and promote it later.
                        # - A session that ever appended video frames can take
                        #   the video-only deferral (a commit with no audio
                        #   while a response runs, on a model that accepts
                        #   video without audio; the runner drops the deferral
                        #   when the payload carries audio). This driver's
                        #   recognition runs never append video, so the flag
                        #   is False for them by construction -- recorded in
                        #   ``post_commit.video_frames_appended``.
                        # - The commit landed while a response was running:
                        #   the runner then drops the forced precreate and the
                        #   terminal input rides the in-flight answer instead,
                        #   so that settle belongs to the old response and the
                        #   answer's own ``response.created`` follows.
                        # The client-side flag is the server's own value, not
                        # a guess: ``SessionConfig.to_payload`` puts this same
                        # ``auto_response`` into the session's ``extra_body``
                        # at connect time and the server derives its
                        # auto-response scheduling from exactly that key.
                        commit_may_defer_create=(
                            not client.config.auto_response or bool(video_frames) or response_in_progress_at_commit
                        ),
                        on_fallback_sent=lambda: fallback_progress.__setitem__("sent", True),
                    )
                else:
                    await wait_for_condition(
                        lambda: _post_commit_model_decision(collector.events, committed_index) is not None,
                        timeout_s=args.timeout_s,
                        label="post-commit model decision or response drain",
                    )
                post_commit_decision = _post_commit_model_decision(collector.events, committed_index)
        except TimeoutError as exc:
            wait_error = str(exc)
        forced_response_fallback_sent = forced_response_fallback_sent or fallback_progress["sent"]
        await acknowledge_collected_playback(client, collector)
        close_error: str | None = None
        await client.close(timeout_s=args.timeout_s)
        # close() returns when the client's reader observed session.closed;
        # the collector is a separate subscriber task, so wait for it to
        # drain before checking, or a normal close reads as a failure.
        try:
            await asyncio.wait_for(consume_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            consume_task.cancel()
        if collector.count("session.closed") == 0:
            close_error = "Timed out waiting for session.closed"

        audio = collector.audio_bytes()
        first_text_at_s = collector.first_received_at(
            "response.output_audio_transcript.delta",
            "response.output_text.delta",
            after_s=commit_sent_at_s,
        )
        first_audio_at_s = collector.first_received_at(
            "response.output_audio.delta",
            after_s=commit_sent_at_s,
        )
        response_created_at_s = collector.first_received_at(
            "response.created",
            after_s=commit_sent_at_s,
        )
        response_done_at_s = collector.first_received_at(
            "response.done",
            after_s=commit_sent_at_s,
        )
        audio_duration_s = len(audio) / (collector.output_sample_rate_hz * 2)
        response_generation_s = (
            response_done_at_s - response_created_at_s
            if response_done_at_s is not None and response_created_at_s is not None
            else None
        )
        transcript_deltas = [
            str(event.get("delta", ""))
            for event in collector.events
            if event.get("type")
            in {
                "response.output_audio_transcript.delta",
                "response.output_text.delta",
            }
        ]
        response_id = collector.response_ids[0] if collector.response_ids else None
        timing = collector.timing_summary(
            after_s=commit_sent_at_s,
            input_committed_at_s=commit_sent_at_s,
            response_id=response_id,
        )
        # The forced final create can be answered by a definitive refusal --
        # the commit retained nothing answerable, so the final answer was
        # already given by an earlier response (or the input was not
        # answerable at all, which the recognition report judges). Only
        # refusals that arrived after the commit are forgiven: an earlier one
        # came from some other ask and still fails the run, as does every
        # other error.
        forgiven_indexes = (
            _post_commit_refusal_indexes(collector.events, committed_index)
            if final_create_response and committed_index is not None
            else frozenset()
        )
        errors = collector.errors(exclude_indexes=forgiven_indexes)
        if wait_error:
            errors.append({"type": "client.timeout", "message": wait_error})
        if close_error:
            errors.append({"type": "client.timeout", "message": close_error})
        model_decision = post_commit_decision or stream_decision
        (output_dir / "events.jsonl").write_text(
            "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in collector.events),
            encoding="utf-8",
        )
        (output_dir / "input_timeline.json").write_text(json.dumps(client.input_timeline, indent=2), encoding="utf-8")
        (output_dir / "output.pcm").write_bytes(audio)
        if audio:
            write_pcm16_wav(
                output_dir / "output.wav",
                audio,
                sample_rate_hz=collector.output_sample_rate_hz,
            )

        result = {
            "validation_scope": "transport_and_lifecycle_only",
            "ok": (
                collector.count("session.created") > 0
                and collector.count("session.closed") > 0
                and not errors
                and model_decision is not None
                and (bool(audio) or not args.require_audio)
            ),
            "model_decision": model_decision,
            "input_stream_started_at_s": input_stream_started_at_s,
            "realtime_pacing": not args.no_realtime_pacing,
            "sampling_temperature": args.temperature,
            "sliding_window_mode": args.sliding_window_mode,
            "basic_window_high_tokens": args.basic_window_high_tokens,
            "basic_window_low_tokens": args.basic_window_low_tokens,
            "context_max_units": args.context_max_units,
            "context_previous_max_tokens": args.context_previous_max_tokens,
            "tail_silence_s": args.tail_silence_s,
            "tail_video_frames_sent": tail_frames_sent,
            "post_commit": {
                "input_committed_event_index": committed_index,
                "decision": post_commit_decision,
                "decision_required": wait_for_post_commit_decision,
                "final_create_response": final_create_response,
                "forced_response_fallback_sent": forced_response_fallback_sent,
                "input_had_residual_model_unit": input_has_residual_model_unit,
                "video_frames_appended": video_frames_appended,
                "response_in_progress_at_commit": response_in_progress_at_commit,
                "response_created_count": _event_count_after(
                    collector.events,
                    "response.created",
                    committed_index,
                ),
                "response_listen_count": _event_count_after(
                    collector.events,
                    "response.listen",
                    committed_index,
                ),
                "response_done_count": _event_count_after(
                    collector.events,
                    "response.done",
                    committed_index,
                ),
            },
            "input_video": args.input_video,
            "input_wav": args.input_wav,
            "video_frames_extracted": len(video_frames),
            "video_frames_sent": frames_sent,
            "video_stacked_frames_extracted": sum(frame is not None for frame in stacked_video_frames),
            "audio_bytes": len(audio),
            "audio_chunk_count": len(stream_writer.audio_chunk_paths),
            "audio_chunk_files": [str(path) for path in stream_writer.audio_chunk_paths],
            "output_sample_rate_hz": collector.output_sample_rate_hz,
            "latency": {
                "ttft_ms": (
                    round((first_text_at_s - commit_sent_at_s) * 1000, 2) if first_text_at_s is not None else None
                ),
                "ttfp_ms": (
                    round((first_audio_at_s - commit_sent_at_s) * 1000, 2) if first_audio_at_s is not None else None
                ),
                "rtf": (
                    round(response_generation_s / audio_duration_s, 4)
                    if response_generation_s is not None and audio_duration_s > 0
                    else None
                ),
                "response_generation_ms": (
                    round(response_generation_s * 1000, 2) if response_generation_s is not None else None
                ),
                "text_stream_ms": (
                    round((response_done_at_s - first_text_at_s) * 1000, 2)
                    if response_done_at_s is not None and first_text_at_s is not None
                    else None
                ),
                "transcript_delta_count": len(transcript_deltas),
                "audio_duration_s": round(audio_duration_s, 3),
                "measurement_origin": "input_audio_buffer.commit send",
            },
            "timing": timing,
            "response_ids": collector.response_ids,
            "transcript": "".join(transcript_deltas),
            "errors": errors,
            "output_dir": str(output_dir),
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8099/v1/realtime?duplex=1")
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument(
        "--input-wav",
        help="Mono 16 kHz PCM16 clip to stream. Optional when --input-video carries the audio track.",
    )
    parser.add_argument(
        "--input-video",
        help=(
            "Video to stream as omni duplex input. PyAV demuxes frames and "
            "vLLM load_audio extracts a 16 kHz mono WAV (unless --input-wav overrides the audio)."
        ),
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=1.0,
        help="Frame extraction rate. The wire contract carries ~1 frame per second of audio.",
    )
    parser.add_argument(
        "--frame-max-side",
        type=int,
        default=0,
        help=(
            "Optional client-side downscale of the longest side, in pixels. "
            "0 (default) keeps the capture resolution so the server "
            "process_image can normalize at scale_resolution=448 from a "
            "higher-quality source. The web demo never resizes on the client."
        ),
    )
    parser.add_argument(
        "--stack-frames",
        type=int,
        default=1,
        help=(
            "Frames sampled per 1 s audio unit. 1 (default) sends the unit's "
            "frame alone. N > 1 also samples the N-1 sub-frames inside the unit "
            "and tiles them into one composite sent next to the base frame, so "
            "motion (arrows, doors, digits changing) is visible at N fps while "
            "the audio stays 1 s per unit. Official uses 5 for high refresh "
            "rate mode; the wire carries 2 images per unit at any N because the "
            "sub-frames share one composite."
        ),
    )
    parser.add_argument(
        "--ref-audio",
        help=(
            "Optional reference WAV for the MiniCPM-o duplex assistant voice. "
            "The server defaults to the model-bundled assets/HT_ref_audio.wav."
        ),
    )
    parser.add_argument("--output-dir", default="/tmp/minicpmo_realtime_duplex_demo")
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument(
        "--instructions",
        help="Session system prompt. Omit to use the model's Streaming Omni Conversation default.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Stage 0 sampling temperature; omit to preserve the model default.",
    )
    parser.add_argument("--no-realtime-pacing", action="store_true")
    parser.add_argument("--sliding-window-mode", choices=("off", "basic", "context"), default="off")
    parser.add_argument("--basic-window-high-tokens", type=int, default=8000)
    parser.add_argument("--basic-window-low-tokens", type=int, default=6000)
    parser.add_argument("--context-max-units", type=int, default=24)
    parser.add_argument("--context-previous-max-tokens", type=int, default=500)
    parser.add_argument(
        "--tail-silence-s",
        type=float,
        default=0,
        help="Additional silent seconds holding the last frame to allow the final answer to finish.",
    )
    parser.add_argument(
        "--final-create-response",
        action="store_true",
        help=(
            "Ask the server for one response on the final commit and drain it "
            "before closing. Native duplex lets the model keep listening, so "
            "without this the last user turn can stay unanswered."
        ),
    )
    parser.add_argument("--require-audio", action="store_true")
    args = parser.parse_args()
    if args.tail_silence_s < 0:
        parser.error("--tail-silence-s must be nonnegative")
    if not 0 < args.basic_window_low_tokens < args.basic_window_high_tokens:
        parser.error("window watermarks must satisfy 0 < low < high")
    if args.context_max_units <= 0 or args.context_previous_max_tokens <= 0:
        parser.error("context window limits must be positive")
    return args


def main() -> None:
    result = asyncio.run(run_demo(parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
