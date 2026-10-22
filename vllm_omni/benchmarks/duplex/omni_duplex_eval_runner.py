# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Single-sample duplex generation.

Model-native duplex models (MiniCPM-o) hear the whole clip and are committed
once at the end. Turn-based models (Qwen3-Omni) run with ``server_vad`` turn
detection: the server cuts the question audio into turns, and the runner waits
for observed speech/responses to settle instead of committing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pybase64 as base64

from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import DuplexSample
from vllm_omni.benchmarks.duplex_session_inputs import (
    FRAME_TRANSPORT_IMAGE_ITEMS,
    SERVER_VAD_FRAME_MS,
    SERVER_VAD_TAIL_S,
    TURN_DETECTION_MODES,
    TURN_DETECTION_NONE,
    TURN_DETECTION_SERVER_VAD,
    ImageItemWindow,
    select_frame_transport,
    server_turns_pending,
    session_turn_detection,
    turn_activity_count,
)
from vllm_omni.experimental.fullduplex.client import (
    PCM16_BYTES_PER_SAMPLE,
    PCM16_SAMPLE_RATE,
    RealtimeDuplexClient,
    RealtimeEventCollector,
    build_realtime_url,
    wait_for,
)

if TYPE_CHECKING:
    from vllm_omni.clients.duplex import EventCollector

from .omni_duplex_eval_clock import extract_timed_chunks, extract_timed_sentences, has_response_chunks
from .omni_duplex_eval_media import (
    iter_av_units,
    iter_jpegs,
    materialize_media,
    mix_pcm16,
    read_audio_pcm16,
    read_soundtrack_pcm16,
    video_duration,
)

#: MiniCPM-o's native duplex system prompt, the historical default.
DEFAULT_INSTRUCTIONS = "Streaming Omni Conversation."
#: Session audio: the instruction alone, or the instruction over the clip's own audio track.
MIX_QUESTION = "question"
MIX_SOUNDTRACK = "soundtrack"
MIX_MODES = (MIX_QUESTION, MIX_SOUNDTRACK)
_SERVER_TURN_SETTLE_S = 2.0
_COMPLETION_SETTLE_S = 2.0
# Events that can still open or extend a response after the commit. Playback
# acknowledgements are omitted: the runner sends one every 250 ms for any
# response that produced audio, and the server echoes each one, so counting
# them would keep this window from ever elapsing.
_COMPLETION_ACTIVITY = frozenset(
    {
        "input_audio_buffer.committed",
        "response.created",
        "response.done",
        "response.listen",
        "response.output_audio.delta",
        "response.output_audio_transcript.delta",
        "response.output_text.delta",
        "response.text.delta",
    }
)


@dataclass(frozen=True)
class GenerateSampleResult:
    """Timed-sentence path plus duplex metrics for one generate sample."""

    output: Path
    request_metrics: list[dict[str, object]] = field(default_factory=list)
    session_metrics: dict[str, object] = field(default_factory=dict)
    timed_sentences: list[dict[str, object]] = field(default_factory=list)
    timed_chunks: list[dict[str, object]] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    output_tokens: int = 0
    audio_bytes: int = 0
    audio_sample_rate: int = 24_000
    error: str = ""


@dataclass(frozen=True)
class PreparedReferenceAudio:
    data_url: str = field(repr=False)
    sha256: str

    @classmethod
    def from_path(cls, path: str | Path) -> PreparedReferenceAudio:
        value = Path(path).expanduser().read_bytes()
        return cls(
            "data:audio/wav;base64," + base64.b64encode(value).decode("ascii"), hashlib.sha256(value).hexdigest()
        )


@dataclass(frozen=True)
class PreparedSample:
    """Decoded input, reusable across readiness, warmup and measured sessions."""

    pcm: bytes = field(repr=False)
    frames: tuple[tuple[float, bytes], ...] = field(repr=False)
    duration: float
    ref_audio: str | None = field(repr=False)
    ref_audio_sha256: str | None
    mix: str = MIX_QUESTION


def prepare_sample(
    sample: DuplexSample,
    *,
    media_dir: Path,
    ref_audio: str | Path | PreparedReferenceAudio | None,
    fps: float,
    mix: str = MIX_QUESTION,
) -> PreparedSample:
    if mix not in MIX_MODES:
        raise ValueError(f"mix must be one of {MIX_MODES}")
    audio_path = materialize_media(sample.question_audio, media_dir, f"{sample.id}_question", ".wav")
    video_path = materialize_media(sample.video, media_dir, sample.id, ".mp4")
    duration = sample.video_duration or video_duration(video_path)
    reference = ref_audio
    if isinstance(reference, str | Path):
        reference = PreparedReferenceAudio.from_path(reference)
    pcm = read_audio_pcm16(audio_path)
    if mix == MIX_SOUNDTRACK:
        # The dataset's question audio is padded to the clip length so it can
        # be overlaid on the clip's own audio at the original timestamps.
        pcm = mix_pcm16(pcm, read_soundtrack_pcm16(video_path))
    return PreparedSample(
        pcm=pcm,
        frames=tuple(iter_jpegs(video_path, fps=fps, duration=duration)),
        duration=duration,
        ref_audio=reference.data_url if reference is not None else None,
        ref_audio_sha256=reference.sha256 if reference is not None else None,
        mix=mix,
    )


def _cached_response_mix(output: Path) -> str:
    """Return the mix recorded for an existing sample; missing metadata is ``question``."""
    meta_path = output.with_name(output.stem + ".meta.json")
    if not meta_path.exists():
        return MIX_QUESTION
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read mix from {meta_path}: {exc}") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"cannot read mix from {meta_path}: metadata must be a JSON object")
    mix = meta.get("mix", MIX_QUESTION)
    if mix not in MIX_MODES:
        raise ValueError(f"{meta_path} records unsupported mix={mix!r}")
    return str(mix)


def _cached_response_has_chunks(output: Path) -> bool:
    """Whether a cached response can be scored by chunk time.

    A bare sentence list has no per-chunk ``current_time``. Reusing it for a
    proactive-reminder sample would score it by sentence start while newly
    written files in the same directory are scored by chunk time.
    """
    try:
        raw = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read response {output}: {exc}") from exc
    return has_response_chunks(raw)


def write_sample_result(result: GenerateSampleResult) -> None:
    """Publish the judge input (RTD reads sentences, PR reads chunks); serving calls this after timing."""
    output = result.output
    output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output.with_name(output.stem + ".meta.json")
    # Publish the judge-visible JSON last. A failed write cannot leave a partial
    # response that the evaluator would mistake for a complete sample.
    meta_path.write_text(json.dumps(result.metadata, indent=2), encoding="utf-8")
    temporary = output.with_suffix(".json.part")
    try:
        response = {"sentences": result.timed_sentences, "chunks": result.timed_chunks}
        temporary.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _event_collector_from_realtime(source: RealtimeEventCollector) -> EventCollector:
    """Replay experimental collector events onto the public EventCollector."""
    from vllm_omni.clients.duplex import EventCollector

    collector = EventCollector()
    for event, received_at_s in zip(source.events, source.event_received_at_s, strict=True):
        collector.add(event, received_at_s=received_at_s)
    for record in source.client_appends:
        sent_at_s = record.get("sent_at_s")
        event_id = record.get("event_id")
        if not isinstance(sent_at_s, int | float) or not isinstance(event_id, str):
            continue
        collector.note_client_append({"sent_at_s": float(sent_at_s), "event_id": event_id})
    return collector


def _with_sample_identity(payload: dict[str, object], sample: DuplexSample) -> dict[str, object]:
    return {"sample_id": sample.id, "split": sample.split, **payload}


def _session_capabilities(client: RealtimeDuplexClient) -> dict[str, object]:
    for event in client.events.events:
        session = event.get("session") if event.get("type") == "session.created" else None
        if isinstance(session, dict) and isinstance(session.get("capabilities"), dict):
            return session["capabilities"]
    return {}


def _is_model_listen_decision(event: dict[str, object]) -> bool:
    response = event.get("response")
    metadata = response.get("metadata") if isinstance(response, dict) else event.get("metadata")
    return isinstance(metadata, dict) and metadata.get("buffering") is not True and metadata.get("model_listen") is True


def _event_response_id(event: dict[str, object]) -> str | None:
    response_id = event.get("response_id")
    if isinstance(response_id, str):
        return response_id
    response = event.get("response")
    if isinstance(response, dict):
        nested = response.get("id")
        if isinstance(nested, str):
            return nested
    return None


def _response_status(event: dict[str, object]) -> str | None:
    response = event.get("response")
    if isinstance(response, dict) and isinstance(response.get("status"), str):
        return str(response["status"])
    status = event.get("status")
    return status if isinstance(status, str) else None


def _completion_activity(events: list[dict[str, object]], start: int) -> int:
    return sum(event.get("type") in _COMPLETION_ACTIVITY for event in events[start:])


def _residual_unit_decided(events: list[dict[str, object]], start: int) -> bool:
    """Whether the unit flushed by the commit has decided.

    A response created before ``input_audio_buffer.committed`` is the in-flight
    reply. Ending it normally promotes the residual unit, so that ``response.done``
    is not the decision; the residual then emits ``response.listen`` or opens a
    new response. A ``cancelled`` ending is not a decision either: MiniCPM-o
    barge-in cancels the open turn and the same input can still open the residual
    response after the quiet window. A ``failed`` ending is. The server does not
    promote another unit after a failure, and a failed response created after the
    commit is itself the residual; the sample is already unsuccessful.
    """
    committed_at = next(
        (index for index in range(start, len(events)) if events[index].get("type") == "input_audio_buffer.committed"),
        None,
    )
    if committed_at is None:
        return False
    created: set[str] = set()
    finished: set[str] = set()
    for event in events[:committed_at]:
        event_type = event.get("type")
        response_id = _event_response_id(event) if event_type in {"response.created", "response.done"} else None
        if response_id is None:
            continue
        if event_type == "response.created":
            created.add(response_id)
        else:
            finished.add(response_id)
    open_at_commit = created - finished
    created_after: set[str] = set()
    finished_open: set[str] = set()
    cancelled_open = False
    failed_open = False
    promotes_residual = False
    for event in events[committed_at + 1 :]:
        event_type = event.get("type")
        if event_type == "response.created":
            response_id = _event_response_id(event)
            if response_id is not None:
                created_after.add(response_id)
            continue
        if event_type == "response.done":
            response_id = _event_response_id(event)
            status = _response_status(event)
            if response_id in open_at_commit:
                finished_open.add(response_id)
                if status == "cancelled":
                    cancelled_open = True
                elif status == "failed":
                    failed_open = True
                else:
                    # A missing status is a normal completion and does promote.
                    promotes_residual = True
                continue
            # The residual response itself failed. A cancellation can still be
            # followed by the response the barge-in opens.
            if response_id in created_after and status != "cancelled":
                return True
            continue
        if event_type == "response.listen" and finished_open >= open_at_commit and _is_model_listen_decision(event):
            return True
    return (
        bool(open_at_commit)
        and finished_open >= open_at_commit
        and failed_open
        and not cancelled_open
        and not promotes_residual
        and not created_after
    )


async def _wait_for_server_turns(client: RealtimeDuplexClient, *, timeout_s: float) -> None:
    """Wait until no server-detected speech or response is open and turns are quiet."""
    deadline = time.monotonic() + timeout_s
    activity = turn_activity_count(client.events.events)
    stable_since = time.monotonic()
    while time.monotonic() < deadline:
        client.raise_if_reader_stopped()
        current = turn_activity_count(client.events.events)
        if current != activity:
            activity, stable_since = current, time.monotonic()
        if not server_turns_pending(client.events.events) and time.monotonic() - stable_since >= _SERVER_TURN_SETTLE_S:
            return
        await asyncio.sleep(0.05)
    raise TimeoutError("Timed out waiting for server-detected turns to finish")


async def generate_sample(
    sample: DuplexSample,
    *,
    url: str,
    model: str,
    ref_audio: str | Path | PreparedReferenceAudio | None,
    output_root: str | Path,
    fps: float = 1.0,
    mix: str = MIX_QUESTION,
    pace: str = "realtime",
    clock: str = "media",
    overwrite: bool = False,
    unit_ms: int = 1000,
    prepared: PreparedSample | None = None,
    defer_artifacts: bool = False,
    additional_headers: dict[str, str] | None = None,
    extra_body: dict[str, object] | None = None,
    instructions: str | None = DEFAULT_INSTRUCTIONS,
    turn_detection: str = TURN_DETECTION_NONE,
) -> GenerateSampleResult:
    """Generate one sample's timed sentences and collect duplex session metrics.

    Timed sentences and ``*.meta.json`` stay the quality-eval artifacts. Duplex
    request/session metrics are returned for the CLI to write
    ``duplex_metrics.json``; they are not written into meta.
    ``server_vad`` uses one detector frame per append, overriding the native
    ``unit_ms`` setting while keeping the decoded video samples unchanged.
    An existing sample is reused only when its metadata records the same ``mix``
    and, for proactive-reminder samples, the response carries ``chunks``.
    Metadata written before mix was recorded counts as ``question``. RTD only
    reads sentences, so its bare sentence lists are still reused.
    """
    if pace not in {"realtime", "as-fast-as-possible"}:
        raise ValueError("pace must be realtime or as-fast-as-possible")
    if turn_detection not in TURN_DETECTION_MODES:
        raise ValueError(f"turn_detection must be one of {TURN_DETECTION_MODES}")
    if mix not in MIX_MODES:
        raise ValueError(f"mix must be one of {MIX_MODES}")
    server_turns = turn_detection == TURN_DETECTION_SERVER_VAD
    if server_turns and pace != "realtime":
        raise ValueError("server_vad requires pace='realtime' to preserve turn and playback timing")
    output = Path(output_root) / sample.split / f"{sample.id}.json"
    if not defer_artifacts and output.exists() and not overwrite:
        cached_mix = _cached_response_mix(output)
        if cached_mix != mix:
            raise ValueError(
                f"existing response {output} was generated with mix={cached_mix!r}, not {mix!r}; "
                "pass --overwrite or choose another response directory"
            )
        if sample.family == "pr" and not _cached_response_has_chunks(output):
            raise ValueError(
                f"existing response {output} has no chunk times; pass --overwrite or choose another response directory"
            )
        return GenerateSampleResult(output=output)
    if prepared is not None and prepared.mix != mix:
        raise ValueError(f"prepared input uses mix={prepared.mix!r}, not {mix!r}")
    if server_turns:
        unit_ms = SERVER_VAD_FRAME_MS
    if prepared is None:
        # Standalone generation shares its loop with other live sessions;
        # decoding a whole clip must not pause their input or playback ACKs.
        prepared = await asyncio.to_thread(
            prepare_sample, sample, media_dir=output.parent / ".media", ref_audio=ref_audio, fps=fps, mix=mix
        )
    pcm = prepared.pcm
    if server_turns:
        # No final commit: trailing silence lets the VAD close the last turn.
        pcm += bytes(round(SERVER_VAD_TAIL_S * PCM16_SAMPLE_RATE) * PCM16_BYTES_PER_SAMPLE)
    realtime = pace == "realtime"
    client = RealtimeDuplexClient(
        build_realtime_url(url, model, autostart=False), additional_headers=additional_headers
    )
    response_done = False
    drain_timeout = None
    close_timeout = None
    stream_start: float | None = None
    async with client:
        await client.configure(
            model,
            ref_audio=prepared.ref_audio,
            instructions=instructions,
            extra_body=extra_body,
            auto_response=not server_turns,
            turn_detection=session_turn_detection(turn_detection),
        )
        frame_transport = select_frame_transport(_session_capabilities(client))
        if not server_turns:
            from vllm_omni.clients.duplex import chunk_period_ms, has_residual_model_unit

            # Keep a residual model unit for the commit to flush, so the
            # model's decision on the end of the clip arrives after the commit.
            period_ms = chunk_period_ms(client.events.events)
            if len(pcm) >= PCM16_BYTES_PER_SAMPLE and not has_residual_model_unit(pcm, chunk_period_ms=period_ms):
                pcm = pcm[:-PCM16_BYTES_PER_SAMPLE]
        image_items = (
            ImageItemWindow(client.send, item_prefix=f"duplexeval_frame_{sample.id}")
            if frame_transport == FRAME_TRANSPORT_IMAGE_ITEMS
            else None
        )
        ack_task = asyncio.create_task(_ack_playback(client))
        try:
            stream_start = time.monotonic()
            await client.stream_av_units(
                iter_av_units(pcm, iter(prepared.frames), unit_ms=unit_ms),
                realtime=realtime,
                frame_sink=image_items.push if image_items is not None else None,
            )

            commit_from = len(client.events.events)
            seen_activity = 0
            stable_since = time.monotonic()

            def responses_finished() -> bool:
                # The residual unit decides after input_audio_buffer.committed.
                # A reply that was already open only promotes that unit when it
                # completes, and Stage1 can still open a response afterwards.
                # Either ending waits until this activity has been quiet.
                nonlocal seen_activity, stable_since
                client.raise_if_reader_stopped()
                events = client.events.events
                activity = _completion_activity(events, commit_from)
                if activity != seen_activity:
                    seen_activity, stable_since = activity, time.monotonic()
                if not all(client.events.response_is_done(response_id) for response_id in client.events.response_ids):
                    return False
                if time.monotonic() - stable_since < _COMPLETION_SETTLE_S:
                    return False
                return _residual_unit_decided(events, commit_from)

            try:
                if server_turns:
                    await _wait_for_server_turns(client, timeout_s=max(20.0, prepared.duration + 20.0))
                    response_done = client.events.count("response.done") > 0
                else:
                    await client.commit()
                    await wait_for(
                        responses_finished,
                        timeout_s=max(20.0, prepared.duration + 20.0),
                        label="response.done",
                    )
                    response_done = True
            except TimeoutError as exc:
                drain_timeout = str(exc)
        finally:
            ack_task.cancel()
            try:
                await ack_task
            except asyncio.CancelledError:
                pass
        await client.acknowledge_playback()
        try:
            await client.close_session(timeout_s=20.0)
        except TimeoutError as exc:
            close_timeout = str(exc)
        events = list(client.events.events)
        collector = _event_collector_from_realtime(client.events)
    timed = [sentence.as_dict() for sentence in extract_timed_sentences(events, clock=clock)]
    chunks = [chunk.as_dict() for chunk in extract_timed_chunks(events, clock=clock)]
    meta = {
        "id": sample.id,
        "split": sample.split,
        "clock": clock if realtime else "invalid",
        "pace": pace,
        "mix": mix,
        "fps": fps,
        "unit_ms": unit_ms,
        "model": model,
        "turn_detection": turn_detection,
        "frame_transport": frame_transport,
        "instructions": instructions,
        "response_done": response_done,
        "drain_timeout": drain_timeout,
        "close_timeout": close_timeout,
        "ref_audio_sha256": prepared.ref_audio_sha256,
    }
    from vllm_omni.benchmarks.duplex_session_metrics import collect_duplex_session_metrics

    bundle = collect_duplex_session_metrics(
        collector,
        stream_start=stream_start if stream_start is not None else 0.0,
        session_id=sample.id,
    )
    errors = []
    for event in client.events.errors():
        error = event.get("error")
        code = error.get("code") if isinstance(error, dict) else event.get("code")
        # As in OmniInteract, an ACK racing a later commit is harmless. All
        # other server errors still make the serving sample unsuccessful.
        if code not in {"playback_ack_too_late", "playback_item_not_found"}:
            errors.append(str(event))
    errors.extend(str(error) for error in (drain_timeout, close_timeout) if error)
    finished = set()
    for event in events:
        response = event.get("response")
        if event.get("type") == "response.done" and isinstance(response, dict):
            finished.add(response.get("id"))
            # Barge-in cancels an individual turn, not the input clip. Its
            # preceding responses and already emitted text remain judge input.
            if response.get("status", "completed") not in {"completed", "cancelled"}:
                errors.append(f"Response did not complete: {response}")
    # A turn may be created while closing the continuous session. Record this
    # boundary in metadata; drain/close timeouts and server errors above still
    # fail the sample instead of silently publishing a broken session.
    meta["unfinished_response_ids"] = sorted(set(client.events.response_ids) - finished)
    result = GenerateSampleResult(
        output=output,
        request_metrics=[_with_sample_identity(metric, sample) for metric in bundle.request_metrics],
        session_metrics=_with_sample_identity(bundle.session_metrics, sample),
        timed_sentences=timed,
        timed_chunks=chunks,
        metadata=meta,
        output_tokens=bundle.output_tokens,
        audio_bytes=len(client.events.audio_bytes()),
        audio_sample_rate=client.events.output_sample_rate_hz,
        error="; ".join(errors),
    )
    if not defer_artifacts:
        write_sample_result(result)
    return result


async def _ack_playback(client: RealtimeDuplexClient) -> None:
    while True:
        await asyncio.sleep(0.25)
        await client.acknowledge_playback()
