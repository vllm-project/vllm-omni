# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded finalized-chunk output pipeline for MiniMax H3.

The remote H3 video VAE can expose temporally finalized decoder chunks.  This
module converts each chunk to the exact uint8 representation used by the
ordinary response path, copies it through a small pinned-host ring, and feeds
one persistent PyAV/libx264 MP4 session.  Consequently, rank zero never has to
assemble or transport a full-resolution floating-point video.

The feature is deliberately opt-in.  ``submit_decoded`` also defers failures
until ``finish`` so a rank-zero media failure cannot make it leave PP8 VAE
collectives while peer ranks are still decoding.
"""

from __future__ import annotations

import io
import os
import queue
import struct
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, cast

import numpy as np
import torch

from vllm_omni.platforms import current_omni_platform

MINIMAX_H3_CHUNKED_CPU_MP4_OUTPUT_ENV = "VLLM_OMNI_MINIMAX_H3_CHUNKED_CPU_MP4_OUTPUT"
MINIMAX_H3_CHUNKED_CPU_MP4_SLOTS_ENV = "VLLM_OMNI_MINIMAX_H3_CHUNKED_CPU_MP4_SLOTS"
MINIMAX_H3_CHUNKED_CPU_MP4_ROUTE = "finalized_chunk_gpu_u8_pinned_d2h_pyav_libx264"
MINIMAX_H3_CHUNKED_CPU_MP4_DEFAULT_CODEC_OPTIONS = {
    "preset": "ultrafast",
    "threads": "0",
}
_MINIMAX_H3_AAC_GLOBAL_HEADER_FLAG = 1 << 22
_MINIMAX_H3_AAC_PAYLOAD_MAGIC = b"H3AACV1\0"
_MINIMAX_H3_AAC_PAYLOAD_HEADER = struct.Struct("<8sIIH")
_MINIMAX_H3_AAC_PACKET_HEADER = struct.Struct("<qqqqqI")


def resolve_chunked_cpu_mp4_slots(raw: str | None = None) -> int:
    """Return the bounded pinned-ring size (only one or two slots are legal)."""

    if raw is None:
        raw = os.environ.get(MINIMAX_H3_CHUNKED_CPU_MP4_SLOTS_ENV, "2")
    try:
        slots = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{MINIMAX_H3_CHUNKED_CPU_MP4_SLOTS_ENV} must be 1 or 2, got {raw!r}") from exc
    if slots not in (1, 2):
        raise ValueError(f"{MINIMAX_H3_CHUNKED_CPU_MP4_SLOTS_ENV} must be 1 or 2, got {slots}")
    return slots


def validate_chunked_cpu_mp4_runtime() -> None:
    """Fail at pipeline startup if PyAV has no usable H.264 encoder."""

    try:
        import av

        av.CodecContext.create("h264", "w")
    except BaseException as exc:
        raise RuntimeError(
            f"{MINIMAX_H3_CHUNKED_CPU_MP4_OUTPUT_ENV}=1 requires PyAV with an H.264/libx264 encoder"
        ) from exc
    resolve_chunked_cpu_mp4_slots()


def resolve_chunked_cpu_mp4_codec_options(
    options: dict[str, str] | None,
) -> dict[str, str]:
    """Mirror the synchronous video API's default-or-replacement contract."""

    if options is None:
        return dict(MINIMAX_H3_CHUNKED_CPU_MP4_DEFAULT_CODEC_OPTIONS)
    if not isinstance(options, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in options.items()
    ):
        raise TypeError("video_codec_options must be a string-to-string mapping")
    return dict(options)


def exact_rgb_u8_bthwc(
    frames: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Crop BCTHW RGB and reproduce the baseline FP32/NumPy uint8 rounding.

    NumPy ``rint`` and ``torch.round`` both use round-to-nearest-even.  The
    explicit FP32 conversion is important: doing the multiply in FP16 changes
    pixels at half-way boundaries.
    """

    if frames.ndim != 5 or int(frames.shape[0]) != 1 or int(frames.shape[1]) != 3:
        raise ValueError(f"expected one normalized BCTHW RGB chunk, got shape={tuple(frames.shape)}")
    if not frames.dtype.is_floating_point:
        raise TypeError(f"expected floating-point RGB chunk, got {frames.dtype}")
    if int(frames.shape[-2]) < height or int(frames.shape[-1]) < width:
        raise ValueError(
            f"decoded RGB chunk is smaller than the requested crop: shape={tuple(frames.shape)}, crop={height}x{width}"
        )
    rgb = frames[0, :, :, :height, :width].permute(1, 2, 3, 0)
    work = rgb.float()
    if not work.is_contiguous():
        work = work.contiguous()
    work.clamp_(0.0, 1.0).mul_(255.0).round_()
    return work.to(torch.uint8)


def _audio_to_numpy(audio: Any | None) -> np.ndarray:
    if audio is None:
        raise ValueError("MiniMax H3 chunked MP4 output requires stereo audio")
    if isinstance(audio, torch.Tensor):
        samples = audio.detach().float().cpu().numpy()
    else:
        samples = np.asarray(audio, dtype=np.float32)
    samples = np.squeeze(samples).astype(np.float32, copy=False)
    if samples.ndim == 2 and samples.shape[0] > samples.shape[1]:
        samples = samples.T
    if samples.ndim != 2 or int(samples.shape[0]) != 2:
        raise ValueError(f"MiniMax H3 audio must be stereo channel-first PCM, got {samples.shape}")
    return np.ascontiguousarray(samples)


@dataclass(frozen=True)
class MiniMaxH3EncodedAudioPacket:
    data: bytes
    pts: int
    dts: int
    duration: int
    time_base: Fraction


@dataclass(frozen=True)
class MiniMaxH3EncodedAudio:
    """AAC packets encoded independently but muxed in the original order."""

    sample_rate: int
    extradata: bytes
    packets: tuple[MiniMaxH3EncodedAudioPacket, ...]


def encode_minimax_h3_aac(
    audio: Any,
    audio_sample_rate: int,
) -> MiniMaxH3EncodedAudio:
    """Encode the fixed H3 stereo AAC stream without opening an MP4 container."""

    import av

    sample_rate = int(audio_sample_rate)
    if sample_rate <= 0:
        raise ValueError(f"audio_sample_rate must be positive, got {sample_rate}")
    samples = _audio_to_numpy(audio)
    codec = av.CodecContext.create("aac", "w")
    codec.sample_rate = sample_rate
    codec.layout = "stereo"
    codec.format = "fltp"
    codec.flags |= _MINIMAX_H3_AAC_GLOBAL_HEADER_FLAG
    codec.open()

    frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="stereo")
    frame.sample_rate = sample_rate
    frame.pts = 0
    frame.time_base = Fraction(1, sample_rate)
    raw_packets = list(codec.encode(frame))
    raw_packets.extend(codec.encode())
    packets: list[MiniMaxH3EncodedAudioPacket] = []
    for packet in raw_packets:
        if packet.pts is None or packet.dts is None or packet.duration is None or packet.time_base is None:
            raise RuntimeError("MiniMax H3 AAC encoder returned incomplete packet timing metadata")
        packets.append(
            MiniMaxH3EncodedAudioPacket(
                data=bytes(packet),
                pts=int(packet.pts),
                dts=int(packet.dts),
                duration=int(packet.duration),
                time_base=Fraction(packet.time_base),
            )
        )
    if not packets:
        raise RuntimeError("MiniMax H3 AAC encoder returned no packets")
    return MiniMaxH3EncodedAudio(
        sample_rate=sample_rate,
        extradata=bytes(codec.extradata or b""),
        packets=tuple(packets),
    )


def serialize_minimax_h3_encoded_audio(audio: MiniMaxH3EncodedAudio) -> bytes:
    """Pack encoded AAC and timing metadata into one broadcast payload."""

    if len(audio.extradata) > 0xFFFF:
        raise ValueError("MiniMax H3 AAC extradata exceeds the payload format")
    output = io.BytesIO()
    output.write(
        _MINIMAX_H3_AAC_PAYLOAD_HEADER.pack(
            _MINIMAX_H3_AAC_PAYLOAD_MAGIC,
            int(audio.sample_rate),
            len(audio.packets),
            len(audio.extradata),
        )
    )
    output.write(audio.extradata)
    for packet in audio.packets:
        output.write(
            _MINIMAX_H3_AAC_PACKET_HEADER.pack(
                int(packet.pts),
                int(packet.dts),
                int(packet.duration),
                int(packet.time_base.numerator),
                int(packet.time_base.denominator),
                len(packet.data),
            )
        )
        output.write(packet.data)
    return output.getvalue()


def deserialize_minimax_h3_encoded_audio(payload: bytes) -> MiniMaxH3EncodedAudio:
    """Validate and unpack one rank-local AAC broadcast payload."""

    view = memoryview(payload)
    header_size = _MINIMAX_H3_AAC_PAYLOAD_HEADER.size
    if len(view) < header_size:
        raise ValueError("MiniMax H3 AAC payload is truncated before its header")
    magic, sample_rate, packet_count, extradata_size = _MINIMAX_H3_AAC_PAYLOAD_HEADER.unpack_from(view)
    if magic != _MINIMAX_H3_AAC_PAYLOAD_MAGIC or sample_rate <= 0 or packet_count <= 0:
        raise ValueError("MiniMax H3 AAC payload header is invalid")
    offset = header_size
    if offset + extradata_size > len(view):
        raise ValueError("MiniMax H3 AAC payload is truncated in codec extradata")
    extradata = bytes(view[offset : offset + extradata_size])
    offset += extradata_size
    packets: list[MiniMaxH3EncodedAudioPacket] = []
    for _ in range(packet_count):
        packet_header_size = _MINIMAX_H3_AAC_PACKET_HEADER.size
        if offset + packet_header_size > len(view):
            raise ValueError("MiniMax H3 AAC payload is truncated in packet metadata")
        pts, dts, duration, numerator, denominator, data_size = _MINIMAX_H3_AAC_PACKET_HEADER.unpack_from(view, offset)
        offset += packet_header_size
        if duration < 0 or denominator == 0 or offset + data_size > len(view):
            raise ValueError("MiniMax H3 AAC packet metadata is invalid")
        packets.append(
            MiniMaxH3EncodedAudioPacket(
                data=bytes(view[offset : offset + data_size]),
                pts=pts,
                dts=dts,
                duration=duration,
                time_base=Fraction(numerator, denominator),
            )
        )
        offset += data_size
    if offset != len(view):
        raise ValueError("MiniMax H3 AAC payload contains trailing bytes")
    return MiniMaxH3EncodedAudio(
        sample_rate=sample_rate,
        extradata=extradata,
        packets=tuple(packets),
    )


@dataclass(frozen=True)
class _PinnedSlot:
    tensor: torch.Tensor
    index: int


@dataclass
class _D2HItem:
    slot: _PinnedSlot
    frame_count: int
    first_frame: int
    ready: torch.cuda.Event
    gpu_owner: torch.Tensor


class MiniMaxH3ChunkedCpuMp4Output:
    """Finalized VAE chunk -> exact GPU uint8 -> pinned D2H -> MP4 sink."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: int,
        audio_sample_rate: int,
        max_chunk_frames: int,
        device: torch.device,
        video_codec_options: dict[str, str] | None = None,
        queue_slots: int | None = None,
    ) -> None:
        if device.type != "cuda":
            raise ValueError(f"MiniMax H3 chunked CPU MP4 output requires a CUDA device, got {device}")
        if device.index is None:
            device = torch.device("cuda", torch.accelerator.current_device_index())
        if (
            min(
                int(width),
                int(height),
                int(fps),
                int(audio_sample_rate),
                int(max_chunk_frames),
            )
            <= 0
        ):
            raise ValueError("width, height, fps, audio_sample_rate, and max_chunk_frames must be positive")
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.audio_sample_rate = int(audio_sample_rate)
        self.device = device
        self._queue_slots = (
            resolve_chunked_cpu_mp4_slots() if queue_slots is None else resolve_chunked_cpu_mp4_slots(str(queue_slots))
        )
        self._video_codec_options = resolve_chunked_cpu_mp4_codec_options(video_codec_options)
        self._copy_stream = torch.get_device_module().Stream(device=device)
        self._free: queue.Queue[_PinnedSlot] = queue.Queue(maxsize=self._queue_slots)
        self._work: queue.Queue[_D2HItem | None] = queue.Queue(maxsize=self._queue_slots)
        self._slot_frame_capacity = int(max_chunk_frames)
        for index in range(self._queue_slots):
            tensor = torch.empty(
                (self._slot_frame_capacity, self.height, self.width, 3),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
            self._free.put(_PinnedSlot(tensor=tensor, index=index))

        self._error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._audio: np.ndarray | MiniMaxH3EncodedAudio | None = None
        self._result: bytes | None = None
        self._closed = False
        self._submitted_frames = 0
        self._expected_frames: int | None = None
        self._next_frame = 0
        self._raw_rgb_bytes = 0
        self._max_inflight = 0
        self._inflight = 0
        self._pipeline_start: float | None = None
        self._producer_backpressure_s = 0.0
        self._gpu_prepare_enqueue_s = 0.0
        self._d2h_wait_s = 0.0
        self._video_encode_mux_s = 0.0
        self._audio_mux_finalize_s = 0.0

        self._thread = threading.Thread(
            target=self._worker,
            name="minimax-h3-pinned-pyav",
            daemon=True,
        )
        self._thread.start()

    def _set_error(self, error: BaseException) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = error

    def _get_error(self) -> BaseException | None:
        with self._error_lock:
            return self._error

    def _ensure_slots(self, frame_count: int) -> None:
        if frame_count < 1:
            raise ValueError("finalized VAE chunks must contain at least one frame")
        if frame_count > self._slot_frame_capacity:
            raise ValueError(
                "finalized VAE chunk exceeds the checkpoint-derived pinned-ring "
                f"capacity: frames={frame_count}, capacity={self._slot_frame_capacity}"
            )

    def submit_decoded(
        self,
        decoded: torch.Tensor,
        first_frame: int,
        total_frames: int,
        processor: Any,
    ) -> None:
        """Queue one chunk without ever escaping an error into VAE collectives."""

        if self._get_error() is not None:
            return
        slot: _PinnedSlot | None = None
        ready: torch.cuda.Event | None = None
        queued = False
        inflight_accounted = False
        try:
            if self._closed:
                raise RuntimeError("cannot submit after the chunked MP4 sink is closed")
            if self._pipeline_start is None:
                self._pipeline_start = time.perf_counter()
            first_frame = int(first_frame)
            total_frames = int(total_frames)
            if first_frame != self._next_frame:
                raise RuntimeError(
                    f"non-contiguous finalized VAE chunk: first_frame={first_frame}, expected={self._next_frame}"
                )
            if total_frames < 1 or first_frame >= total_frames:
                raise ValueError(f"invalid finalized VAE range {first_frame}/{total_frames}")
            if self._expected_frames not in (None, total_frames):
                raise RuntimeError(
                    f"finalized VAE total changed between chunks: {self._expected_frames} -> {total_frames}"
                )
            self._expected_frames = total_frames

            submit_start = time.perf_counter()
            if decoded.ndim != 5:
                raise ValueError(f"expected a rank-5 finalized decoder chunk, got shape={tuple(decoded.shape)}")
            if decoded.device.type != "cuda" or decoded.device.index != self.device.index:
                raise ValueError(
                    "finalized decoder chunks must remain on the sink CUDA device: "
                    f"chunk={decoded.device}, sink={self.device}"
                )
            frame_count = int(decoded.shape[2])
            self._ensure_slots(frame_count)
            wait_start = time.perf_counter()
            slot = self._free.get()
            self._producer_backpressure_s += time.perf_counter() - wait_start

            # Record the producer before switching streams. The callback owns
            # ``decoded`` only until this method returns.
            producer_ready = torch.get_device_module().Event()
            producer_ready.record(torch.get_device_module().current_stream(device=decoded.device))
            with torch.get_device_module().stream(self._copy_stream):
                self._copy_stream.wait_event(producer_ready)
                decoded.record_stream(self._copy_stream)
                frames = processor.revert_tensor(decoded)
                if frames.ndim == 4:
                    frames = frames.unsqueeze(0).transpose(1, 2)
                if frames.device != decoded.device:
                    raise ValueError(
                        "MiniMax H3 processor moved a finalized chunk off its CUDA "
                        f"device: {decoded.device} -> {frames.device}"
                    )
                rgb_u8 = exact_rgb_u8_bthwc(
                    frames,
                    height=self.height,
                    width=self.width,
                )
            if int(rgb_u8.shape[0]) != frame_count:
                raise RuntimeError(
                    f"MiniMax H3 processor changed finalized chunk length: {frame_count} -> {int(rgb_u8.shape[0])}"
                )
            if first_frame + frame_count > total_frames:
                raise ValueError(
                    f"finalized VAE chunk exceeds advertised total: {first_frame}+{frame_count}>{total_frames}"
                )

            ready = torch.get_device_module().Event()
            with torch.get_device_module().stream(self._copy_stream):
                slot.tensor[:frame_count].copy_(rgb_u8, non_blocking=True)
                frames.record_stream(self._copy_stream)
                rgb_u8.record_stream(self._copy_stream)
                ready.record(self._copy_stream)
            self._inflight += 1
            inflight_accounted = True
            self._max_inflight = max(self._max_inflight, self._inflight)
            self._work.put(
                _D2HItem(
                    slot=slot,
                    frame_count=frame_count,
                    first_frame=first_frame,
                    ready=ready,
                    gpu_owner=rgb_u8,
                )
            )
            queued = True
            self._submitted_frames += frame_count
            self._next_frame += frame_count
            self._raw_rgb_bytes += frame_count * self.height * self.width * 3
            self._gpu_prepare_enqueue_s += time.perf_counter() - submit_start
        except BaseException as exc:
            if slot is not None and not queued:
                if ready is not None:
                    with suppress(BaseException):
                        ready.synchronize()
                if inflight_accounted:
                    self._inflight -= 1
                self._free.put(slot)
            self._set_error(exc)

    def _worker(self) -> None:
        stop_seen = False
        try:
            import av

            current_omni_platform.set_device(self.device)
            buffer = io.BytesIO()
            with cast(Any, av.open(buffer, mode="w", format="mp4")) as container:
                video_stream = cast(
                    Any,
                    container.add_stream(
                        "h264",
                        rate=Fraction(self.fps).limit_denominator(10_000),
                    ),
                )
                video_stream.width = self.width
                video_stream.height = self.height
                video_stream.pix_fmt = "yuv420p"
                options: dict[str, object] = {"crf": "18"}
                options.update(self._video_codec_options)
                video_stream.options = options

                audio_stream = cast(
                    Any,
                    container.add_stream("aac", rate=self.audio_sample_rate),
                )
                # H3 always emits stereo audio. Set the layout before video
                # packets can cause PyAV/FFmpeg to open the codec context;
                # recent PyAV versions reject layout changes after that point.
                audio_stream.layout = "stereo"

                while True:
                    item = self._work.get()
                    try:
                        if item is None:
                            stop_seen = True
                            break
                        wait_start = time.perf_counter()
                        item.ready.synchronize()
                        self._d2h_wait_s += time.perf_counter() - wait_start
                        array = item.slot.tensor[: item.frame_count].numpy()
                        encode_start = time.perf_counter()
                        for frame_data in array:
                            frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
                            for packet in video_stream.encode(frame):
                                container.mux(packet)
                        self._video_encode_mux_s += time.perf_counter() - encode_start
                    finally:
                        if item is not None:
                            self._inflight -= 1
                            self._free.put(item.slot)
                        self._work.task_done()

                finalize_start = time.perf_counter()
                for packet in video_stream.encode():
                    container.mux(packet)
                audio = self._audio
                if isinstance(audio, MiniMaxH3EncodedAudio):
                    if audio.sample_rate != self.audio_sample_rate:
                        raise ValueError(
                            "MiniMax H3 encoded audio sample rate changed during output: "
                            f"{audio.sample_rate}/{self.audio_sample_rate}"
                        )
                    if bytes(audio_stream.codec_context.extradata or b"") != audio.extradata:
                        raise RuntimeError("MiniMax H3 independently encoded AAC extradata does not match MP4 stream")
                    for encoded_packet in audio.packets:
                        packet = av.Packet(encoded_packet.data)
                        packet.pts = encoded_packet.pts
                        packet.dts = encoded_packet.dts
                        packet.duration = encoded_packet.duration
                        packet.time_base = encoded_packet.time_base
                        packet.stream = audio_stream
                        container.mux(packet)
                elif audio is not None:
                    audio_frame = av.AudioFrame.from_ndarray(
                        audio,
                        format="fltp",
                        layout="stereo",
                    )
                    audio_frame.sample_rate = self.audio_sample_rate
                    audio_frame.pts = 0
                    audio_frame.time_base = Fraction(1, self.audio_sample_rate)
                    for packet in audio_stream.encode(audio_frame):
                        container.mux(packet)
                    for packet in audio_stream.encode():
                        container.mux(packet)
                self._audio_mux_finalize_s += time.perf_counter() - finalize_start
            self._result = buffer.getvalue()
        except BaseException as exc:
            self._set_error(exc)
            # Return every queued slot so a producer blocked by bounded
            # backpressure cannot strand PP8 peers in a later callback. Once
            # the sentinel was consumed, however, finalize/close failures
            # must return immediately: waiting for a second sentinel would
            # deadlock ``finish`` in its thread join.
            if not stop_seen:
                while True:
                    item = self._work.get()
                    self._work.task_done()
                    if item is None:
                        break
                    # A media-thread failure can race with a D2H submitted by
                    # the callback. Keep the pinned destination alive until
                    # that copy is done before returning its slot.
                    with suppress(BaseException):
                        item.ready.synchronize()
                    self._inflight -= 1
                    self._free.put(item.slot)

    def finish(
        self,
        audio: Any | None,
        audio_sample_rate: int,
    ) -> tuple[bytes, dict[str, float]]:
        """Flush the persistent codec session and return only final MP4 bytes."""

        if self._closed:
            raise RuntimeError("MiniMax H3 chunked CPU MP4 sink is already closed")
        if int(audio_sample_rate) != self.audio_sample_rate:
            self._closed = True
            self._work.put(None)
            self._thread.join()
            raise ValueError(
                f"MiniMax H3 audio sample rate changed during output: {audio_sample_rate}/{self.audio_sample_rate}"
            )
        self._closed = True
        audio_start = time.perf_counter()
        if isinstance(audio, MiniMaxH3EncodedAudio):
            self._audio = audio
        else:
            try:
                self._audio = _audio_to_numpy(audio)
            except BaseException:
                self._work.put(None)
                self._thread.join()
                raise
        audio_prepare_s = time.perf_counter() - audio_start
        self._work.put(None)
        self._thread.join()

        error = self._get_error()
        if error is not None:
            raise RuntimeError("MiniMax H3 chunked CPU MP4 pipeline failed") from error
        if self._result is None:
            raise RuntimeError("MiniMax H3 chunked CPU MP4 pipeline produced no MP4")
        if self._expected_frames is None or self._submitted_frames != self._expected_frames:
            raise RuntimeError(
                "MiniMax H3 chunked CPU MP4 frame count mismatch: "
                f"submitted={self._submitted_frames}, expected={self._expected_frames}"
            )
        pipeline_s = 0.0 if self._pipeline_start is None else time.perf_counter() - self._pipeline_start
        return self._result, {
            "chunked_cpu_mp4_frames": float(self._submitted_frames),
            "chunked_cpu_mp4_rgb_u8_bytes": float(self._raw_rgb_bytes),
            "chunked_cpu_mp4_bytes": float(len(self._result)),
            "chunked_cpu_mp4_queue_slots": float(self._queue_slots),
            "chunked_cpu_mp4_max_inflight": float(self._max_inflight),
            "chunked_cpu_mp4_producer_backpressure_s": self._producer_backpressure_s,
            "chunked_cpu_mp4_gpu_prepare_enqueue_s": self._gpu_prepare_enqueue_s,
            "chunked_cpu_mp4_d2h_wait_s": self._d2h_wait_s,
            "chunked_cpu_mp4_video_encode_mux_s": self._video_encode_mux_s,
            "chunked_cpu_mp4_audio_prepare_s": audio_prepare_s,
            "chunked_cpu_mp4_audio_mux_finalize_s": self._audio_mux_finalize_s,
            "chunked_cpu_mp4_audio_preencoded": float(isinstance(self._audio, MiniMaxH3EncodedAudio)),
            "chunked_cpu_mp4_audio_packet_bytes": float(
                sum(len(packet.data) for packet in self._audio.packets)
                if isinstance(self._audio, MiniMaxH3EncodedAudio)
                else 0
            ),
            "chunked_cpu_mp4_pipeline_s": pipeline_s,
        }

    def abort(self) -> None:
        """Drain and stop the worker without masking an upstream exception."""

        if self._closed:
            return
        self._closed = True
        self._work.put(None)
        self._thread.join()


__all__ = [
    "MINIMAX_H3_CHUNKED_CPU_MP4_OUTPUT_ENV",
    "MINIMAX_H3_CHUNKED_CPU_MP4_DEFAULT_CODEC_OPTIONS",
    "MINIMAX_H3_CHUNKED_CPU_MP4_ROUTE",
    "MINIMAX_H3_CHUNKED_CPU_MP4_SLOTS_ENV",
    "MiniMaxH3ChunkedCpuMp4Output",
    "MiniMaxH3EncodedAudio",
    "MiniMaxH3EncodedAudioPacket",
    "deserialize_minimax_h3_encoded_audio",
    "encode_minimax_h3_aac",
    "exact_rgb_u8_bthwc",
    "resolve_chunked_cpu_mp4_slots",
    "resolve_chunked_cpu_mp4_codec_options",
    "serialize_minimax_h3_encoded_audio",
    "validate_chunked_cpu_mp4_runtime",
]
