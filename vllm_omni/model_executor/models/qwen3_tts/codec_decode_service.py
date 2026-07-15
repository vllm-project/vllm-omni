# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import struct
import threading
import time
import uuid
from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from multiprocessing import shared_memory

import torch
import torch.nn as nn
from vllm.logger import init_logger

DecodeBatchRecordCallback = Callable[
    [int, int, list[int]],
    None,
]

logger = init_logger(__name__)

_CODEC_HANDLE_MAGIC = 0x51434F4445430001
_CODEC_SHM_MAGIC = 0x51434F4445435348
_CODEC_SHM_HEADER_BYTES = 64
_CODEC_SHM_MIN_PAYLOAD_BYTES = 4096
_CODEC_SHM_STATE_PENDING = 0
_CODEC_SHM_STATE_READY = 1
_CODEC_SHM_STATE_ERROR = 2
_CODEC_SHM_DTYPE_FLOAT32 = 1
_I64 = struct.Struct("=q")


def _codec_shm_name(handle_id: int) -> str:
    return f"vq3c_{int(handle_id):x}"


def _shm_i64(buf: memoryview, offset: int) -> int:
    return int(_I64.unpack_from(buf, offset)[0])


def _set_shm_i64(buf: memoryview, offset: int, value: int) -> None:
    _I64.pack_into(buf, offset, int(value))


def _close_shm(shm: shared_memory.SharedMemory) -> None:
    try:
        shm.close()
    except BufferError:
        logger.debug("Failed to close Qwen3 codec shared memory %s", shm.name, exc_info=True)


def _unlink_shm(shm: shared_memory.SharedMemory) -> None:
    try:
        shm.unlink()
    except FileNotFoundError:
        pass


def _init_codec_shm_header(shm: shared_memory.SharedMemory, payload_capacity: int) -> None:
    buf = shm.buf
    try:
        _set_shm_i64(buf, 0, _CODEC_SHM_MAGIC)
        _set_shm_i64(buf, 8, _CODEC_SHM_STATE_PENDING)
        _set_shm_i64(buf, 16, _CODEC_SHM_DTYPE_FLOAT32)
        _set_shm_i64(buf, 24, 0)
        _set_shm_i64(buf, 32, 0)
        _set_shm_i64(buf, 40, 0)
        _set_shm_i64(buf, 48, int(payload_capacity))
    finally:
        del buf


def _validate_codec_shm_header(buf: memoryview, handle_id: int) -> None:
    if _shm_i64(buf, 0) != _CODEC_SHM_MAGIC:
        raise RuntimeError(f"Invalid Qwen3 codec shared memory header for handle {handle_id}")


def _codec_shm_payload_capacity_from_spec(spec: dict[str, int]) -> int:
    actual_frames = max(0, int(spec.get("actual_frames", 0)))
    ctx_frames = max(0, int(spec.get("ctx_frames", 0)))
    upsample = max(1, int(spec.get("upsample", 1)))
    output_frames = max(0, actual_frames - ctx_frames)
    return max(_CODEC_SHM_MIN_PAYLOAD_BYTES, output_frames * upsample * 4)


def register_codec_shm_handle(handle_id: int, payload_capacity: int) -> None:
    size = _CODEC_SHM_HEADER_BYTES + max(_CODEC_SHM_MIN_PAYLOAD_BYTES, int(payload_capacity))
    name = _codec_shm_name(handle_id)
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        stale = shared_memory.SharedMemory(name=name, create=False)
        try:
            stale.unlink()
        finally:
            _close_shm(stale)
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    try:
        _init_codec_shm_header(shm, size - _CODEC_SHM_HEADER_BYTES)
    finally:
        _close_shm(shm)


def _new_codec_handle_id() -> int:
    # Keep the value positive and within int64 so it can travel as a tensor
    # through the existing msgspec EngineCore output path.
    return uuid.uuid4().int & ((1 << 63) - 1)


def make_codec_handle_tensor(
    handle_id: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    return torch.tensor([_CODEC_HANDLE_MAGIC, int(handle_id)], dtype=torch.int64, device=device)


def empty_codec_handle_tensor(*, device: torch.device | str | None = None) -> torch.Tensor:
    return torch.zeros((2,), dtype=torch.int64, device=device)


def is_codec_handle_tensor(value: object) -> bool:
    if not isinstance(value, torch.Tensor):
        return False
    if value.dtype != torch.int64 or value.numel() < 2 or value.numel() % 2 != 0:
        return False
    flat = value.detach().reshape(-1).cpu()
    return bool((flat[0::2] == _CODEC_HANDLE_MAGIC).all())


def _codec_handle_ids(handle: torch.Tensor) -> list[int]:
    ids = [int(v) for v in handle.detach().reshape(-1).cpu()[1::2].tolist()]
    return [handle_id for handle_id in ids if handle_id > 0]


def write_codec_shm_result_tensor(handle_id: int, tensor: torch.Tensor) -> None:
    cpu_tensor = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().reshape(-1)
    payload = cpu_tensor.numpy().tobytes()
    nbytes = len(payload)
    shm = shared_memory.SharedMemory(name=_codec_shm_name(handle_id), create=False)
    buf = shm.buf
    try:
        _validate_codec_shm_header(buf, handle_id)
        capacity = _shm_i64(buf, 48)
        if nbytes > capacity:
            raise RuntimeError(
                f"Qwen3 codec shared memory handle {handle_id} capacity {capacity} is too small for {nbytes} bytes"
            )
        buf[_CODEC_SHM_HEADER_BYTES : _CODEC_SHM_HEADER_BYTES + nbytes] = payload
        _set_shm_i64(buf, 16, _CODEC_SHM_DTYPE_FLOAT32)
        _set_shm_i64(buf, 24, int(cpu_tensor.numel()))
        _set_shm_i64(buf, 32, nbytes)
        _set_shm_i64(buf, 40, 0)
        _set_shm_i64(buf, 8, _CODEC_SHM_STATE_READY)
    finally:
        del buf
        _close_shm(shm)


def write_codec_shm_error(handle_id: int, error: BaseException) -> None:
    shm = shared_memory.SharedMemory(name=_codec_shm_name(handle_id), create=False)
    buf = shm.buf
    try:
        _validate_codec_shm_header(buf, handle_id)
        capacity = _shm_i64(buf, 48)
        message = str(error).encode("utf-8", errors="replace")[:capacity]
        buf[_CODEC_SHM_HEADER_BYTES : _CODEC_SHM_HEADER_BYTES + len(message)] = message
        _set_shm_i64(buf, 24, 0)
        _set_shm_i64(buf, 32, 0)
        _set_shm_i64(buf, 40, len(message))
        _set_shm_i64(buf, 8, _CODEC_SHM_STATE_ERROR)
    finally:
        del buf
        _close_shm(shm)


def _attach_codec_shm(handle_id: int) -> shared_memory.SharedMemory | None:
    try:
        return shared_memory.SharedMemory(name=_codec_shm_name(handle_id), create=False)
    except FileNotFoundError:
        return None


def _read_codec_shm_payload(
    shm: shared_memory.SharedMemory,
    handle_id: int,
) -> tuple[torch.Tensor | None, str | None]:
    buf = shm.buf
    try:
        _validate_codec_shm_header(buf, handle_id)
        state = _shm_i64(buf, 8)
        if state == _CODEC_SHM_STATE_ERROR:
            error_nbytes = max(0, _shm_i64(buf, 40))
            message = bytes(buf[_CODEC_SHM_HEADER_BYTES : _CODEC_SHM_HEADER_BYTES + error_nbytes]).decode(
                "utf-8", errors="replace"
            )
            return (
                None,
                message or f"Qwen3 codec shared memory handle {handle_id} failed",
            )
        if state != _CODEC_SHM_STATE_READY:
            raise RuntimeError(f"Qwen3 codec shared memory handle {handle_id} is not ready")
        dtype_code = _shm_i64(buf, 16)
        if dtype_code != _CODEC_SHM_DTYPE_FLOAT32:
            raise RuntimeError(f"Unsupported Qwen3 codec shared memory dtype code {dtype_code} for handle {handle_id}")
        numel = max(0, _shm_i64(buf, 24))
        nbytes = max(0, _shm_i64(buf, 32))
        capacity = max(0, _shm_i64(buf, 48))
        if nbytes > capacity or nbytes != numel * 4:
            raise RuntimeError(
                f"Invalid Qwen3 codec shared memory payload for handle {handle_id}: "
                f"numel={numel}, nbytes={nbytes}, capacity={capacity}"
            )
        if numel == 0:
            return torch.zeros(0, dtype=torch.float32), None
        payload = bytes(buf[_CODEC_SHM_HEADER_BYTES : _CODEC_SHM_HEADER_BYTES + nbytes])
        tensor = torch.frombuffer(bytearray(payload), dtype=torch.float32).clone().reshape(-1)
        return tensor, None
    finally:
        del buf


def _resolve_codec_shm_result(
    handle_id: int,
    *,
    deadline: float,
    poll_s: float,
) -> torch.Tensor | None:
    shm = _attach_codec_shm(handle_id)
    if shm is None:
        return None

    try:
        while True:
            buf = shm.buf
            try:
                _validate_codec_shm_header(buf, handle_id)
                state = _shm_i64(buf, 8)
            finally:
                del buf
            if state in {_CODEC_SHM_STATE_READY, _CODEC_SHM_STATE_ERROR}:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for Qwen3 codec audio handle {handle_id}")
            time.sleep(max(0.0001, float(poll_s)))

        tensor, error_message = _read_codec_shm_payload(shm, handle_id)
        _unlink_shm(shm)
        if error_message is not None:
            raise RuntimeError(error_message)
        assert tensor is not None
        return tensor
    finally:
        _close_shm(shm)


async def _resolve_codec_shm_result_async(
    handle_id: int,
    *,
    deadline: float,
    poll_s: float,
) -> torch.Tensor | None:
    shm = _attach_codec_shm(handle_id)
    if shm is None:
        return None

    try:
        while True:
            buf = shm.buf
            try:
                _validate_codec_shm_header(buf, handle_id)
                state = _shm_i64(buf, 8)
            finally:
                del buf
            if state in {_CODEC_SHM_STATE_READY, _CODEC_SHM_STATE_ERROR}:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for Qwen3 codec audio handle {handle_id}")
            await asyncio.sleep(max(0.0001, float(poll_s)))

        tensor, error_message = _read_codec_shm_payload(shm, handle_id)
        _unlink_shm(shm)
        if error_message is not None:
            raise RuntimeError(error_message)
        assert tensor is not None
        return tensor
    finally:
        _close_shm(shm)


def resolve_codec_handle_tensor(
    handle: torch.Tensor,
    *,
    timeout_s: float = 300.0,
    poll_s: float = 0.001,
) -> torch.Tensor:
    if not is_codec_handle_tensor(handle):
        raise ValueError(f"Invalid Qwen3 codec audio handle: {handle!r}")

    ids = _codec_handle_ids(handle)
    if not ids:
        return torch.zeros(0, dtype=torch.float32)

    deadline = time.monotonic() + max(0.0, float(timeout_s))
    chunks: list[torch.Tensor] = []
    for handle_id in ids:
        shm_result = _resolve_codec_shm_result(handle_id, deadline=deadline, poll_s=poll_s)
        if shm_result is None:
            raise FileNotFoundError(
                f"Missing Qwen3 codec shared memory for handle {handle_id}; "
                "the handle may be invalid or was not registered"
            )
        chunks.append(shm_result)
    if len(chunks) == 1:
        return chunks[0]
    return torch.cat(chunks, dim=-1)


async def resolve_codec_handle_tensor_async(
    handle: torch.Tensor,
    *,
    timeout_s: float = 300.0,
    poll_s: float = 0.001,
) -> torch.Tensor:
    if not is_codec_handle_tensor(handle):
        raise ValueError(f"Invalid Qwen3 codec audio handle: {handle!r}")

    ids = _codec_handle_ids(handle)
    if not ids:
        return torch.zeros(0, dtype=torch.float32)

    deadline = time.monotonic() + max(0.0, float(timeout_s))
    chunks: list[torch.Tensor] = []
    for handle_id in ids:
        shm_result = await _resolve_codec_shm_result_async(handle_id, deadline=deadline, poll_s=poll_s)
        if shm_result is None:
            raise FileNotFoundError(
                f"Missing Qwen3 codec shared memory for handle {handle_id}; "
                "the handle may be invalid or was not registered"
            )
        chunks.append(shm_result)
    if len(chunks) == 1:
        return chunks[0]
    return torch.cat(chunks, dim=-1)


@dataclass
class PyTorchCodecDecodeServiceStats:
    submitted_requests: int = 0
    decoded_batches: int = 0
    decoded_frames: int = 0
    padded_frames: int = 0
    submit_queue_ns: int = 0
    batch_wait_ns: int = 0
    decode_gpu_ns: int = 0
    output_assemble_ns: int = 0
    queued_jobs: int = 0
    dynamic_batches: int = 0
    actual_frames: Counter[int] = field(default_factory=Counter)
    bucket_groups: Counter[tuple[int, int]] = field(default_factory=Counter)


class PyTorchCodecDecodeService:
    """Synchronous PyTorch codec decode backend boundary.

    This service centralizes the core PyTorch decode and batching logic used by
    both the synchronous baseline path and the deferred shared-memory service.
    """

    def __init__(
        self,
        *,
        decoder: nn.Module,
        num_quantizers: int,
        decode_chunk_frames: int,
        decode_left_context_frames: int,
        decode_variable_chunk_batch_min_frames: int,
        decode_batch_max_size: int,
        decode_batch_bucket_frames: Sequence[int] | None = None,
    ) -> None:
        self.decoder = decoder
        self.num_quantizers = int(num_quantizers)
        self.decode_chunk_frames = int(decode_chunk_frames)
        self.decode_left_context_frames = int(decode_left_context_frames)
        self.decode_variable_chunk_batch_min_frames = int(decode_variable_chunk_batch_min_frames)
        self.decode_batch_max_size = int(decode_batch_max_size)
        self.decode_batch_bucket_frames = sorted(
            int(frames) for frames in (decode_batch_bucket_frames or []) if int(frames) > 0
        )
        self.stats = PyTorchCodecDecodeServiceStats()

    def decode_group_chunks(
        self,
        group_chunks: Sequence[Sequence[tuple[int, torch.Tensor]]],
        *,
        record_batch: DecodeBatchRecordCallback | None = None,
    ) -> dict[int, torch.Tensor]:
        submit_start_ns = time.perf_counter_ns()
        normalized_group_chunks = [list(group_chunk) for group_chunk in group_chunks if group_chunk]
        submitted_requests = sum(len(group_chunk) for group_chunk in normalized_group_chunks)
        self.stats.submitted_requests += submitted_requests
        self.stats.submit_queue_ns += time.perf_counter_ns() - submit_start_ns

        wav_rows_by_index: dict[int, torch.Tensor] = {}
        for group_chunk in normalized_group_chunks:
            wav_rows = self._decode_group_chunk(group_chunk, record_batch=record_batch)

            assemble_start_ns = time.perf_counter_ns()
            for row, (j, _) in enumerate(group_chunk):
                wav_rows_by_index[j] = wav_rows[row]
            self.stats.output_assemble_ns += time.perf_counter_ns() - assemble_start_ns
        return wav_rows_by_index

    def _decode_group_chunk(
        self,
        group_chunk: list[tuple[int, torch.Tensor]],
        *,
        record_batch: DecodeBatchRecordCallback | None,
    ) -> torch.Tensor:
        actual_frames = [int(codes_qf.shape[1]) for _, codes_qf in group_chunk]
        max_actual_frames = max(actual_frames)
        target_frames = self._get_decode_batch_bucket_frames(max_actual_frames)
        is_equal_length_batch = all(frames == target_frames for frames in actual_frames)
        use_variable_length_batch = (
            len(group_chunk) > 1
            and not is_equal_length_batch
            and not self.decode_batch_bucket_frames
            and target_frames >= self.decode_variable_chunk_batch_min_frames
            and hasattr(self.decoder, "batched_chunked_decode")
        )
        codes_bqf = self._pack_codes(group_chunk, target_frames, is_equal_length_batch)

        self.stats.decoded_batches += 1
        self.stats.decoded_frames += len(group_chunk) * target_frames
        self.stats.padded_frames += sum(target_frames - frames for frames in actual_frames)
        self.stats.actual_frames.update(actual_frames)
        self.stats.bucket_groups[(len(group_chunk), target_frames)] += 1
        if record_batch is not None:
            record_batch(len(group_chunk), target_frames, actual_frames)

        decode_start_ns = time.perf_counter_ns()
        # Grad/inference mode is thread-local. Dynamic batching runs codec
        # decode from a worker thread, so do not rely on the caller's
        # model-runner no-grad context being inherited here.
        with torch.no_grad():
            try:
                if use_variable_length_batch:
                    wav_batch = self.decoder.batched_chunked_decode(
                        codes_bqf,
                        actual_frames,
                        chunk_size=self.decode_chunk_frames,
                        left_context_size=self.decode_left_context_frames,
                        max_batch_size=self.decode_batch_max_size,
                    )
                else:
                    wav_batch = self.decoder.chunked_decode(
                        codes_bqf,
                        chunk_size=self.decode_chunk_frames,
                        left_context_size=self.decode_left_context_frames,
                    )
            except TypeError:
                wav_batch = self.decoder.chunked_decode(codes_bqf)
        self.stats.decode_gpu_ns += time.perf_counter_ns() - decode_start_ns

        if wav_batch.dim() == 3 and wav_batch.shape[1] == 1:
            wav_rows = wav_batch[:, 0, :]
        elif wav_batch.dim() == 2:
            wav_rows = wav_batch
        else:
            raise ValueError(
                f"Code2Wav decoder returned unexpected shape {tuple(wav_batch.shape)} for batch size {len(group_chunk)}"
            )
        if wav_rows.shape[0] != len(group_chunk):
            raise ValueError(
                f"Code2Wav decoder returned batch size {wav_rows.shape[0]} for input batch size {len(group_chunk)}"
            )
        return wav_rows

    def _pack_codes(
        self,
        group_chunk: list[tuple[int, torch.Tensor]],
        target_frames: int,
        is_equal_length_batch: bool,
    ) -> torch.Tensor:
        if len(group_chunk) == 1:
            return group_chunk[0][1].unsqueeze(0)
        if is_equal_length_batch:
            return torch.stack([codes_qf for _, codes_qf in group_chunk], dim=0)

        first = group_chunk[0][1]
        codes_bqf = first.new_zeros((len(group_chunk), self.num_quantizers, target_frames))
        for row, (_, codes_qf) in enumerate(group_chunk):
            codes_bqf[row, :, : codes_qf.shape[1]] = codes_qf
        return codes_bqf

    def _get_decode_batch_bucket_frames(self, actual_frames: int) -> int:
        for bucket_frames in self.decode_batch_bucket_frames:
            if actual_frames <= bucket_frames:
                return bucket_frames
        return int(actual_frames)


@dataclass
class _QueuedShmDecodeJob:
    group_chunks: list[list[tuple[int, torch.Tensor]]]
    record_batch: DecodeBatchRecordCallback | None
    trim_specs: dict[int, dict[str, int]]
    handle_ids: dict[int, int]
    submit_ns: int
    input_ready: torch.cuda.Event | None = None


class ShmCodecDecodeService:
    """Threaded dynamic codec decode service with shared-memory output handles.

    This service lets Stage1 return a small
    int64 handle through the existing tensor-only EngineCore output path while a
    background worker batches and decodes codec chunks, then writes final audio
    tensors to per-handle shared memory for the serving layer to resolve.
    """

    def __init__(
        self,
        *,
        decoder: nn.Module,
        num_quantizers: int,
        decode_chunk_frames: int,
        decode_left_context_frames: int,
        decode_variable_chunk_batch_min_frames: int,
        decode_batch_max_size: int,
        max_queue_delay_us: int,
        max_queue_jobs: int,
        decode_batch_bucket_frames: Sequence[int] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self._sync_service = PyTorchCodecDecodeService(
            decoder=decoder,
            num_quantizers=num_quantizers,
            decode_chunk_frames=decode_chunk_frames,
            decode_left_context_frames=decode_left_context_frames,
            decode_variable_chunk_batch_min_frames=decode_variable_chunk_batch_min_frames,
            decode_batch_max_size=decode_batch_max_size,
            decode_batch_bucket_frames=decode_batch_bucket_frames,
        )
        self.device = torch.device(device) if device is not None else None
        self._max_queue_delay_s = max(0, int(max_queue_delay_us)) / 1_000_000.0
        self._max_queue_jobs = max(1, int(max_queue_jobs))
        self._pending: deque[_QueuedShmDecodeJob] = deque()
        self._closed = False
        self._condition = threading.Condition()
        self._stream: torch.cuda.Stream | None = None
        self._stats_enabled = False
        self._stats_log_every = 0
        if self.device is not None and self.device.type == "cuda":
            with torch.cuda.device(self.device):
                self._stream = torch.cuda.Stream(device=self.device)
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="qwen3-codec-shm-batcher",
            daemon=True,
        )
        self._worker.start()

    @property
    def decoder(self) -> nn.Module:
        return self._sync_service.decoder

    @property
    def stats(self) -> PyTorchCodecDecodeServiceStats:
        return self._sync_service.stats

    def shutdown(self, *, timeout: float | None = None) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if threading.current_thread() is not self._worker:
            self._worker.join(timeout=timeout)

    def __del__(self) -> None:
        try:
            self.shutdown(timeout=1.0)
        except Exception:
            pass

    def decode_group_chunks_async(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("ShmCodecDecodeService returns handles; use decode_group_chunks_to_handles_async")

    def decode_group_chunks_to_handles_async(
        self,
        group_chunks: Sequence[Sequence[tuple[int, torch.Tensor]]],
        *,
        trim_specs: dict[int, dict[str, int]],
        record_batch: DecodeBatchRecordCallback | None = None,
    ) -> dict[int, torch.Tensor]:
        submit_start_ns = time.perf_counter_ns()
        normalized_group_chunks = [list(group_chunk) for group_chunk in group_chunks if group_chunk]
        handle_ids: dict[int, int] = {}
        for index in trim_specs:
            handle_id = _new_codec_handle_id()
            handle_ids[int(index)] = handle_id
        handles = {index: make_codec_handle_tensor(handle_id) for index, handle_id in handle_ids.items()}
        for index, handle_id in handle_ids.items():
            register_codec_shm_handle(
                handle_id,
                _codec_shm_payload_capacity_from_spec(trim_specs.get(index, {})),
            )
        if not normalized_group_chunks:
            for handle_id in handle_ids.values():
                self._write_result(handle_id, torch.zeros(0, dtype=torch.float32))
            return handles

        input_ready: torch.cuda.Event | None = None
        if self._stream is not None:
            assert self.device is not None
            with torch.cuda.device(self.device):
                input_ready = torch.cuda.Event(enable_timing=False)
                input_ready.record(torch.cuda.current_stream(self.device))

        job = _QueuedShmDecodeJob(
            group_chunks=normalized_group_chunks,
            record_batch=record_batch,
            trim_specs={int(k): {str(sk): int(sv) for sk, sv in v.items()} for k, v in trim_specs.items()},
            handle_ids=handle_ids,
            submit_ns=time.perf_counter_ns(),
            input_ready=input_ready,
        )
        with self._condition:
            if self._closed:
                exc = RuntimeError("Qwen3 codec shm batcher is shut down")
                for handle_id in handle_ids.values():
                    self._safe_write_error(handle_id, exc)
                return handles
            self._pending.append(job)
            self.stats.queued_jobs += 1
            self.stats.submit_queue_ns += time.perf_counter_ns() - submit_start_ns
            self._condition.notify()
        return handles

    def _worker_loop(self) -> None:
        while True:
            jobs = self._collect_jobs()
            if not jobs:
                return
            self._decode_jobs(jobs)

    def _collect_jobs(self) -> list[_QueuedShmDecodeJob]:
        with self._condition:
            while not self._pending and not self._closed:
                self._condition.wait()
            if not self._pending and self._closed:
                return []

            deadline = time.perf_counter() + self._max_queue_delay_s
            while not self._closed and len(self._pending) < self._max_queue_jobs and self._max_queue_delay_s > 0:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)

            jobs: list[_QueuedShmDecodeJob] = []
            while self._pending and len(jobs) < self._max_queue_jobs:
                jobs.append(self._pending.popleft())
            return jobs

    def _decode_jobs(self, jobs: list[_QueuedShmDecodeJob]) -> None:
        decode_start_ns = time.perf_counter_ns()
        for job in jobs:
            self.stats.batch_wait_ns += max(0, decode_start_ns - job.submit_ns)

        try:
            merged_group_chunks, index_map = self._merge_jobs(jobs)
            record_batch = self._first_record_batch_callback(jobs)

            def _record_batch(group_size: int, bucket_frames: int, actual_frames: list[int]) -> None:
                if record_batch is not None:
                    record_batch(group_size, bucket_frames, actual_frames)

            if self._stream is not None:
                assert self.device is not None
                with torch.cuda.device(self.device):
                    with torch.cuda.stream(self._stream):
                        for job in jobs:
                            if job.input_ready is not None:
                                self._stream.wait_event(job.input_ready)
                        wav_rows_by_global_index = self._sync_service.decode_group_chunks(
                            merged_group_chunks,
                            record_batch=_record_batch if record_batch is not None else None,
                        )
                        self._stream.synchronize()
            else:
                wav_rows_by_global_index = self._sync_service.decode_group_chunks(
                    merged_group_chunks,
                    record_batch=_record_batch if record_batch is not None else None,
                )

            for global_index, wav_row in wav_rows_by_global_index.items():
                job, local_index = index_map[global_index]
                handle_id = job.handle_ids.get(local_index)
                spec = job.trim_specs.get(local_index)
                if handle_id is None or spec is None:
                    continue
                wav = self._trim_wav_row(wav_row, spec)
                self._write_result(handle_id, wav)
            self.stats.dynamic_batches += 1
            self._maybe_log_stats()
        except Exception as exc:
            for job in jobs:
                for handle_id in job.handle_ids.values():
                    self._safe_write_error(handle_id, exc)

    def _write_result(self, handle_id: int, tensor: torch.Tensor) -> None:
        write_codec_shm_result_tensor(handle_id, tensor)

    def _write_error(self, handle_id: int, error: BaseException) -> None:
        write_codec_shm_error(handle_id, error)

    def _safe_write_error(self, handle_id: int, error: BaseException) -> None:
        try:
            self._write_error(handle_id, error)
        except Exception:
            logger.exception("Failed to write Qwen3 codec decode error for handle %s", handle_id)

    def _maybe_log_stats(self) -> None:
        if (
            not self._stats_enabled
            or self._stats_log_every <= 0
            or self.stats.dynamic_batches % self._stats_log_every != 0
        ):
            return
        decoded_batches = max(1, self.stats.decoded_batches)
        avg_decode_batch_size = self.stats.submitted_requests / decoded_batches
        avg_jobs_per_dynamic_batch = self.stats.queued_jobs / max(1, self.stats.dynamic_batches)
        pad_ratio = self.stats.padded_frames / max(1, self.stats.decoded_frames)
        logger.info(
            "Shm Code2Wav service stats: dynamic_batches=%d queued_jobs=%d "
            "decoded_batches=%d submitted_requests=%d avg_decode_batch_size=%.2f "
            "avg_jobs_per_dynamic_batch=%.2f padded_frames=%d decoded_frames=%d "
            "pad_ratio=%.2f%% top_actual_frames=%s top_bucket_groups=%s",
            self.stats.dynamic_batches,
            self.stats.queued_jobs,
            self.stats.decoded_batches,
            self.stats.submitted_requests,
            avg_decode_batch_size,
            avg_jobs_per_dynamic_batch,
            self.stats.padded_frames,
            self.stats.decoded_frames,
            100.0 * pad_ratio,
            self.stats.actual_frames.most_common(12),
            self.stats.bucket_groups.most_common(12),
        )

    @staticmethod
    def _trim_wav_row(wav_row: torch.Tensor, spec: dict[str, int]) -> torch.Tensor:
        ctx_frames = int(spec.get("ctx_frames", 0))
        actual_frames = int(spec.get("actual_frames", 0))
        upsample = int(spec.get("upsample", 1))
        start = max(0, ctx_frames * upsample)
        end = max(start, actual_frames * upsample)
        if start >= wav_row.shape[0]:
            return torch.zeros(0, dtype=torch.float32, device=wav_row.device)
        wav = wav_row[start : min(end, wav_row.shape[0])]
        if wav.dtype != torch.float32:
            wav = wav.to(torch.float32)
        return wav.reshape(-1)

    def _merge_jobs(
        self,
        jobs: list[_QueuedShmDecodeJob],
    ) -> tuple[list[list[tuple[int, torch.Tensor]]], dict[int, tuple[_QueuedShmDecodeJob, int]]]:
        grouped_by_target_frames: dict[int, list[tuple[int, torch.Tensor]]] = {}
        index_map: dict[int, tuple[_QueuedShmDecodeJob, int]] = {}
        next_global_index = 0
        for job in jobs:
            for group_chunk in job.group_chunks:
                if not group_chunk:
                    continue
                max_frames = max(int(codes_qf.shape[1]) for _, codes_qf in group_chunk)
                target_frames = self._sync_service._get_decode_batch_bucket_frames(max_frames)
                merged_group = grouped_by_target_frames.setdefault(target_frames, [])
                for local_index, codes_qf in group_chunk:
                    global_index = next_global_index
                    next_global_index += 1
                    merged_group.append((global_index, codes_qf))
                    index_map[global_index] = (job, local_index)

        max_batch_size = self._sync_service.decode_batch_max_size
        merged_group_chunks: list[list[tuple[int, torch.Tensor]]] = []
        for group_chunk in grouped_by_target_frames.values():
            if max_batch_size <= 0:
                merged_group_chunks.append(group_chunk)
                continue
            for start in range(0, len(group_chunk), max_batch_size):
                merged_group_chunks.append(group_chunk[start : start + max_batch_size])
        return merged_group_chunks, index_map

    @staticmethod
    def _first_record_batch_callback(jobs: list[_QueuedShmDecodeJob]) -> DecodeBatchRecordCallback | None:
        for job in jobs:
            if job.record_batch is not None:
                return job.record_batch
        return None
