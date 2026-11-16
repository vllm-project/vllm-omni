# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unified native MRV2 state for Zonos3 AR and in-process packed Codec."""

from __future__ import annotations

import math
import secrets
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, apply_top_k_top_p_pytorch
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.input_batch import InputBatch, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

from vllm_omni.utils.cuda_stream import create_cuda_stream
from vllm_omni.utils.device_copy import index_to_device, to_device_nonblocking
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
from vllm_omni.worker_v2.warmup import is_warmup_request

from .ar_config import Zonos3Architecture
from .modeling_zonos3_mini import Zonos3BackboneBatch, Zonos3ForCausalLM, Zonos3PrefillSpan
from .sampling_state import EOS, Zonos3ARSamplingState


@dataclass(frozen=True)
class Zonos3SamplingOptions:
    temperature: float
    top_k: int
    top_p: float
    min_p: float
    repetition_penalty: float
    repetition_window: int
    repetition_codebooks: int
    min_frames: int
    seed: int
    ignore_eos: bool
    eoa_codebook0: bool

    @classmethod
    def from_params(
        cls, params: SamplingParams, config: Zonos3Architecture, history_capacity: int
    ) -> Zonos3SamplingOptions:
        extra = params.extra_args or {}
        if not isinstance(extra, Mapping) or set(extra) - {
            "repetition_window",
            "repetition_codebooks",
            "min_frames",
            "eoa_policy",
            "tts_local_seed",
        }:
            raise ValueError("Invalid Zonos3 sampling options")
        if (
            params.logprobs is not None
            or params.prompt_logprobs is not None
            or params.logit_bias
            or params.allowed_token_ids
            or params.bad_words
            or params.structured_outputs
            or params.stop
            or params.frequency_penalty
            or params.presence_penalty
            or params.min_tokens
            or params.n != 1
            or any(token != EOS for token in params.stop_token_ids)
        ):
            raise ValueError("Unsupported logical-token sampling constraint for Zonos3 RVQ generation")
        policy = extra.get("eoa_policy", "any")
        if policy not in ("any", "codebook0"):
            raise ValueError("Zonos3 eoa_policy must be any or codebook0")
        codebooks = int(extra.get("repetition_codebooks", 6))
        request_seed = extra.get("tts_local_seed", params.seed)
        seed = int(request_seed) if request_seed is not None else secrets.randbits(63)
        seed = seed & ((1 << 64) - 1)
        if seed >= 1 << 63:
            seed -= 1 << 64
        values = (params.temperature, params.top_p, params.min_p, params.repetition_penalty)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Zonos3 sampling scalars must be finite")
        return cls(
            temperature=max(float(params.temperature), 0.0),
            top_k=min(params.top_k if params.top_k > 0 else config.codebook_size, config.audio_vocab),
            top_p=float(params.top_p) if params.top_p > 0 else 1.0,
            min_p=max(float(params.min_p), 0.0),
            repetition_penalty=max(float(params.repetition_penalty), 1.0),
            repetition_window=min(max(int(extra.get("repetition_window", 96)), 0), history_capacity),
            repetition_codebooks=10 if codebooks < 0 else min(codebooks, 10),
            min_frames=max(int(extra.get("min_frames", 0)), 0),
            seed=seed,
            ignore_eos=bool(params.ignore_eos),
            eoa_codebook0=policy == "codebook0",
        )


@dataclass
class _SamplingSettings:
    temperature: UvaBackedTensor
    top_k: UvaBackedTensor
    top_p: UvaBackedTensor
    min_p: UvaBackedTensor
    repetition_penalty: UvaBackedTensor
    repetition_window: UvaBackedTensor
    repetition_codebooks: UvaBackedTensor
    min_frames: UvaBackedTensor
    seed: UvaBackedTensor


class Zonos3RVQBuffer:
    """Bounded delayed-row ring and GPU-only chunk readiness, indexed by AR slot.

    Codec stream is the sole writer. Completed slots are absorbing; inactive
    lanes target distinct scratch slots, including zero-length normal calls.
    """

    def __init__(self, capacity: int, max_chunk: int, device: torch.device) -> None:
        self.capacity, self.max_chunk = capacity, max_chunk
        # Nine delayed codebooks plus the newly arriving row must coexist.
        self.ring = max_chunk + 10
        self.rows = torch.full((2 * capacity, self.ring, 10), -1, dtype=torch.long, device=device)
        self.received = torch.zeros(2 * capacity, dtype=torch.long, device=device)
        self.consumed_frames = torch.zeros_like(self.received)
        self.ready_frames = torch.zeros_like(self.received)
        self.primer = torch.zeros_like(self.received)
        self.skip_samples = torch.zeros_like(self.received)
        self.eoa = torch.full_like(self.received, -1)
        self.flushed = torch.zeros(2 * capacity, dtype=torch.bool, device=device)
        self.chunk_index = torch.zeros_like(self.received)
        self.thresholds = torch.full((2 * capacity, 32), 15, dtype=torch.long, device=device)
        self.first_codebooks = torch.ones_like(self.received)
        self.columns = torch.arange(10, device=device)

    def reset(
        self, slot: int, primer: torch.Tensor, chunk_size=15, ramp=(1, 2, 4, 8, 15), first=1, samples_per_frame=2940
    ) -> None:
        if (
            not 1 <= chunk_size <= self.max_chunk
            or len(ramp) > 31
            or any(not 1 <= n <= self.max_chunk for n in ramp)
            or not 1 <= first <= 10
        ):
            raise ValueError("Codec chunk settings exceed the configured static capacity")
        n = primer.shape[0]
        if n + (ramp[0] if ramp else chunk_size) > self.max_chunk:
            raise ValueError("Silence primer plus initial Codec chunk exceeds static capacity")
        self.rows[slot].fill_(-1)
        self.rows[slot, :n].copy_(primer)
        for tensor in (self.consumed_frames, self.ready_frames, self.chunk_index):
            tensor[slot : slot + 1].zero_()
        self.received[slot : slot + 1].fill_(n)
        self.primer[slot : slot + 1].fill_(n)
        self.skip_samples[slot : slot + 1].fill_(n * samples_per_frame)
        self.eoa[slot : slot + 1].fill_(-1)
        self.flushed[slot : slot + 1].fill_(False)
        self.first_codebooks[slot : slot + 1].fill_(first)
        thresholds = [*ramp, *([chunk_size] * (32 - len(ramp)))]
        self.thresholds[slot].copy_(index_to_device(thresholds, self.rows.device))

    def prepare(self, codes, slots, active, final, token_capacity):
        active = active & ~self.flushed[slots]
        received = self.received[slots]
        address = received.remainder(self.ring)
        previous = self.rows[slots, address]
        self.rows[slots, address] = torch.where(active[:, None], codes, previous)
        last_eoa = torch.where((codes == 4096) & active[:, None], self.columns, -1).amax(-1)
        observed = (received - last_eoa).clamp_min(0)
        old_eoa = self.eoa[slots]
        eoa = torch.where(last_eoa >= 0, torch.where(old_eoa >= 0, torch.minimum(old_eoa, observed), observed), old_eoa)
        self.eoa.index_copy_(0, slots, eoa)
        received = received + active.long()
        self.received.index_copy_(0, slots, received)
        complete = (received - 9).clamp_min(0)
        index = self.chunk_index[slots]
        partial = (index == 0) & ~final
        ready = torch.where(partial, (received - self.first_codebooks[slots] + 1).clamp_min(0), complete)
        ready = torch.where(eoa >= 0, torch.minimum(ready, eoa), ready)
        self.ready_frames.index_copy_(0, slots, ready)
        consumed = self.consumed_frames[slots]
        threshold = self.thresholds[slots, index.clamp_max(31)]
        run = ((ready - torch.maximum(consumed, self.primer[slots]) >= threshold) | final) & ~self.flushed[slots]
        lengths = torch.where(run, (ready - consumed).clamp_min(0), 0)
        torch._assert_async((lengths <= self.max_chunk).all(), "Codec chunk exceeds static capacity")
        flush = final & run
        self.consumed_frames.index_copy_(0, slots, consumed + lengths)
        self.chunk_index.index_copy_(0, slots, index + run.long())
        self.flushed.index_copy_(0, slots, self.flushed[slots] | flush)
        # Stable compaction preserves native request order among ready rows.
        # The capacity is fixed; only lengths/slots/order change on device.
        order = torch.argsort((~run).long(), stable=True)
        source_slots = slots[order]
        lengths, flush = lengths[order], flush[order]
        consumed, received = consumed[order], received[order]
        offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
        row = torch.arange(token_capacity, device=codes.device)
        request = torch.searchsorted(offsets[1:], row, right=True).clamp_max(slots.numel() - 1)
        frame = consumed[request] + row - offsets[request]
        raw_index = frame[:, None] + self.columns
        packed = self.rows[source_slots[request, None], raw_index.remainder(self.ring), self.columns]
        valid = (row < offsets[-1])[:, None] & (raw_index < received[request, None]) & (packed >= 0) & (packed < 4096)
        packed = torch.where(valid, packed, -1)
        packed[:, 0].clamp_min_(0)
        destinations = torch.where(
            run[order], source_slots, self.capacity + torch.arange(slots.numel(), device=slots.device)
        )
        return packed, destinations, lengths, flush, self.skip_samples[source_slots], order


@dataclass
class _AudioLease:
    active: bool = True
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class Zonos3AudioSnapshot:
    """Owned PCM and boundaries retained through native output materialization.

    Each in-flight step retains its allocations and lease objects. Neither a
    graph replay nor slot reuse can overwrite or resurrect a cancelled output.
    """

    audio: torch.Tensor
    offsets: torch.Tensor
    counts: torch.Tensor
    skip: torch.Tensor
    leases: list[_AudioLease | None]
    order: torch.Tensor
    event: torch.cuda.Event | None = None
    pcm_copy_stream: torch.cuda.Stream | None = None

    def to_cpu(self, copy_stream, copy_tensor):
        if self.event is not None:
            copy_stream.wait_event(self.event)
        return Zonos3AudioSnapshot(
            self.audio,
            copy_tensor(self.offsets),
            copy_tensor(self.counts),
            copy_tensor(self.skip),
            self.leases,
            copy_tensor(self.order),
            pcm_copy_stream=self.pcm_copy_stream,
        )

    def get_output(self):
        boundaries, counts, skip = self.offsets.tolist(), self.counts.tolist(), self.skip.tolist()
        # Native get_output runs after its metadata copy event, on the output
        # materializer. Only actual PCM crosses PCIe; capacity-tail garbage and
        # no-ready steps never trigger a large D2H. This wait does not control
        # Codec readiness or run on the GPU submission thread.
        total = boundaries[-1]
        if self.audio.device.type == "cuda" and total:
            assert self.pcm_copy_stream is not None
            host = torch.empty(total, dtype=self.audio.dtype, device="cpu", pin_memory=True)
            with torch.cuda.stream(self.pcm_copy_stream):
                host.copy_(self.audio[:total], non_blocking=True)
                copied = torch.cuda.Event()
                copied.record()
            copied.synchronize()
            self.audio = host
        sample_rate = torch.tensor(44100, dtype=torch.int32)
        output: list[dict[str, torch.Tensor] | None] = [None] * len(self.leases)
        for packed_row, native_row in enumerate(self.order.tolist()):
            lease, n, crop = self.leases[native_row], counts[packed_row], skip[packed_row]
            if lease is not None and n > crop:
                with lease.lock:
                    if lease.active:
                        start = boundaries[packed_row]
                        output[native_row] = {"model_outputs": self.audio[start + crop : start + n], "sr": sample_rate}
        return output


@dataclass
class _Prompt:
    embeddings: torch.Tensor
    has_audio_prefix: bool


@dataclass
class _PromptInput:
    frames: torch.Tensor
    condition_start: int = 0
    conditions: tuple[int, ...] = ()
    speaker_row: int | None = None
    speaker: torch.Tensor | None = None
    waveform: torch.Tensor | None = None
    sample_rate: int | None = None
    has_audio_prefix: bool = False
    warmup: bool = False


class Zonos3MiniModelState(OmniModelState):
    supports_prompt_embeds = False
    # vLLM attaches an all-valid grammar mask to startup warmup requests and
    # uses sampler-warmup SamplingParams (including logprobs > this model's
    # two logical tokens). Zonos3 determines CONTINUE/EOS in its own GPU step;
    # route only synthetic warmup requests through that result as well.
    sample_determined_tokens_during_warmup = True

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: Zonos3ForCausalLM,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        if vllm_config.model_config.enable_prompt_embeds or encoder_cache is not None:
            raise ValueError("Zonos3 conditioning must use its request metadata, not generic prompt/MM embeddings")
        if vllm_config.speculative_config is not None:
            raise ValueError("Zonos3 V1 does not support speculative scheduler tokens")
        super().__init__(vllm_config, model, encoder_cache, device)
        self.has_preprocess = True
        self._embed_dim = model.config.dim
        if self._static_inputs_embeds is None:
            self._static_inputs_embeds = torch.zeros(
                (self.max_num_tokens, self._embed_dim), dtype=self.dtype, device=device
            )
        self.gdn_states = model.allocate_gdn_states(2 * self.max_num_reqs)
        self.sampling_state = Zonos3ARSamplingState(
            2 * self.max_num_reqs, model.config.eoa_id, model.config.audio_pad_id, device
        )
        self.next_embeddings = torch.zeros((2 * self.max_num_reqs, self._embed_dim), dtype=self.dtype, device=device)
        self.history = torch.full((self.max_num_reqs, 10, self.max_model_len), -1, dtype=torch.int64, device=device)
        self.settings = _SamplingSettings(
            **{
                name: UvaBackedTensor(self.max_num_reqs, dtype=dtype)
                for name, dtype in (
                    ("temperature", torch.float32),
                    ("top_k", torch.int32),
                    ("top_p", torch.float32),
                    ("min_p", torch.float32),
                    ("repetition_penalty", torch.float32),
                    ("repetition_window", torch.int32),
                    ("repetition_codebooks", torch.int32),
                    ("min_frames", torch.int32),
                    ("seed", torch.int64),
                )
            }
        )
        self.options: list[Zonos3SamplingOptions | None] = [None] * self.max_num_reqs
        self._sampling_pending: set[int] = set()
        self._generators: dict[int, torch.Generator] = {}
        self._sample_output: SamplerOutput | None = None
        self.generation_limit = torch.zeros(self.max_num_reqs, dtype=torch.long, device=device)
        self._slots = torch.empty(self.max_num_reqs, dtype=torch.int64, device=device)
        self._scratch_slots = torch.arange(self.max_num_reqs, 2 * self.max_num_reqs, device=device)
        self._decode_rows = torch.arange(self.max_num_reqs, device=device)
        self._batches: dict[int, Zonos3BackboneBatch] = {}
        self._prompts: dict[str, _Prompt] = {}
        self._lengths: dict[str, int] = {}
        self._request_slots: dict[str, int] = {}
        self._decode_started: set[str] = set()
        self._startup = True
        self._audio_leases: dict[str, _AudioLease] = {}
        self._audio_output: Zonos3AudioSnapshot | None = None
        self.codec_stream = None
        self.audio_copy_stream = None
        self.codec_graph = None
        # Numerical backbone fixtures have no release Codec attached.
        if hasattr(model, "codec"):
            from .cuda_graph_wrapper import ZACCodecGraphWrapper

            max_chunk = int(getattr(vllm_config.model_config.hf_config, "zonos3_codec_max_chunk", 31))
            self.rvq = Zonos3RVQBuffer(self.max_num_reqs, max_chunk, device)
            self.codec_graph = ZACCodecGraphWrapper(
                model.codec.network, self.rvq, self.max_num_reqs, max_chunk, vllm_config.model_config.enforce_eager
            )
            if device.type == "cuda":
                self.codec_stream = create_cuda_stream(device)
                self.audio_copy_stream = torch.cuda.Stream(device=device)
                self.codec_stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(self.codec_stream):
                    self.codec_graph.capture()
                self.codec_stream.synchronize()

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        if new_req_data.num_computed_tokens:
            raise ValueError("Zonos3 cannot restore a KV-only prefix or recomputed/preempted GDN state")
        if new_req_data.req_id in self._request_slots:
            raise ValueError("Zonos3 request was reinitialized without completion cleanup")
        params = new_req_data.sampling_params
        if is_warmup_request(new_req_data.req_id):
            # Upstream startup requests can ask for text-only logprobs/grammar.
            # They are not serving requests and have no physical RVQ contract.
            # Zonos3 emits only CONTINUE/EOS logical IDs, so the upstream text
            # sampler's dummy logprob request (typically top-20) can exceed
            # this tiny logical vocabulary during GPU startup warmup.
            params = SamplingParams(temperature=0.9, top_k=50, top_p=0.9, min_p=0.1, seed=0)
            params.logprobs = None
            params.prompt_logprobs = None
        options = Zonos3SamplingOptions.from_params(params, self.model.config, self.max_model_len)
        super().add_request(req_index, new_req_data)
        req_id = new_req_data.req_id
        self._request_slots[req_id] = req_index
        self._lengths[req_id] = len(new_req_data.prompt_token_ids)
        self.generation_limit[req_index : req_index + 1].fill_(
            min(params.max_tokens or self.max_model_len, self.max_model_len - self._lengths[req_id])
        )
        self._audio_leases[req_id] = _AudioLease()
        if self.codec_graph is not None:
            info = self.intermediate_buffer.buffers[req_index]
            count = info.get("silence_prefill_frames", 0)
            frames = info.get("frame_ids")
            primer = frames[-count:, :10] if count else torch.empty(0, 10, dtype=torch.long)
            chunk = info.get("codec_chunking", {})
            with torch.cuda.stream(self.codec_stream) if self.codec_stream is not None else nullcontext():
                self.codec_graph.codec.reset(index_to_device([req_index], device=self.device))
                self.rvq.reset(
                    req_index,
                    to_device_nonblocking(primer, self.device),
                    chunk.get("chunk_size", 15),
                    chunk.get("chunk_ramp", (1, 2, 4, 8, 15)),
                    chunk.get("first_chunk_codebooks", 1),
                )
        self._startup = self._startup and is_warmup_request(req_id)
        for state in self.gdn_states.values():
            state.reset([req_index])
        slots = index_to_device([req_index], self.device)
        self.sampling_state.reset(slots, torch.zeros(1, dtype=torch.bool, device=self.device))
        self.next_embeddings.index_fill_(0, slots, 0)
        self.history.index_fill_(0, slots, -1)
        self._stage_sampling_options(req_index, options)

    def _stage_sampling_options(self, req_index: int, options: Zonos3SamplingOptions) -> None:
        self.options[req_index] = options
        for setting in fields(self.settings):
            getattr(self.settings, setting.name).np[req_index] = getattr(options, setting.name)
        self._sampling_pending.add(req_index)
        if self.device.type == "cpu":
            self._generators[req_index] = torch.Generator().manual_seed(options.seed)

    def apply_staged_writes(self) -> None:
        super().apply_staged_writes()
        if not self._sampling_pending:
            return
        for setting in fields(self.settings):
            getattr(self.settings, setting.name).copy_to_uva()
        slots = sorted(self._sampling_pending)
        indices = index_to_device(slots, self.device)
        self.sampling_state.ignore_eos.index_copy_(
            0, indices, index_to_device([self.options[i].ignore_eos for i in slots], self.device, dtype=torch.bool)
        )
        self.sampling_state.eoa_codebook0.index_copy_(
            0, indices, index_to_device([self.options[i].eoa_codebook0 for i in slots], self.device, dtype=torch.bool)
        )
        self._sampling_pending.clear()

    def _drop(self, req_id: str) -> None:
        lease = self._audio_leases.pop(req_id, None)
        if lease is not None:
            with lease.lock:
                lease.active = False
        self._prompts.pop(req_id, None)
        self._lengths.pop(req_id, None)
        slot = self._request_slots.pop(req_id, None)
        if slot is not None:
            self.options[slot] = None
            self._generators.pop(slot, None)
            self._sampling_pending.discard(slot)
        self._decode_started.discard(req_id)

    def remove_request(self, req_index_or_id: int | str) -> None:
        index = self._resolve_req_index(req_index_or_id)
        if index is not None:
            req_id = self.intermediate_buffer.buffers[index].get("req_id")
            if req_id is not None:
                self._drop(req_id)
        super().remove_request(req_index_or_id)

    def on_requests_finished(self, req_ids: set[str]) -> None:
        for req_id in req_ids:
            self._drop(req_id)
        super().on_requests_finished(req_ids)

    def on_request_preempted(self, req_id: str, req_index: int) -> None:
        del req_index
        raise RuntimeError(f"Zonos3 AR does not support preemption/state recomputation: {req_id}")

    def _prompt_input(self, req_id: str, info: dict[str, Any]) -> _PromptInput:
        length = self._lengths[req_id]
        if is_warmup_request(req_id):
            return _PromptInput(torch.zeros(length, 11, dtype=torch.int64), warmup=True)
        if not info:
            raise ValueError("Missing Zonos3 prompt conditioning metadata")
        ids = info.get("frame_ids")
        c = self.model.config
        if (
            not isinstance(ids, torch.Tensor)
            or ids.device.type != "cpu"
            or ids.dtype not in (torch.int32, torch.int64)
            or ids.shape != (length, 11)
        ):
            raise ValueError("Zonos3 placeholder length must exactly match CPU prompt frame_ids [length, 11]")
        if bool(((ids[:, :10] < 0) | (ids[:, :10] >= c.audio_vocab)).any()) or bool(
            ((ids[:, 10] < 0) | (ids[:, 10] >= c.text_rows)).any()
        ):
            raise ValueError("Zonos3 conditioning contains out-of-vocabulary frame IDs")
        speaker, waveform = info.get("speaker_embedding"), info.get("reference_waveform")
        sample_rate = info.get("reference_sample_rate")
        if waveform is not None:
            if (
                speaker is not None
                or self.model.speaker_encoder is None
                or not isinstance(waveform, torch.Tensor)
                or waveform.device.type != "cpu"
            ):
                raise ValueError("Zonos3 raw reference requires the model's ownmect_s2 encoder and CPU waveform")
            max_samples = int(c.speaker_reference_max_seconds * int(sample_rate or 0))
            if (
                not isinstance(sample_rate, int)
                or sample_rate <= 0
                or waveform.ndim not in (1, 2)
                or waveform.numel() == 0
                or waveform.shape[-1] > max_samples
            ):
                raise ValueError(
                    "Zonos3 raw speaker reference must be non-empty and no longer than "
                    f"{c.speaker_reference_max_seconds:g} seconds"
                )
        row = info.get("speaker_row")
        if speaker is not None or waveform is not None:
            if (
                not isinstance(row, int)
                or not 0 <= row < length
                or (
                    speaker is not None
                    and (
                        not isinstance(speaker, torch.Tensor)
                        or speaker.device.type != "cpu"
                        or speaker.shape != (c.speaker_embedding_dim,)
                    )
                )
            ):
                raise ValueError("Invalid Zonos3 speaker embedding row")
        buckets, start = info.get("condition_buckets"), info.get("condition_start")
        if (
            not isinstance(buckets, (list, tuple))
            or len(buckets) != c.condition_count
            or not isinstance(start, int)
            or not 0 <= start <= length - c.condition_count
        ):
            raise ValueError("Invalid Zonos3 condition rows")
        for bucket, count in zip(buckets, c.condition_bucket_counts, strict=True):
            if not isinstance(bucket, int) or not 0 <= bucket <= count:
                raise ValueError("Invalid Zonos3 condition bucket ID")
        return _PromptInput(
            ids, start, tuple(buckets), row, speaker, waveform, sample_rate, bool(info.get("has_audio_prefix", False))
        )

    def preprocess_batch(self, req_infos: Sequence[tuple[str, dict[str, Any]]]) -> list[_Prompt]:
        """One pinned frame transfer and embedding pass for all new prompts.

        Prompt caches own immutable views of this cohort's embedding allocation.
        Host staging is fresh per call; the caching pinned allocator retains it
        until queued copies complete, without a reusable-buffer overwrite race.
        """
        if not req_infos:
            return []
        plans = [self._prompt_input(req_id, info) for req_id, info in req_infos]
        frames = torch.cat([plan.frames for plan in plans])
        embeddings = self.model.multi_embedder(to_device_nonblocking(frames, self.device))
        offsets = np.cumsum([0, *(len(p.frames) for p in plans)])
        # Variable-length raw waveforms retain separate encoder calls. Plain
        # speaker vectors and resulting GPU vectors each share one projection.
        cpu_speakers, gpu_speakers = [], []
        for i, plan in enumerate(plans):
            if plan.warmup:
                embeddings[offsets[i] : offsets[i + 1]].zero_()
            row = int(offsets[i]) + (plan.speaker_row or 0)
            if plan.waveform is not None:
                feature = self.model.speaker_encoder(plan.waveform, plan.sample_rate).squeeze(0)
                gpu_speakers.append((row, feature))
            elif plan.speaker is not None:
                cpu_speakers.append((row, plan.speaker))
        for group in (cpu_speakers, gpu_speakers):
            if group:
                features = to_device_nonblocking(torch.stack([value for _, value in group]), self.device).to(self.dtype)
                rows = index_to_device([row for row, _ in group], self.device)
                embeddings.index_copy_(0, rows, self.model.speaker_projection(features))
        real = [(i, p) for i, p in enumerate(plans) if not p.warmup]
        if real:
            positions = [
                [int(offsets[i]) + p.condition_start + j for i, p in real]
                for j in range(self.model.config.condition_count)
            ]
            buckets = [[p.conditions[j] for _, p in real] for j in range(self.model.config.condition_count)]
            indices = to_device_nonblocking(torch.tensor([positions, buckets], dtype=torch.int64), self.device)
            for j, table in enumerate(self.model.condition_embeddings):
                embeddings.index_copy_(0, indices[0, j], table(indices[1, j]))
        return [_Prompt(embeddings[offsets[i] : offsets[i + 1]], p.has_audio_prefix) for i, p in enumerate(plans)]

    def build_prompt(self, req_id: str, info: dict[str, Any]) -> _Prompt:
        return self.preprocess_batch([(req_id, info)])[0]

    def preprocess_decode_batch(
        self, input_batch: InputBatch, embeddings: torch.Tensor, rows: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather next-frame embeddings once; pure decode needs no host staging."""
        if rows.size == input_batch.num_reqs:
            slots = input_batch.idx_mapping[: input_batch.num_reqs]
            destination = self._decode_rows[: input_batch.num_reqs]
            torch.index_select(self.next_embeddings, 0, slots, out=embeddings[: input_batch.num_reqs])
        elif rows.size:
            metadata = np.stack((input_batch.query_start_loc_np[rows], input_batch.idx_mapping_np[rows])).astype(
                np.int64
            )
            staged = to_device_nonblocking(torch.from_numpy(metadata), self.device)
            destination, slots = staged.unbind(0)
            embeddings.index_copy_(0, destination, self.next_embeddings.index_select(0, slots))
        else:
            slots = destination = self._decode_rows[:0]
        self._decode_started.update(input_batch.req_ids[i] for i in rows)
        return slots, destination

    def _batch(self, tokens: int, decode: bool) -> Zonos3BackboneBatch:
        batch = self._batches.get(tokens)
        if batch is None:
            batch = Zonos3BackboneBatch(
                self.gdn_states,
                self.model.allocate_workspace(tokens),
                self._slots[:tokens],
                torch.arange(tokens, device=self.device),
                decode_only=decode,
            )
        batch.decode_only = decode
        return batch

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        if not 0 < num_reqs <= self.max_num_reqs or num_tokens > self.max_num_tokens:
            raise ValueError("Invalid Zonos3 native dummy batch shape")
        slots = list(range(self.max_num_reqs, self.max_num_reqs + num_reqs))
        for state in self.gdn_states.values():
            state.reset(slots)
        self._static_inputs_embeds[:num_tokens].zero_()
        batch = self._batch(num_tokens, num_reqs == num_tokens)
        batch.dummy_slots = tuple(slots)
        # Both FULL and PIECEWISE graphs keep raw workspace addresses. Retain
        # every capture bucket and reuse its exact bank during live replay.
        # Arbitrary uncaptured eager shapes are still transient in _batch().
        self._batches[num_tokens] = batch
        if batch.decode_only:
            self._slots[:num_tokens].copy_(index_to_device(slots, self.device))
            batch.decode_slots = self._slots[:num_tokens]
            batch.prefill_spans = ()
        else:
            lengths = [num_tokens // num_reqs] * num_reqs
            for i in range(num_tokens % num_reqs):
                lengths[-1 - i] += 1
            spans, start = [], 0
            for slot, length in zip(slots, lengths, strict=True):
                spans.append(Zonos3PrefillSpan(start, length, slot, 0))
                start += length
            batch.prefill_spans = tuple(spans)
            batch.decode_slots = torch.empty(0, dtype=torch.int64, device=self.device)
            batch.decode_rows = batch.decode_slots
        self.model.bind_gdn_batch(batch)
        if self.model._compiled_gdn:
            return {"inputs_embeds": self._static_inputs_embeds[:num_tokens], "zonos3_workspace": batch.workspace.bank}
        return {"inputs_embeds": self._static_inputs_embeds[:num_tokens], "zonos3_batch": batch}

    def prepare_inputs(self, input_batch: InputBatch, req_states: RequestState) -> dict[str, Any]:
        result = super().prepare_inputs(input_batch, req_states)
        if self._startup and all(req_id not in self._request_slots for req_id in input_batch.req_ids):
            result.update(self.prepare_dummy_inputs(input_batch.num_reqs, input_batch.num_tokens_after_padding))
        return result

    def run_preprocess(
        self,
        input_batch: InputBatch,
        model_inputs: dict[str, Any],
        req_states: RequestState | None = None,
        mtp_batch_descriptor_dispatcher: Callable[[int], Any] | None = None,
    ) -> None:
        tokens = input_batch.num_tokens_after_padding
        embeddings = self._static_inputs_embeds[:tokens]
        embeddings.zero_()
        one_token = bool(np.all(input_batch.num_scheduled_tokens == 1))
        decode_only = one_token and input_batch.decode_graph_eligible
        batch = self._batch(tokens, decode_only)
        batch.dummy_slots = ()
        expected_slots = np.fromiter(
            (self._request_slots.get(req_id, -1) for req_id in input_batch.req_ids), dtype=np.int64
        )
        if not np.array_equal(expected_slots, input_batch.idx_mapping_np):
            raise RuntimeError("Zonos3 request/slot identity mismatch")
        prefill_rows = np.flatnonzero(input_batch.is_prefilling_np)
        decode_rows = np.flatnonzero(~input_batch.is_prefilling_np)
        if np.any(input_batch.num_scheduled_tokens[decode_rows] != 1) or any(
            input_batch.req_ids[i] not in self._prompts for i in decode_rows
        ):
            raise ValueError("Zonos3 AR decode needs one logical token after conditioning prefill")
        spans, new = [], []
        for row in prefill_rows:
            req_id = input_batch.req_ids[row]
            if req_id in self._decode_started:
                raise RuntimeError("Zonos3 cannot resume prefill after AR decode")
            length = int(input_batch.num_scheduled_tokens[row])
            offset = int(input_batch.num_computed_prefill_tokens_np[row])
            if offset + length > self._lengths[req_id] or int(input_batch.prefill_len_np[row]) != self._lengths[req_id]:
                raise ValueError("Zonos3 scheduled prefill differs from its exact logical prompt length")
            if req_id not in self._prompts:
                new.append((req_id, self.intermediate_buffer.buffers[int(expected_slots[row])]))
            spans.append(
                Zonos3PrefillSpan(int(input_batch.query_start_loc_np[row]), length, int(expected_slots[row]), offset)
            )
        if new:
            prompts = self.preprocess_batch(new)
            self._prompts.update((req_id, prompt) for (req_id, _), prompt in zip(new, prompts, strict=True))
            prefix_slots = [
                self._request_slots[req_id]
                for (req_id, _), prompt in zip(new, prompts, strict=True)
                if prompt.has_audio_prefix
            ]
            if prefix_slots:
                # A GPU fill replaces per-request Python-bool assignment, which
                # otherwise performs a blocking pageable scalar transfer.
                self.sampling_state.has_audio_prefix.index_fill_(0, index_to_device(prefix_slots, self.device), True)
        pieces = [
            self._prompts[input_batch.req_ids[i]].embeddings[span.previous_tokens : span.previous_tokens + span.length]
            for i, span in zip(prefill_rows, spans, strict=True)
        ]
        if len(pieces) == 1:
            span = spans[0]
            embeddings[span.start : span.start + span.length].copy_(pieces[0])
        elif pieces:
            destinations = np.concatenate(
                [np.arange(span.start, span.start + span.length, dtype=np.int64) for span in spans]
            )
            rows = to_device_nonblocking(torch.from_numpy(destinations), self.device)
            embeddings.index_copy_(0, rows, torch.cat(pieces))
        live_slots, live_rows = self.preprocess_decode_batch(input_batch, embeddings, decode_rows)
        if decode_only:
            count = input_batch.num_reqs
            if tokens > self.max_num_reqs:
                raise ValueError("Zonos3 decode graph padding exceeds its configured request capacity")
            self._slots[:count].copy_(input_batch.idx_mapping[:count])
            self._slots[count:tokens].copy_(self._scratch_slots[: tokens - count])
            batch.decode_slots, batch.prefill_spans = self._slots[:tokens], ()
            for span in spans:
                for state in self.gdn_states.values():
                    state.prefill_tokens[span.slot] = span.previous_tokens + 1
        else:
            batch.prefill_spans = tuple(spans)
            batch.decode_slots, batch.decode_rows = live_slots, live_rows
        model_inputs["inputs_embeds"] = embeddings
        self.model.bind_gdn_batch(batch)
        if self.model._compiled_gdn:
            model_inputs["zonos3_workspace"] = batch.workspace.bank
        else:
            model_inputs["zonos3_batch"] = batch

    def _setting(self, name: str, slots: torch.Tensor) -> torch.Tensor:
        backed: UvaBackedTensor = getattr(self.settings, name)
        if backed.gpu.device != slots.device:
            values = backed.np[slots.detach().cpu().numpy()]
            return torch.from_numpy(values.copy()).to(slots.device)
        return backed.gpu.index_select(0, slots)

    def probabilities(self, logits: torch.Tensor, slots: torch.Tensor, width: int) -> torch.Tensor:
        """Per-codebook filters, using the same upstream kernels as before."""
        c = self.model.config
        logits = logits.float()
        positions = self.sampling_state.position.index_select(0, slots)
        ids = torch.arange(c.audio_vocab, device=logits.device)
        invalid = (ids >= c.codebook_size) & (ids != c.eoa_id)
        minimum = self._setting("min_frames", slots)
        stop_blocked = (minimum[:, None] > 0) & (positions[:, None] < minimum[:, None] + self.sampling_state.columns)
        logits = logits.masked_fill(
            invalid[None, None, :] | (stop_blocked[:, :, None] & (ids == c.eoa_id)), float("-inf")
        )
        if width:
            offsets = torch.arange(width, device=logits.device)
            indices = (positions[:, None] - 1 - offsets) % self.max_model_len
            history = self.history[
                slots[:, None, None], self.sampling_state.columns[None, :, None], indices[:, None, :]
            ]
            window = self._setting("repetition_window", slots)
            books = self._setting("repetition_codebooks", slots)
            valid = (history >= 0) & (history < c.codebook_size)
            valid = valid & (offsets[None, None, :] < torch.minimum(positions, window)[:, None, None])
            valid = valid & (self.sampling_state.columns[None, :, None] < books[:, None, None])
            repeated = torch.zeros_like(logits, dtype=torch.int32)
            repeated.scatter_add_(-1, history.clamp(min=0, max=c.audio_vocab - 1), valid.to(torch.int32))
            penalty = self._setting("repetition_penalty", slots)[:, None, None]
            adjusted = torch.where(logits > 0, logits / penalty, logits * penalty)
            logits = torch.where(repeated > 0, adjusted, logits)
        temperature = self._setting("temperature", slots)[:, None, None]
        logits = logits / torch.where(temperature > 0, temperature, 1.0)
        # Triton expects contiguous per-codebook settings, including B=1.
        k = self._setting("top_k", slots).repeat_interleave(10)
        p = self._setting("top_p", slots).repeat_interleave(10)
        flat = logits.reshape(-1, c.audio_vocab)
        if flat.is_cuda:
            logits = apply_top_k_top_p(flat, k, p, has_topp_only_rows=True)
        else:
            logits = apply_top_k_top_p_pytorch(flat, k.to(flat.device), p.to(flat.device))
        logits = logits.view(-1, 10, c.audio_vocab)
        probs = logits.softmax(-1)
        min_p = self._setting("min_p", slots)[:, None, None]
        probs = probs.masked_fill(probs < min_p * probs.amax(-1, keepdim=True), 0)
        total = probs.sum(-1, keepdim=True)
        fallback = torch.zeros_like(probs).scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
        return torch.where(total > 0, probs / total.clamp(min=1e-8), fallback)

    def _draw_cpu(self, probs: torch.Tensor, indices: list[int], active: torch.Tensor) -> torch.Tensor:
        result = torch.full((len(indices), 10), self.model.config.audio_pad_id, dtype=torch.int64)
        for row, slot in enumerate(indices):
            if bool(active[row]):
                if self.options[slot].temperature == 0:
                    result[row] = probs[row].argmax(-1)
                else:
                    result[row] = torch.multinomial(
                        probs[row, :, : self.model.config.audio_pad_id + 1], 1, generator=self._generators[slot]
                    ).flatten()
        return result

    def postprocess_model_output(
        self, model_output: Any, input_batch: InputBatch, req_states: RequestState
    ) -> tuple[torch.Tensor, dict]:
        hidden, _ = super().postprocess_model_output(model_output, input_batch, req_states)
        if input_batch.num_draft_tokens or input_batch.logits_indices.numel() != input_batch.num_reqs:
            raise ValueError("Zonos3 RVQ sampling does not support speculative scheduler tokens")
        if self._sampling_pending:
            raise RuntimeError("Zonos3 sampling staged writes must be applied before postprocess")
        slots = input_batch.idx_mapping
        options = [self.options[int(slot)] for slot in input_batch.idx_mapping_np]
        if any(option is None for option in options):
            raise RuntimeError("Zonos3 sampling request was not initialized")
        if hidden.device.type == "cuda":
            sampled, rejected = get_num_sampled_and_rejected(
                input_batch.seq_lens.new_ones(input_batch.num_reqs),
                input_batch.seq_lens,
                input_batch.cu_num_logits,
                slots,
                req_states.prefill_len.gpu,
            )
        else:
            prefill_len = req_states.prefill_len.gpu.to(device=hidden.device)
            sampled = (input_batch.seq_lens >= prefill_len.index_select(0, slots)).to(torch.int32)
            rejected = torch.zeros_like(sampled)
        logits = self.model.multi_output(hidden.index_select(0, input_batch.logits_indices))
        width = max(option.repetition_window for option in options)
        if hidden.device.type == "cuda":
            from .sampling_kernels import sample_rvq

            codes = sample_rvq(
                logits,
                slots,
                self.sampling_state,
                self.settings,
                self.history,
                width,
                self.model.config.codebook_size,
                self.model.config.eoa_id,
            )
        elif hidden.device.type == "cpu":
            probs = self.probabilities(logits, slots, width)
            active = (sampled > 0) & ~(
                (self.sampling_state.eoa_frame.index_select(0, slots) >= 0)
                & (self.sampling_state.drain_remaining.index_select(0, slots) == 0)
            )
            codes = self._draw_cpu(probs, [int(slot) for slot in input_batch.idx_mapping_np], active)
        else:
            raise ValueError("Zonos3 sampling supports CUDA serving and CPU numerical checks")
        # Partial-prefill and already-finished rows are inactive. The logical
        # EOS step still contributes its last delayed row before final flush.
        step = self.sampling_state.accept(slots, codes, sampled > 0)
        if hidden.device.type == "cuda":
            from .sampling_kernels import commit_feedback

            commit_feedback(self.model, slots, step, self.sampling_state, self.history, self.next_embeddings)
        else:
            position = self.sampling_state.position.index_select(0, slots) - step.active.long()
            ring = position[:, None] % self.max_model_len
            old = self.history[slots[:, None], self.sampling_state.columns[None, :], ring]
            self.history[slots[:, None], self.sampling_state.columns[None, :], ring] = torch.where(
                step.active[:, None], step.codes, old
            )
            frame = torch.cat((step.codes, torch.full_like(step.codes[:, :1], self.model.config.text_vocab)), -1)
            embeddings = self.model.multi_embedder(frame)
            previous = self.next_embeddings.index_select(0, slots)
            self.next_embeddings.index_copy_(0, slots, torch.where(step.active[:, None], embeddings, previous))
        self._sample_output = SamplerOutput(step.logical_token_ids.long(), None, None, sampled, rejected)
        if self.codec_graph is not None:
            # Own native metadata before the next async step can reuse its buffers.
            owned_slots = slots.clone()
            # accept() allocates fresh code/active tensors; only native slots
            # borrow a reusable runner buffer. Retain them on the Codec stream.
            owned_codes, owned_active = step.codes, step.active
            final = step.finished | (
                (self.sampling_state.position[slots] >= self.generation_limit[slots]) & (sampled > 0)
            )
            sampled_event = None
            if self.codec_stream is not None:
                sampled_event = torch.cuda.Event()
                sampled_event.record()
            with torch.cuda.stream(self.codec_stream) if self.codec_stream is not None else nullcontext():
                if sampled_event is not None:
                    self.codec_stream.wait_event(sampled_event)
                audio, offsets, counts, skip, order = self.codec_graph(owned_codes, owned_slots, owned_active, final)
                crop = torch.minimum(skip, counts)
                self.rvq.skip_samples.index_copy_(0, owned_slots[order], skip - crop)
                snapshot = Zonos3AudioSnapshot(
                    audio.clone(),
                    offsets.clone(),
                    counts.clone(),
                    crop,
                    [self._audio_leases.get(req_id) for req_id in input_batch.req_ids],
                    order.clone(),
                    torch.cuda.Event() if self.codec_stream is not None else None,
                    self.audio_copy_stream,
                )
                if self.codec_stream is not None:
                    snapshot.event.record()
                    for tensor in (owned_slots, owned_codes, owned_active, final):
                        tensor.record_stream(self.codec_stream)
            self._audio_output = snapshot
        return hidden, {"zonos3_audio": True}

    def prepare_streaming_audio_output(self, input_batch, req_states, outputs):
        output, self._audio_output = self._audio_output, None
        return output

    def sample_determined_tokens(self, batch: InputBatch, sampler: Sampler) -> SamplerOutput:
        """Deliver the CONTINUE/EOS decided in postprocess without text sampling."""
        output = self._sample_output
        if output is None or output.sampled_token_ids.shape != (batch.num_reqs, 1):
            raise RuntimeError("Zonos3 determined tokens require the current step's postprocess output")
        return output

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        startup_dummy = self._startup and all(req_id not in self._request_slots for req_id in input_batch.req_ids)
        warmup_only = startup_dummy or (
            bool(input_batch.req_ids) and all(is_warmup_request(req_id) for req_id in input_batch.req_ids)
        )
        if (
            not for_capture
            and not warmup_only
            and not input_batch.has_prefill
            and not self.vllm_config.model_config.enforce_eager
            and cudagraph_mode is not CUDAGraphMode.FULL
        ):
            raise RuntimeError("Zonos3 steady-state AR decode requires one native FULL Backbone graph replay")
        return super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
            ubatch_idx=ubatch_idx,
            model_specific_attn_metadata=model_specific_attn_metadata,
        )
