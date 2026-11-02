# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""OmniModelState — generic ModelState base for all Omni model stages.

Extends ``DefaultModelState`` with:

* Cross-stage intermediate buffer (``OmniIntermediateBuffer``)
* ``model_intermediate_buffer`` / ``runtime_additional_information`` injection
  into ``model_inputs`` via ``prepare_inputs()``
* ``OmniOutput`` → ``(text_hidden, multimodal_outputs)`` post-processing
"""

from __future__ import annotations

import inspect
import threading
import types
from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

from vllm_omni.model_executor.models.output_templates import OmniOutput, OwnedBatchTensor
from vllm_omni.utils.device_copy import index_to_device
from vllm_omni.worker_v2.model_states.intermediate_buffer import (
    OmniIntermediateBuffer,
)
from vllm_omni.worker_v2.model_states.mtp_model_state import FrameBatch, MTPState
from vllm_omni.worker_v2.streaming_audio import StreamingAudioSnapshot
from vllm_omni.worker_v2.warmup import is_warmup_request

logger = init_logger(__name__)


_rope_patch_lock = threading.Lock()


def _default_mrope_positions(
    self_model: Any,
    input_tokens: list[int],
    mm_features: list,
) -> tuple[np.ndarray, int]:
    """Return 3D sequential positions with zero delta.

    For non-vision Omni models (e.g. TTS Talker), all 3 M-RoPE
    dimensions use the same sequential positions. Delta=0 keeps decode
    positions sequential, identical to the 1D case but broadcast to 3 dims.
    """
    n = len(input_tokens)
    row = _SEQUENTIAL_POSITIONS[:n] if n <= _SEQUENTIAL_POSITIONS.shape[0] else np.arange(n, dtype=np.int32)
    # A read-only view of one row; RopeState copies it into its staged writes.
    return np.broadcast_to(row, (3, n)), 0


_SEQUENTIAL_POSITIONS = np.arange(1 << 16, dtype=np.int32)
_SEQUENTIAL_POSITIONS.flags.writeable = False


def _make_safe_get_rope(orig_get_rope):
    from vllm.v1.worker.gpu.mm.rope import RopeState

    def _safe_get_rope(model_config: Any, mdl: Any, **kwargs: Any) -> Any:
        try:
            result = orig_get_rope(model_config, mdl, **kwargs)
        except AssertionError:
            result = None

        needs_mrope = bool(getattr(model_config, "uses_mrope", False))
        if result is not None and (not needs_mrope or getattr(result, "num_dims", 0) >= 3):
            return result
        if not needs_mrope:
            return None
        if not hasattr(mdl, "get_mrope_input_positions"):
            mdl.get_mrope_input_positions = types.MethodType(_default_mrope_positions, mdl)
        return RopeState(num_dims=3, **kwargs)

    return _safe_get_rope


class OmniModelState(DefaultModelState):
    """Generic Omni ``ModelState`` — works for **all** Omni model stages.

    Model-owned preprocess and postprocess hooks supply stage-specific behavior.
    """

    # Eager-frame defaults for states built without __init__ (tests, adapters).
    _decode_preprocess_is_identity = False
    _has_mtp = False
    preprocess_keeps_mm_inputs = False
    _supports_prefill_token_mask = False

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        # DefaultModelState.__init__ calls get_rope_state() which asserts
        # isinstance(model, SupportsMRoPE).  Two categories of Omni models:
        #
        # 1. Models that implement SupportsMRoPE (e.g. Qwen3-Omni Thinker):
        #    get_rope_state() succeeds normally, _safe_get_rope is a no-op.
        #    These models get correct 3D M-RoPE positions from the runner.
        #
        # 2. Models that do NOT implement SupportsMRoPE (e.g. Qwen3-TTS
        #    Talker, Code2Wav, FishSpeech): get_rope_state() would assert.
        #    These models compute their own position encoding internally
        #    (via model.forward kwargs or fixed 1D positions from
        #    InputBatch.positions), so rope_state = None is correct —
        #    DefaultModelState.prepare_inputs returns {} when rope_state
        #    is None, and upstream execute_model falls back to
        #    InputBatch.positions (1D sequential).
        # Patch get_rope_state to handle Omni models that declare
        # M-RoPE in config (mrope_section) but do not implement the
        # SupportsMRoPE interface.  For these models we create a
        # RopeState with 3D sequential positions (matching V1 MR).
        #
        # The patch is applied via a class-level lock to prevent
        # concurrent OmniModelState instances (e.g. different stages
        # in a thread pool) from overwriting each other's patch.
        from vllm.v1.worker.gpu.model_states import default as _default_mod

        with _rope_patch_lock:
            orig_get_rope = _default_mod.get_rope_state
            _default_mod.get_rope_state = _make_safe_get_rope(orig_get_rope)
            try:
                super().__init__(vllm_config, model, encoder_cache, device)
            finally:
                _default_mod.get_rope_state = orig_get_rope
        max_num_reqs = self.scheduler_config.max_num_seqs
        self.intermediate_buffer = OmniIntermediateBuffer(max_num_reqs)
        self.has_preprocess: bool = getattr(model, "has_preprocess", False)
        self.has_postprocess: bool = getattr(model, "has_postprocess", False)
        self.have_multimodal_outputs: bool = getattr(model, "have_multimodal_outputs", False)
        self._supports_prefill_token_mask = bool(getattr(model, "supports_prefill_token_mask", False))
        self._decode_preprocess = self._resolve_decode_preprocess(model)
        self._decode_preprocess_is_identity = bool(getattr(model, "mrv2_decode_preprocess_is_identity", False))
        # Talker's codec_embedding dim may differ from hf_text_config.hidden_size; probe real dim.
        self._embed_dim = self._get_embed_dim(model, device) if self.has_preprocess else 0

        # A preprocess model may also take native multimodal inputs (MiniCPM-o's
        # duplex Thinker serves media chat). The encoder runner rewrites every
        # active row of its buffer each step, so that buffer is the static one.
        encoder_runner = getattr(self, "encoder_runner", None)
        encoder_embeds = getattr(encoder_runner, "inputs_embeds", None)
        self.preprocess_keeps_mm_inputs = bool(
            self.has_preprocess
            and self.supports_mm_inputs
            and getattr(model, "preprocess_keeps_mm_inputs", False)
            and isinstance(encoder_embeds, torch.Tensor)
            and tuple(encoder_embeds.shape) == (self.max_num_tokens, self._embed_dim)
        )

        # Static inputs_embeds buffer for FULL CUDA graph — preprocess fills it in-place each step.
        self._static_inputs_embeds: torch.Tensor | None = None
        if self.preprocess_keeps_mm_inputs:
            self._static_inputs_embeds = encoder_embeds
        elif self._embed_dim > 0:
            self._static_inputs_embeds = torch.zeros(
                (self.max_num_tokens, self._embed_dim),
                dtype=self.dtype,
                device=device,
            )

        self._mtp_state = MTPState(self)
        # Capability flag, not an eager/deferred timing choice. Built-in MTP
        # completes frames in post_sample; preprocess consumes cached inputs.
        self._has_mtp = self._mtp_state._mtp_input_ids is not None and bool(getattr(model, "has_talker_mtp", False))
        if self._has_mtp:
            # The model then leaves frame codes/validity and the decode MTP
            # inputs to post_sample instead of building them per row.
            model.eager_frames_active = True
            logger.info("Eager Talker-MTP frames enabled")

    def set_first_audio_sink(self, sink: Any) -> None:
        self._mtp_state.set_first_audio_sink(sink)

    def close_first_audio_sender(self) -> None:
        sender = self._mtp_state.first_audio_sender
        if sender is not None:
            sender.close()

    def post_sample(
        self,
        input_batch: InputBatch,
        text_hidden: torch.Tensor,
        sampled_token_ids: torch.Tensor,
        multimodal_outputs: dict[str, Any] | None,
        mtp_batch_descriptor_dispatcher: Callable[[int], Any] | None = None,
    ) -> None:
        if self._has_mtp and multimodal_outputs:
            self._mtp_state.post_sample(
                input_batch,
                text_hidden,
                sampled_token_ids,
                multimodal_outputs,
                mtp_batch_descriptor_dispatcher,
            )

    def prepare_streaming_audio_output(
        self, input_batch: InputBatch, req_states: RequestState, outputs: dict[str, Any]
    ) -> StreamingAudioSnapshot | None:
        return self._mtp_state.prepare_audio_output(input_batch, req_states, outputs)

    def get_request_errors(self) -> tuple[torch.Tensor, str] | None:
        """Optional owned per-row failure mask and reason for the current step.

        The runner copies the mask with its other outputs and only reads it
        after D2H completes. A failure terminates that request, not the batch.
        """
        return None

    def on_requests_finished(self, req_ids: set[str]) -> None:
        self._mtp_state.finish_audio(req_ids)

    def on_request_preempted(self, req_id: str, req_index: int) -> None:
        self._mtp_state.suspend_audio(req_id, req_index)

    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        """Let the model replace or wrap the MRv2 sampler (``mrv2_custom_sampler``)."""
        hook = getattr(self.model, "mrv2_custom_sampler", None)
        if callable(hook):
            return hook(sampler)
        return super().custom_sampler(sampler)

    @staticmethod
    def _resolve_decode_preprocess(model: nn.Module) -> Callable | None:
        """Adapt existing V1 hooks once, outside the per-step hot path.

        Keep the V2 hook precedence for models with an optimized implementation.
        Both hooks return the same five values; only the embedding argument is
        optional in the original contract. Errors inside a hook propagate.
        """
        hook = getattr(model, "preprocess_decode_batch_mrv2", None)
        if not callable(hook):
            hook = getattr(model, "preprocess_decode_batch", None)
        if not callable(hook):
            return None
        parameters = inspect.signature(hook).parameters
        if "input_embeds" in parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            return hook

        def decode_preprocess(*, input_ids, input_embeds, req_infos):
            return hook(input_ids=input_ids, req_infos=req_infos)

        return decode_preprocess

    def _is_mtp_graph_runner(self) -> bool:
        return self._mtp_state._is_mtp_graph_runner()

    def _get_mtp_capture_sizes(self) -> list[int]:
        return self._mtp_state._get_mtp_capture_sizes()

    def capture_mtp_graphs(self, dispatch_batch_descriptor, *, restore_capturing=False) -> None:
        self._mtp_state.capture_mtp_graphs(dispatch_batch_descriptor, restore_capturing=restore_capturing)

    @staticmethod
    def _get_embed_dim(model: nn.Module, device: torch.device) -> int:
        """Return the embedding dim that ``embed_input_ids`` produces (may differ from hf_text_config)."""
        if hasattr(model, "embed_input_ids"):
            dummy = torch.zeros(1, dtype=torch.long, device=device)
            with torch.no_grad():
                out = model.embed_input_ids(dummy)
            return out.shape[-1]
        return 0

    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        self.intermediate_buffer.add_request(req_index, new_req_data)
        self._mtp_state.resume_audio(new_req_data.req_id, req_index)
        self._initialize_upstream_warmup_buffer(req_index, new_req_data.req_id)

    def _initialize_upstream_warmup_buffer(self, req_index: int, req_id: str) -> None:
        """Satisfy model-declared output contracts for vLLM warmup requests."""
        if not is_warmup_request(req_id):
            return

        validity_key = getattr(self.model, "mtp_validity_key", None)
        if validity_key is None:
            return

        validity = torch.zeros((), dtype=torch.bool)
        buffer = self.intermediate_buffer.buffers[req_index]
        if isinstance(validity_key, tuple) and len(validity_key) == 2:
            buffer.setdefault(validity_key[0], {})[validity_key[1]] = validity
        elif isinstance(validity_key, str):
            buffer[validity_key] = validity
        else:
            raise TypeError(
                f"mtp_validity_key must be a string or 2-tuple, got {type(validity_key).__name__}: {validity_key!r}"
            )

    def _resolve_req_index(self, req_index_or_id: int | str) -> int | None:
        if isinstance(req_index_or_id, int):
            return req_index_or_id

        return self.intermediate_buffer.req_id_to_index.get(req_index_or_id)

    def remove_request(self, req_index_or_id: int | str) -> None:
        req_index = self._resolve_req_index(req_index_or_id)
        if req_index is None:
            return
        req_id = self.intermediate_buffer.buffers[req_index].get("req_id")
        if req_id is not None:
            self._mtp_state.release_request(req_id, req_index)
        self.intermediate_buffer.remove_request(req_index)

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

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
        if (
            for_capture
            and input_batch.max_query_len is None
            and not self.vllm_config.compilation_config.cudagraph_mode.separate_routine()
        ):
            # vLLM 0.30 distributes dummy tokens evenly across requests. For
            # an unconstrained FULL graph that split is not a query-length
            # bound: replay may put the entire token bucket in one request.
            # Attention launch parameters are fixed at capture, so use the
            # bucket's worst-case query length, not the dummy per-row length.
            # Separate decode graphs must retain their uniform query bound:
            # FA3's captured GQA layout must match the runtime AOT schedule.
            # Keep explicit bounds (e.g. varlen decode) and runtime metadata.
            input_batch = replace(input_batch, max_query_len=input_batch.num_tokens)
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

    @staticmethod
    def _build_prefill_token_mask(input_batch: InputBatch) -> bool | torch.Tensor:
        """Use CPU phase metadata, including one-token and resumed prefills.

        Query boundaries describe the actual flattened tokens, unlike scheduled
        counts which can be upper bounds under adaptive verification. Tokens
        within each row must be contiguous; sparse scheduling needs an explicit
        token-position-aligned mask instead.
        """
        if not input_batch.has_prefill:
            return False
        # These CPU arrays are already reordered into batch order. prefill_len
        # also includes safe codec output tokens replayed after preemption.
        starts = input_batch.query_start_loc_np[: input_batch.num_reqs + 1]
        counts = np.diff(starts)
        num_prefill = np.minimum(
            np.maximum(
                input_batch.prefill_len_np[: input_batch.num_reqs]
                - input_batch.num_computed_prefill_tokens_np[: input_batch.num_reqs],
                0,
            ),
            counts,
        )
        if not np.any(num_prefill):
            return False
        if np.array_equal(num_prefill, counts):
            return True
        mask = np.zeros(input_batch.num_tokens, dtype=np.bool_)
        for start, count in zip(starts[:-1], num_prefill):
            mask[start : start + count] = True
        return torch.from_numpy(mask).to(device=input_batch.input_ids.device)

    def prepare_inputs_embeds(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor:
        # Models that rearrange multimodal prompts per request (CosyVoice3)
        # need the request boundaries the V1 runner passes as query_start_loc.
        if (
            not self.supports_mm_inputs
            or self.mm_pruner is not None
            or not getattr(self.model, "supports_embed_input_ids_query_start_loc", False)
        ):
            return super().prepare_inputs_embeds(scheduled_encoder_inputs, input_batch, req_states)
        self.execute_mm_encoder(scheduled_encoder_inputs)
        mm_embeds, is_mm_embed = self.gather_mm_embeddings(input_batch)
        kwargs: dict[str, Any] = {"multimodal_embeddings": mm_embeds, "is_multimodal": is_mm_embed}
        if self._supports_prefill_token_mask:
            kwargs["prefill_token_mask"] = self._build_prefill_token_mask(input_batch)
        if mm_embeds:
            kwargs["query_start_loc"] = input_batch.query_start_loc_np[: input_batch.num_reqs + 1].tolist()
        embeds = self.model.embed_input_ids(input_batch.input_ids[: input_batch.num_tokens], **kwargs)
        inputs_embeds = self.encoder_runner.inputs_embeds
        inputs_embeds[: embeds.shape[0]] = embeds
        if self.prompt_embeds_state is not None:
            self.prompt_embeds_state.apply(input_batch, req_states.num_computed_tokens.gpu, inputs_embeds)
        return inputs_embeds[: input_batch.num_tokens_after_padding]

    def prepare_inputs(self, input_batch: InputBatch, req_states: RequestState) -> dict[str, Any]:
        # Forward-only stages have no preprocess hook. Honor their declared
        # GPU inputs before gathering the model's runtime view, so payload
        # tensors do not fall back to per-request blocking device copies.
        if not self.has_preprocess and getattr(self.model, "batched_gpu_staging_keys", None):
            self._stage_batched_preprocess_inputs(
                [int(input_batch.idx_mapping_np[i]) for i in range(input_batch.num_reqs)],
                input_batch.input_ids.device,
            )
        base = super().prepare_inputs(input_batch, req_states)
        buffer_list = self.intermediate_buffer.gather(input_batch)
        base["model_intermediate_buffer"] = buffer_list
        base["runtime_additional_information"] = buffer_list
        base["seq_token_counts"] = np.asarray(input_batch.num_scheduled_tokens[: input_batch.num_reqs]).tolist()
        # Return static inputs_embeds so FULL graph replay uses the same
        # tensor address that was captured.  Preprocess fills it in-place.
        if self._static_inputs_embeds is not None:
            base["inputs_embeds"] = self._static_inputs_embeds[: input_batch.num_tokens_after_padding]
        return base

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        base = super().prepare_dummy_inputs(num_reqs, num_tokens)
        dummy_buffer: list[dict[str, Any]] = [{} for _ in range(num_reqs)]
        base["model_intermediate_buffer"] = dummy_buffer
        base["runtime_additional_information"] = dummy_buffer
        if num_reqs > 0:
            per_req = num_tokens // num_reqs
            remainder = num_tokens % num_reqs
            counts = [per_req] * num_reqs
            counts[-1] += remainder
            base["seq_token_counts"] = counts
        else:
            base["seq_token_counts"] = []
        # Return static inputs_embeds for FULL graph capture so the graph
        # captures this tensor's address.
        if self._static_inputs_embeds is not None:
            base["inputs_embeds"] = self._static_inputs_embeds[:num_tokens]
        return base

    # ------------------------------------------------------------------
    # Pre-forward: per-request preprocess + batched MTP
    # ------------------------------------------------------------------

    @staticmethod
    def _get_req_state_value(field: Any, req_idx: int) -> int | None:
        values = getattr(field, "np", field)
        try:
            return int(values[req_idx])
        except Exception:
            return None

    @staticmethod
    def _get_input_batch_num_computed(input_batch: InputBatch, req_idx: int, batch_idx: int) -> int | None:
        for attr in ("num_computed_tokens_cpu", "num_computed_tokens_np"):
            values = getattr(input_batch, attr, None)
            if values is None:
                continue
            try:
                if len(values) == getattr(input_batch, "num_reqs", len(values)):
                    return int(values[batch_idx])
                return int(values[req_idx])
            except Exception:
                continue
        return None

    @staticmethod
    def _preprocess_result_needs_writeback(original: torch.Tensor, updated: torch.Tensor) -> bool:
        return updated is not original

    @staticmethod
    def _batch_move_tensor_rows(
        tensors: list[torch.Tensor],
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Move compatible variable-length rows with one device transfer."""
        lengths = [int(tensor.shape[0]) for tensor in tensors]
        staged = torch.cat(tensors, dim=0).to(device=device, non_blocking=True)
        return list(staged.split(lengths, dim=0))

    def _stage_batched_preprocess_inputs(
        self,
        req_indices: list[int],
        device: torch.device,
    ) -> None:
        staging_keys: set[tuple[str, str]] = getattr(self.model, "batched_gpu_staging_keys", set())
        for key in staging_keys:
            if not isinstance(key, tuple) or len(key) != 2:
                raise TypeError(f"batched_gpu_staging_keys entries must be 2-tuples, got {key!r}")
            type_key, qualifier = key
            groups: dict[
                tuple[torch.device, torch.dtype, tuple[int, ...]], list[tuple[dict[str, Any], torch.Tensor]]
            ] = {}
            for req_idx in req_indices:
                nested = self.intermediate_buffer.buffers[req_idx].get(type_key)
                if not isinstance(nested, dict):
                    continue
                tensor = nested.get(qualifier)
                if not isinstance(tensor, torch.Tensor) or tensor.device == device:
                    continue
                group_key = tensor.device, tensor.dtype, tuple(tensor.shape[1:])
                groups.setdefault(group_key, []).append((nested, tensor))

            for group in groups.values():
                staged_rows = self._batch_move_tensor_rows([tensor for _nested, tensor in group], device)
                for (nested, _tensor), staged in zip(group, staged_rows, strict=True):
                    nested[qualifier] = staged

    def _write_batched_gpu_preprocess_updates(
        self,
        req_indices: list[int],
        updates_by_req: list[dict[str, Any]],
        gpu_keys: set[Any],
    ) -> None:
        """Snapshot compatible resident preprocess values once per cohort."""
        if not req_indices or len(req_indices) != len(updates_by_req):
            return

        for key in gpu_keys:
            nested_key = isinstance(key, tuple) and len(key) == 2
            values: list[torch.Tensor] = []
            for updates in updates_by_req:
                if nested_key:
                    type_key, qualifier = key
                    nested = updates.get(type_key)
                    value = nested.get(qualifier) if isinstance(nested, dict) else None
                else:
                    value = updates.get(key)
                if not isinstance(value, torch.Tensor):
                    break
                values.append(value)
            if len(values) != len(updates_by_req):
                continue

            first = values[0]
            if any(
                value.shape != first.shape or value.dtype != first.dtype or value.device != first.device
                for value in values[1:]
            ):
                continue

            stacked = torch.stack(values, dim=0)
            self.intermediate_buffer.update_gpu_tensor_rows(
                req_indices,
                key,
                stacked,
                keepdim=False,
            )

            for updates in updates_by_req:
                if nested_key:
                    type_key, qualifier = key
                    nested = updates[type_key]
                    del nested[qualifier]
                    if not nested:
                        del updates[type_key]
                else:
                    del updates[key]

    @staticmethod
    def _split_settled_rows(
        input_batch: InputBatch,
        req_indices: list[int],
        settled: np.ndarray,
    ) -> tuple[list[tuple[int, int, int, str]], list[tuple[int, int]]] | None:
        """Vectorized settled-row test of ``run_preprocess``.

        In steady decode almost every row is settled: it schedules one token
        and ``settled`` maps its slot to its request id. ``settled`` entries
        are dropped together with the slot buffer, so comparing against the
        row's request id equals the per-row ``buf["req_id"]`` test. Returns the
        settled rows and the remaining ``(row, slot)`` pairs, both in row
        order, or ``None`` when no row is settled.
        """
        n = input_batch.num_reqs
        slots = np.asarray(input_batch.idx_mapping_np[:n])
        req_ids = np.empty(n, dtype=object)
        req_ids[:] = input_batch.req_ids[:n]
        is_settled = (np.asarray(input_batch.num_scheduled_tokens[:n]) == 1) & (settled[slots] == req_ids)
        if not is_settled.any():
            return None
        rows = np.flatnonzero(is_settled)
        settled_rows = list(
            zip(
                rows.tolist(),
                slots[rows].tolist(),
                np.asarray(input_batch.query_start_loc_np[:n])[rows].tolist(),
                req_ids[rows].tolist(),
            )
        )
        remaining = [(i, req_indices[i]) for i in np.flatnonzero(~is_settled).tolist()]
        return settled_rows, remaining

    def run_preprocess(
        self,
        input_batch: InputBatch,
        model_inputs: dict[str, Any],
        req_states: RequestState | None = None,
        mtp_batch_descriptor_dispatcher: Callable[[int], Any] | None = None,
    ) -> None:
        """Prepare backbone inputs and record rows for current-step MTP.

        Modifies ``model_inputs["input_ids"]`` and ``model_inputs["inputs_embeds"]``
        in-place. Decode rows consume cached frame embeddings; MTP itself runs
        after token sampling through post_sample.
        """
        if not self.has_preprocess:
            return

        input_ids = model_inputs.get("input_ids")
        if input_ids is None:
            input_ids = input_batch.input_ids

        embeds = model_inputs.get("inputs_embeds")
        if embeds is None:
            embeds = self.model.embed_input_ids(input_batch.input_ids[: input_batch.num_tokens])
            model_inputs["inputs_embeds"] = embeds
        elif (
            not self.preprocess_keeps_mm_inputs
            and self._static_inputs_embeds is not None
            and embeds.data_ptr() == self._static_inputs_embeds.data_ptr()
        ):
            # FULL graph replay requires a stable inputs_embeds address. Refresh
            # the active rows from the current token ids before model-specific
            # preprocessing; otherwise decode reuses embeddings left by the
            # previous step at the same static address.
            num_active_tokens = int(getattr(input_batch, "num_tokens", input_ids.shape[0]))
            token_embeds = self.model.embed_input_ids(input_ids[:num_active_tokens])
            embeds[:num_active_tokens].copy_(token_embeds)

        gpu_keys: set[str] = getattr(self.model, "gpu_resident_buffer_keys", set())
        decode_mtp_rows: list[tuple[int, int]] = []
        text_step_batches: list[torch.Tensor] = []

        req_indices = np.asarray(input_batch.idx_mapping_np[: input_batch.num_reqs]).tolist()
        self._stage_batched_preprocess_inputs(req_indices, embeds.device)

        preprocess_entries: list[tuple[int, int, int, int, dict[str, Any], bool]] = []
        settled = self._mtp_state.settled if self._has_mtp else None
        settled_rows: list[tuple[int, int, int, str]] = []
        replay_rows: list[tuple[int, int, str, bool]] = []
        # Declared identity decode has no per-row updates or hook work.
        skip_decode_rows = self._decode_preprocess_is_identity and not self._has_mtp
        is_prefilling_np = getattr(input_batch, "is_prefilling_np", None) if skip_decode_rows else None
        rows: Iterable[tuple[int, int]] = enumerate(req_indices)
        if settled is not None and is_prefilling_np is None and not self._mtp_state.has_pending_replay():
            split = self._split_settled_rows(input_batch, req_indices, settled)
            if split is not None:
                settled_rows, rows = split
        for i, req_idx in rows:
            if is_prefilling_np is not None and not is_prefilling_np[i]:
                continue
            buf = self.intermediate_buffer.buffers[req_idx]
            if not buf or "req_id" not in buf:
                continue
            start = int(input_batch.query_start_loc_np[i])
            n_tok = int(input_batch.num_scheduled_tokens[i])
            if (
                self._has_mtp
                # Only the Qwen3-TTS Talker has an in-Talker streaming codec.
                and getattr(self.model, "stream_decoder", None) is not None
                and self._mtp_state.replay_inputs(
                    buf["req_id"],
                    req_idx,
                    int(input_batch.num_computed_tokens_np[i]),
                    input_ids[start : start + n_tok],
                    embeds[start : start + n_tok],
                )
            ):
                if input_batch.num_computed_prefill_tokens_np[i] + n_tok >= input_batch.prefill_len_np[i]:
                    replay_rows.append((i, req_idx, buf["req_id"], True))
                continue
            if settled is not None and n_tok == 1 and settled[req_idx] == buf["req_id"]:
                settled_rows.append((i, req_idx, start, buf["req_id"]))
                continue
            if is_warmup_request(buf["req_id"]):
                continue

            info = {key: value for key, value in buf.items() if isinstance(key, str)}
            # Match the V1 runner: the sampler's seed does not seed random
            # draws performed inside model.forward (e.g. VoxCPM2 CFM noise).
            sampling_params = info.get("sampling_params")
            info["_omni_seed"] = getattr(sampling_params, "seed", None)
            info["_omni_max_tokens"] = getattr(sampling_params, "max_tokens", None)
            prompt_len = None
            num_computed_tokens = None
            if req_states is not None:
                prompt_len = self._get_req_state_value(getattr(req_states, "prompt_len", None), req_idx)
                num_computed_tokens = self._get_req_state_value(
                    getattr(req_states, "num_computed_tokens", None),
                    req_idx,
                )
            if num_computed_tokens is None:
                num_computed_tokens = self._get_input_batch_num_computed(input_batch, req_idx, i)
            if prompt_len is not None:
                info["_omni_prompt_len"] = int(prompt_len)
            if num_computed_tokens is not None:
                info["_omni_num_computed_tokens"] = int(num_computed_tokens)
            if prompt_len is not None and num_computed_tokens is not None:
                info["_omni_is_prefill"] = int(num_computed_tokens) < int(prompt_len)

            is_prefill = (
                int(num_computed_tokens) < int(prompt_len)
                if prompt_len is not None and num_computed_tokens is not None
                else n_tok > 1
            )
            preprocess_entries.append((i, req_idx, start, n_tok, info, is_prefill))

        if self._has_mtp:
            # Rows whose sample is kept this step: decode, or the final prefill chunk.
            self._mtp_state.recorded = (
                input_batch,
                FrameBatch(
                    [
                        (i, req_idx, str(info["req_id"]), is_prefill)
                        for i, req_idx, _start, n_tok, info, is_prefill in preprocess_entries
                        if not is_prefill
                        or int(info.get("_omni_num_computed_tokens", 0)) + n_tok >= int(info.get("_omni_prompt_len", 0))
                    ]
                    + [(i, req_idx, str(req_id), False) for i, req_idx, _start, req_id in settled_rows]
                    + replay_rows
                ),
                input_ids,
            )

        prefill_updates_owned = bool(getattr(self.model, "preprocess_prefill_updates_owned", False))
        # Rows the model preprocesses for the whole batch at once skip the per-request preprocess.
        batched_indices: set[int] = set()
        preprocess_prefill_rows = getattr(self.model, "preprocess_prefill_rows_mrv2", None)
        prefill_entries = [entry for entry in preprocess_entries if entry[5]]
        if callable(preprocess_prefill_rows) and prefill_entries:
            prefill_results = preprocess_prefill_rows(
                entries=[(start, n_tok, info) for _i, _req_idx, start, n_tok, info, _p in prefill_entries],
                input_ids=input_ids,
                input_embeds=embeds,
            )
            if len(prefill_results) != len(prefill_entries):
                raise RuntimeError(
                    "Batched prefill preprocess returned the wrong update count: "
                    f"expected={len(prefill_entries)} actual={len(prefill_results)}"
                )
            for entry, updates in zip(prefill_entries, prefill_results, strict=True):
                if updates is not None:
                    self.intermediate_buffer.update(entry[1], updates, gpu_keys, owned=prefill_updates_owned)
                    batched_indices.add(entry[0])

        preprocess_batch_mrv2 = getattr(self.model, "preprocess_batch_mrv2", None)
        if callable(preprocess_batch_mrv2):
            prefill_infos = [
                self.intermediate_buffer.buffers[req_idx]
                for i, req_idx, _start, _n_tok, _info, is_prefill in preprocess_entries
                if is_prefill and i not in batched_indices
            ]
            if prefill_infos:
                preprocess_batch_mrv2(req_infos=prefill_infos, device=embeds.device)
                for i, req_idx, _start, _n_tok, info, is_prefill in preprocess_entries:
                    if not is_prefill or i in batched_indices:
                        continue
                    runtime_info = {key: value for key, value in info.items() if key.startswith("_omni_")}
                    info.update(self.intermediate_buffer.buffers[req_idx])
                    info.update(runtime_info)

        if not hasattr(self, "_decode_preprocess"):
            self._decode_preprocess = self._resolve_decode_preprocess(self.model)
        batch_decode_preprocess = self._decode_preprocess
        decode_entries = [entry for entry in preprocess_entries if entry[3] == 1 and not entry[5]]
        if callable(batch_decode_preprocess) and decode_entries:
            batch_size = len(decode_entries)
            starts = [entry[2] for entry in decode_entries]
            is_contiguous_cohort = all(entry[0] == row and entry[2] == row for row, entry in enumerate(decode_entries))
            if is_contiguous_cohort:
                batch_ids = input_ids[:batch_size]
                batch_embeds = embeds[:batch_size]
                batch_offsets = None
            else:
                # A pageable host tensor would block on the stream here.
                batch_offsets = index_to_device(starts, input_ids.device)
                batch_ids = input_ids.index_select(0, batch_offsets)
                batch_embeds = embeds.index_select(0, batch_offsets)

            new_ids, new_embeds, batch_hidden, batch_text_step, updates_by_req = batch_decode_preprocess(
                input_ids=batch_ids,
                input_embeds=batch_embeds,
                req_infos=[entry[4] for entry in decode_entries],
            )
            if len(updates_by_req) != batch_size:
                raise RuntimeError(
                    "Batched Omni preprocess returned the wrong update count: "
                    f"expected={batch_size} actual={len(updates_by_req)}"
                )
            if batch_hidden.shape[0] != batch_size or batch_text_step.shape[0] != batch_size:
                raise RuntimeError(
                    "Batched Omni preprocess changed the request axis: "
                    f"expected={batch_size} hidden={batch_hidden.shape[0]} text_step={batch_text_step.shape[0]}"
                )

            if self._preprocess_result_needs_writeback(batch_ids, new_ids):
                if batch_offsets is None:
                    input_ids[:batch_size].copy_(new_ids.reshape(-1)[:batch_size])
                else:
                    input_ids.index_copy_(0, batch_offsets, new_ids.reshape(-1)[:batch_size])
            if self._preprocess_result_needs_writeback(batch_embeds, new_embeds):
                reshaped_embeds = new_embeds.reshape(batch_size, -1)
                if batch_offsets is None:
                    embeds[:batch_size].copy_(reshaped_embeds)
                else:
                    embeds.index_copy_(0, batch_offsets, reshaped_embeds)

            self._write_batched_gpu_preprocess_updates(
                [entry[1] for entry in decode_entries],
                updates_by_req,
                gpu_keys,
            )
            for entry, updates in zip(decode_entries, updates_by_req, strict=True):
                i, req_idx, start, _, _, _ = entry
                if self._has_mtp:
                    decode_mtp_rows.append((i, start))
                batched_indices.add(i)
                self.intermediate_buffer.update(req_idx, updates, gpu_keys)
            if self._has_mtp:
                text_step_batches.append(batch_text_step.reshape(batch_size, -1))

        for i, req_idx, start, n_tok, info, _is_prefill in preprocess_entries:
            if i in batched_indices:
                continue

            ids_slice = input_ids[start : start + n_tok]
            emb_slice = embeds[start : start + n_tok]
            new_ids, new_emb, updates = self.model.preprocess(ids_slice, emb_slice, **info)

            # Write back in-place
            seg = min(n_tok, new_ids.shape[0])
            if self._preprocess_result_needs_writeback(ids_slice, new_ids):
                input_ids[start : start + seg] = new_ids[:seg]
            if self._preprocess_result_needs_writeback(emb_slice, new_emb):
                embeds[start : start + seg] = new_emb[:seg]

            # Existing model hooks return (past hidden, text contribution).
            # Current-step MTP only needs the text contribution here.
            mtp_inputs = updates.pop("mtp_inputs", None)
            if mtp_inputs is not None and n_tok == 1:
                decode_mtp_rows.append((i, start))
                text_step_batches.append(mtp_inputs[1].reshape(1, -1))

            # A model declaring fresh prefill outputs skips the per-request snapshot copies.
            self.intermediate_buffer.update(req_idx, updates, gpu_keys, owned=_is_prefill and prefill_updates_owned)

        if self._has_mtp:
            settled_check = getattr(self.model, "eager_decode_settled", None)
            if callable(settled_check):
                for _i, req_idx, _start, n_tok, info, is_prefill in preprocess_entries:
                    buf = self.intermediate_buffer.buffers[req_idx]
                    if not is_prefill and n_tok == 1 and settled_check(buf):
                        self._mtp_state.mark_settled(req_idx, buf["req_id"])
            if settled_rows:
                self._mtp_state._apply_settled_frames(settled_rows, embeds)
        if decode_mtp_rows and self._has_mtp:
            text_steps = text_step_batches[0] if len(text_step_batches) == 1 else torch.cat(text_step_batches, dim=0)
            self._mtp_state.apply_frame_embeddings(decode_mtp_rows, embeds, input_batch, text_steps)
        elif decode_mtp_rows:
            raise RuntimeError(
                "Deferred MTP is not supported by OmniModelState; declare has_talker_mtp "
                "for current-step MTP or implement model-specific input preparation."
            )

        if self._has_mtp:
            self._mtp_state.record_inputs(input_batch, embeds)

    def publish_sampled_embeddings(
        self, input_batch: InputBatch, sampled_token_ids: torch.Tensor
    ) -> tuple[dict[str, Any], torch.cuda.Event] | None:
        """Embedding of the token each row sampled this step.

        A producer whose consumer is fed the embeddings of its generated tokens
        (``publishes_sampled_embeddings`` on the model) can then hand each token
        over with the step that sampled it instead of one step later, when that
        token's forward would capture the same embedding. Published beside the
        packed snapshot as ``embed.sampled``: one ``[1, H]`` row per row whose
        sample is kept, an empty tensor for rows still inside their prefill.
        """
        if not getattr(self.model, "publishes_sampled_embeddings", False):
            return None
        num_reqs = input_batch.num_reqs
        if not num_reqs:
            return None
        if sampled_token_ids.reshape(num_reqs, -1).shape[1] != 1:
            return None  # speculative steps sample several tokens per row
        computed = input_batch.num_computed_prefill_tokens_np[:num_reqs]
        scheduled = np.asarray(input_batch.num_scheduled_tokens[:num_reqs])
        finishing = computed + scheduled >= input_batch.prefill_len_np[:num_reqs]
        kept = ~input_batch.is_prefilling_np[:num_reqs] | finishing
        rows = np.flatnonzero(kept).tolist()
        device = sampled_token_ids.device
        # Preserve the empty per-row prefill marker even when every request
        # is prefilling; absence would look like a broken decode handoff.
        sampled: list[torch.Tensor] = [torch.empty(0)] * num_reqs
        if rows:
            first = sampled_token_ids.reshape(num_reqs, -1)[:, 0].index_select(0, index_to_device(rows, device))
            embeds = self.model.embed_input_ids(first.long())
            for k, row in enumerate(rows):
                sampled[row] = embeds[k : k + 1]
        done = torch.cuda.Event()
        done.record()
        extra: dict[str, Any] = {"embed": {"sampled": sampled}}
        return extra, done

    # ------------------------------------------------------------------
    # Post-forward: per-request postprocess
    # ------------------------------------------------------------------

    def run_postprocess(self, hidden_states: torch.Tensor, input_batch: InputBatch) -> None:
        """Per-request postprocess after model forward.

        Extracts per-request updates from hidden_states and writes them
        back to the intermediate buffer (e.g. ``last_talker_hidden``).
        """
        if not self.has_postprocess:
            return
        if self._has_mtp and not getattr(self.model, "eager_frames_need_postprocess", True):
            # The last hidden it stores only fed the deferred decode MTP.
            return
        gpu_keys: set[str] = getattr(self.model, "gpu_resident_buffer_keys", set())
        batch_postprocess = getattr(self.model, "postprocess_batch_mrv2", None)
        if callable(batch_postprocess) and input_batch.num_reqs:
            req_indices = [int(input_batch.idx_mapping_np[i]) for i in range(input_batch.num_reqs)]
            buffers = [self.intermediate_buffer.buffers[req_idx] for req_idx in req_indices]
            if all(buf and "req_id" in buf and not is_warmup_request(buf["req_id"]) for buf in buffers):
                query_start_loc = getattr(input_batch, "query_start_loc", None)
                if (
                    isinstance(query_start_loc, torch.Tensor)
                    and query_start_loc.device == hidden_states.device
                    and query_start_loc.shape[0] >= input_batch.num_reqs + 1
                ):
                    last_token_indices = query_start_loc[1 : input_batch.num_reqs + 1] - 1
                else:
                    last_token_indices = torch.as_tensor(
                        [
                            int(input_batch.query_start_loc_np[i]) + int(input_batch.num_scheduled_tokens[i]) - 1
                            for i in range(input_batch.num_reqs)
                        ],
                        device=hidden_states.device,
                        dtype=torch.long,
                    )
                output_key, values = batch_postprocess(
                    hidden_states=hidden_states,
                    last_token_indices=last_token_indices,
                )
                if output_key not in gpu_keys:
                    raise RuntimeError(f"Batched Omni postprocess output must be GPU-resident: key={output_key!r}")
                owned_rows = isinstance(values, OwnedBatchTensor)
                raw_values = values.tensor if owned_rows else values
                if (
                    not isinstance(raw_values, torch.Tensor)
                    or raw_values.ndim == 0
                    or raw_values.shape[0] != input_batch.num_reqs
                ):
                    actual = raw_values.shape[0] if isinstance(raw_values, torch.Tensor) and raw_values.ndim else 0
                    raise RuntimeError(
                        "Batched Omni postprocess changed the request axis: "
                        f"expected={input_batch.num_reqs} actual={actual}"
                    )
                if owned_rows:
                    # Owned batch: share row views, no second snapshot.
                    self.intermediate_buffer.update_owned_gpu_tensor_rows(
                        req_indices,
                        output_key,
                        raw_values,
                        keepdim=False,
                    )
                else:
                    # Borrowed output (graph buffer or shared scratch): snapshot once.
                    self.intermediate_buffer.update_gpu_tensor_rows(
                        req_indices,
                        output_key,
                        raw_values,
                        keepdim=False,
                    )
                return
        for i in range(input_batch.num_reqs):
            req_idx = int(input_batch.idx_mapping_np[i])
            buf = self.intermediate_buffer.buffers[req_idx]
            if not buf or "req_id" not in buf:
                continue
            if is_warmup_request(buf["req_id"]):
                continue
            start = int(input_batch.query_start_loc_np[i])
            n_tok = int(input_batch.num_scheduled_tokens[i])
            h_slice = hidden_states[start : start + n_tok]
            info = {key: value for key, value in buf.items() if isinstance(key, str) and key != "hidden_states"}
            updates = self.model.postprocess(h_slice, **info)
            if updates:
                self.intermediate_buffer.update(req_idx, updates, gpu_keys)

    # ------------------------------------------------------------------
    # Output post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _request_token_spans(input_batch: InputBatch) -> list[tuple[int, int]]:
        """``(start, end)`` token rows of each request, built with NumPy instead of a per-row loop."""
        n = input_batch.num_reqs
        starts = np.asarray(input_batch.query_start_loc_np[:n], dtype=np.int64)
        ends = starts + np.asarray(input_batch.num_scheduled_tokens[:n], dtype=np.int64)
        return list(zip(starts.tolist(), ends.tolist()))

    def postprocess_model_output(
        self,
        model_output: Any,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> tuple[torch.Tensor, dict]:
        """Convert raw model output to ``(text_hidden, multimodal_outputs)``.

        Handles ``OmniOutput`` unwrapping and ``make_omni_output``
        conversion.
        """
        make_output_mrv2 = getattr(self.model, "make_omni_output_mrv2", None)
        if not isinstance(model_output, OmniOutput) and callable(make_output_mrv2) and self.have_multimodal_outputs:
            # Device-side output: the hook reads this step's batch and request
            # state directly instead of per-request host views.
            model_output = make_output_mrv2(
                model_output,
                input_batch=input_batch,
                req_states=req_states,
                model_intermediate_buffer=self.intermediate_buffer.gather(input_batch),
            )
        if not isinstance(model_output, OmniOutput) and hasattr(self.model, "make_omni_output"):
            if isinstance(model_output, (list, tuple)) or self.have_multimodal_outputs:
                buffer_list = self.intermediate_buffer.gather(input_batch)
                make_output_kwargs = {
                    "model_intermediate_buffer": buffer_list,
                    "request_token_spans": self._request_token_spans(input_batch),
                }
                make_output_kwargs["runtime_additional_information"] = buffer_list
                model_output = self.model.make_omni_output(model_output, **make_output_kwargs)

        if isinstance(model_output, OmniOutput):
            text_hidden = model_output.text_hidden_states
            multimodal_outputs = cast(dict[str, Any], model_output.multimodal_outputs or {})
        elif isinstance(model_output, (list, tuple)):
            text_hidden = model_output[0]
            multimodal_outputs = {}
        else:
            text_hidden = model_output
            multimodal_outputs = {}

        return text_hidden, multimodal_outputs
