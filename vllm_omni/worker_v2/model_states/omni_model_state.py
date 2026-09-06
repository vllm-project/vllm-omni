"""OmniModelState — generic ModelState base for all Omni model stages.

Extends ``DefaultModelState`` with:

* Cross-stage intermediate buffer (``OmniIntermediateBuffer``)
* ``model_intermediate_buffer`` / ``runtime_additional_information`` injection
  into ``model_inputs`` via ``prepare_inputs()``
* ``OmniOutput`` → ``(text_hidden, multimodal_outputs)`` post-processing
* Plugin lifecycle dispatch (``OmniModelStatePlugin``)
"""

from __future__ import annotations

import threading
import types
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.worker.utils import AttentionGroup

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker.sampling_utils import get_tts_local_seed
from vllm_omni.worker_v2.model_states.intermediate_buffer import (
    OmniIntermediateBuffer,
)
from vllm_omni.worker_v2.model_states.plugin import OmniModelStatePlugin

logger = init_logger(__name__)


_rope_patch_lock = threading.Lock()


def _default_mrope_positions(
    self_model: Any,
    input_tokens: list[int],
    mm_features: list,
) -> tuple[torch.Tensor, int]:
    """Return 3D sequential positions with zero delta.

    For non-vision Omni models (e.g. TTS Talker), all 3 M-RoPE
    dimensions use the same sequential positions. Delta=0 keeps decode
    positions sequential, identical to the 1D case but broadcast to 3 dims.
    """
    n = len(input_tokens)
    pos = torch.arange(n, dtype=torch.long)
    return pos.unsqueeze(0).expand(3, -1), 0


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
        return RopeState(num_dims=3, has_delta=True, **kwargs)

    return _safe_get_rope


class OmniModelState(DefaultModelState):
    """Generic Omni ``ModelState`` — works for **all** Omni model stages.

    Model-specific behaviour is injected via ``OmniModelStatePlugin``
    instances or subclasses; this class itself is model-agnostic.
    """

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
        self.plugins: list[OmniModelStatePlugin] = []
        self._talker_mtp_generators: dict[str, torch.Generator] = {}
        # Talker's codec_embedding dim may differ from hf_text_config.hidden_size; probe real dim.
        self._embed_dim = self._get_embed_dim(model, device) if self.has_preprocess else 0

        # Static inputs_embeds buffer for FULL CUDA graph — preprocess fills it in-place each step.
        self._static_inputs_embeds: torch.Tensor | None = None
        if self._embed_dim > 0:
            self._static_inputs_embeds = torch.zeros(
                (self.max_num_tokens, self._embed_dim),
                dtype=self.dtype,
                device=device,
            )

        # Static MTP buffers so _run_batched_mtp uses .copy_() instead of torch.cat().
        self._mtp_input_ids: torch.Tensor | None = None
        self._mtp_input_embeds: torch.Tensor | None = None
        self._mtp_hidden: torch.Tensor | None = None
        self._mtp_text_step: torch.Tensor | None = None
        self._mtp_offsets: torch.Tensor | None = None
        self._talker_mtp_runner: Any | None = None
        if self._embed_dim > 0 and hasattr(model, "talker_mtp"):
            max_bs = max_num_reqs
            self._mtp_input_ids = torch.zeros(max_bs, dtype=torch.long, device=device)
            self._mtp_input_embeds = torch.zeros((max_bs, self._embed_dim), dtype=self.dtype, device=device)
            self._mtp_hidden = torch.zeros((max_bs, self._embed_dim), dtype=self.dtype, device=device)
            self._mtp_text_step = torch.zeros((max_bs, self._embed_dim), dtype=self.dtype, device=device)
            self._mtp_offsets = torch.zeros(max_bs, dtype=torch.long, device=device)
            self._talker_mtp_runner = self._init_talker_mtp_runner(model)

        if hasattr(model, "get_omni_plugins"):
            for plugin in model.get_omni_plugins():
                self.register_plugin(plugin)

    def _init_talker_mtp_runner(self, model: nn.Module) -> Any:
        talker_mtp = getattr(model, "talker_mtp", None)
        if talker_mtp is None:
            return None

        compilation_config = self.vllm_config.compilation_config
        cudagraph_mode = getattr(compilation_config, "cudagraph_mode", CUDAGraphMode.NONE)
        if bool(getattr(model, "talker_mtp_disable_graph", False)):
            logger.info("Skipping talker_mtp graph wrapper because the model marks it graph-unsafe.")
            return talker_mtp
        has_separate_talker = getattr(model, "talker", None) is not None
        graph_safe = bool(getattr(model, "talker_mtp_graph_safe", False))
        if cudagraph_mode is not None and cudagraph_mode.has_full_cudagraphs() and (has_separate_talker or graph_safe):
            graph_wrapper_cls = current_omni_platform.get_graph_wrapper_cls()
            return graph_wrapper_cls(talker_mtp, self.vllm_config, runtime_mode=CUDAGraphMode.FULL)
        return talker_mtp

    def _is_talker_mtp_graph_runner(self) -> bool:
        runner = getattr(self, "_talker_mtp_runner", None)
        if runner is None:
            return False
        graph_wrapper_cls = current_omni_platform.get_graph_wrapper_cls()
        return isinstance(runner, graph_wrapper_cls)

    def capture_talker_mtp_graphs(self, dispatch_batch_descriptor: Callable[[int], Any]) -> None:
        if not self._is_talker_mtp_graph_runner():
            return
        if (
            self._mtp_input_ids is None
            or self._mtp_input_embeds is None
            or self._mtp_hidden is None
            or self._mtp_text_step is None
        ):
            return
        if getattr(self.model, "talker_mtp_accepts_req_infos", False):
            logger.warning("Skipping talker_mtp graph capture because this model requires per-request req_infos.")
            return

        from vllm.compilation.monitor import set_cudagraph_capturing_enabled
        from vllm.distributed.parallel_state import graph_capture
        from vllm.forward_context import set_forward_context

        compilation_config = self.vllm_config.compilation_config
        capture_sizes = self._get_talker_mtp_capture_sizes()
        num_warmups = compilation_config.cudagraph_num_of_warmups
        capture_kwargs = self._get_talker_mtp_base_sampling_kwargs()
        logger.info("Capturing MRv2 talker_mtp graphs for sizes %s", capture_sizes)

        set_cudagraph_capturing_enabled(True)
        try:
            with torch.inference_mode(), graph_capture(device=self.device):
                for bsz in capture_sizes:
                    batch_descriptor = dispatch_batch_descriptor(int(bsz))
                    num_tokens = int(getattr(batch_descriptor, "num_tokens", bsz))
                    ids = self._mtp_input_ids[:num_tokens]
                    emb = self._mtp_input_embeds[:num_tokens]
                    hidden = self._mtp_hidden[:num_tokens]
                    text_step = self._mtp_text_step[:num_tokens]

                    for _ in range(num_warmups):
                        with set_forward_context(
                            None,
                            self.vllm_config,
                            cudagraph_runtime_mode=CUDAGraphMode.NONE,
                            batch_descriptor=batch_descriptor,
                        ):
                            self._call_talker_mtp_runner(ids, emb, hidden, text_step, **capture_kwargs)

                    with set_forward_context(
                        None,
                        self.vllm_config,
                        cudagraph_runtime_mode=CUDAGraphMode.FULL,
                        batch_descriptor=batch_descriptor,
                    ):
                        self._call_talker_mtp_runner(ids, emb, hidden, text_step, **capture_kwargs)
                    torch.accelerator.synchronize()

            logger.info("Captured MRv2 talker_mtp graphs for %d sizes", len(capture_sizes))
        except RuntimeError as e:
            raise RuntimeError(f"MRv2 talker_mtp graph capture failed: {e}") from e
        finally:
            set_cudagraph_capturing_enabled(False)

    def _get_talker_mtp_capture_sizes(self) -> list[int]:
        """Return graph buckets that a request-batched Talker can reach."""
        max_num_reqs = int(self.scheduler_config.max_num_seqs)
        return sorted(
            {
                int(size)
                for size in self.vllm_config.compilation_config.cudagraph_capture_sizes
                if 0 < int(size) <= max_num_reqs
            },
            reverse=True,
        )

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
    # Attention metadata
    # ------------------------------------------------------------------

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: Any,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        return super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture,
        )

    # ------------------------------------------------------------------
    # Plugin management
    # ------------------------------------------------------------------

    def register_plugin(self, plugin: OmniModelStatePlugin) -> None:
        self.plugins.append(plugin)

    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        self.intermediate_buffer.add_request(req_index, new_req_data)
        self._initialize_upstream_warmup_buffer(req_index, new_req_data.req_id)
        for plugin in self.plugins:
            plugin.on_add_request(req_index, new_req_data)

    def _initialize_upstream_warmup_buffer(self, req_index: int, req_id: str) -> None:
        """Satisfy model-declared output contracts for vLLM warmup requests."""
        if not str(req_id).startswith("_warmup_"):
            return

        validity_key = getattr(self.model, "talker_mtp_validity_key", None)
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
                "talker_mtp_validity_key must be a string or 2-tuple, "
                f"got {type(validity_key).__name__}: {validity_key!r}"
            )

    def _resolve_req_index(self, req_index_or_id: int | str) -> int | None:
        if isinstance(req_index_or_id, int):
            return req_index_or_id

        for idx, buffer in enumerate(self.intermediate_buffer.buffers):
            if buffer.get("req_id") == req_index_or_id:
                return idx
        return None

    def remove_request(self, req_index: int | str) -> None:
        req_index = self._resolve_req_index(req_index)
        if req_index is None:
            return
        req_id = self.intermediate_buffer.buffers[req_index].get("req_id")
        if req_id is not None:
            getattr(self, "_talker_mtp_generators", {}).pop(req_id, None)
        self.intermediate_buffer.remove_request(req_index)
        for plugin in self.plugins:
            plugin.on_remove_request(req_index)

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def prepare_inputs(self, input_batch: InputBatch, req_states: RequestState) -> dict[str, Any]:
        base = super().prepare_inputs(input_batch, req_states)
        buffer_list = self.intermediate_buffer.gather(input_batch)
        base["model_intermediate_buffer"] = buffer_list
        if not getattr(self.model, "requires_native_model_intermediate_buffer", False):
            base["runtime_additional_information"] = buffer_list
        base["seq_token_counts"] = [int(input_batch.num_scheduled_tokens[i]) for i in range(input_batch.num_reqs)]
        # Return static inputs_embeds so FULL graph replay uses the same
        # tensor address that was captured.  Preprocess fills it in-place.
        if self._static_inputs_embeds is not None:
            base["inputs_embeds"] = self._static_inputs_embeds[: input_batch.num_tokens_after_padding]
        for plugin in self.plugins:
            base.update(plugin.prepare_extra_inputs(input_batch, req_states))
        return base

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        base = super().prepare_dummy_inputs(num_reqs, num_tokens)
        dummy_buffer = [{} for _ in range(num_reqs)]
        base["model_intermediate_buffer"] = dummy_buffer
        if not getattr(self.model, "requires_native_model_intermediate_buffer", False):
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
        staging_keys = getattr(self.model, "batched_gpu_staging_keys", set())
        for key in staging_keys:
            if not isinstance(key, tuple) or len(key) != 2:
                raise TypeError(f"batched_gpu_staging_keys entries must be 2-tuples, got {key!r}")
            type_key, qualifier = key
            groups: dict[tuple[torch.dtype, tuple[int, ...]], list[tuple[dict[str, Any], torch.Tensor]]] = {}
            for req_idx in req_indices:
                nested = self.intermediate_buffer.buffers[req_idx].get(type_key)
                if not isinstance(nested, dict):
                    continue
                tensor = nested.get(qualifier)
                if not isinstance(tensor, torch.Tensor) or tensor.device == device:
                    continue
                group_key = tensor.dtype, tuple(tensor.shape[1:])
                groups.setdefault(group_key, []).append((nested, tensor))

            for group in groups.values():
                staged_rows = self._batch_move_tensor_rows([tensor for _nested, tensor in group], device)
                for (nested, _tensor), staged in zip(group, staged_rows, strict=True):
                    nested[qualifier] = staged

    def run_preprocess(
        self,
        input_batch: InputBatch,
        model_inputs: dict[str, Any],
        req_states: RequestState | None = None,
        mtp_batch_descriptor_dispatcher: Callable[[int], Any] | None = None,
    ) -> None:
        """Per-request preprocess + MTP before model forward.

        Modifies ``model_inputs["input_ids"]`` and ``model_inputs["inputs_embeds"]``
        in-place.  Collects decode-step MTP inputs and runs a single batched MTP
        forward at the end.

        Skipped when the model declares ``preprocess_in_forward = True``,
        meaning it handles preprocess internally inside forward().
        """
        if not self.has_preprocess:
            return
        # Model does preprocess+MTP inside forward() — skip external preprocess.
        if getattr(self.model, "preprocess_in_forward", False):
            return

        input_ids = model_inputs.get("input_ids")
        if input_ids is None:
            input_ids = input_batch.input_ids

        embeds = model_inputs.get("inputs_embeds")
        if embeds is None:
            embeds = self.model.embed_input_ids(input_batch.input_ids[: input_batch.num_tokens])
            model_inputs["inputs_embeds"] = embeds
        elif self._static_inputs_embeds is not None and embeds.data_ptr() == self._static_inputs_embeds.data_ptr():
            # FULL graph replay requires a stable inputs_embeds address. Refresh
            # the active rows from the current token ids before model-specific
            # preprocessing; otherwise decode reuses embeddings left by the
            # previous step at the same static address.
            num_active_tokens = int(getattr(input_batch, "num_tokens", input_ids.shape[0]))
            token_embeds = self.model.embed_input_ids(input_ids[:num_active_tokens])
            embeds[:num_active_tokens].copy_(token_embeds)

        gpu_keys: set[str] = getattr(self.model, "gpu_resident_buffer_keys", set())
        mtp_batches: list[tuple[int, int, tuple[torch.Tensor, torch.Tensor]]] = []
        prepacked_mtp_inputs: tuple[torch.Tensor, torch.Tensor] | None = None
        deferred_text_projection_rows: list[int] = []
        project_text_steps = getattr(self.model, "project_talker_text_steps", None)

        req_indices = [int(input_batch.idx_mapping_np[i]) for i in range(input_batch.num_reqs)]
        self._stage_batched_preprocess_inputs(req_indices, embeds.device)

        preprocess_entries: list[tuple[int, int, int, int, dict[str, Any], bool]] = []
        for i, req_idx in enumerate(req_indices):
            buf = self.intermediate_buffer.buffers[req_idx]
            if not buf or "req_id" not in buf:
                continue
            if str(buf["req_id"]).startswith("_warmup_"):
                continue

            start = int(input_batch.query_start_loc_np[i])
            n_tok = int(input_batch.num_scheduled_tokens[i])

            info = {key: value for key, value in buf.items() if isinstance(key, str)}
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
            if callable(project_text_steps):
                info["_omni_defer_talker_text_projection"] = True

            is_prefill = (
                int(num_computed_tokens) < int(prompt_len)
                if prompt_len is not None and num_computed_tokens is not None
                else n_tok > 1
            )
            preprocess_entries.append((i, req_idx, start, n_tok, info, is_prefill))

        preprocess_batch_mrv2 = getattr(self.model, "preprocess_batch_mrv2", None)
        if callable(preprocess_batch_mrv2):
            prefill_infos = [
                self.intermediate_buffer.buffers[req_idx]
                for _i, req_idx, _start, _n_tok, _info, is_prefill in preprocess_entries
                if is_prefill
            ]
            if prefill_infos:
                preprocess_batch_mrv2(req_infos=prefill_infos, device=embeds.device)
                for _i, req_idx, _start, _n_tok, info, is_prefill in preprocess_entries:
                    if not is_prefill:
                        continue
                    runtime_info = {key: value for key, value in info.items() if key.startswith("_omni_")}
                    info.update(self.intermediate_buffer.buffers[req_idx])
                    info.update(runtime_info)

        batch_decode_preprocess = getattr(self.model, "preprocess_decode_batch_mrv2", None)
        decode_entries = [entry for entry in preprocess_entries if entry[3] == 1 and not entry[5]]
        batched_decode_indices: set[int] = set()
        if callable(batch_decode_preprocess) and decode_entries:
            batch_size = len(decode_entries)
            starts = [entry[2] for entry in decode_entries]
            is_contiguous_cohort = all(entry[0] == row and entry[2] == row for row, entry in enumerate(decode_entries))
            if is_contiguous_cohort:
                batch_ids = input_ids[:batch_size]
                batch_embeds = embeds[:batch_size]
                batch_offsets = None
            else:
                batch_offsets = torch.as_tensor(starts, device=input_ids.device, dtype=torch.long)
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

            for row, (entry, updates) in enumerate(zip(decode_entries, updates_by_req, strict=True)):
                i, req_idx, start, _n_tok, _info, _is_prefill = entry
                mtp_batches.append(
                    (
                        i,
                        start,
                        (batch_hidden[row : row + 1], batch_text_step[row : row + 1]),
                    )
                )
                self.intermediate_buffer.update(req_idx, updates, gpu_keys)
                batched_decode_indices.add(i)
            prepacked_mtp_inputs = batch_hidden, batch_text_step

        for i, req_idx, start, n_tok, info, _is_prefill in preprocess_entries:
            if i in batched_decode_indices:
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

            # Collect MTP inputs for decode steps (n_tok == 1 with mtp_inputs)
            mtp_inputs = updates.pop("mtp_inputs", None)
            text_step_requires_projection = bool(updates.pop("mtp_text_step_requires_projection", False))
            if mtp_inputs is not None and n_tok == 1:
                mtp_batches.append((i, start, mtp_inputs))
                if text_step_requires_projection:
                    deferred_text_projection_rows.append(len(mtp_batches) - 1)

            self.intermediate_buffer.update(req_idx, updates, gpu_keys)

        if deferred_text_projection_rows:
            if not callable(project_text_steps):
                raise RuntimeError("Deferred Talker text projection requires project_talker_text_steps().")
            raw_text_steps = torch.cat(
                [mtp_batches[row][2][1].reshape(1, -1) for row in deferred_text_projection_rows],
                dim=0,
            )
            projected_text_steps = project_text_steps(raw_text_steps)
            if projected_text_steps.shape[0] != len(deferred_text_projection_rows):
                raise RuntimeError(
                    "Batched Talker text projection changed the request axis: "
                    f"expected={len(deferred_text_projection_rows)} actual={projected_text_steps.shape[0]}"
                )
            for projected_row, batch_row in enumerate(deferred_text_projection_rows):
                req_idx, start, (past_hidden, original_text_step) = mtp_batches[batch_row]
                text_step = projected_text_steps[projected_row]
                if original_text_step.ndim > 1:
                    text_step = text_step.reshape(1, -1)
                mtp_batches[batch_row] = (req_idx, start, (past_hidden, text_step))

        if mtp_batches and hasattr(self.model, "talker_mtp"):
            if prepacked_mtp_inputs is None:
                self._run_batched_mtp(
                    mtp_batches,
                    input_ids,
                    embeds,
                    input_batch,
                    gpu_keys,
                    mtp_batch_descriptor_dispatcher,
                )
            else:
                self._run_batched_mtp(
                    mtp_batches,
                    input_ids,
                    embeds,
                    input_batch,
                    gpu_keys,
                    mtp_batch_descriptor_dispatcher,
                    prepacked_mtp_inputs=prepacked_mtp_inputs,
                )

    def _pack_talker_mtp_batch(
        self,
        mtp_batches: list[tuple[int, int, tuple[torch.Tensor, torch.Tensor]]],
        input_ids: torch.Tensor,
        embeds: torch.Tensor,
        offsets: torch.Tensor | None = None,
        prepacked_mtp_inputs: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack per-request Talker state with a bounded number of GPU ops."""
        bsz = len(mtp_batches)
        if offsets is None:
            offsets = torch.as_tensor(
                [start for _i, start, _mtp in mtp_batches],
                device=input_ids.device,
                dtype=torch.long,
            )
        if prepacked_mtp_inputs is None:
            hidden_rows = [past_hidden.reshape(1, -1) for _i, _start, (past_hidden, _step) in mtp_batches]
            text_step_rows = [text_step.reshape(1, -1) for _i, _start, (_hidden, text_step) in mtp_batches]
            packed_hidden = None
            packed_text_step = None
        else:
            packed_hidden, packed_text_step = prepacked_mtp_inputs
            packed_hidden = packed_hidden.reshape(bsz, -1)
            packed_text_step = packed_text_step.reshape(bsz, -1)

        if self._mtp_input_ids is not None and bsz <= self._mtp_input_ids.shape[0]:
            batch_ids = self._mtp_input_ids[:bsz]
            batch_embeds = self._mtp_input_embeds[:bsz]
            batch_hidden = self._mtp_hidden[:bsz]
            batch_text_step = self._mtp_text_step[:bsz]
            if input_ids.dtype == batch_ids.dtype:
                torch.index_select(input_ids, 0, offsets, out=batch_ids)
            else:
                batch_ids.copy_(input_ids.index_select(0, offsets))
            torch.index_select(embeds, 0, offsets, out=batch_embeds)
            if prepacked_mtp_inputs is None:
                torch.cat(hidden_rows, dim=0, out=batch_hidden)
                torch.cat(text_step_rows, dim=0, out=batch_text_step)
            else:
                batch_hidden.copy_(packed_hidden)
                batch_text_step.copy_(packed_text_step)
            return batch_ids, batch_embeds, batch_hidden, batch_text_step, offsets

        if prepacked_mtp_inputs is None:
            packed_hidden = torch.cat(hidden_rows, dim=0)
            packed_text_step = torch.cat(text_step_rows, dim=0)
        return (
            input_ids.index_select(0, offsets),
            embeds.index_select(0, offsets),
            packed_hidden,
            packed_text_step,
            offsets,
        )

    def _talker_mtp_batch_offsets(
        self,
        mtp_batches: list[tuple[int, int, tuple[torch.Tensor, torch.Tensor]]],
        input_batch: InputBatch,
        device: torch.device,
    ) -> torch.Tensor:
        """Reuse MRv2's device query offsets for a contiguous decode prefix.

        The steady-state Talker batch contains one decode token from every
        scheduled request, in scheduler order. ``InputBatch.query_start_loc``
        already contains the exact device-side gather offsets for that shape;
        rebuilding it from a Python list creates a small synchronous H2D
        transfer on every decode step.
        """
        bsz = len(mtp_batches)
        query_start_loc = getattr(input_batch, "query_start_loc", None)
        static_offsets = self._mtp_offsets
        if (
            isinstance(query_start_loc, torch.Tensor)
            and query_start_loc.device == device
            and isinstance(static_offsets, torch.Tensor)
            and static_offsets.device == device
            and bsz <= static_offsets.shape[0]
            and all(batch_index == expected for expected, (batch_index, _start, _mtp) in enumerate(mtp_batches))
        ):
            offsets = static_offsets[:bsz]
            offsets.copy_(query_start_loc[:bsz], non_blocking=True)
            return offsets

        return torch.as_tensor(
            [start for _i, start, _mtp in mtp_batches],
            device=device,
            dtype=torch.long,
        )

    def _run_batched_mtp(
        self,
        mtp_batches: list[tuple[int, int, tuple[torch.Tensor, torch.Tensor]]],
        input_ids: torch.Tensor,
        embeds: torch.Tensor,
        input_batch: InputBatch,
        gpu_keys: set[str],
        mtp_batch_descriptor_dispatcher: Callable[[int], Any] | None = None,
        prepacked_mtp_inputs: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """Batch MTP forward for all decode-step requests.

        Uses pre-allocated static buffers to avoid per-step torch.cat
        memory allocations.
        """
        from vllm.forward_context import set_forward_context

        bsz = len(mtp_batches)
        batch_offsets = self._talker_mtp_batch_offsets(
            mtp_batches,
            input_batch,
            input_ids.device,
        )
        batch_ids, batch_emb, batch_hidden, batch_step, batch_offsets = self._pack_talker_mtp_batch(
            mtp_batches,
            input_ids,
            embeds,
            batch_offsets,
            prepacked_mtp_inputs,
        )

        req_indices = [int(input_batch.idx_mapping_np[i]) for i, _start, _mtp in mtp_batches]
        buffers = [self.intermediate_buffer.buffers[req_idx] for req_idx in req_indices]
        req_ids = [str(buffer.get("req_id")) for buffer in buffers]
        generators = [
            self._get_talker_mtp_generator(
                req_id,
                buffer.get("sampling_params"),
                batch_ids.device,
            )
            for req_id, buffer in zip(req_ids, buffers, strict=True)
        ]

        use_graph_runner = self._is_talker_mtp_graph_runner()
        has_explicit_generator = any(generator is not None for generator in generators)
        batch_descriptor = None
        cudagraph_mode = CUDAGraphMode.NONE
        num_tokens = bsz
        if use_graph_runner and not has_explicit_generator and mtp_batch_descriptor_dispatcher is not None:
            batch_descriptor = mtp_batch_descriptor_dispatcher(bsz)
            cudagraph_mode = getattr(batch_descriptor, "cg_mode", CUDAGraphMode.FULL)
            num_tokens = int(getattr(batch_descriptor, "num_tokens", bsz))
            batch_ids = self._mtp_input_ids[:num_tokens]
            batch_emb = self._mtp_input_embeds[:num_tokens]
            batch_hidden = self._mtp_hidden[:num_tokens]
            batch_step = self._mtp_text_step[:num_tokens]
        elif use_graph_runner and not has_explicit_generator:
            cudagraph_mode = CUDAGraphMode.FULL
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_mode,
            batch_descriptor=batch_descriptor,
        ):
            new_emb, codes = self._call_talker_mtp_with_sampling(
                batch_ids,
                batch_emb,
                batch_hidden,
                batch_step,
                buffers=buffers,
                req_ids=req_ids,
                generators=generators,
            )

        embeds.index_copy_(0, batch_offsets, new_emb[:bsz].reshape(bsz, -1))
        audio_key = getattr(self.model, "talker_mtp_output_key", ("codes", "audio"))
        validity_key = getattr(self.model, "talker_mtp_validity_key", None)
        valid_rows = None
        if codes is not None and validity_key is not None:
            valid_rows = torch.ones((bsz,), dtype=torch.bool, device=codes.device)
        if codes is not None and audio_key in gpu_keys:
            self.intermediate_buffer.update_gpu_tensor_rows(
                req_indices,
                audio_key,
                codes[:bsz],
            )
            if valid_rows is not None:
                self.intermediate_buffer.update_gpu_tensor_rows(
                    req_indices,
                    validity_key,
                    valid_rows,
                    keepdim=False,
                )
            return
        for j, (i, _start, _) in enumerate(mtp_batches):
            if codes is None:
                continue
            req_idx = int(input_batch.idx_mapping_np[i])
            if isinstance(audio_key, tuple) and len(audio_key) == 2:
                updates = {audio_key[0]: {audio_key[1]: codes[j : j + 1]}}
            elif isinstance(audio_key, str):
                updates = {audio_key: codes[j : j + 1]}
            else:
                raise TypeError(
                    f"talker_mtp_output_key must be a string or 2-tuple, got {type(audio_key).__name__}: {audio_key!r}"
                )
            if valid_rows is not None:
                if isinstance(validity_key, tuple) and len(validity_key) == 2:
                    updates.setdefault(validity_key[0], {})[validity_key[1]] = valid_rows[j]
                elif isinstance(validity_key, str):
                    updates[validity_key] = valid_rows[j]
                else:
                    raise TypeError(
                        "talker_mtp_validity_key must be a string or 2-tuple, "
                        f"got {type(validity_key).__name__}: {validity_key!r}"
                    )
            self.intermediate_buffer.update(req_idx, updates, gpu_keys)

    def _get_talker_mtp_base_sampling_kwargs(self) -> dict[str, Any]:
        subtalker_params = getattr(self.vllm_config.model_config, "subtalker_sampling_params", None)
        if not isinstance(subtalker_params, dict):
            subtalker_params = {}
        return {
            "do_sample": subtalker_params.get("do_sample"),
            "temperature": subtalker_params.get("temperature"),
            "top_k": subtalker_params.get("top_k"),
            "top_p": subtalker_params.get("top_p"),
        }

    def _get_talker_mtp_sampling_kwargs(
        self,
        buffers: list[dict[str, Any]],
        req_ids: list[str],
        generators: list[torch.Generator | None],
    ) -> dict[str, Any]:
        kwargs = self._get_talker_mtp_base_sampling_kwargs()

        if len(generators) == 1:
            if generators[0] is not None:
                kwargs["generator"] = generators[0]
        elif any(generator is not None for generator in generators):
            kwargs["generators"] = generators

        if getattr(self.model, "talker_mtp_accepts_req_infos", False):
            kwargs["req_ids"] = req_ids
            kwargs["req_infos"] = buffers

        return kwargs

    def _get_talker_mtp_generator(
        self,
        req_id: str,
        sampling_params: Any,
        device: torch.device,
    ) -> torch.Generator | None:
        seed = get_tts_local_seed(sampling_params)
        if seed is None:
            return None

        cache = getattr(self, "_talker_mtp_generators", None)
        if cache is None:
            cache = {}
            self._talker_mtp_generators = cache
        generator = cache.get(req_id)
        if generator is None or generator.device != device:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            cache[req_id] = generator
        return generator

    def _call_talker_mtp_with_sampling(
        self,
        batch_ids: torch.Tensor,
        batch_emb: torch.Tensor,
        batch_hidden: torch.Tensor,
        batch_step: torch.Tensor,
        *,
        buffers: list[dict[str, Any]],
        req_ids: list[str],
        generators: list[torch.Generator | None],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        kwargs = self._get_talker_mtp_sampling_kwargs(buffers, req_ids, generators)

        if (
            len(req_ids) > 1
            and "generators" in kwargs
            and not getattr(self.model, "talker_mtp_accepts_per_row_generators", False)
        ):
            emb_chunks = []
            code_chunks = []
            for row, (req_id, buffer, generator) in enumerate(zip(req_ids, buffers, generators, strict=True)):
                row_kwargs = self._get_talker_mtp_sampling_kwargs(
                    [buffer],
                    [req_id],
                    [generator],
                )
                row_emb, row_codes = self._call_talker_mtp_runner(
                    batch_ids[row : row + 1],
                    batch_emb[row : row + 1],
                    batch_hidden[row : row + 1],
                    batch_step[row : row + 1],
                    **row_kwargs,
                )
                emb_chunks.append(row_emb)
                if row_codes is not None:
                    code_chunks.append(row_codes)
            new_emb = torch.cat(emb_chunks)
            codes = torch.cat(code_chunks) if code_chunks else None
            return new_emb, codes

        return self._call_talker_mtp_runner(
            batch_ids,
            batch_emb,
            batch_hidden,
            batch_step,
            **kwargs,
        )

    def _call_talker_mtp_runner(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        last_hidden: torch.Tensor,
        text_step: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Explicit generators are request state, while the outer whole-MTP
        # graph owns one captured RNG stream. Keep the transformer/code-
        # predictor graphs, but call the raw MTP function so per-row streams
        # remain deterministic and independent of scheduler batch makeup.
        has_explicit_generator = kwargs.get("generator") is not None or kwargs.get("generators") is not None
        runner = self.model.talker_mtp if has_explicit_generator else getattr(self, "_talker_mtp_runner", None)
        runner = runner or self.model.talker_mtp
        return runner(input_ids, input_embeds, last_hidden, text_step, **kwargs)

    # ------------------------------------------------------------------
    # Post-forward: per-request postprocess
    # ------------------------------------------------------------------

    def run_postprocess(self, hidden_states: torch.Tensor, input_batch: InputBatch) -> None:
        """Per-request postprocess after model forward.

        Extracts per-request updates from hidden_states and writes them
        back to the intermediate buffer (e.g. ``last_talker_hidden``).

        Skipped when the model declares ``preprocess_in_forward = True``
        (the flag covers both pre- and post-processing — both run inside
        the model's forward()).
        """
        if not self.has_postprocess:
            return
        # preprocess_in_forward also covers postprocess — both run inside forward()
        if getattr(self.model, "preprocess_in_forward", False):
            return
        gpu_keys: set[str] = getattr(self.model, "gpu_resident_buffer_keys", set())
        batch_postprocess = getattr(self.model, "postprocess_batch_mrv2", None)
        if callable(batch_postprocess) and input_batch.num_reqs:
            req_indices = [int(input_batch.idx_mapping_np[i]) for i in range(input_batch.num_reqs)]
            buffers = [self.intermediate_buffer.buffers[req_idx] for req_idx in req_indices]
            if all(buf and "req_id" in buf and not str(buf["req_id"]).startswith("_warmup_") for buf in buffers):
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
                if not isinstance(values, torch.Tensor) or values.shape[0] != input_batch.num_reqs:
                    actual = values.shape[0] if isinstance(values, torch.Tensor) and values.ndim else 0
                    raise RuntimeError(
                        "Batched Omni postprocess changed the request axis: "
                        f"expected={input_batch.num_reqs} actual={actual}"
                    )
                self.intermediate_buffer.update_gpu_tensor_rows(
                    req_indices,
                    output_key,
                    values,
                    keepdim=False,
                )
                return
        for i in range(input_batch.num_reqs):
            req_idx = int(input_batch.idx_mapping_np[i])
            buf = self.intermediate_buffer.buffers[req_idx]
            if not buf or "req_id" not in buf:
                continue
            if str(buf["req_id"]).startswith("_warmup_"):
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

    def postprocess_model_output(
        self,
        model_output: Any,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> tuple[torch.Tensor, dict]:
        """Convert raw model output to ``(text_hidden, multimodal_outputs)``.

        Handles ``OmniOutput`` unwrapping and ``make_omni_output``
        conversion, then dispatches to registered plugins.
        """
        if not isinstance(model_output, OmniOutput) and hasattr(self.model, "make_omni_output"):
            if isinstance(model_output, (list, tuple)) or self.have_multimodal_outputs:
                buffer_list = self.intermediate_buffer.gather(input_batch)
                make_output_kwargs = {"model_intermediate_buffer": buffer_list}
                if not getattr(self.model, "requires_native_model_intermediate_buffer", False):
                    make_output_kwargs["runtime_additional_information"] = buffer_list
                model_output = self.model.make_omni_output(model_output, **make_output_kwargs)

        if isinstance(model_output, OmniOutput):
            text_hidden = model_output.text_hidden_states
            multimodal_outputs: dict = model_output.multimodal_outputs or {}
        elif isinstance(model_output, (list, tuple)):
            text_hidden = model_output[0]
            multimodal_outputs = {}
        else:
            text_hidden = model_output
            multimodal_outputs = {}

        for plugin in self.plugins:
            text_hidden, multimodal_outputs = plugin.postprocess(
                text_hidden, multimodal_outputs, input_batch, req_states
            )

        return text_hidden, multimodal_outputs
