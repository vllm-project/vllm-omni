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
        except (AssertionError, TypeError):
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
        if self._embed_dim > 0 and hasattr(model, "talker_mtp"):
            max_bs = max_num_reqs
            self._mtp_input_ids = torch.zeros(max_bs, dtype=torch.long, device=device)
            self._mtp_input_embeds = torch.zeros((max_bs, self._embed_dim), dtype=self.dtype, device=device)
            self._mtp_hidden = torch.zeros((max_bs, self._embed_dim), dtype=self.dtype, device=device)
            self._mtp_text_step = torch.zeros((max_bs, self._embed_dim), dtype=self.dtype, device=device)

        if hasattr(model, "get_omni_plugins"):
            for plugin in model.get_omni_plugins():
                self.register_plugin(plugin)

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
    # Attention metadata: use actual max_seq_len, not max_model_len
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
        """Override to use actual max_seq_len instead of max_model_len.

        Upstream DefaultModelState uses ``self.max_model_len`` as
        ``max_seq_len`` for attention metadata.  This causes
        FlashAttention to use a different scheduler/tiling strategy
        than V1 MR (which uses the actual maximum sequence length).
        The different tiling changes the float accumulation order in
        bf16, causing numerically different attention outputs that
        snowball through the transformer layers and produce completely
        different logits—leading to 30x more decode steps for TTS.
        """
        from vllm.v1.worker.gpu.attn_utils import build_attn_metadata

        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens

        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()

        # Use actual max seq_len (matching V1 MR behavior) instead of
        # max_model_len.  For CUDA graph capture, use max_model_len to
        # ensure the captured kernel covers all possible seq_lens.
        if for_capture:
            max_seq_len = self.max_model_len
        else:
            max_seq_len = int(input_batch.seq_lens[:num_reqs].max().item())

        return build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
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
        for plugin in self.plugins:
            plugin.on_add_request(req_index, new_req_data)

    def remove_request(self, req_index: int) -> None:
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

    def run_preprocess(self, input_batch: InputBatch, model_inputs: dict[str, Any]) -> None:
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

        embeds = model_inputs.get("inputs_embeds")
        if embeds is None:
            embeds = self.model.embed_input_ids(input_batch.input_ids[: input_batch.num_tokens])
            model_inputs["inputs_embeds"] = embeds

        input_ids = model_inputs.get("input_ids")
        if input_ids is None:
            input_ids = input_batch.input_ids

        gpu_keys: set[str] = getattr(self.model, "gpu_resident_buffer_keys", set())
        mtp_batches: list[tuple[int, int, tuple[torch.Tensor, torch.Tensor]]] = []

        for i in range(input_batch.num_reqs):
            req_idx = int(input_batch.idx_mapping_np[i])
            buf = self.intermediate_buffer.buffers[req_idx]

            # Skip warmup/dummy requests that have no real buffer data.
            # Warmup creates fake requests without the metadata that
            # model.preprocess() requires (e.g. additional_information.text).
            if not buf or "req_id" not in buf:
                continue

            start = int(input_batch.query_start_loc_np[i])
            n_tok = int(input_batch.num_scheduled_tokens[i])

            ids_slice = input_ids[start : start + n_tok]
            emb_slice = embeds[start : start + n_tok]

            try:
                info = {key: value for key, value in buf.items() if isinstance(key, str)}
                new_ids, new_emb, updates = self.model.preprocess(ids_slice, emb_slice, **info)
            except Exception:
                logger.warning(
                    "preprocess failed for req_idx=%d (req_id=%s); skipping preprocess for this request",
                    req_idx,
                    buf.get("req_id", "?"),
                    exc_info=True,
                )
                continue

            # Write back in-place
            seg = min(n_tok, new_ids.shape[0])
            input_ids[start : start + seg] = new_ids[:seg]
            embeds[start : start + seg] = new_emb[:seg]

            # Collect MTP inputs for decode steps (n_tok == 1 with mtp_inputs)
            mtp_inputs = updates.pop("mtp_inputs", None)
            if mtp_inputs is not None and n_tok == 1:
                mtp_batches.append((i, start, mtp_inputs))

            self.intermediate_buffer.update(req_idx, updates, gpu_keys)

        if mtp_batches and hasattr(self.model, "talker_mtp"):
            self._run_batched_mtp(mtp_batches, input_ids, embeds, input_batch, gpu_keys)

    def _run_batched_mtp(
        self,
        mtp_batches: list[tuple[int, int, tuple[torch.Tensor, torch.Tensor]]],
        input_ids: torch.Tensor,
        embeds: torch.Tensor,
        input_batch: InputBatch,
        gpu_keys: set[str],
    ) -> None:
        """Batch MTP forward for all decode-step requests.

        Uses pre-allocated static buffers to avoid per-step torch.cat
        memory allocations.
        """
        from vllm.forward_context import set_forward_context

        bsz = len(mtp_batches)

        # Fill static buffers via .copy_() — no allocation.
        if self._mtp_input_ids is not None and bsz <= self._mtp_input_ids.shape[0]:
            for j, (_i, start, (past_hidden, text_step)) in enumerate(mtp_batches):
                self._mtp_input_ids[j] = input_ids[start]
                self._mtp_input_embeds[j].copy_(embeds[start])
                self._mtp_hidden[j].copy_(past_hidden.reshape(-1)[: self._mtp_hidden.shape[1]])
                self._mtp_text_step[j].copy_(text_step.reshape(-1)[: self._mtp_text_step.shape[1]])
            batch_ids = self._mtp_input_ids[:bsz]
            batch_emb = self._mtp_input_embeds[:bsz]
            batch_hidden = self._mtp_hidden[:bsz]
            batch_step = self._mtp_text_step[:bsz]
        else:
            # Fallback to torch.cat if static buffers not available.
            ids_list, emb_list, hidden_list, step_list = [], [], [], []
            for _i, start, (past_hidden, text_step) in mtp_batches:
                ids_list.append(input_ids[start : start + 1])
                emb_list.append(embeds[start : start + 1])
                hidden_list.append(past_hidden)
                step_list.append(text_step)
            batch_ids = torch.cat(ids_list)
            batch_emb = torch.cat(emb_list)
            batch_hidden = torch.cat(hidden_list)
            batch_step = torch.cat(step_list)

        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=bsz,
        ):
            new_emb, codes = self.model.talker_mtp(
                batch_ids,
                batch_emb,
                batch_hidden,
                batch_step,
            )

        audio_key = getattr(self.model, "talker_mtp_output_key", ("codes", "audio"))
        for j, (i, start, _) in enumerate(mtp_batches):
            embeds[start : start + 1] = new_emb[j : j + 1]
            req_idx = int(input_batch.idx_mapping_np[i])
            if isinstance(audio_key, tuple) and len(audio_key) == 2:
                updates = {audio_key[0]: {audio_key[1]: codes[j : j + 1]}}
            elif isinstance(audio_key, str):
                updates = {audio_key: codes[j : j + 1]}
            else:
                raise TypeError(
                    f"talker_mtp_output_key must be a string or 2-tuple, got {type(audio_key).__name__}: {audio_key!r}"
                )
            self.intermediate_buffer.update(req_idx, updates, gpu_keys)

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
        for i in range(input_batch.num_reqs):
            req_idx = int(input_batch.idx_mapping_np[i])
            buf = self.intermediate_buffer.buffers[req_idx]
            if not buf or "req_id" not in buf:
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
                try:
                    model_output = self.model.make_omni_output(
                        model_output,
                        model_intermediate_buffer=buffer_list,
                        runtime_additional_information=buffer_list,
                    )
                except Exception:
                    _desc = type(model_output).__name__
                    if isinstance(model_output, (list, tuple)):
                        _desc += f"(len={len(model_output)})"
                    logger.warning(
                        "make_omni_output failed for %s; multimodal outputs will be empty",
                        _desc,
                        exc_info=True,
                    )

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
