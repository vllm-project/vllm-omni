# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import time
from collections.abc import Mapping
from copy import copy, deepcopy
from typing import Any, NamedTuple

import numpy as np
import torch
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import CUDAGraphMode
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.forward_context import BatchDescriptor
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    ECConnectorOutput,
    SamplerOutput,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput, PerLayerAttnMetadata
from vllm.v1.worker.mamba_utils import preprocess_mamba
from vllm.v1.worker.ubatch_utils import maybe_create_ubatch_slices
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper, get_graph_params

# yapf conflicts with isort for this block
# yapf: disable
from vllm_ascend.ops.rotary_embedding import update_cos_sin
from vllm_ascend.utils import enable_sp, global_stream
from vllm_ascend.worker.model_runner_v1 import graph_capture

from vllm_omni.data_entry_keys import flatten_payload
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import OmniKVTransferManager
from vllm_omni.distributed.omni_connectors.utils.config import get_stage_connector_role, stage_sends_async_output
from vllm_omni.experimental.fullduplex.model_executor import DuplexSamplingRunnerMixin
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.platforms.npu.minicpmo_fia_pad import STATE as FIA_PAD_STATE
from vllm_omni.platforms.npu.minicpmo_fia_pad import fia_pad_mode, talker_gate
from vllm_omni.platforms.npu.worker.npu_model_runner import OmniNPUModelRunner
from vllm_omni.utils.mm_outputs import build_mm_cpu, partition_payload_list, to_payload_element
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin
from vllm_omni.worker.sampling_utils import sanitize_min_tokens_stop_ids


def _ensure_tensor_values(payload: dict[str, object]) -> dict[str, torch.Tensor]:
    """Convert a flattened payload to strictly ``dict[str, torch.Tensor]``.

    Non-tensor scalars (int, float, bool) are wrapped with ``torch.tensor()``.
    Values that cannot be safely converted are dropped with a warning.
    This enforces the tensor-only invariant required by the
    ``OmniEngineCoreOutput.multimodal_output`` wire field and msgspec
    serialization. Mirrors ``gpu_ar_model_runner._ensure_tensor_values``.
    """
    result: dict[str, torch.Tensor] = {}
    for key, val in payload.items():
        if isinstance(val, torch.Tensor):
            result[key] = val
        elif isinstance(val, (int, float, bool)):
            result[key] = torch.tensor(val)
        elif isinstance(val, (list, tuple)):
            try:
                result[key] = torch.tensor(val)
            except (ValueError, TypeError, RuntimeError):
                logger.warning(
                    "Dropping non-tensorizable multimodal output key '%s' (type=%s) from wire payload.",
                    key,
                    type(val).__name__,
                )
        else:
            logger.warning(
                "Dropping non-tensor multimodal output key '%s' (type=%s) from wire payload.",
                key,
                type(val).__name__,
            )
    return result


# TTS prefill bypass placeholder: the stage-0 token is echo-overwritten by
# the orchestrator and sits OUTSIDE the llm2tts [tts_bos, tts_eos) slice, so
# any id that is not 151703 (<|tts_bos|>) keeps stage-1 conditioning identical.
# 0 is far from the 1516xx special range.
_BYPASS_SKIP_LOGITS_EOS_ID = 151645  # <|im_end|>: the greedy token the bypassed
# prefill step would sample; check_stop() stops the stage-0 request on it
# (Request.max_tokens is copied before the orchestrator caps it, so the
# EOS hit — not the cap — is what terminates the bypass request).


class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: SchedulerOutput
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: AscendCommonAttentionMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    attn_metadata: PerLayerAttnMetadata
    positions: torch.Tensor
    ec_connector_output: ECConnectorOutput | None
    cudagraph_stats: CUDAGraphStat | None
    batch_desc: BatchDescriptor
    multimodal_outputs: Any # Omni-Specific

class NPUARModelRunner(OmniNPUModelRunner, OmniConnectorModelRunnerMixin, DuplexSamplingRunnerMixin):
    """Autoregressive NPU model runner that returns hidden states per request."""
    _skip_logits_step = False  # TTS prefill bypass: skip lm_head+sampler for the step

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        # each model stage has their own hidden size
        self.hidden_size = self.model_config.hf_text_config.hidden_size
        self.inputs_embeds = self._make_buffer(self.max_num_tokens, self.hidden_size, dtype=self.dtype, numpy=False)
        # Initialize KV cache manager (preserve vllm_config fallback behavior)
        self.kv_transfer_manager = OmniKVTransferManager.from_vllm_config(self.vllm_config, self.model_config)
        self._async_chunk = getattr(self.model_config, "async_chunk", False)
        _OMNI_CONNECTOR_INIT_ARCHS = {
            "Qwen3OmniMoeForConditionalGeneration",
            "Qwen2_5OmniForConditionalGeneration",
            "CovoAudioForConditionalGeneration",
            "MiMoAudioModel",
            "Qwen3TTSTalkerForConditionalGeneration",
            "Qwen3TTSCode2Wav",
            "CosyVoice3Model",
            "DyninOmniForConditionalGeneration",
            "IndexTTS2TalkerForConditionalGeneration",
        }
        # Mirrors gpu_ar_model_runner: an arch missing from the hardcoded allowlist
        # still needs connectors when the deploy config hands the stage a
        # sender/receiver role (e.g. MiniCPM-o 4.5, whose archs are not listed but
        # whose YAML wires stage 1 -> stage 2). Without the role check the
        # full-payload (``--no-async-chunk``) handoff never initializes: nothing
        # accumulates, nothing flushes, and the downstream stage starves silently.
        if (
            getattr(self.model_config, "model_arch", None) in _OMNI_CONNECTOR_INIT_ARCHS
            or get_stage_connector_role(self.model_config) is not None
        ):
            self.init_omni_connectors(
                model_config=self.model_config,
                kv_transfer_manager=self.kv_transfer_manager,
            )
        self._downstream_payload_cache: dict[str, bool] = {}
        self._init_duplex_sampling_state()
        # [Omni] Single-request decode cache for _build_attention_metadata.
        # Entry: (key, attn_metadata); spec_decode_common is always None in
        # the eligible regime (speculative_config is None).
        self._cached_attn_meta: tuple[tuple, PerLayerAttnMetadata] | None = None
        # [minicpm-challenge: A-tier FIA pad] per-engine gate + persistent
        # buffers (see vllm_omni/platforms/npu/minicpmo_fia_pad.py). The
        # talker-only gate keeps stage0 (which shares this runner class) and
        # stage2 fully stock.
        FIA_PAD_STATE.mode = fia_pad_mode()
        if FIA_PAD_STATE.mode != 0 and talker_gate(self.model_config):
            FIA_PAD_STATE.enabled = True
            FIA_PAD_STATE.degraded = FIA_PAD_STATE.mode == 2
            dev = self.device
            FIA_PAD_STATE.klen_dev = torch.zeros(1, dtype=torch.int32, device=dev)
            FIA_PAD_STATE.klen_host = torch.zeros(1, dtype=torch.int32, pin_memory=True)
            FIA_PAD_STATE.klen_host[0] = FIA_PAD_STATE.KV_PAD
            FIA_PAD_STATE.klen_dev.copy_(FIA_PAD_STATE.klen_host)
            FIA_PAD_STATE.arange_buf = (
                torch.arange(FIA_PAD_STATE.KV_MAX, dtype=torch.int32, device=dev)
                .view(1, 1, 1, -1)
                .contiguous()
            )
            FIA_PAD_STATE.cmp_buf = torch.zeros(
                1, 1, 1, FIA_PAD_STATE.KV_MAX, dtype=torch.bool, device=dev
            )
            FIA_PAD_STATE.mask_buf = torch.ones(
                1, 1, 1, FIA_PAD_STATE.KV_MAX, dtype=torch.int8, device=dev
            )
            logger.info(
                "[fia_pad] stage1 talker gate OPEN (mode=%d degraded=%s)",
                FIA_PAD_STATE.mode,
                FIA_PAD_STATE.degraded,
            )

    def load_model(self, *args, **kwargs) -> None:
        super().load_model(*args, **kwargs)
        self._resolve_duplex_sampling_hook(force=True)

    # minicpm-challenge: prep-fast (Slice B2b). Steady-state decode fast
    # path for _prepare_inputs; see the docstring below. Falls back to the
    # parent implementation for any frame that is not exactly one running
    # request with one scheduled token (prefill, chunked, spec, PCP, ...).
    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[torch.Tensor, "SpecDecodeMetadata | None", int]:
        import os

        if os.environ.get("MINICPMO_PREP_FAST", "1") != "0":
            try:
                r = self._prep_fast(scheduler_output, num_scheduled_tokens)
                if r is not None:
                    return r
            except Exception:
                # Leave no stale optimization state behind; the parent
                # implementation rewrites every buffer from scratch.
                self._pf_state = None
                if not getattr(self, "_pf_err_logged", False):
                    self._pf_err_logged = True
                    try:
                        import logging

                        logging.getLogger(
                            "vllm.omni.prep_fast"
                        ).exception("prep-fast: falling back to parent")
                    except Exception:
                        pass
        return super()._prepare_inputs(scheduler_output, num_scheduled_tokens)

    def _prep_fast(
        self, scheduler_output: "SchedulerOutput", num_scheduled_tokens: np.ndarray
    ) -> tuple[torch.Tensor, None, int] | None:
        """1-req / 1-token decode fast path. None => use parent path."""
        import os

        ib = self.input_batch
        dbg = os.environ.get("MINICPMO_PREP_FAST_DEBUG") == "1"

        def bail(reason: str):
            if dbg:
                seen = getattr(self, "_pf_dbg", None)
                if seen is None:
                    seen = self._pf_dbg = {"n": 0, "reasons": {}}
                if reason not in seen["reasons"]:
                    seen["reasons"][reason] = True
                    seen["n"] += 1
                    if seen["n"] <= 5:
                        try:
                            import logging

                            logging.getLogger(
                                "vllm.omni.prep_fast"
                            ).info("prep-fast guard: %s", reason)
                        except Exception:
                            pass
            return None

        if ib.num_reqs != 1:
            return bail(f"num_reqs={ib.num_reqs}")
        if num_scheduled_tokens[0] != 1:
            return bail(f"nsched={num_scheduled_tokens[0]}")
        if scheduler_output.total_num_scheduled_tokens != 1:
            return bail("total!=1")
        if scheduler_output.scheduled_spec_decode_tokens:
            return bail("spec_tokens")
        if getattr(self, "speculative_config", None):
            return bail("spec_config")
        if getattr(self, "pcp_size", 1) > 1 or getattr(self, "use_cp", False):
            return bail("pcp/cp")
        if getattr(self, "uses_mrope", False) or getattr(
            self, "uses_xdrope_dim", 0
        ) > 0:
            return bail("mrope/xdrope")
        if getattr(self, "_has_gdn", False):
            return bail("gdn")
        if getattr(self, "enable_prompt_embeds", False) or ib.req_prompt_embeds:
            return bail("prompt_embeds")
        if getattr(self, "use_async_spec_decode", False):
            return bail("async_spec")
        pos0 = int(ib.num_computed_tokens_cpu[0])
        if pos0 == 0:
            return bail("pos0=0")

        st = getattr(self, "_pf_state", None)
        if st is None:
            st = self._pf_state = {
                "bt_rows": None,       # last committed row snapshot per group
                "slot": "unverified",  # unverified | ok | bad
                "ids_np": hasattr(self.input_ids, "np"),
            }

        # Side effects the rest of the frame depends on.
        self._build_attn_state(1, num_scheduled_tokens, num_scheduled_tokens)
        self.with_prefill = False  # DecodeOnly branch
        self.query_lens = torch.from_numpy(num_scheduled_tokens)

        bt = ib.block_table
        groups = getattr(bt, "block_tables", None)
        if groups is None:
            groups = [bt]
        if st["bt_rows"] is None or len(st["bt_rows"]) != len(groups):
            st["bt_rows"] = [None] * len(groups)

        # Block table: upstream commits the full-width row of every KV-cache
        # group each frame; for decode a row only changes when a new block is
        # appended. Skip a group's copy when its row 0 content is unchanged
        # (content compare, not a flag).
        for gi, g in enumerate(groups):
            nb = int(g.num_blocks_per_row[0])
            row = g.block_table.np[0, :nb]
            cached = st["bt_rows"][gi]
            if (
                cached is None
                or row.shape != cached.shape
                or not np.array_equal(row, cached)
            ):
                g.block_table.copy_to_gpu(1)
                st["bt_rows"][gi] = row.copy()

        # input_ids: under async scheduling the scheduler has not committed
        # the previous step's sampled token to token_ids_cpu, so for a
        # continuing request (prev row >= 0) the token is copied on device
        # from prev_sampled_token_ids — the same slice the upstream
        # common-case optimization uses. Otherwise it is a single scalar
        # read from the host token table.
        self._compute_prev_positions(1)
        pv = int(self.prev_positions.np[0])
        if pv >= 0 and ib.prev_sampled_token_ids is not None:
            self.input_ids.gpu[:1].copy_(
                ib.prev_sampled_token_ids[pv : pv + 1, 0], non_blocking=True
            )
        else:
            tid = ib.token_ids_cpu[0, pos0]
            if st["ids_np"]:
                self.input_ids.np[0] = tid
            else:
                self.input_ids.cpu[0] = tid
            self.input_ids.copy_to_gpu(1)

        # Constant-for-decode buffers: upstream re-writes and re-copies these
        # every frame; the copies are n=1/2 slices (never the full buffer).
        self.query_pos.np[0] = 0
        self.query_pos.copy_to_gpu(1)
        qsl = self.query_start_loc
        qsl.np[0] = 0
        qsl.np[1] = 1
        qsl.copy_to_gpu(2)
        self.req_indices.np[0] = 0
        self.req_indices.copy_to_gpu(1)
        self.num_scheduled_tokens.np[0] = 1
        self.num_scheduled_tokens.copy_to_gpu(1)

        # seq_lens[:1] == optimistic value (computed + 1); write the pinned
        # host buffer once and do a single non_blocking H2D slice copy
        # instead of the upstream device-side add chain. The gpu tails
        # (qsl -1 fill / seq_lens 0 / num_accepted all-ones) are maintained
        # by every parent-path frame and are not written by anything else.
        opt = self.optimistic_seq_lens_cpu
        opt[0] = pos0 + 1
        self.seq_lens[:1].copy_(opt[:1], non_blocking=True)
        if FIA_PAD_STATE.enabled:
            # [minicpm-challenge: A-tier FIA pad] publish the current klen to
            # the device scalar the captured in-graph mask rebuild reads
            # (enqueued on the compute stream, ordered before the replay).
            FIA_PAD_STATE.klen_host[0] = pos0 + 1
            FIA_PAD_STATE.klen_dev.copy_(
                FIA_PAD_STATE.klen_host, non_blocking=True
            )
        self.num_computed_tokens[:1].copy_(
            ib.num_computed_tokens_cpu_tensor[:1], non_blocking=True
        )
        self._positions_np_buf[0] = pos0
        self.positions[:1].copy_(
            self._positions_cpu_buf[:1], non_blocking=True
        )

        # num_accepted_tokens: under async scheduling upstream synchronizes
        # the recorded event and mirrors the per-request accepted count (1
        # for non-spec decode) into the pinned buffer, then copies it up.
        # The gpu tail stays all-ones from the surrounding parent frames.
        evt = getattr(self, "num_accepted_tokens_event", None)
        if evt is not None:
            evt.synchronize()
            na = self.num_accepted_tokens
            if pv >= 0:
                na.np[0] = ib.num_accepted_tokens_cpu[pv]
                ib.num_accepted_tokens_cpu[0] = na.np[0]
            else:
                na.np[0] = 1
                ib.num_accepted_tokens_cpu[0] = 1
            na.copy_to_gpu(1)

        # Discard bookkeeping: scalar equivalent of the upstream mask math.
        nt = self.requests[ib.req_ids[0]].num_tokens
        mask0 = (pos0 + 1) < nt
        if mask0:
            self.discard_request_indices.np[0] = 0
            self.discard_request_indices.copy_to_gpu(1)
            self.num_discarded_requests = 1
        else:
            self.num_discarded_requests = 0
        self.discard_request_mask.np[0] = mask0
        self.discard_request_mask.copy_to_gpu(1)

        # Slot mapping: the general path launches the jit kernel once per
        # KV-cache group per frame (vllm_ascend.worker.block_table wraps
        # vllm core's _compute_slot_mapping_kernel). For one token each
        # group's slot is a host scalar following the kernel's math:
        #   vbi, voff = divmod(pos, physical_block_size)
        #   bt_idx = vbi * blocks_per_phys_block + voff // block_size
        #   slot = bt[row, bt_idx] * block_size + voff % block_size
        # Verified against the real kernel on the first fast frame per boot
        # (torch.equal, per group); any mismatch keeps the kernel running.
        for g in groups:
            pbs = g.physical_block_size
            lbs = g.block_size
            bpp = g.blocks_per_phys_block
            vbi, voff = divmod(pos0, pbs)
            bi = vbi * bpp + voff // lbs
            slot = int(g.block_table.np[0, bi]) * lbs + (voff % lbs)
            sm = g.slot_mapping
            if st["slot"] == "ok":
                sm.np[0] = slot
                sm.copy_to_gpu(1)
            else:
                g.compute_slot_mapping(
                    1, self.query_start_loc.gpu[:2], self.positions[:1]
                )
                if st["slot"] == "unverified":
                    exp = torch.tensor(
                        [slot], dtype=sm.gpu.dtype, device=sm.gpu.device
                    )
                    if not bool(torch.equal(sm.gpu[:1], exp)):
                        st["slot"] = "bad"
        if st["slot"] == "unverified":
            st["slot"] = "ok"

        logits_indices = self.query_start_loc.gpu[1:2] - 1
        return logits_indices, None, 1

    def _update_states(self, scheduler_output: SchedulerOutput):
        deferred_state_corrections_fn = super()._update_states(scheduler_output)
        self._update_duplex_sampling_states(scheduler_output)
        return deferred_state_corrections_fn

    def _make_buffer(self, *size, dtype, numpy=True):
        # Prevent ray from pinning the buffer due to large size
        from vllm_omni.distributed.ray_utils.utils import (
            calculate_total_bytes,
            maybe_disable_pin_memory_for_ray,
        )

        total_bytes = calculate_total_bytes(size, dtype)

        # Use the context manager to temporarily disable pinning if needed
        with maybe_disable_pin_memory_for_ray(self, total_bytes):
            return super()._make_buffer(*size, dtype=dtype, numpy=numpy)

    #  -------------------------------------- Omni-new -------------------------------------------------
    def capture_model(self) -> int:
        npugraph_memory_bytes = super().capture_model()
        self._capture_talker_mtp_graphs()
        return npugraph_memory_bytes

    def _capture_talker_mtp_graphs(self) -> None:
        if not self.has_talker_mtp or not isinstance(self.talker_mtp, ACLGraphWrapper):
            return

        from vllm.compilation.monitor import set_cudagraph_capturing_enabled

        capture_sizes = sorted(self.compilation_config.cudagraph_capture_sizes, reverse=True)
        num_warmups = self.compilation_config.cudagraph_num_of_warmups
        logger.info("Capturing talker_mtp graphs for sizes %s", capture_sizes)

        set_cudagraph_capturing_enabled(True)
        try:
            with torch.inference_mode(), graph_capture(device=self.device):
                for bsz in capture_sizes:
                    _, batch_desc, _, _, _ = self._determine_batch_execution_and_padding(
                        num_tokens=bsz,
                        num_reqs=bsz,
                        num_scheduled_tokens_np=np.ones(bsz, dtype=np.int32),
                        max_num_scheduled_tokens=1,
                        use_cascade_attn=False,
                    )
                    n = batch_desc.num_tokens
                    ids = self.talker_mtp_input_ids.gpu[:n]
                    emb = self.talker_mtp_inputs_embeds.gpu[:n]
                    hid = self.last_talker_hidden.gpu[:n]
                    ts = self.text_step.gpu[:n]

                    for _ in range(num_warmups):
                        with set_ascend_forward_context(
                            None,
                            self.vllm_config,
                            aclgraph_runtime_mode=CUDAGraphMode.NONE,
                            batch_descriptor=batch_desc,
                        ):
                            self.talker_mtp(ids, emb, hid, ts)

                    with set_ascend_forward_context(
                        None,
                        self.vllm_config,
                        aclgraph_runtime_mode=CUDAGraphMode.FULL,
                        batch_descriptor=batch_desc,
                    ):
                        self.talker_mtp(ids, emb, hid, ts)
                    torch.npu.synchronize()

            logger.info("Captured talker_mtp graphs for %d sizes", len(capture_sizes))
        except RuntimeError as e:
            raise RuntimeError(
                f"talker_mtp graph capture failed for a model that declared talker_mtp_graph_safe=True: {e}"
            ) from e
        finally:
            set_cudagraph_capturing_enabled(False)

    def _model_needs_full_prefix_hidden_states(self) -> bool:
        """See gpu_ar_model_runner._model_needs_full_prefix_hidden_states."""
        model = getattr(self, "model", None)
        return bool(getattr(model, "requires_full_prefix_cached_hidden_states", True))

    def _maybe_update_prefix_cache(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: dict,
        num_tokens_unpadded: int,
        num_tokens_padded: int,
    ):
        if self.omni_prefix_cache is not None and get_pp_group().is_last_rank:
            if multimodal_outputs is not None and not isinstance(multimodal_outputs, Mapping):
                logger.warning_once(
                    "prefix caching expects mm outputs to be a dict, but got %s",
                    type(multimodal_outputs),
                )

            hs_for_cache = hidden_states if self._model_needs_full_prefix_hidden_states() else None
            self.omni_prefix_cache.update_omni_tensor_prefix_cache(
                hidden_states=hs_for_cache,
                multimodal_outputs=flatten_payload(multimodal_outputs) if multimodal_outputs else multimodal_outputs,
                num_tokens_unpadded=num_tokens_unpadded,
                slot_mapping=self.input_batch.block_table[0].slot_mapping.cpu,
                num_tokens_padded=num_tokens_padded,
            )

    def _maybe_get_combined_prefix_cache_tensors(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: dict,
        num_scheduled_tokens: dict[str, int],
    ) -> tuple[dict[str, torch.Tensor] | None, dict | None]:
        combined_hidden_states, combined_multimodal_outputs = None, None
        if self.omni_prefix_cache is not None:
            if self._model_needs_full_prefix_hidden_states():
                combined_hidden_states = self.omni_prefix_cache.get_merged_hidden_states(
                    query_start_loc=self.query_start_loc.cpu,
                    input_batch=self.input_batch,
                    hidden_states=hidden_states,
                    num_scheduled_tokens=num_scheduled_tokens,
                )
            combined_multimodal_outputs = self.omni_prefix_cache.get_merged_multimodal_states(
                query_start_loc=self.query_start_loc.cpu,
                input_batch=self.input_batch,
                multimodal_outputs=flatten_payload(multimodal_outputs) if multimodal_outputs else multimodal_outputs,
                num_scheduled_tokens=num_scheduled_tokens,
            )
        return combined_hidden_states, combined_multimodal_outputs

    @staticmethod
    def _resolve_req_hidden_states(
        hidden_states_cpu: torch.Tensor,
        combined_hidden_states: dict[str, torch.Tensor] | None,
        rid: str,
        start: int,
        end: int,
    ):
        if combined_hidden_states is not None:
            if rid not in combined_hidden_states:
                raise RuntimeError("Request IDs in the batch are missing from the merged states!")
            return combined_hidden_states[rid]
        return hidden_states_cpu[start:end]


    def _build_multimodal_outputs(
        self,
        per_req_payloads: list[dict[str, object] | None] | None,
    ) -> list[dict[str, torch.Tensor] | None] | None:
        if self.vllm_config.model_config.engine_output_type == "text":
            return None
        if per_req_payloads is None:
            return None
        wire_payloads: list[dict[str, torch.Tensor] | None] = []
        for payload in per_req_payloads:
            if not payload:
                wire_payloads.append(None)
            else:
                wire_payloads.append(_ensure_tensor_values(payload))
        if all(item is None for item in wire_payloads):
            return None
        return wire_payloads


    def _skip_logits_for_batch(self) -> bool:
        """True when EVERY request in the persistent batch carries the
        orchestrator's skip_logits tag (TTS prefill bypass; the sampled
        token is discarded downstream). Any miss -> normal lm_head path."""
        num_reqs = self.input_batch.num_reqs
        if num_reqs == 0:
            return False
        req_ids = self.input_batch.req_ids
        for i in range(num_reqs):
            info = self.model_intermediate_buffer.get(req_ids[i])
            if not isinstance(info, dict):
                req_state = self.requests.get(req_ids[i])
                info = getattr(req_state, "additional_information_cpu", None)
            if not isinstance(info, dict) or info.get("skip_logits") is not True:
                return False
        return True

    def _request_final_stage_id(self, req_id: str) -> int | None:
        info = self.model_intermediate_buffer.get(req_id)
        if not isinstance(info, dict):
            req_state = self.requests.get(req_id)
            info = getattr(req_state, "additional_information_cpu", None)
        if not isinstance(info, dict):
            return None
        val = info.get("omni_final_stage_id")
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    def _request_needs_downstream_stage_payload(self, req_id: str) -> bool:
        cached = self._downstream_payload_cache.get(req_id)
        if cached is not None:
            return cached
        final_stage_id = self._request_final_stage_id(req_id)
        needs_payload = final_stage_id is None or final_stage_id > 0
        self._downstream_payload_cache[req_id] = needs_payload
        return needs_payload

    def _resolve_pooler_payload_req_ids(self, req_ids_output_copy: list[str]) -> tuple[str, list[str]]:
        downstream_req_ids = [rid for rid in req_ids_output_copy if self._request_needs_downstream_stage_payload(rid)]
        engine_output_type = (self.vllm_config.model_config.engine_output_type or "").lower()
        # Single-stage AR TTS models (e.g. VoxCPM2) finish on this stage but still
        # need multimodal payloads for final audio postprocess/output.
        if engine_output_type == "audio" and not downstream_req_ids:
            downstream_req_ids = req_ids_output_copy
        return engine_output_type, downstream_req_ids

    @staticmethod
    def _sparse_mm_req_ids(multimodal_outputs: Any) -> list[str] | None:
        if not isinstance(multimodal_outputs, dict):
            return None
        meta = multimodal_outputs.get("meta")
        req_ids = None
        sparse_audio = False
        if isinstance(meta, dict):
            req_ids = meta.get("req_id")
            sparse_audio = NPUARModelRunner._is_sparse_audio_marker(meta.get("sparse_audio"))
        if req_ids is None:
            req_ids = multimodal_outputs.get("meta.req_id")
            sparse_audio = NPUARModelRunner._is_sparse_audio_marker(multimodal_outputs.get("meta.sparse_audio"))
        if not sparse_audio:
            return None
        if not isinstance(req_ids, list):
            return None
        return [rid for rid in req_ids if isinstance(rid, str)]

    @staticmethod
    def _is_sparse_audio_marker(value: Any) -> bool:
        if isinstance(value, list):
            return any(str(item).lower() in ("1", "true", "yes", "on") for item in value)
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)
    #  -------------------------------------- Omni-new -------------------------------------------------

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> OmniModelRunnerOutput | IntermediateTensors | None:
        if self.vllm_config.model_config.enable_return_routed_experts:
            capturer = self.routed_experts_capturer
            if capturer is not None and hasattr(capturer, "finalize_pending_copy"):
                capturer.finalize_pending_copy()
        if self.ascend_config.profiling_chunk_config.enabled:
            self._sync_device()
            self._execution_start_time = time.perf_counter()
        if self.execute_model_state is not None:
            raise RuntimeError("State error: sample_tokens() must be called after execute_model() returns None.")

        #  -------------------------------------- Omni-new -------------------------------------------------
        # [Omni] Handle KV transfer BEFORE updating states (which removes finished requests)
        if not getattr(self, "_warmup_state_cleared", False):
            self._warmup_state_cleared = True
            if hasattr(self.model, "_clear_warmup_state"):
                self.model._clear_warmup_state()

        # [Omni] Handle KV transfer BEFORE updating states (which removes finished requests)
        finished_reqs = getattr(scheduler_output, "finished_requests_needing_kv_transfer", {})
        if finished_reqs and hasattr(self.model, "get_kv_transfer_metadata"):
            for req_id, data in finished_reqs.items():
                try:
                    req_idx = self.input_batch.req_id_to_index.get(req_id)
                    num_computed = (
                        int(self.input_batch.num_computed_tokens_cpu[req_idx]) if req_idx is not None else None
                    )
                    model_meta = self.model.get_kv_transfer_metadata(
                        req_id,
                        num_computed_tokens=num_computed,
                    )
                    if model_meta:
                        existing = data.get("custom_metadata") or {}
                        existing.update(model_meta)
                        data["custom_metadata"] = existing
                except Exception as e:
                    logger.warning(f"Failed to get custom metadata from model for {req_id}: {e}")
        self.kv_extracted_req_ids = self.kv_transfer_manager.handle_finished_requests_kv_transfer(
            finished_reqs=finished_reqs,
            kv_caches=self.kv_caches,
            block_size=self.cache_config.block_size,
            cache_dtype=str(self.cache_config.cache_dtype),
            request_id_resolver=self._resolve_global_request_id,
        )
        #  -------------------------------------- Omni-new -------------------------------------------------
        if hasattr(self, "_omni_connector"):
            for request in getattr(scheduler_output, "pending_input_registrations", []):
                self.register_chunk_recv(request)
            self.recv_full_payload_inputs(scheduler_output)
            if self._pending_full_payload_send:
                flush_ids = set(getattr(scheduler_output, "finished_req_ids", set()))
                flush_ids.update({rid for rid in self._pending_full_payload_send if rid not in self.requests})
                if flush_ids:
                    self.flush_full_payload_outputs(flush_ids)
        # self._draft_token_ids is None when `input_fits_in_drafter=False`
        # and there is no draft tokens scheduled. so it need to update the
        # spec_decoding info in scheduler_output with async_scheduling.
        # use deepcopy to avoid the modification has influence on the
        # scheduler_output in engine core process.
        # TODO(Ronald1995): deepcopy is expensive when there is a large
        # number of requests, optimize it later.
        if ((
            self.use_async_scheduling
            and self.num_spec_tokens
            and self._draft_token_ids is None  # type: ignore[has-type]
        ) or (
            # NOTE: This branch specifically triggers a deepcopy during the prefill phase
            # only for PCP (Parallel Context Processing) + Multi-Modal (MM) scenarios.
            # It does not affect other use cases. This is a temporary workaround and
            # will be removed once upstream vLLM provides native support for PCP + MM.
            self.pcp_size > 1
            and self.supports_mm_inputs
            and get_pp_group().is_first_rank
            and not self.model_config.is_encoder_decoder
        )):
            scheduler_output = deepcopy(scheduler_output)

        #  -------------------------------------- Omni-new -------------------------------------------------
        if has_kv_transfer_group():
            kv_connector_metadata = scheduler_output.kv_connector_metadata
            if kv_connector_metadata is not None:
                get_kv_transfer_group().handle_preemptions(kv_connector_metadata)
        #  -------------------------------------- Omni-new -------------------------------------------------

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with record_function_or_nullcontext("prepare input"):
            with self.synchronize_input_prep():
                # Update persistent batch states.
                deferred_state_corrections_fn = self._update_states(scheduler_output)

                #  -------------------------------------- Omni-new -------------------------------------------------
                if scheduler_output.finished_req_ids and hasattr(self.model, "on_requests_finished"):
                    self.model.on_requests_finished(scheduler_output.finished_req_ids)
                #  -------------------------------------- Omni-new -------------------------------------------------

                if has_ec_transfer() and get_ec_transfer().is_producer:
                    with self.maybe_get_ec_connector_output(
                        scheduler_output,
                        encoder_cache=self.encoder_cache,
                    ) as ec_connector_output:
                        self._execute_mm_encoder(scheduler_output)

                        kv_ids = self.kv_extracted_req_ids
                        self.kv_extracted_req_ids = None

                        output = make_empty_encoder_model_runner_output(scheduler_output)
                        if kv_ids:
                            output = copy(output)
                            output.kv_extracted_req_ids = kv_ids
                        return self.attach_omni_connector_output(output)

                # `<= 0`: upstream can schedule a negative span, which is truthy (#5196).
                if num_scheduled_tokens <= 0:
                    if (
                        self.parallel_config.distributed_executor_backend == "external_launcher"
                        and self.parallel_config.data_parallel_size > 1
                    ):
                        # this is a corner case when both external launcher
                        # and DP are enabled, num_scheduled_tokens could be
                        # 0, and has_unfinished_requests in the outer loop
                        # returns True. before returning early here we call
                        # dummy run to ensure coordinate_batch_across_dp
                        # is called into to avoid out of sync issues.
                        self._dummy_run(1)

                    kv_ids = self.kv_extracted_req_ids
                    self.kv_extracted_req_ids = None

                    if not has_kv_transfer_group():
                        output = EMPTY_MODEL_RUNNER_OUTPUT
                    else:
                        output = self.kv_connector_no_forward(scheduler_output, self.vllm_config)

                    if kv_ids:
                        output = copy(output)
                        output.kv_extracted_req_ids = kv_ids

                    return self.attach_omni_connector_output(output)
                if self.cache_config.kv_sharing_fast_prefill:
                    assert not self.num_prompt_logprobs, (
                        "--kv-sharing-fast-prefill produces incorrect "
                        "logprobs for prompt tokens, tokens, please disable "
                        "it when the requests need prompt logprobs"
                    )

                num_reqs = self.input_batch.num_reqs
                req_ids = self.input_batch.req_ids
                tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
                num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
                max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())

                (
                    logits_indices,
                    spec_decode_metadata,
                    total_num_scheduled_tokens,
                ) = self._prepare_inputs(
                    scheduler_output,
                    num_scheduled_tokens_np,
                )

                num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens
                if self.pcp_size > 1:
                    num_tokens_unpadded = self.pcp_manager.total_num_sampled_tokens_pcp
                cascade_attn_prefix_lens = None
                # Disable cascade attention when using microbatching (DBO)
                if self.cascade_attn_enabled and not self.parallel_config.enable_dbo:
                    # Pre-compute cascade attention prefix lengths
                    cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                        num_scheduled_tokens_np,
                        self.input_batch.num_computed_tokens_cpu[:num_reqs],
                        scheduler_output.num_common_prefix_blocks,
                    )

                (
                    cudagraph_mode,
                    batch_desc,
                    should_ubatch,
                    num_tokens_across_dp,
                    cudagraph_stats,
                ) = self._determine_batch_execution_and_padding(
                    num_tokens=num_tokens_unpadded,
                    num_reqs=num_reqs,
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    max_num_scheduled_tokens=max_num_scheduled_tokens,
                    use_cascade_attn=cascade_attn_prefix_lens is not None,
                    force_eager=self.model_config.enforce_eager,
                    num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
                )

                logger.debug(
                    "Running batch with cudagraph_mode: %s, batch_descriptor: %s, "
                    "should_ubatch: %s, num_tokens_across_dp: %s",
                    cudagraph_mode,
                    batch_desc,
                    should_ubatch,
                    num_tokens_across_dp,
                )

                num_tokens_padded = batch_desc.num_tokens
                num_reqs_padded = batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
                ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                    should_ubatch,
                    num_scheduled_tokens_np,
                    num_tokens_padded,
                    num_reqs_padded,
                    self.parallel_config.num_ubatches,
                )

                pad_attn = cudagraph_mode == CUDAGraphMode.FULL

                # NOTE(Angazenn): According to https://github.com/vllm-project/vllm/pull/30877,
                # there should be a corresponding 'postprocess_mamba'. However, it is called inside
                # '_update_states_after_model_execute', which is not overridden in vLLM-Ascend.
                # We simply utilize the implementation in vLLM.
                if self.cache_config.mamba_cache_mode == "align":
                    # preprocess_mamba reads req_state.num_computed_tokens (CPU)
                    # to decide copy operations, so we must apply deferred
                    # corrections before it runs.
                    if deferred_state_corrections_fn:
                        deferred_state_corrections_fn()
                        deferred_state_corrections_fn = None
                    preprocess_mamba(
                        scheduler_output,
                        self.kv_cache_config,
                        self.cache_config,
                        self.mamba_state_idx,
                        self.input_batch,
                        self.requests,
                        self.compilation_config.static_forward_context,
                        self.model.get_mamba_state_copy_func(),
                        self._get_mamba_copy_bufs(),
                    )
                    # preprocess_mamba resets num_accepted_tokens_cpu to 1
                    # for requests whose state was copied to a new block.
                    # Re-sync to GPU so the mamba kernel reads from the
                    # correct initial state slot (init_token_idx = 0).
                    self.num_accepted_tokens.np[:num_reqs] = self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                    self.num_accepted_tokens.copy_to_gpu(num_reqs)

                use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
                ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

                if (
                    cudagraph_mode == CUDAGraphMode.FULL
                    or (enable_sp() and not self.model_config.use_mla)
                    and self.pcp_size * self.dcp_size == 1
                ):
                    # Currently, Graph Mode and SP will both pad num_tokens,
                    # Another possible condition is num_tokens_padded != num_tokens_unpadded
                    # but this scope is way too big and the consequences are unpredictable
                    num_reqs_padded = self._pad_query_start_loc_for_fia(
                        self.query_start_loc,
                        num_tokens_padded,
                        num_reqs_padded,
                        num_reqs,
                        cudagraph_mode,
                        batch_desc.num_reqs,
                    )

                (attn_metadata, spec_decode_common_attn_metadata) = self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded
                    if not (self.use_cp and self.pcp_manager.pcp_use_hybrid_attn)
                    else total_num_scheduled_tokens,
                    num_tokens_padded=num_tokens_padded,
                    num_reqs=num_reqs,
                    num_reqs_padded=num_reqs_padded,
                    max_query_len=max_num_scheduled_tokens,
                    ubatch_slices=ubatch_slices_attn,
                    logits_indices=logits_indices,
                    use_spec_decode=use_spec_decode,
                    num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                )

            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(
                scheduler_output,
                num_tokens_padded
                if not (self.use_cp and self.pcp_manager.pcp_use_hybrid_attn)
                else total_num_scheduled_tokens,
                intermediate_tensors,
            )

            #  -------------------------------------- Omni-new -------------------------------------------------
            if hasattr(self.model, "prepare_runner_inputs"):
                input_ids, positions = self.model.prepare_runner_inputs(
                    input_ids=input_ids,
                    positions=positions,
                    inputs_embeds=inputs_embeds,
                    req_ids=req_ids[:num_reqs],
                    num_computed_tokens=self.input_batch.num_computed_tokens_cpu[:num_reqs],
                    num_scheduled_tokens=num_scheduled_tokens_np[:num_reqs],
                    input_ids_buffer=self.input_ids.gpu[:num_tokens_padded],
                )
            #  -------------------------------------- Omni-new -------------------------------------------------

            # update global cos, sin
            update_cos_sin(positions)

        if self.dynamic_eplb:
            with record_function_or_nullcontext("EPLB weight D2D"):
                self.eplb_updator.forward_before()

        # Set cudagraph mode to none if calc_kv_scales is true.
        # KV scales calculation involves dynamic operations that are incompatible
        # with CUDA graph capture.
        if self.calculate_kv_scales:  # type: ignore[has-type]
            cudagraph_mode = CUDAGraphMode.NONE
            # Mark KV scales as calculated after the first forward pass
            self.calculate_kv_scales = False  # type: ignore[has-type]
        # prevent debugger is None
        if self.debugger is not None:
            dbg_cfg = getattr(self.debugger, "config", None)
            dump_level = str(getattr(dbg_cfg, "level", "L1")).upper() if dbg_cfg is not None else "L1"
            if dump_level in ("L0", "MIX"):
                self.debugger.start(model=self.model)
            else:
                self.debugger.start()
        if self.ascend_config.enable_async_exponential:
            self.sampler.do_async_exponential(
                b_s=logits_indices.shape[0],
                head_dim=self.model_config.get_vocab_size(),
                generators=self.input_batch.sampling_metadata.generators,
            )

        # Encoder-decoder models can only compile the pure decode steps where no
        # encoder inputs are present. Use eager for the first pass.
        num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)
        has_encoder_input = self.model_config.is_encoder_decoder and num_encoder_reqs > 0

        # Run forward pass
        clear_kv_metadata = self.speculative_config is None
        with (
            record_function_or_nullcontext("forward"),
            set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                aclgraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
                num_actual_tokens=scheduler_output.total_num_scheduled_tokens,
                model_instance=self.model,
                max_tokens_across_pcp=0 if self.pcp_size == 1 else self.pcp_manager.max_num_tokens_across_pcp,
                skip_compiled=has_encoder_input,
            ),
            self.maybe_get_kv_connector_output(
                scheduler_output,
                **(
                    {"defer_finalize": not clear_kv_metadata}
                ),
            ) as kv_connector_output,
        ):
            hidden_states = self._model_forward(
                num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs
            )
        with record_function_or_nullcontext("post process"):
            #  -------------------------------------- Omni-new -------------------------------------------------
            # [Omni] Map pending ropes metadata to req_ids.
            flush_pending_metadata = getattr(self.model, "flush_pending_metadata", None)
            if callable(flush_pending_metadata):
                flush_pending_metadata(req_ids[:num_reqs])

            hidden_states, multimodal_outputs = self.extract_multimodal_outputs(hidden_states)

            if multimodal_outputs is not None:
                keys_or_type = (
                    list(multimodal_outputs.keys())
                    if isinstance(multimodal_outputs, Mapping)
                    else type(multimodal_outputs)
                )
                logger.debug(f"[AR] execute_model: multimodal_outputs keys = {keys_or_type}")
            else:
                logger.debug("[AR] execute_model: multimodal_outputs is None")
            #  -------------------------------------- Omni-new -------------------------------------------------
            aux_hidden_states = None
            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = hidden_states
            if self.pcp_size > 1:
                # NOTE we must `slice` hidden_states because pcp_allgather_restore_idx
                # ignores the padding from CUDA Graph.
                hidden_states = self.pcp_manager.get_restore_hidden_states(hidden_states)
                if aux_hidden_states is not None:
                    aux_hidden_states = [
                        self.pcp_manager.get_restore_hidden_states(aux_hidden_states_pcp)
                        for aux_hidden_states_pcp in aux_hidden_states
                    ]

            #  -------------------------------------- Omni-new -------------------------------------------------
            self._maybe_update_prefix_cache(
                hidden_states=hidden_states,
                multimodal_outputs=multimodal_outputs,
                num_tokens_unpadded=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded,
            )
            #  -------------------------------------- Omni-new -------------------------------------------------

            if not self.broadcast_pp_output:
                # Common case.
                if not get_pp_group().is_last_rank:
                    # Return the intermediate tensors.
                    assert isinstance(hidden_states, IntermediateTensors)
                    hidden_states.kv_connector_output = kv_connector_output
                    self.kv_connector_output = kv_connector_output
                    if self.debugger is not None:
                        self.debugger.stop()
                        self.debugger.step()
                    return hidden_states
                if self.is_pooling_model:
                    # Return the pooling output.
                    output = self._pool(
                        hidden_states, num_scheduled_tokens, num_scheduled_tokens_np, kv_connector_output
                    )
                    output.kv_connector_output = kv_connector_output
                    if self.debugger is not None:
                        self.debugger.stop()
                        self.debugger.step()
                    return output

                sample_hidden_states = hidden_states[logits_indices]
                #  -------------------------------------- Omni-new -------------------------------------------------
                self._skip_logits_step = spec_decode_metadata is None and self._skip_logits_for_batch()
                if self._skip_logits_step:
                    # TTS prefill bypass: every request in this batch discards
                    # its sampled token (orchestrator echo overwrite), so skip
                    # the lm_head GEMM; a zero-logits tensor keeps every
                    # downstream shape/None consumer intact.
                    logits = torch.zeros(
                        (sample_hidden_states.shape[0], self.model_config.get_vocab_size()),
                        dtype=torch.float32,
                        device=sample_hidden_states.device,
                    )
                else:
                    # Try with sampling_metadata first; fall back to without for models that don't support it
                    try:
                        logits = self.model.compute_logits(
                            sample_hidden_states, sampling_metadata=self.input_batch.sampling_metadata
                        )
                    except TypeError:
                        logits = self.model.compute_logits(sample_hidden_states)
                #  -------------------------------------- Omni-new -------------------------------------------------
            else:
                # Rare case.
                assert not self.is_pooling_model

                if not get_pp_group().is_last_rank:
                    sample_hidden_states = hidden_states[logits_indices]
                    get_pp_group().send_tensor_dict(hidden_states.tensors, all_gather_group=get_tp_group())
                    logits = None
                else:
                    sample_hidden_states = hidden_states[logits_indices]
                    #  -------------------------------------- Omni-new -------------------------------------------------
                    # Try with sampling_metadata first; fall back to without for models that don't support it
                    try:
                        logits = self.model.compute_logits(
                            sample_hidden_states, sampling_metadata=self.input_batch.sampling_metadata
                        )
                    except TypeError:
                        logits = self.model.compute_logits(sample_hidden_states)
                    #  -------------------------------------- Omni-new -------------------------------------------------

                model_output_broadcast_data: dict[str, Any] = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()
                broadcasted = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert broadcasted is not None
                logits = broadcasted["logits"]

            # Apply structured output bitmasks if present
            self.execute_model_state = ExecuteModelState(
                scheduler_output,
                logits,
                spec_decode_metadata,
                spec_decode_common_attn_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                attn_metadata,
                positions,
                ec_connector_output,
                cudagraph_stats,
                batch_desc,
                multimodal_outputs, # Omni-specific
            )
            self.kv_connector_output = kv_connector_output

        # Now the batch has been launched we can wait for corrections from the
        # previous model forward without breaking async scheduling.
        if deferred_state_corrections_fn:
            deferred_state_corrections_fn()

        if self.vllm_config.model_config.enable_return_routed_experts and hasattr(self, "_positions_cpu"):
            self._omni_routed_experts_d2h(scheduler_output)

        return None

    #  -------------------------------------- Omni-new -------------------------------------------------
    def _attn_meta_cache_eligible(self) -> bool:
        # One-time config guards: any feature that changes what the upstream
        # builder produces or adds per-frame side effects disables the cache.
        return (
            self.speculative_config is None
            and not self.use_async_spec_decode
            and self.pcp_size == 1
            and not self.use_cp
            and not self._has_gdn
            and not self.enable_hamming_sparse
            and not self.is_mm_prefix_lm
            and not self.model_config.enable_return_routed_experts
            and not self.cache_config.kv_sharing_fast_prefill
            and len(self.kv_cache_config.kv_cache_groups) == 1
        )

    def _build_attention_metadata(
        self,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
        num_tokens_padded: int | None = None,
        num_reqs_padded: int | None = None,
        ubatch_slices=None,
        logits_indices: torch.Tensor | None = None,
        use_spec_decode: bool = False,
        for_cudagraph_capture: bool = False,
        num_scheduled_tokens=None,
        num_scheduled_tokens_np: "np.ndarray | None" = None,
        cascade_attn_prefix_lens=None,
    ):
        # [Omni] Fast path: in the steady single-request decode regime every
        # tensor field of the built metadata is a live view of a persistent
        # in-place buffer, so the dict can be reused; only the serialized
        # python lists (seq_lens_list / actual_seq_lengths_q) plus the padding
        # side effects must be refreshed each frame.
        cacheable = (
            self._attn_meta_cache_eligible()
            and not for_cudagraph_capture
            and not use_spec_decode
            and ubatch_slices is None
            and cascade_attn_prefix_lens is None
            and num_reqs == 1
            and num_tokens == 1
            and max_query_len == 1
            and num_tokens_padded is not None
            and num_reqs_padded is not None
            and self.attn_state == AscendAttentionState.DecodeOnly
        )
        key = None
        if cacheable:
            req_id = self.input_batch.req_ids[0]
            key = (req_id, num_tokens, num_tokens_padded, num_reqs, num_reqs_padded, max_query_len)
            cached = self._cached_attn_meta
            if cached is not None and cached[0] == key:
                self._refresh_cached_attention_metadata(
                    cached[1], num_reqs, num_reqs_padded, num_tokens, num_tokens_padded
                )
                return cached[1], None

        result = super()._build_attention_metadata(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            max_query_len=max_query_len,
            num_tokens_padded=num_tokens_padded,
            num_reqs_padded=num_reqs_padded,
            ubatch_slices=ubatch_slices,
            logits_indices=logits_indices,
            use_spec_decode=use_spec_decode,
            for_cudagraph_capture=for_cudagraph_capture,
            num_scheduled_tokens=num_scheduled_tokens,
            num_scheduled_tokens_np=num_scheduled_tokens_np,
            cascade_attn_prefix_lens=cascade_attn_prefix_lens,
        )
        if (
            FIA_PAD_STATE.enabled
            and FIA_PAD_STATE.capturing_pad
            and for_cudagraph_capture
            and num_tokens == 1
        ):
            # [minicpm-challenge: A-tier FIA pad] forge the bucket-1 FULL
            # capture metadata: kv=512 descriptor (seq_lens_list),
            # sparse_mode=0 (causal=False) + persistent wide mask, narrow
            # block_table view. Only this fresh capture-time object is
            # touched; the steady B4 cache is bypassed for captures and
            # keeps stock fields for every eager fallback frame.
            attn_metadata_cap, spec_common_cap = result
            if spec_common_cap is None:
                bucket = FIA_PAD_STATE.KV_PAD
                for meta in attn_metadata_cap.values():
                    if (
                        getattr(meta, "attn_state", None)
                        != AscendAttentionState.DecodeOnly
                    ):
                        continue
                    meta.causal = False
                    meta.attn_mask = FIA_PAD_STATE.mask_buf
                    if meta.block_tables is not None:
                        meta.block_tables = meta.block_tables[:, : bucket // 128]
                    meta.seq_lens_list = [bucket]
                FIA_PAD_STATE.pad_captured = True
                FIA_PAD_STATE.baked_kv = bucket
                logger.info(
                    "[fia_pad] bucket-1 capture forged (kv=%d, sparse0 + wide mask)",
                    bucket,
                )
        if key is not None:
            attn_metadata, spec_common = result
            if spec_common is None:
                self._cached_attn_meta = (key, attn_metadata)
        return result

    def _refresh_cached_attention_metadata(
        self,
        attn_metadata: PerLayerAttnMetadata,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens: int,
        num_tokens_padded: int,
    ) -> None:
        # Live buffers are refreshed in place by _prepare_inputs (and by
        # _pad_query_start_loc_for_fia) before this runs each frame.
        qsl_cpu = self.query_start_loc.cpu
        if callable(qsl_cpu):
            qsl_cpu = qsl_cpu()
        qsl_cpu = qsl_cpu[: num_reqs_padded + 1]
        seq_lens_live = self.optimistic_seq_lens_cpu[:num_reqs_padded]
        # Replicate the builder's padding side effects on the live buffers
        # exactly (empty slices are aten no-ops in the unpadded regime).
        slot_mapping = self.input_batch.block_table[0].slot_mapping.gpu[:num_tokens_padded]
        slot_mapping[num_tokens:num_tokens_padded].fill_(-1)
        blk_table_tensor = self.input_batch.block_table[0].get_device_tensor()[:num_reqs_padded]
        blk_table_tensor[num_reqs:num_reqs_padded].fill_(0)
        seen: set[int] = set()
        for meta in attn_metadata.values():
            if id(meta) in seen:
                continue
            seen.add(id(meta))
            # Serialized host copies: the only values the FULL-graph replay
            # update loop re-reads every frame. Must be rebuilt per frame.
            meta.actual_seq_lengths_q = qsl_cpu[1:].tolist()
            meta.seq_lens_list = seq_lens_live.tolist()
            # Live views: rebind so the object always tracks the persistent
            # buffers (identical to what a fresh build would install).
            meta.seq_lens = seq_lens_live
            meta.seq_lens_cpu = seq_lens_live
            if meta.query_start_loc is not None:
                # Keep the device copy byte-identical for any eager fallback
                # frame, without the per-frame pin_memory() allocation.
                meta.query_start_loc.copy_(qsl_cpu, non_blocking=True)

    def _update_full_graph_params_if_needed(
        self,
        forward_context,
        num_tokens_padded: int,
        positions: torch.Tensor | None,
    ) -> None:
        # [minicpm-challenge: A-tier FIA pad] replace the per-frame 20-layer
        # FIA graph_task_update loop (~2.3ms host) with an ExternalEvent
        # re-arm: the forged capture already bakes kv=bucket + sparse0 +
        # wide mask + narrow block_table view, and the in-graph mask rebuild
        # tracks klen, so steady frames need no re-issue at all. Promotion
        # (klen crossed the bucket) and demotion (new short request while a
        # high bucket is baked) run ONE forged stock update pass.
        st = FIA_PAD_STATE
        in_full = (
            forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
            and not forward_context.capturing
            and not self.use_sparse
            and not self.use_compress
        )
        if (
            st.enabled
            and not st.degraded
            and in_full
            and num_tokens_padded == 1
            and st.pad_captured
            and positions is not None
        ):
            events = get_graph_params().events.get(1)
            klen = int(self.optimistic_seq_lens_cpu[0])
            if events:
                if klen > st.baked_kv:
                    nb = next((b for b in st.BUCKETS if b >= klen), st.KV_MAX)
                    try:
                        self._fia_pad_forge_update(forward_context, nb, positions)
                        st.baked_kv = nb
                        logger.info("[fia_pad] promoted baked kv -> %d (klen=%d)", nb, klen)
                        return
                    except Exception:
                        logger.exception("[fia_pad] promote forge failed; stock update")
                elif st.baked_kv > st.KV_PAD and klen <= st.KV_PAD:
                    try:
                        self._fia_pad_forge_update(forward_context, st.KV_PAD, positions)
                        st.baked_kv = st.KV_PAD
                        logger.info("[fia_pad] demoted baked kv -> 512 (request switch)")
                        return
                    except Exception:
                        logger.exception("[fia_pad] demote forge failed; stock update")
                else:
                    for ev in events:
                        ev.record(self.update_stream)
                    if not st.skip_logged:
                        st.skip_logged = True
                        logger.info("[fia_pad] steady skip engaged (baked_kv=%d)", st.baked_kv)
                    return
        super()._update_full_graph_params_if_needed(forward_context, num_tokens_padded, positions)
        if st.enabled and not st.degraded and in_full and num_tokens_padded == 1:
            # A stock pass re-baked kv from the live metadata (fall-through or
            # failed forge); track it so the skip gate stays truthful until
            # the next forge heals the descriptor.
            st.baked_kv = int(self.optimistic_seq_lens_cpu[0])

    def _fia_pad_forge_update(self, forward_context, new_bucket: int, positions) -> None:
        # Temporarily swap the live per-layer metadata's kv descriptor and
        # block_table to the forged bucket values, run ONE stock update pass
        # (re-bakes the graph's kv + block_table descriptors), then restore
        # the python attributes -- the enqueued update ops keep the forged
        # device views (single request -> single bt row, so the narrow view
        # shares the persistent buffer's base pointer).
        attn_metadata = forward_context.attn_metadata
        ncols = new_bucket // 128
        # Validate every layer group BEFORE mutating anything: layers in the
        # same KV group share one metadata object (dedupe by id, or the
        # restore would write the forged view back over the original and the
        # block_table could never widen again), and a view too narrow for the
        # target bucket must bail out before any device call -- torch slicing
        # silently clamps it, the FIA update fails inside the graph task
        # group, poisons it (107033 on every later update) and kills the
        # engine.
        metas = []
        seen: set[int] = set()
        for meta in attn_metadata.values():
            if (
                getattr(meta, "attn_state", None)
                != AscendAttentionState.DecodeOnly
            ):
                continue
            if id(meta) in seen:
                continue
            seen.add(id(meta))
            bt = meta.block_tables
            if bt is not None and bt.shape[1] < ncols:
                raise ValueError(
                    f"block_table view too narrow for bucket {new_bucket} "
                    f"({bt.shape[1]} < {ncols} cols)"
                )
            metas.append(meta)
        saved = [(m, m.seq_lens_list, m.block_tables) for m in metas]
        for m in metas:
            m.seq_lens_list = [new_bucket]
            if m.block_tables is not None:
                m.block_tables = m.block_tables[:, :ncols]
        try:
            super()._update_full_graph_params_if_needed(forward_context, 1, positions)
        finally:
            for meta, seq_lens_saved, bt_saved in saved:
                meta.seq_lens_list = seq_lens_saved
                meta.block_tables = bt_saved
    #  -------------------------------------- Omni-new -------------------------------------------------

    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: Any,
    ):
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            model_sample = getattr(self.model, "sample", None)
            self.input_batch.update_async_output_token_ids()
            if logits is not None and callable(model_sample) and getattr(self.model, "prefer_model_sampler", False):
                # Apply logit bias (min_tokens, allowed_token_ids) before
                # the custom model sampler — the standard GPU sampler does
                # this internally, but prefer_model_sampler bypasses it.
                if hasattr(self.sampler, "logit_bias_state"):
                    self.sampler.logit_bias_state.apply_logit_bias(
                        logits,
                        self.input_batch.expanded_idx_mapping,
                        self.input_batch.idx_mapping_np,
                        self.input_batch.positions[self.input_batch.logits_indices],
                    )
                prepared_sampling_metadata = self._sampling_metadata_for_model_sampler(sampling_metadata)
                self._apply_duplex_sampling(logits, prepared_sampling_metadata)
                sampler_output = model_sample(logits, prepared_sampling_metadata)
                if sampler_output is not None:
                    return sampler_output
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        return super()._sample(logits, spec_decode_metadata)

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> OmniModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        #  -------------------------------------- Omni-new -------------------------------------------------
        kv_extracted_req_ids = getattr(self, "kv_extracted_req_ids", None)
        self.kv_extracted_req_ids = None
        combined_hidden_states = None
        combined_multimodal_outputs = None
        mm_cpu = {}
        #  -------------------------------------- Omni-new -------------------------------------------------


        if self.execute_model_state is None:
            # Nothing to do (PP non-final rank case), output isn't used.
            # receive sampled token ids from the last PP rank when using
            # async scheduling + pipeline parallelism so downstream code
            # (e.g., PCP input preparation) can access them.
            if self.use_async_scheduling and get_pp_group().world_size > 1:
                self._pp_receive_prev_sampled_token_ids_to_input_batch()
            if not kv_connector_output:
                return None  # noqa
            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return self.attach_omni_connector_output(EMPTY_MODEL_RUNNER_OUTPUT)

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return self.attach_omni_connector_output(output)

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            attn_metadata,
            positions,
            ec_connector_output,
            cudagraph_stats,
            batch_desc,
            multimodal_outputs, # Omni-Specific
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None
        hidden_seq_len = int(hidden_states.shape[0])
        scheduled_seq_len = int(scheduler_output.total_num_scheduled_tokens)

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            # here we are different from gpu_model_runner,
            # the apply_grammar_bitmask uses torch.compile to optimize this,ascend does not support it now
            logits_dtype = logits.dtype
            logits = logits.to("cpu").float()
            apply_grammar_bitmask(scheduler_output, grammar_output, self.input_batch, logits)
            logits = logits.to(self.device).to(logits_dtype)

        #  -------------------------------------- Omni-new -------------------------------------------------
        # Correct padding values of prompt_token_ids to match the logits vocabulary size.
        if logits is not None and not self.input_batch.sampling_metadata.no_penalties:
            smd = self.input_batch.sampling_metadata
            if smd.prompt_token_ids is not None:
                logits_vocab = logits.shape[-1]
                if self.input_batch.vocab_size > logits_vocab:
                    smd.prompt_token_ids = smd.prompt_token_ids.clamp(max=logits_vocab)

        # Drop min-tokens stop ids the head cannot emit (e.g. the text
        # tokenizer EOS folded into all_stop_token_ids on a narrow codec
        # talker head); they would index_put_ out of bounds (#4962).
        if logits is not None:
            sanitize_min_tokens_stop_ids(
                self.input_batch.sampling_metadata.logitsprocs,
                logits.shape[-1],
            )
        #  -------------------------------------- Omni-new -------------------------------------------------


        with record_function_or_nullcontext("sample_token"):
            if self._skip_logits_step:
                # Bypass: the real token is echo-overwritten by the orchestrator;
                # emit a fixed placeholder outside the tts_bos/tts_end id set so
                # the llm2tts span slice (prompt + this token) is unchanged.
                sampler_output = SamplerOutput(
                    sampled_token_ids=torch.full(
                        (sample_hidden_states.shape[0], 1),
                        _BYPASS_SKIP_LOGITS_EOS_ID,
                        dtype=torch.int32,
                        device=sample_hidden_states.device,
                    ),
                    logprobs_tensors=None,
                )
            else:
                sampler_output = self._sample(logits, spec_decode_metadata)

        if self.need_accepted_tokens:
            if self.sampling_done_event is None:
                self.sampling_done_event = torch.npu.Event()

            assert self.sampling_done_event is not None
            self.sampling_done_event.record()

        self.valid_sampled_token_count_gpu: torch.Tensor | None = None # type: ignore[no-redef]

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            self._draft_token_ids = self.propose_draft_token_ids(
                sampled_token_ids,
                self.input_batch.sampling_metadata,
                scheduler_output,
                spec_decode_metadata,
                spec_decode_common_attn_metadata,
                positions,
                scheduler_output.total_num_scheduled_tokens,
                hidden_states,
                aux_hidden_states,
                sample_hidden_states,
                batch_desc,
            )
            self._copy_draft_token_ids_to_cpu(scheduler_output)

        (
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = self._bookkeeping_sync(
            scheduler_output,
            sampler_output,
            logits,
            hidden_states,
            scheduler_output.total_num_scheduled_tokens,
            spec_decode_metadata,
        )

        with record_function_or_nullcontext("draft_token"):
            if self.speculative_config:
                use_padded_batch = (
                    self.speculative_config
                    and (self.speculative_config.use_eagle() or self.speculative_config.uses_draft_model())
                    and not self.speculative_config.disable_padded_drafter_batch
                )
                if use_padded_batch:
                    # EAGLE speculative decoding can use the GPU sampled tokens
                    # as inputs, and does not need to wait for bookkeeping to finish.
                    propose_draft_token_ids(sampler_output.sampled_token_ids)
                if self.speculative_config and not use_padded_batch:
                    # ngram and other speculative decoding methods use the sampled
                    # tokens on the CPU, so they are run after bookkeeping.
                    propose_draft_token_ids(valid_sampled_token_ids)

            # vLLM v0.18 defers KV connector finalization during target-model
            # forward when speculative decoding is enabled. Finalize here after
            # draft model runs so KV pool save/put can complete.
            if self.speculative_config is not None:
                self.finalize_kv_connector()

        routed_experts_lists = None
        if self.model_config.enable_return_routed_experts:
            capturer = self.routed_experts_capturer
            if capturer is not None and hasattr(self.input_batch, "num_tokens_no_spec"):
                routed_experts_lists = self._omni_extract_routed_experts(scheduler_output)

        #  -------------------------------------- Omni-new -------------------------------------------------
        engine_output_type, downstream_req_ids = self._resolve_pooler_payload_req_ids(req_ids_output_copy)
        sparse_mm_req_ids = self._sparse_mm_req_ids(multimodal_outputs)
        sparse_mm_index = {rid: i for i, rid in enumerate(sparse_mm_req_ids or [])}
        if engine_output_type == "audio" and sparse_mm_req_ids is not None:
            sparse_req_id_set = set(sparse_mm_req_ids)
            downstream_req_ids = [rid for rid in req_ids_output_copy if rid in sparse_req_id_set]
        needs_pooler_payload = len(downstream_req_ids) > 0
        downstream_req_id_set = set(downstream_req_ids)
        hidden_states_cpu = None
        req_hidden_states_cpu: dict[str, torch.Tensor] | None = None
        audio_sparse_output = engine_output_type == "audio" and sparse_mm_req_ids is not None
        needs_scheduled_hidden_payload = needs_pooler_payload and (
            self.omni_prefix_cache is None or not self._model_needs_full_prefix_hidden_states()
        )
        if needs_scheduled_hidden_payload:
            num_valid_tokens = min(
                int(scheduler_output.total_num_scheduled_tokens),
                int(hidden_states.shape[0]),
            )
            if audio_sparse_output:
                pass
            elif len(downstream_req_ids) == len(req_ids_output_copy):
                hidden_states_cpu = hidden_states[:num_valid_tokens].detach().to("cpu").contiguous()
            else:
                req_hidden_states_cpu = {}
        num_scheduled_tokens_np = getattr(self, "_omni_num_scheduled_tokens_np", None)
        if num_scheduled_tokens_np is None:
            req_ids = self.input_batch.req_ids
            num_scheduled_tokens_np = np.array(
                [scheduler_output.num_scheduled_tokens[rid] for rid in req_ids],
                dtype=np.int32,
            )
        query_start_loc_cpu = self.query_start_loc.cpu
        if callable(query_start_loc_cpu):
            query_start_loc_cpu = query_start_loc_cpu()

        pooler_output: list[dict[str, object]] | None = None
        if needs_pooler_payload:
            combined_hidden_states = None
            combined_multimodal_outputs = None
            mm_cpu = None
            if self.omni_prefix_cache is not None:
                (
                    combined_hidden_states,
                    combined_multimodal_outputs,
                ) = self._maybe_get_combined_prefix_cache_tensors(
                    hidden_states,
                    multimodal_outputs,
                    scheduler_output.num_scheduled_tokens,
                )
            if self.omni_prefix_cache is None or combined_multimodal_outputs is None:
                mm_cpu = build_mm_cpu(
                    flatten_payload(multimodal_outputs) if multimodal_outputs else multimodal_outputs
                )

            self._process_additional_information_updates(
                hidden_states,
                multimodal_outputs,
                num_scheduled_tokens_np,
                scheduler_output,
                combined_hidden_states,
                combined_multimodal_outputs,
                req_ids_filter=downstream_req_id_set,
            )

            if req_hidden_states_cpu is not None and combined_hidden_states is None:
                for rid in downstream_req_ids:
                    idx = req_id_to_index_output_copy[rid]
                    start = int(query_start_loc_cpu[idx])
                    sched = int(num_scheduled_tokens_np[idx])
                    end = start + sched
                    req_hidden_states_cpu[rid] = hidden_states[start:end].detach().to("cpu").contiguous()

            pooler_output = []
            for rid in req_ids_output_copy:
                if rid not in downstream_req_id_set:
                    pooler_output.append({})
                    continue
                idx = req_id_to_index_output_copy[rid]
                start = int(query_start_loc_cpu[idx])
                sched = int(num_scheduled_tokens_np[idx])
                end = start + sched
                payload: dict[str, object] = {}
                if not audio_sparse_output:
                    if req_hidden_states_cpu is not None and combined_hidden_states is None:
                        req_hidden_states = req_hidden_states_cpu[rid]
                    else:
                        req_hidden_states = self._resolve_req_hidden_states(
                            hidden_states_cpu,
                            combined_hidden_states,
                            rid,
                            start,
                            end,
                        )
                    payload["hidden"] = req_hidden_states

                mm_payload: dict[str, object] = {}
                if combined_multimodal_outputs or mm_cpu:
                    if combined_multimodal_outputs:
                        # Prefix cache enabled; all items have already been processed
                        # and split apart for each request as needed, and all tensors
                        # have already been detached to the CPU.  Lists are kept as
                        # passthrough data for consistent behavior in postprocess.
                        # Recurse into nested dicts so list-valued sub-keys (e.g.
                        # embed.tts_bos = [tensor]) are unwrapped to bare tensors
                        # at the leaves; downstream flatten_payload then yields a
                        # wire-clean dict[str, torch.Tensor].
                        def _unwrap_lists(v):
                            if isinstance(v, list):
                                return v[idx] if idx < len(v) else v[0]
                            if isinstance(v, dict):
                                return {k: _unwrap_lists(sv) for k, sv in v.items()}
                            return v

                        for mm_key in combined_multimodal_outputs.keys():
                            mm_payload[mm_key] = _unwrap_lists(combined_multimodal_outputs[mm_key][rid])
                    else:
                        for mm_key, mm_val in mm_cpu.items():
                            if mm_key in {"meta.req_id", "meta.sparse_audio"}:
                                continue
                            if audio_sparse_output and isinstance(mm_val, list):
                                sparse_idx = sparse_mm_index.get(rid)
                                if sparse_idx is None:
                                    continue
                                if sparse_idx >= len(mm_val):
                                    logger.warning(
                                        "Sparse multimodal payload mismatch for request %s: index %d >= %d.",
                                        rid,
                                        sparse_idx,
                                        len(mm_val),
                                    )
                                    continue
                                sparse_val = mm_val[sparse_idx]
                                mm_payload[mm_key] = (
                                    sparse_val.clone() if isinstance(sparse_val, torch.Tensor) else sparse_val
                                )
                                continue
                            mm_payload[mm_key] = to_payload_element(
                                element=mm_val,
                                idx=idx,
                                start=start,
                                end=end,
                                pass_lists_through=False,
                                seq_len=hidden_seq_len,
                                scheduled_seq_len=scheduled_seq_len,
                            )
                    payload.update(mm_payload)
                pooler_output.append(flatten_payload(payload))

        pooler_output = pooler_output or []
        if self._async_chunk and stage_sends_async_output(self.model_config):
            pooler_inter, pooler_client = partition_payload_list(pooler_output)
        else:
            # Non-async-chunk ships the full payload to the next stage via
            # inter_stage_outputs (the NPU runner has no separate full-payload
            # accumulate). #4527's (None, pooler_output) starved it. (PR #4792)
            pooler_inter, pooler_client = pooler_output, pooler_output

        # [Omni] Full-payload send-side accumulation. Mirrors gpu_ar_model_runner.py.
        if pooler_inter and self._should_accumulate_full_payload_output():
            with record_function_or_nullcontext("omni_output_builder:accumulate_full_payload_output"):
                for i, rid in enumerate(req_ids_output_copy):
                    req_state = self.requests.get(rid)
                    if req_state is not None and pooler_inter[i]:
                        self.accumulate_full_payload_output(rid, pooler_inter[i], req_state)

        inter_stage_outputs = self._build_multimodal_outputs(pooler_inter)
        multimodal_outputs = (
            inter_stage_outputs if pooler_client is pooler_inter else self._build_multimodal_outputs(pooler_client)
        )
        model_runner_output = OmniModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=None,
            multimodal_outputs=multimodal_outputs,
            inter_stage_outputs=inter_stage_outputs,
            kv_connector_output=kv_connector_output,
            ec_connector_output=ec_connector_output if self.supports_mm_inputs else None,
            cudagraph_stats=cudagraph_stats,
        )
        model_runner_output.kv_extracted_req_ids = kv_extracted_req_ids
        model_runner_output.routed_experts = routed_experts_lists
        with record_function_or_nullcontext("omni_output_builder:get_omni_connector_output"):
            model_runner_output.omni_connector_output = self.get_omni_connector_output()
        #  -------------------------------------- Omni-new -------------------------------------------------

        if self.ascend_config.profiling_chunk_config.enabled and hasattr(self, "_execution_start_time"):
            self._sync_device()
            model_runner_output.execution_time_ms = (time.perf_counter() - self._execution_start_time) * 1000.0

        if self.dynamic_eplb:
            with record_function_or_nullcontext("EPLB update"):
                self.eplb_updator.forward_end()

        if self.debugger is not None:
            self.debugger.stop()
            self.debugger.step()

        if self.need_accepted_tokens:
            assert self.sampling_done_event is not None
            with (
                record_function_or_nullcontext("async_state_update"),
                torch.npu.stream(global_stream()),
            ):
                global_stream().wait_event(self.sampling_done_event)
                self._update_states_after_model_execute(sampler_output.sampled_token_ids, scheduler_output)

        # In async scheduling + PP, broadcast sampled token ids from the
        # last PP rank so other PP ranks can receive them without going
        # through the scheduler/engine IPC path.
        if self.use_async_scheduling:
            pp = get_pp_group()
            if pp.world_size > 1 and pp.is_last_rank:
                self._pp_broadcast_prev_sampled_token_ids(sampler_output.sampled_token_ids)

        if not self.use_async_scheduling:
            return model_runner_output
        async_output = AsyncGPUModelRunnerOutput(
            model_runner_output=model_runner_output,
            sampled_token_ids=sampler_output.sampled_token_ids,
            logprobs_tensors=sampler_output.logprobs_tensors,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self.async_output_copy_stream,
            vocab_size=self.input_batch.vocab_size,
        )
        self.input_batch.set_async_sampled_token_ids(
            async_output.sampled_token_ids_cpu,
            async_output.async_copy_ready_event,
        )
        return async_output

    #  -------------------------------------- Omni-new -------------------------------------------------
    def _resolve_global_request_id(self, req_id: str) -> str:
        """Resolve global request ID from request state."""
        req_state = self.requests.get(req_id)
        if not req_state:
            return req_id

        add_info = self.model_intermediate_buffer.get(req_id, {})
        global_id = add_info.get("global_request_id")
        if global_id:
            if isinstance(global_id, list) and global_id:
                global_id = global_id[0]
            if isinstance(global_id, bytes):
                return global_id.decode("utf-8")
            return str(global_id)
        return req_id
    #  -------------------------------------- Omni-new -------------------------------------------------


# minicpm-challenge: profile-skip hook (Slice B2); see profile_skip.py.
try:
    from vllm_omni.platforms.npu.profile_skip import install_profile_skip

    install_profile_skip()
except Exception:
    pass
