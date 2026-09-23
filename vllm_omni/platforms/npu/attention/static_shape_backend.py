# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend attention backend that captures a decode step with constant host args.

``FULL_DECODE_ONLY`` captures a whole decode step, but
``npu_fused_infer_attention_score`` takes ``actual_seq_lengths_kv`` as a host
``SymInt[]``, so its tiling is baked into the captured task. The KV length grows
by one every step, so vllm-ascend re-issues the op for every layer on every step
(``update_full_graph_params`` -> ``graph_task_update_begin``/``_end``). On the
20-layer, 768-hidden MiniCPM-o Talker that rebind *is* the step: it dominates
the stage-1 main-thread wall, which is what this module removes.

The mechanism lives in :mod:`static_shape_decode`; this module is only the seam. It
is a subclass rather than a patch because every place vllm-ascend needs to be
told about it is a factory vLLM-Omni already controls:

    AscendAttentionBackend.get_impl_cls / get_builder_cls   -> here
    acl_graph.update_full_graph_params                      -> goes through
                                                               get_impl_cls()
    NPUOmniPlatform.get_attn_backend_cls                    -> selects this
    NPUARModelRunner._capture_cudagraphs                    -> captures buckets
    ACLGraphWrapper.__call__ key + replay barrier           -> wrapped by
                                                               :func:`static_shape_decode.install_into_ascend_aclgraph`

That matters: vllm-ascend is used exactly as the image ships it, so anything
that would have to be applied *to* vllm-ascend is out of reach and a factory
seam has to be used instead.
"""

from __future__ import annotations

import torch
import torch_npu
from vllm.config import CUDAGraphMode
from vllm.logger import init_logger
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import (
    SWA_INT_MAX,
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
)

from vllm_omni.platforms.npu.attention import static_shape_decode, static_shape_prefill

logger = init_logger(__name__)


class OmniStaticShapeMetadataBuilder(AscendAttentionMetadataBuilder):
    """Attach the device-side state a captured step needs, and pick the bucket."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._omni_buckets = static_shape_decode.buckets_for(
            static_shape_decode.capacity_for(
                self.vllm_config.model_config.max_model_len,
                self.vllm_config.cache_config.block_size,
            ),
            self.vllm_config.cache_config.block_size,
        )
        spec = getattr(self.vllm_config, "speculative_config", None)
        self._omni_max_q_len = 1
        if spec is not None and getattr(spec, "num_speculative_tokens", 0):
            self._omni_max_q_len = 1 + int(spec.num_speculative_tokens)
        # The runtime bucket is part of the dispatch key: it is only answerable
        # where decode steps run as one FULL graph (FULL / FULL_DECODE_ONLY /
        # FULL_AND_PIECEWISE), because those are the only modes that capture
        # per-capacity decode graphs to look up. Under PIECEWISE or eager,
        # setting it sends the wrapper after a per-capacity entry dict that was
        # never captured, and the runtime capture attempt crashes vLLM's
        # capturing-window check ("Cannot capture CUDA graphs in an
        # inappropriate time"). With empty buckets build() leaves the bucket
        # unset and every consumer (wrapper __call__, static-shape attention,
        # prefill hook) takes its stock path -- the behaviour this stage would
        # have had without the static-shape seam at all.
        mode = self.vllm_config.compilation_config.cudagraph_mode
        if self._omni_buckets and mode not in (
            CUDAGraphMode.FULL,
            CUDAGraphMode.FULL_DECODE_ONLY,
            CUDAGraphMode.FULL_AND_PIECEWISE,
        ):
            logger.info(
                "[minicpmo] static-shape decode graphs need FULL decode graphs "
                "(cudagraph_mode=%s); keeping the stock attention path",
                mode,
            )
            self._omni_buckets = ()

    def build(self, *args, **kwargs):
        attn_metadata = super().build(*args, **kwargs)
        common = kwargs.get("common_attn_metadata")
        if common is None:
            # Positional call: (common_prefix_len, common_attn_metadata, ...).
            common = next((a for a in args if hasattr(a, "block_table_tensor")), None)
        if common is None:
            static_shape_decode.set_runtime_bucket(None)
            return attn_metadata

        # `seq_lens` on the metadata is a CPU tensor; a captured graph has to
        # refresh its bias from the live device buffer instead.
        attn_metadata.seq_lens_device = common.seq_lens
        # build() materializes padded copies of seq_lens/block_table for the FIA
        # batch-size check when the batch is full. Those are fresh tensors every
        # step, so a graph cannot bake them in. The block table is the one that
        # is not caught by the row check in `_static_shape_applies`; identity against
        # the runtime's own persistent buffer is the reliable test.
        attn_metadata.fia_inputs_persistent = attn_metadata.block_tables is common.block_table_tensor
        attn_metadata.static_shape_mask_recorded = False
        attn_metadata.static_shape_pse_recorded = False

        # Whether this step is a decode cannot be read off `attn_state`. A step
        # that schedules several tokens per request comes back labelled
        # `ChunkedPrefill`: vllm-ascend calls it `SpecDecoding` and then
        # rewrites that to `ChunkedPrefill` for every proposer except `mtp`
        # (`model_runner_v1._build_attn_state`). The shape is the honest test --
        # the same number of query positions for every request, none of them
        # still consuming a prompt -- and it is what the captured graph and the
        # causal bias actually require.
        attn_metadata.static_shape_decode_step = (
            attn_metadata.attn_state == AscendAttentionState.DecodeOnly
            or self._is_uniform_multi_query_decode(attn_metadata, common)
        )

        # Settle the capacity before the forward runs: it is part of the ACL
        # graph dispatch key, so the wrapper has to be able to look it up.
        if self._omni_buckets and attn_metadata.static_shape_decode_step:
            seq_lens_list = attn_metadata.seq_lens_list or [1]
            max_seq_len = getattr(common, "max_seq_len", None) or max(seq_lens_list)
            static_shape_decode.set_runtime_bucket(
                static_shape_decode.select_bucket(int(max_seq_len), self._omni_buckets)
            )
        else:
            static_shape_decode.set_runtime_bucket(None)
            # A single fresh prefill inside a captured bucket replays a graph
            # instead of paying ~18 ms of eager dispatch; the hook refuses
            # everything else (chunked, batched, oversized, other stages).
            static_shape_prefill.maybe_mark_step(
                attn_metadata,
                AscendAttentionState.PrefillNoCache,
                int(self.vllm_config.cache_config.block_size),
            )
        return attn_metadata

    def _is_uniform_multi_query_decode(self, attn_metadata, common) -> bool:
        if self._omni_max_q_len <= 1:
            return False
        cumulative = getattr(attn_metadata, "actual_seq_lengths_q", None)
        if not cumulative:
            return False
        rows = len(cumulative)
        q_len = int(cumulative[0])
        if q_len <= 1 or q_len > self._omni_max_q_len:
            return False
        if int(cumulative[-1]) != rows * q_len:
            return False
        for index, value in enumerate(cumulative):
            if int(value) != (index + 1) * q_len:
                return False
        is_prefilling = getattr(common, "is_prefilling", None)
        if is_prefilling is None:
            return False
        num_reqs = int(getattr(common, "num_reqs", rows))
        return not bool(is_prefilling[:num_reqs].any())


class OmniStaticShapeAttentionBackendImpl(AscendAttentionBackendImpl):
    """Decode attention whose host-side arguments never change between steps."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # vllm-ascend builds differ here: newer ones set these quant flags on
        # the impl base class, the build this image ships does not. Default
        # them so the constant-tiling path stays on the non-quant branch.
        for _flag in ("enable_c8_quant", "enable_hamming_sparse"):
            if not hasattr(self, _flag):
                setattr(self, _flag, False)
        self._omni_capacity = static_shape_decode.capacity_for(
            self.vllm_config.model_config.max_model_len,
            self.vllm_config.cache_config.block_size,
        )
        # +2 matches the query_start_loc buffer, which carries the FIA padding
        # request on a full batch.
        self._omni_max_rows = int(self.vllm_config.scheduler_config.max_num_seqs) + 2
        self._omni_buckets = static_shape_decode.buckets_for(
            self._omni_capacity, self.vllm_config.cache_config.block_size
        )
        # A step that schedules several tokens per sequence (the Talker's
        # multi-frame decode declares them through a speculative_config) is
        # still a uniform decode, and the fixed-capacity mask covers it as
        # long as the bias is causal among those queries too.
        # Every length up to the declared one, because the scheduler may trim a
        # step to fewer tokens than the config asks for, and a bias buffer that
        # does not already exist cannot be allocated during capture -- it would
        # come out of the graph's private pool.
        spec = getattr(self.vllm_config, "speculative_config", None)
        max_q_len = 1
        if spec is not None and getattr(spec, "num_speculative_tokens", 0):
            max_q_len = 1 + int(spec.num_speculative_tokens)
        self._omni_q_lens = tuple(range(1, max_q_len + 1))
        if self._omni_capacity:
            # At model load, not during capture: an allocation made while a graph
            # is capturing comes out of that graph's private pool.
            static_shape_decode.prewarm(
                self._omni_max_rows,
                self.num_heads,
                self._omni_buckets,
                self.vllm_config.model_config.dtype,
                torch.device("npu", torch.npu.current_device()),
                self._omni_q_lens,
            )

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        # The vllm-ascend build this targets dropped ``num_dcp_pcp_tokens`` from
        # the upstream signature (PCP removed from MRV1, see #12592).
        draft_attn_metadatas=None,
    ):
        """Skip the per-step rebind for a step that captured no updatable tasks."""
        if static_shape_decode.is_static_shape_graph(num_tokens):
            metadata = forward_context.attn_metadata
            first = next(iter(metadata.values())) if isinstance(metadata, dict) else metadata
            static_shape_decode.sync_captured_inputs(
                num_tokens,
                getattr(first, "seq_lens_device", None),
                getattr(first, "block_tables", None),
            )
            return
        AscendAttentionBackendImpl.update_graph_params(
            update_stream,
            forward_context,
            num_tokens,
            vllm_config,
            speculative_config,
            draft_attn_metadatas,
        )

    def _static_shape_applies(self, attn_metadata, block_table, block_size) -> bool:
        """Whether this layer can capture a decode step with constant host args."""
        if not self._omni_capacity:
            return False
        if not getattr(attn_metadata, "static_shape_decode_step", False):
            return False
        if not attn_metadata.causal or not getattr(attn_metadata, "fia_inputs_persistent", False):
            return False
        if getattr(attn_metadata, "seq_lens_device", None) is None or block_table is None:
            return False
        # Every one of these steers the op onto a different argument set that the
        # constant-tiling assumption does not cover.
        if self.sinks is not None or self.sliding_window is not None:
            return False
        if self.enable_c8_quant or self.enable_hamming_sparse:
            return False
        if _EXTRA_CTX.is_draft_model:
            return False
        capacity = static_shape_decode.current_capacity()
        if capacity is None or capacity not in self._omni_buckets:
            return False
        if block_size * block_table.shape[1] < capacity:
            return False
        # FIA takes its batch size from actual_seq_lengths, so the block table and
        # the bias have to have exactly that many rows, not merely enough.
        rows = len(attn_metadata.actual_seq_lengths_q)
        if block_table.shape[0] != rows or attn_metadata.seq_lens_device.shape[0] != rows:
            return False
        if rows > self._omni_max_rows:
            return False
        # The bias is causal among the step's own queries as well as against
        # the sequence length, so several queries per request are fine -- but
        # only if every request contributed the *same* number, since the shift
        # baked into the mask is one number for the whole step.
        # actual_seq_lengths_q is a cumulative sum.
        if self._uniform_query_len(attn_metadata) is None:
            return False
        return True

    def _uniform_query_len(self, attn_metadata) -> int | None:
        """Queries per request when every request contributed the same count."""
        cumulative = attn_metadata.actual_seq_lengths_q
        rows = len(cumulative)
        if rows == 0:
            return None
        q_len = int(cumulative[0])
        if q_len <= 0 or q_len not in self._omni_q_lens:
            return None
        if int(cumulative[-1]) != rows * q_len:
            return None
        for index, value in enumerate(cumulative):
            if int(value) != (index + 1) * q_len:
                return None
        return q_len

    def _static_shape_graph_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        num_tokens: int,
    ):
        """Capture one attention layer with no updatable task attached to it.

        The general FIA serves this step; it needs its host arguments constant
        to be replayable, so the op declares the full KV capacity and the live
        sequence length arrives through ``pse_shift``, a device tensor the
        graph refreshes for itself.
        """
        capacity = static_shape_decode.current_capacity()
        rows = len(attn_metadata.actual_seq_lengths_q)
        q_len = self._uniform_query_len(attn_metadata) or 1
        pse_all = static_shape_decode.get_pse_buffer(
            self._omni_max_rows, self.num_heads, capacity, query.dtype, query.device, q_len
        )
        pse = pse_all[:rows]

        seq_lens_device = attn_metadata.seq_lens_device

        if not attn_metadata.static_shape_mask_recorded:
            attn_metadata.static_shape_mask_recorded = True
            static_shape_decode.remember_captured_inputs(
                num_tokens,
                capacity,
                seq_lens_device,
                block_table,
                getattr(attn_metadata, "slot_mapping", None),
            )
            logger.info(
                "[minicpmo] static-shape decode capture: graph_size=%d rows=%d q_len=%d "
                "kv_capacity=%d heads=%d attention=FIA",
                num_tokens,
                rows,
                q_len,
                capacity,
                self.num_heads,
            )

        # The bias is per step, not per layer, and only the FIA path reads it,
        # so it is recorded by the first layer that actually wants it.
        if not attn_metadata.static_shape_pse_recorded:
            static_shape_decode.emit_mask_refresh(pse, seq_lens_device, self.num_heads, capacity, q_len)
            attn_metadata.static_shape_pse_recorded = True

        softmax_lse = static_shape_decode.keep_alive(
            num_tokens, capacity, torch.empty(1, dtype=query.dtype, device=query.device)
        )
        fia_kwargs = dict(
            query=query,
            key=key,
            value=value,
            pse_shift=pse,
            atten_mask=None,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
            actual_seq_lengths_kv=[capacity] * rows,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=0,
            pre_tokens=SWA_INT_MAX,
            next_tokens=0,
        )
        workspace = static_shape_decode.graph_workspace(
            num_tokens,
            capacity,
            (rows, self.num_heads, self.num_kv_heads, self.head_size, block_size, capacity, query.dtype),
            lambda: torch_npu._npu_fused_infer_attention_score_get_max_workspace(**fia_kwargs),
        )
        torch_npu.npu_fused_infer_attention_score.out(workspace=workspace, out=[output, softmax_lse], **fia_kwargs)
        return output.view(num_tokens, self.num_heads, self.head_size), num_tokens

    def _static_shape_prefill_fia(self, query, key, value, attn_metadata, output):
        """Capture one prefill attention layer with all-constant host arguments.

        Runs only while a graph is being captured (and during its warmups):
        the runtime replays the recorded tasks and never re-enters here. The
        call mirrors the eager ``PrefillNoCache`` branch exactly -- causal FIA
        against the singleton mask, KV length = query length, no block table --
        with the workspace and the softmax LSE held outside the graph pool, so
        the capture strands nothing (an allocation made while capturing comes out
        of the graph's private pool).
        """
        bucket = int(attn_metadata.static_shape_prefill_bucket)
        fia_kwargs = dict(
            query=query[:bucket],
            key=key[:bucket],
            value=value[:bucket],
            atten_mask=attn_metadata.attn_mask,
            input_layout="TND",
            block_size=128,
            actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
            actual_seq_lengths_kv=attn_metadata.actual_seq_lengths_q,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )
        workspace = static_shape_prefill.workspace(
            bucket,
            lambda: torch_npu._npu_fused_infer_attention_score_get_max_workspace(**fia_kwargs),
        )
        lse = static_shape_prefill.lse_buffer(bucket, query.dtype, query.device)
        torch_npu.npu_fused_infer_attention_score.out(workspace=workspace, out=[output, lse], **fia_kwargs)
        return output.view(bucket, self.num_heads, self.head_size), bucket

    def full_graph_fia(self, query, key, value, attn_metadata, output, layer=None):
        if getattr(attn_metadata, "static_shape_prefill_bucket", None) is not None:
            return self._static_shape_prefill_fia(query, key, value, attn_metadata, output)
        if self._omni_capacity:
            fixed_key, fixed_value, block_size, block_table, _ = self._get_fia_params(key, value, attn_metadata)
            if self._static_shape_applies(attn_metadata, block_table, block_size):
                return self._static_shape_graph_fia(
                    query,
                    fixed_key,
                    fixed_value,
                    attn_metadata,
                    output,
                    block_table,
                    block_size,
                    attn_metadata.actual_seq_lengths_q[-1],
                )
        # `_get_fia_params` only reshapes cache views, so recomputing it in the
        # base implementation costs nothing beyond two `view` calls, and only on
        # the paths static-shape declines.
        return super().full_graph_fia(query, key, value, attn_metadata, output, layer)


class OmniStaticShapeAttentionBackend(AscendAttentionBackend):
    """The stock Ascend backend with the static-shape decode impl and builder."""

    @staticmethod
    def get_impl_cls() -> type[OmniStaticShapeAttentionBackendImpl]:
        return OmniStaticShapeAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type[OmniStaticShapeMetadataBuilder]:
        return OmniStaticShapeMetadataBuilder
