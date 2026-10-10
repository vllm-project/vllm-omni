# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Registered MRV2 sampler for Lychee's text, merge, speech/control tick."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger
from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.model_executor.models.lychee_fd.output_ring import LycheeOutputRing
from vllm_omni.model_executor.output_snapshot import OutputCopyLifetimeError
from vllm_omni.worker_v2.omni_sampler import (
    OmniSampler,
    OmniSamplingContext,
    OmniSamplingOutput,
    StandardSample,
)
from vllm_omni.worker_v2.output_snapshot import RequestOutputSnapshot

if TYPE_CHECKING:
    from vllm_omni.worker_v2.model_states.lychee_model_state import LycheeModelState


logger = init_logger(__name__)


class LycheeSampler(OmniSampler):
    """Keep all three channels in the registered sampling result contract."""

    def __init__(self, base_sampler: Any, model_state: LycheeModelState) -> None:
        super().__init__(base_sampler)
        self.model_state = model_state
        self._sampling_context: OmniSamplingContext | None = None
        self._output_ring: LycheeOutputRing | None = None

    @contextmanager
    def set_sampling_context(self, context: OmniSamplingContext) -> Iterator[None]:
        if self._sampling_context is not None:
            raise RuntimeError("Lychee sampling transaction is already active")
        self._sampling_context = context
        try:
            yield
        except Exception as exc:
            if not isinstance(exc, OutputCopyLifetimeError):
                self.abort_unbound_output_leases()
            # Main KV has already advanced before this context is entered.
            # Postprocess, standard sampling, merge, side sampling, or output
            # construction failure must forbid reuse of every scheduled row.
            try:
                self.model_state.mark_primary_continuation_failed(input_batch=context.input_batch)
            except Exception:
                logger.exception("Failed to poison Lychee rows after post-main failure")
            raise
        finally:
            self._sampling_context = None

    def __call__(self, logits: torch.Tensor, input_batch: InputBatch):
        return self.base_sampler(self.model_state.constrain_primary_logits(logits, input_batch), input_batch)

    def abort_unbound_output_leases(self) -> None:
        if self._output_ring is not None:
            self._output_ring.abort_unbound()

    def _client_delta_rows(self, input_batch: InputBatch) -> tuple[bool, ...]:
        # Capture the transport choice now; async finalization must not read
        # request slots that the scheduler can cancel or reuse.
        buffer = getattr(self.model_state, "intermediate_buffer", None)
        if buffer is None:
            return (False,) * input_batch.num_reqs
        return tuple(
            isinstance(duplex := buffer.buffers[int(index)].get("duplex"), dict) and duplex.get("data_plane") is True
            for index in input_batch.idx_mapping_np[: input_batch.num_reqs]
        )

    @staticmethod
    def _output_finalizer(
        input_batch: InputBatch, *, compact: bool = False, client_delta_rows: tuple[bool, ...] | None = None
    ):
        # Capture row ownership before the runner reuses InputBatch metadata.
        final_rows = (
            tuple(range(input_batch.num_reqs))
            if compact
            else tuple(int(row) for row in input_batch.query_start_loc_np[1 : input_batch.num_reqs + 1].copy() - 1)
        )
        delta_rows = client_delta_rows or (False,) * len(final_rows)
        if len(delta_rows) != len(final_rows):
            raise ValueError("Lychee output transport flags do not match the request batch")

        def finalize(snapshot: dict[str, Any], num_sampled: list[int]) -> RequestOutputSnapshot:
            if len(num_sampled) != len(final_rows):
                raise ValueError("Lychee output counts do not match the request batch")
            payloads: list[dict[str, Any] | None] = []
            clients: list[dict[str, Any] | None] = []
            for row, count, delta in zip(final_rows, num_sampled, delta_rows):
                if count <= 0:
                    payloads.append(None)
                    clients.append(None)
                    continue
                payload = {key: value[int(row) : int(row) + 1] for key, value in snapshot.items()}
                payloads.append(payload)
                clients.append({f"chunk.{key}": value for key, value in payload.items()} if delta else payload)
            # Both transports need the decision fields. Native output suppresses
            # the pooler mirror, so the client still sees exactly one payload.
            return RequestOutputSnapshot(inter_stage=payloads, client=clients)

        return finalize

    def sample_step(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        req_states: RequestState,
        grammar_output: GrammarOutput | None,
        standard_sample: StandardSample,
    ) -> OmniSamplingOutput:
        context = self._sampling_context
        if context is None or context.input_batch is not input_batch:
            raise RuntimeError("Lychee sampling requires its current forward context")
        sampler_output, num_sampled, num_rejected = standard_sample(hidden_states, input_batch, grammar_output)
        sampler_output.sampled_token_ids = self.model_state.constrain_primary_sample(
            sampled_token_ids=sampler_output.sampled_token_ids,
            num_sampled=num_sampled,
            input_batch=input_batch,
        )
        with context.forward_context():
            outputs = self.model_state.continue_after_primary_sample(
                sampled_token_ids=sampler_output.sampled_token_ids,
                num_sampled=num_sampled,
                multimodal_outputs=context.multimodal_outputs or {},
                input_batch=input_batch,
                req_states=req_states,
            )
        delta_rows = self._client_delta_rows(input_batch)
        if self._output_ring is None:
            config = getattr(self.model_state, "vllm_config", None)
            self._output_ring = LycheeOutputRing(
                max_num_reqs=int(getattr(self.model_state, "max_num_reqs", input_batch.num_reqs)),
                capacity=2 * int(getattr(config, "max_concurrent_batches", 2)),
                device=next(iter(outputs.values())).device,
            )
        final_rows = input_batch.query_start_loc_np[1 : input_batch.num_reqs + 1].copy() - 1
        packed = self._output_ring.pack(outputs, final_rows)
        return OmniSamplingOutput(
            sampler_output,
            num_sampled,
            num_rejected,
            multimodal_outputs=packed,
            include_hidden_states=False,
            owns_multimodal_outputs=True,
            finalize_multimodal=self._output_finalizer(input_batch, compact=True, client_delta_rows=delta_rows),
        )
