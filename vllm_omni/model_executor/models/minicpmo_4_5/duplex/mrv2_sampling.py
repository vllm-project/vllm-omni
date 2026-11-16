# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Adapt the MiniCPM duplex policy to MRv2's request slots and sampler output."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import torch
from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from vllm_omni.model_executor.duplex_sampling import DuplexSamplingHelper
from vllm_omni.utils.device_copy import index_to_device
from vllm_omni.worker_v2.omni_sampler import OmniSampler


def _keep_thinker_payload(payload: dict[str, Any], num_sampled: list[int]) -> dict[str, Any]:
    return payload


def _v1_shaped_runner(input_batch: Any, infos: dict[str, Any]) -> SimpleNamespace:
    """Present MRv2 request params in the V1 runner shape DuplexSamplingHelper reads."""
    params = [
        (infos.get(str(request_id)) or {}).get("sampling_params") or SimpleNamespace()
        for request_id in input_batch.req_ids
    ]
    return SimpleNamespace(
        input_batch=SimpleNamespace(
            req_ids=input_batch.req_ids,
            temperature_cpu=[getattr(sp, "temperature", None) for sp in params],
            top_k_cpu=[getattr(sp, "top_k", None) for sp in params],
            top_p_cpu=[getattr(sp, "top_p", None) for sp in params],
        ),
        model_intermediate_buffer=infos,
        requests={
            str(request_id): SimpleNamespace(sampling_params=sp) for request_id, sp in zip(input_batch.req_ids, params)
        },
    )


class MiniCPMO45DuplexSampler(OmniSampler):
    """Keep policy RNG/session state separate from the stock MRv2 sampler."""

    def __init__(self, base_sampler: Any, model: Any) -> None:
        super().__init__(base_sampler)
        self.model = model
        self.generators: dict[str, torch.Generator] = {}
        self._helper = DuplexSamplingHelper()
        self._histories: dict[str, tuple[tuple[int, int, int | None], list[int]]] = {}
        self._pending_history: Any = None
        model._mrv2_duplex_sampler = self

    def forget_requests(self, request_ids) -> None:
        for request_id in request_ids:
            self.generators.pop(request_id, None)
            self._histories.pop(request_id, None)
            self._helper.active_request_ids.discard(request_id)
            getattr(self.model, "_mrv2_sampling_infos", {}).pop(request_id, None)

    def _commit_history(self):
        pending, self._pending_history = self._pending_history, None
        if pending is None:
            return
        entries, host, ready = pending
        if ready is not None:
            # Wait only for the previous sample, never for this step's forward.
            ready.synchronize()
        for (req_id, cached), token in zip(entries, host.tolist()):
            if self._histories.get(req_id) is cached:
                cached[1].append(int(token))

    def _defer_history(self, rows, sampled_ids):
        pending = getattr(self.model, "_minicpmo45_duplex_pending_samples", None)
        if pending is not None and pending.row_idxs == list(range(len(rows))):
            # The normal policy already snapshots final IDs with its latch
            # updates. Share that owned host buffer/event; do not copy twice.
            host, ready = pending.host[:, 0], pending.event
        else:
            # Mixed lookahead/re-emitted EOS rows or the synchronous policy
            # fallback still need one owned snapshot of the final result.
            tokens = sampled_ids.reshape(-1)
            host = torch.empty(tokens.shape, dtype=tokens.dtype, pin_memory=tokens.device.type == "cuda")
            host.copy_(tokens, non_blocking=True)
            ready = torch.Event(device=tokens.device) if tokens.device.type == "cuda" else None
            if ready is not None:
                ready.record()
        self._pending_history = ([(row.request_id, self._histories[row.request_id]) for row in rows], host, ready)

    def _metadata(self, input_batch, rows, infos, device):
        """Reuse prior sampled IDs instead of synchronously reading the device ledger.

        Without speculative decoding, the CPU scheduled lengths are exact.
        Only admission/replay with an uncached output prefix needs a ledger read.
        The normal prefill and decode paths use host metadata and the previous
        step's asynchronous sampled-ID snapshot.
        """
        self._commit_history()
        histories, params, generators = [], [], {}
        for local_row, row in enumerate(rows):
            slot = int(input_batch.idx_mapping_np[row.row_idx])
            prompt_len = int(self.req_states.prompt_len.np[slot])
            end = int(input_batch.num_computed_tokens_np[row.row_idx]) + int(
                input_batch.num_scheduled_tokens[row.row_idx]
            )
            key = (slot, prompt_len, row.seq)
            previous_key, history = self._histories.get(row.request_id, (None, []))
            if key != previous_key:
                history = []
            elif end - prompt_len < len(history):
                del history[max(0, end - prompt_len) :]
            start = prompt_len + len(history)
            if end > start:
                history.extend(self.req_states.all_token_ids.gpu[slot, start:end].tolist())
            self._histories[row.request_id] = (key, history)
            histories.append(list(history))
            sp = infos[row.request_id]["sampling_params"]
            params.append(sp)
            if sp.seed is not None:
                generator = self.generators.get(row.request_id)
                if generator is None:
                    generator = torch.Generator(device=device).manual_seed(sp.seed)
                    self.generators[row.request_id] = generator
                generators[local_row] = generator
        return SimpleNamespace(
            output_token_ids=histories,
            generators=generators,
            temperature=torch.tensor([sp.temperature for sp in params]),
            top_k=torch.tensor([sp.top_k for sp in params]),
            top_p=torch.tensor([sp.top_p for sp in params]),
            all_greedy=all(sp.temperature <= 0 for sp in params),
        )

    def sample_step(self, hidden_states, input_batch, req_states, grammar_output, standard_sample):
        # Publish the Thinker payload through the MRv2 output-channel contract:
        # its latent row ledger is inter-stage only. The full-payload fallback
        # mirrors it into multimodal_output too, so every llm2tts row would be
        # accumulated twice and the per-unit ledger lookup would fail.
        output = super().sample_step(hidden_states, input_batch, req_states, grammar_output, standard_sample)
        return replace(output, include_hidden_states=False, finalize_multimodal=_keep_thinker_payload)

    def __call__(self, logits: torch.Tensor, input_batch: Any) -> SamplerOutput:
        infos = getattr(self.model, "_mrv2_sampling_infos", {})
        helper = self._helper
        runner = _v1_shaped_runner(input_batch, infos)
        for request_id in input_batch.req_ids:
            helper.refresh_active_request(runner, request_id)
        rows = helper.rows(runner)
        if rows and input_batch.num_draft_tokens:
            raise NotImplementedError("MiniCPM-o MRv2 duplex sampling does not support speculative decoding")
        # Partial prefills are discarded by the runner and must not mutate the
        # policy latches or advance their generators.
        rows = tuple(
            row
            for row in rows
            if not input_batch.is_prefilling_np[row.row_idx]
            or int(input_batch.num_computed_prefill_tokens_np[row.row_idx])
            + int(input_batch.num_scheduled_tokens[row.row_idx])
            >= int(input_batch.prefill_len_np[row.row_idx])
        )
        metadata = self._metadata(input_batch, rows, infos, logits.device) if rows else None
        row_idxs = [row.row_idx for row in rows]
        if logits.shape[0] == input_batch.num_reqs and row_idxs == list(range(input_batch.num_reqs)):
            # Every row follows the policy: skip the index upload and the
            # full-vocabulary gather. As on V1, the policy's force-listen mask
            # then lands on the logits a fallback sampler would read.
            selected, policy_logits = None, logits
        else:
            selected = index_to_device(row_idxs, logits.device)
            policy_logits = logits.index_select(0, selected)
        self.model.prepare_duplex_sampling(
            policy_logits, metadata, tuple(replace(row, row_idx=i) for i, row in enumerate(rows))
        )
        if not rows:
            return self.base_sampler(logits, input_batch)
        policy_output = self.model.sample(policy_logits, metadata)
        if policy_output is None:
            return self.base_sampler(logits, input_batch)
        self._defer_history(rows, policy_output.sampled_token_ids)
        if len(rows) != input_batch.num_reqs:
            # The stock sampler owns counts/logprobs for any ordinary rows.
            output = self.base_sampler(logits, input_batch)
            output.sampled_token_ids.index_copy_(0, selected, policy_output.sampled_token_ids.long())
            return output
        counts, rejected = get_num_sampled_and_rejected(
            input_batch.seq_lens.new_ones(input_batch.num_reqs),
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.req_states.prefill_len.gpu,
        )
        return SamplerOutput(
            sampled_token_ids=policy_output.sampled_token_ids.long(),
            logprobs_tensors=policy_output.logprobs_tensors,
            num_nans=None,
            num_sampled=counts,
            num_rejected=rejected,
        )
