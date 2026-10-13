# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Token-only Chatterbox output and Original guidance around upstream sampling."""

from collections.abc import MutableMapping, Sequence
from dataclasses import replace

import torch
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler as GpuSampler

from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig
from vllm_omni.worker_v2.omni_sampler import OmniSampler

SPEECH_START_TOKEN = ChatterboxConfig().start_speech_token
SPEECH_STOP_TOKEN = ChatterboxConfig().stop_speech_token
# Request ID -> (pair key, role, weight). The model owns registration/cleanup.
CFGPairRegistry = MutableMapping[str, tuple[str, str, float]]


class ChatterboxCFGSampler:
    """Blend paired rows, delegate sampling, and synchronize companion tokens.

    Registry roles are ``cond`` and ``uncond``. Both requests
    must be scheduled atomically and have identical speech token histories.
    A conditional request with zero weight needs no companion. Synthetic
    MRv2 warmup rows bypass pairing and still receive the speech token mask.
    Shared negative-infinity restrictions from V1 logit bias are preserved
    during blending, including min_tokens and allowed_token_ids masks.
    The wrapper deliberately owns no nn.Module state or request lifecycle.
    """

    def __init__(self, base_sampler: Sampler | GpuSampler, pairs: CFGPairRegistry):
        self.base_sampler = base_sampler
        self.pairs = pairs

    def prepare_logits(
        self,
        logits: torch.Tensor,
        req_ids: Sequence[str],
        *,
        warmup: bool = False,
        pair_metadata: Sequence[tuple[str, str, float]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """Resolve current batch rows by request identity and apply guidance."""
        if logits.ndim != 2 or logits.shape[0] != len(req_ids):
            raise ValueError("Chatterbox CFG requires one logits row per request")
        if logits.shape[1] <= SPEECH_STOP_TOKEN:
            raise ValueError("Chatterbox CFG requires the Original speech vocabulary")
        groups: dict[str, dict[str, tuple[int, float]]] = {}
        for row, req_id in enumerate(req_ids):
            # The runner also executes kernel warmup as synthetic requests
            # under its reserved _warmup_ namespace, with nonzero prefills.
            if warmup or req_id.startswith("_warmup_"):
                continue
            if pair_metadata is None and req_id not in self.pairs:
                raise ValueError(f"Missing Chatterbox CFG registration: {req_id}")
            key, role, weight = self.pairs[req_id] if pair_metadata is None else pair_metadata[row]
            if role not in ("cond", "uncond"):
                raise ValueError(f"Unsupported Chatterbox CFG role: {role}")
            group = groups.setdefault(key, {})
            if role in group:
                raise ValueError(f"Duplicate Chatterbox CFG role in pair: {key}")
            group[role] = (row, weight)
        row_pairs = []
        guided = logits.to(torch.float32, copy=True)
        for key, group in groups.items():
            if len(group) == 1 and "cond" in group and group["cond"][1] == 0.0:
                continue
            if len(group) != 2:
                raise ValueError(f"Missing Chatterbox CFG companion in batch: {key}")
            cond, weight = group["cond"]
            uncond, companion_weight = group["uncond"]
            if weight != companion_weight:
                raise ValueError(f"Mismatched Chatterbox CFG weights: {key}")
            conditional, unconditional = logits[cond].float(), logits[uncond].float()
            shared_mask = torch.isneginf(conditional) & torch.isneginf(unconditional)
            conditional = conditional.masked_fill(shared_mask, 0.0)
            unconditional = unconditional.masked_fill(shared_mask, 0.0)
            blended = conditional + weight * (conditional - unconditional)
            blended.masked_fill_(shared_mask, -torch.inf)
            guided[cond] = blended
            guided[uncond] = blended
            row_pairs.append((cond, uncond))
        guided[:, SPEECH_START_TOKEN] = -torch.inf
        guided[:, SPEECH_STOP_TOKEN + 1 :] = -torch.inf
        return guided, row_pairs

    def sample(
        self, logits: torch.Tensor, sampling_metadata: SamplingMetadata, per_req_extra_args: list[dict]
    ) -> SamplerOutput:
        """V1 model sampler entry point; preserve upstream sampling metadata."""
        pairs = [extra["chatterbox_cfg"] for extra in per_req_extra_args]
        guided, row_pairs = self.prepare_logits(logits, [pair[0] for pair in pairs], pair_metadata=pairs)
        output = self.base_sampler(logits=guided, sampling_metadata=sampling_metadata)
        for cond, uncond in row_pairs:
            output.sampled_token_ids[uncond].copy_(output.sampled_token_ids[cond])
        return output


class ChatterboxMRv2Sampler(OmniSampler):
    """Keep sampled tokens and owned metadata; S3Gen consumes no T3 hidden states."""

    def sample_step(self, hidden_states, input_batch, req_states, grammar_output, standard_sample):
        output = super().sample_step(hidden_states, input_batch, req_states, grammar_output, standard_sample)
        # make_omni_output creates fresh CPU flags every step, outside graph
        # capture. No hidden-state copy or graph-output lifetime guard is needed.
        return replace(output, include_hidden_states=False, owns_multimodal_outputs=True)


class ChatterboxMRv2CFGSampler(ChatterboxMRv2Sampler):
    """MRv2 adapter with upstream request lifecycle and logits processing."""

    def __init__(self, base_sampler: GpuSampler, pairs: CFGPairRegistry):
        super().__init__(base_sampler)
        self.cfg_sampler = ChatterboxCFGSampler(base_sampler, pairs)

    def __call__(self, logits: torch.Tensor, input_batch: InputBatch) -> SamplerOutput:
        # InputBatch.make_dummy supplies zero prefill lengths and random IDs.
        # Every real Chatterbox prompt contains conditioning and has positive
        # length. Inspect CPU metadata, avoiding a device sync on every token.
        guided, row_pairs = self.cfg_sampler.prepare_logits(
            logits, input_batch.req_ids, warmup=not input_batch.prefill_len_np.any()
        )
        output = self.base_sampler(guided, input_batch)
        for cond, uncond in row_pairs:
            output.sampled_token_ids[uncond].copy_(output.sampled_token_ids[cond])
        return output
