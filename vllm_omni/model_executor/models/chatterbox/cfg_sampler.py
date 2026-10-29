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

from vllm_omni.transformers_utils.configs.chatterbox import START_SPEECH_TOKEN, STOP_SPEECH_TOKEN
from vllm_omni.worker_v2.omni_sampler import OmniSampler

# Request ID -> (pair key, role, weight). The model owns registration/cleanup.
CFGPairRegistry = MutableMapping[str, tuple[str, str, float]]


class ChatterboxCFGSampler:
    """Blend paired rows, delegate sampling, and synchronize companion tokens.

    Registry roles are ``cond`` and ``uncond``. Both requests
    must be scheduled atomically and have identical speech token histories.
    A conditional request with zero weight needs no companion. Synthetic
    MRv2 warmup rows bypass pairing and still receive the speech token mask.
    A negative-infinity restriction from V1 logit bias on either row, such as
    a min_tokens or allowed_token_ids mask, bans that id from the blend. The
    companion's sampling parameters are a clone of the conditional request's
    with only ``extra_args`` replaced (``ExpandedPrompt.apply_overrides``), so
    both rows carry the same restrictions and the blend bans no id that
    sampling the conditional row alone would allow.
    The stage that owns the registry registers and releases its pairs.
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
        if logits.shape[1] <= STOP_SPEECH_TOKEN:
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
            # An id either row restricts is banned from the blend: left in,
            # cond - (-inf) would give it a +inf logit.
            banned = torch.isneginf(conditional) | torch.isneginf(unconditional)
            conditional = conditional.masked_fill(banned, 0.0)
            unconditional = unconditional.masked_fill(banned, 0.0)
            blended = conditional + weight * (conditional - unconditional)
            blended.masked_fill_(banned, -torch.inf)
            guided[cond] = blended
            guided[uncond] = blended
            row_pairs.append((cond, uncond))
        # Two deliberate departures from upstream's T3.inference, which
        # samples over all 8194 ids; both renormalize the distribution over
        # the remaining ids. The start-of-speech id is never sampled: every
        # placeholder prompt id is that token, so the repetition penalty
        # reaches no real speech token through the prompt. Ids above the stop
        # token are not speech codes: the stage handoff drops every id at or
        # above speech_token_limit, because S3Gen has no embedding for it, so
        # a sampled one would silently vanish from the audio; masking makes
        # it impossible instead. Measured with both masks applied: the H100
        # recipe's matched comparison with upstream (Chatterbox-H100.md,
        # "Comparison with the canonical upstream implementation") keeps
        # Original's WER and SIM within its predeclared guards.
        guided[:, START_SPEECH_TOKEN] = -torch.inf
        guided[:, STOP_SPEECH_TOKEN + 1 :] = -torch.inf
        return guided, row_pairs

    def sample(
        self, logits: torch.Tensor, sampling_metadata: SamplingMetadata, per_req_extra_args: list[dict]
    ) -> SamplerOutput:
        """V1 model sampler entry point; preserve upstream sampling metadata."""
        # Every row is a scheduled request: prepare_original_prompt tags each
        # one, guided or not. The V1 runner's warmup (_dummy_sampler_run)
        # calls its own sampler directly and never reaches this method.
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
        # capture, so the sampler owns them as they are.
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
