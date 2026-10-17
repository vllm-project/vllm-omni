# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Frozen V1 sampler for B1 before/after validation, not a serving implementation.

Methods copied from CosyVoice3Model at
4c5541cfc17143f80bdb89bbb7a5840b08bb52c6. Do not update alongside the optimized
sampler: this reference must retain the old parameter reads and RNG behavior.
"""

from collections.abc import Sequence
from dataclasses import replace

import torch
import torch.nn as nn

# Initialize Omni's process-wide RNG patch before binding vLLM's sampler.
import vllm_omni  # isort: skip # noqa: F401

from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import random_sample
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.cosyvoice3.runtime import cosyvoice3_standard_sampling

BASELINE_COMMIT = "4c5541cfc17143f80bdb89bbb7a5840b08bb52c6"


class ReferenceCosyVoice3Sampler(nn.Module):
    _sampling_eps = 1e-5

    @staticmethod
    def _req_scalar(param: torch.Tensor | None, req_idx: int, default: float | int) -> float | int:
        if param is None or param.numel() == 0:
            return default
        index = min(req_idx, int(param.numel()) - 1)
        value = param.reshape(-1)[index].item()
        if isinstance(default, int):
            return int(value)
        return float(value)

    @staticmethod
    def _random_sample_one(probs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        return random_sample(probs.unsqueeze(0), {} if generator is None else {0: generator}).reshape(())

    @classmethod
    def _nucleus_sample_one(
        cls,
        weighted_scores: torch.Tensor,
        *,
        top_p: float,
        top_k: int,
        generator: torch.Generator | None,
    ) -> int:
        """Vectorized nucleus + top-k sampling.

        Distribution-equivalent to the reference iterative implementation: the
        keep-set is identical (token i is kept iff
        ``cumsum(sorted_probs)[i] - sorted_probs[i] < top_p`` AND ``i < top_k``)
        and the renormalized sampling distribution matches, but the exact token
        drawn for a given seed is NOT guaranteed to match. The reference draws
        via ``multinomial`` over the stacked kept subset while this draws over
        the full sorted vector (zeroed outside the keep-set), so the generator
        advances over different-sized inputs and may yield a different sample.
        The win: no per-token ``.item()`` D2H syncs from the Python loop —
        those dominated the sampler CPU time in profiling.
        """
        probs = weighted_scores.softmax(dim=0)
        sorted_prob, sorted_idx = probs.sort(descending=True, stable=True)
        cum_before = sorted_prob.cumsum(dim=0) - sorted_prob
        mask = cum_before < top_p
        if top_k > 0:
            n = sorted_prob.shape[0]
            mask = mask & (torch.arange(n, device=mask.device) < min(int(top_k), n))
        weights = sorted_prob * mask.to(sorted_prob.dtype)
        # First token always passes (cum_before[0] = 0 < top_p for any top_p > 0),
        # so ``weights`` is guaranteed to have at least one nonzero entry. The
        # final ``.item()`` is the ONLY D2H sync per call.
        sample_idx = cls._random_sample_one(weights, generator=generator)
        return int(sorted_idx[sample_idx].item())

    @classmethod
    def _ras_sample_one(
        cls,
        weighted_scores: torch.Tensor,
        decoded_tokens: Sequence[int],
        *,
        top_p: float,
        top_k: int,
        win_size: int,
        tau_r: float,
        generator: torch.Generator | None,
    ) -> int:
        top_id = cls._nucleus_sample_one(
            weighted_scores,
            top_p=top_p,
            top_k=top_k,
            generator=generator,
        )
        if win_size > 0 and decoded_tokens:
            recent = torch.as_tensor(
                list(decoded_tokens[-win_size:]),
                device=weighted_scores.device,
                dtype=torch.long,
            )
            rep_num = int((recent == top_id).sum().item())
            if rep_num >= win_size * tau_r:
                weighted_scores = weighted_scores.clone()
                original_score = weighted_scores[top_id].clone()
                weighted_scores[top_id] = float("-inf")
                weighted_scores[top_id] = torch.where(
                    torch.isfinite(weighted_scores).any(),
                    weighted_scores[top_id],
                    original_score,
                )
                top_id = int(cls._random_sample_one(weighted_scores.softmax(dim=0), generator=generator).item())
        return top_id

    def _cosyvoice3_ras_enabled(self, sampling_metadata: SamplingMetadata) -> bool:
        if self.model_stage != "cosyvoice3_talker":
            return False
        if sampling_metadata.max_num_logprobs is not None:
            return False
        if sampling_metadata.temperature is None:
            return False
        if bool(sampling_metadata.bad_words_token_ids):
            return False
        if torch.any(sampling_metadata.frequency_penalties != 0):
            return False
        if torch.any(sampling_metadata.presence_penalties != 0):
            return False
        return True

    def _ras_sample_batch(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        *,
        default_top_p: float,
        default_top_k: int,
        win_size: int,
        tau_r: float,
    ) -> torch.Tensor:
        """Batch random RAS without per-row scalar parameter/token transfers.

        Rejection flags cross to the host once per batch so only requests that
        actually reject a token consume a second RNG draw. Per-request seeded
        generators retain their ownership when rejection compacts the batch.
        """
        batch_size = logits.shape[0]

        def parameter(value: torch.Tensor | None, default: float | int) -> torch.Tensor:
            if value is None or value.numel() == 0:
                return torch.full((batch_size,), default, device=logits.device)
            value = value.reshape(-1).to(device=logits.device)
            if value.numel() < batch_size:
                value = torch.cat((value, value[-1:].expand(batch_size - value.numel())))
            return value[:batch_size]

        temperature = parameter(sampling_metadata.temperature, 1.0)
        top_p = parameter(sampling_metadata.top_p, default_top_p)
        top_k = parameter(sampling_metadata.top_k, default_top_k)
        weighted_scores = torch.log_softmax(logits / temperature.clamp_min(self._sampling_eps).unsqueeze(1), dim=1)
        sorted_probs, sorted_ids = weighted_scores.softmax(dim=1).sort(dim=1, descending=True, stable=True)
        # Keep top-p based on the full distribution, before top-k masking.
        keep = (sorted_probs.cumsum(dim=1) - sorted_probs) < top_p.unsqueeze(1)
        ranks = torch.arange(logits.shape[1], device=logits.device)
        keep &= (top_k.unsqueeze(1) <= 0) | (ranks.unsqueeze(0) < top_k.unsqueeze(1))
        weights = sorted_probs * keep.to(sorted_probs.dtype)
        generators = {i: g for i, g in sampling_metadata.generators.items() if i < batch_size}
        draws = random_sample(weights, generators).reshape(-1, 1).long()
        sampled = sorted_ids.gather(1, draws).squeeze(1)

        histories = sampling_metadata.output_token_ids
        if win_size > 0 and any(histories[:batch_size]):
            recent = []
            for i in range(batch_size):
                row = list(histories[i][-win_size:]) if i < len(histories) else []
                recent.append([-1] * (win_size - len(row)) + row)
            history = torch.tensor(recent, dtype=torch.long, device=logits.device)
            repeated = ((history == sampled.unsqueeze(1)).sum(dim=1) >= win_size * tau_r) & (history >= 0).any(dim=1)
            rejected_rows = [i for i, reject in enumerate(repeated.tolist()) if reject]
            if rejected_rows:
                rows = torch.tensor(rejected_rows, device=logits.device, dtype=torch.long)
                scores = weighted_scores.index_select(0, rows)
                rejected_ids = sampled.index_select(0, rows).unsqueeze(1)
                original = scores.gather(1, rejected_ids)
                scores.scatter_(1, rejected_ids, float("-inf"))
                replacement = torch.where(
                    torch.isfinite(scores).any(dim=1, keepdim=True),
                    torch.full_like(original, float("-inf")),
                    original,
                )
                scores.scatter_(1, rejected_ids, replacement)
                compact_generators = {i: generators[row] for i, row in enumerate(rejected_rows) if row in generators}
                # RAS rejection samples the complete remaining distribution,
                # without applying top-k/top-p a second time.
                replacement_ids = random_sample(scores.softmax(dim=1), compact_generators).reshape(-1).long()
                sampled.index_copy_(0, rows, replacement_ids)
        return sampled.to(torch.int32)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        if logits is None or logits.numel() == 0:
            return None
        if self.model_stage != "cosyvoice3_talker":
            return None

        if cosyvoice3_standard_sampling(self.config):
            sampler = getattr(self, "_talker_sampler", None)
            if sampler is None:
                sampler = Sampler()
                self._talker_sampler = sampler
            # Penalize generated tokens only, never text or reference
            # speech in the multimodal prompt. Padding is ignored by vLLM.
            if sampling_metadata.prompt_token_ids is not None:
                sampling_metadata = replace(
                    sampling_metadata,
                    prompt_token_ids=torch.full_like(sampling_metadata.prompt_token_ids, logits.shape[-1]),
                )
            return sampler(logits=logits, sampling_metadata=sampling_metadata)

        if not self._cosyvoice3_ras_enabled(sampling_metadata):
            sampler = getattr(self, "_talker_sampler", None)
            if sampler is None:
                sampler = Sampler()
                self._talker_sampler = sampler
            return sampler(logits=self._full_vocab_logits(logits), sampling_metadata=sampling_metadata)

        logits = logits.to(torch.float32)
        mask = sampling_metadata.allowed_token_ids_mask
        if mask is not None and mask.shape[-1] != logits.shape[-1]:
            sampling_metadata = replace(sampling_metadata, allowed_token_ids_mask=mask[..., : logits.shape[-1]])
        # Apply logits processors directly — RAS handles its own repetition
        # logic.  We avoid instantiating Sampler() here because its import
        # chain pulls in flashinfer / GPU deps that fail in CPU-only tests.
        if sampling_metadata.allowed_token_ids_mask is not None:
            logits.masked_fill_(sampling_metadata.allowed_token_ids_mask, float("-inf"))
        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            logits = processor.apply(logits)
        # The text tokenizer vocabulary is much larger than the speech head.
        # compute_logits pads its output for the generic sampler/processors;
        # RAS only needs speech codes and the merged stop token. Keep the
        # original token indices, and trim only after full-vocabulary masks
        # and processors have run. The generic sampler fallback above retains
        # its full-vocabulary contract (including logprobs and bad words).
        speech_head_size = int(self.config.llm["speech_token_size"]) + 200
        logits = logits[..., :speech_head_size]
        finite_logits = torch.isfinite(logits)
        if not finite_logits.any(dim=-1).all().item():
            raise ValueError("CosyVoice3 sampling received a row with no finite logits")
        logits.masked_fill_(~finite_logits, float("-inf"))

        sampling_cfg = dict(self.config.llm.get("sampling", {}))
        default_top_p = float(sampling_cfg.get("top_p", 0.8))
        default_top_k = int(sampling_cfg.get("top_k", 25))
        win_size = int(sampling_cfg.get("win_size", 10))
        tau_r = float(sampling_cfg.get("tau_r", 0.1))

        if sampling_metadata.all_random:
            sampled = self._ras_sample_batch(
                logits,
                sampling_metadata,
                default_top_p=default_top_p,
                default_top_k=default_top_k,
                win_size=win_size,
                tau_r=tau_r,
            )
            return SamplerOutput(sampled_token_ids=sampled.unsqueeze(-1), logprobs_tensors=None)

        sampled_ids: list[int] = []
        for req_idx in range(int(logits.shape[0])):
            row_logits = logits[req_idx]

            temperature = float(self._req_scalar(sampling_metadata.temperature, req_idx, 1.0))
            if temperature < self._sampling_eps:
                sampled_ids.append(int(torch.argmax(row_logits).item()))
                continue

            top_p = float(self._req_scalar(sampling_metadata.top_p, req_idx, default_top_p))
            top_k = int(self._req_scalar(sampling_metadata.top_k, req_idx, default_top_k))
            generator = sampling_metadata.generators.get(req_idx)
            weighted_scores = torch.log_softmax(row_logits / max(temperature, self._sampling_eps), dim=0)
            decoded_tokens = (
                sampling_metadata.output_token_ids[req_idx] if req_idx < len(sampling_metadata.output_token_ids) else []
            )
            sampled_ids.append(
                self._ras_sample_one(
                    weighted_scores,
                    decoded_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    win_size=win_size,
                    tau_r=tau_r,
                    generator=generator,
                )
            )

        sampled = torch.tensor(sampled_ids, device=logits.device, dtype=torch.int32)
        return SamplerOutput(sampled_token_ids=sampled.unsqueeze(-1), logprobs_tensors=None)

    def _full_vocab_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Pad the speech head to the text vocabulary for the generic sampler contract."""
        pad_size = int(self.config.vocab_size) - logits.size(-1)
        if pad_size <= 0:
            return logits
        return torch.cat([logits, logits.new_full(logits.shape[:-1] + (pad_size,), float("-inf"))], dim=-1)
