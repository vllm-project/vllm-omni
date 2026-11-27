# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tag talker (stage 1 of Sophon-Realtime-32B-Tag): CosyVoice3 LM with
*bistream* text input.

The stock CosyVoice3 talker gets the whole text up front. Here the thinker's text
arrives while it is being written (cumulative CosyVoice3 token ids in
``ids.output``, see ``stage_input_processors.sophon_realtime``), and the talker
follows CosyVoice3's ``inference_bistream`` schedule, which the Tag talker was
trained with. The schedule itself lives in ``core.bistream`` (framework
independent); this class maps its input items to embeddings and samples with
RAS under its constraints.

Prompt: ``[sos] + "You are a helpful assistant.<|endofprompt|>"`` (no prompt
speech tokens: the voice only conditions the flow, whose prompt reaches the
vocoder with the thinker's first chunk).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from types import MethodType
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.v1.outputs import SamplerOutput

from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import CosyVoice3Model
from vllm_omni.model_executor.models.cosyvoice3.tokenizer import get_qwen_tokenizer
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.sophon_realtime.core.bistream import (
    BistreamSchedule,
    BistreamTokens,
    InputItem,
)

logger = init_logger(__name__)


def random_sample(probs: torch.Tensor, generators: dict[int, torch.Generator]) -> torch.Tensor:
    """One draw per row from (unnormalized) ``probs``: exponential-race sampling, with the request's own generator
    for seeded rows (the scheme vLLM's sampler used before 0.34)."""
    q = torch.empty_like(probs)
    if len(generators) != probs.shape[0]:
        q.exponential_()
    for i, generator in generators.items():
        q[i].exponential_(generator=generator)
    return (probs / q).argmax(dim=-1).view(-1)


LM_PROMPT = "You are a helpful assistant.<|endofprompt|>"
MAX_TRACKED_REQUESTS = 512


def _first(x, default=None):
    if isinstance(x, (list, tuple)):
        return x[0] if x else default
    return default if x is None else x


def _flag(x) -> bool:
    if isinstance(x, torch.Tensor):
        return bool(x.reshape(-1)[0].item()) if x.numel() else False
    if isinstance(x, (list, tuple)):
        return bool(x and _flag(x[0]))
    return bool(x)


def _text_ids(info: dict) -> list[int] | None:
    ids = info.get("ids")
    ids = ids.get("output") if isinstance(ids, dict) else None
    if ids is None:
        return None
    if isinstance(ids, torch.Tensor):
        ids = ids.reshape(-1).tolist()
    return [int(x) for x in ids]


def _flatten_info(info: dict) -> dict:
    extra = info.get("additional_information")
    if not isinstance(extra, dict):
        return info
    merged = {k: v for k, v in info.items() if k != "additional_information"}
    for k, v in extra.items():
        merged.setdefault(k, v)
    return merged


class SophonTagTalker(CosyVoice3Model):
    # Sampling keeps per-request bistream state, so it runs on the host after
    # each step: no deferred RNG and no runner-side token override.
    model_sampler_can_defer_rng = False
    supports_sampled_token_ids_cpu_override = False

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        llm = self.config.llm
        mix_text, mix_speech = (int(x) for x in llm.get("mix_ratio", [5, 15]))
        self.tokens = BistreamTokens(int(llm["speech_token_size"]), mix_text, mix_speech)
        self.has_preprocess = self.model_stage == "cosyvoice3_talker"
        self._streams: dict[str, BistreamSchedule] = {}
        self._last_fed: dict[str, torch.Tensor] = {}
        self._batch_req_ids: list[str] = []
        # Model Runner V2 calls mrv2_sampling_context (which names the batch's requests) only for models that ask
        # for output token ids; the bistream schedule needs the request of every row.
        self.logitsprocs_need_output_token_ids = self.model_stage == "cosyvoice3_talker"
        if self.model_stage == "cosyvoice3_talker":
            tokenizer = get_qwen_tokenizer(
                os.path.join(self.model_dir, self.config.qwen_pretrain_path),
                skip_special_tokens=True,
                version="cosyvoice3",
            )
            self._prompt_ids = tokenizer.encode(LM_PROMPT, allowed_special="all")

    # ------------------------------------------------------------------ embeddings
    def _speech_embed(self, ids) -> torch.Tensor:
        weight = self.model.speech_embedding.weight
        return weight[torch.as_tensor(list(ids), device=weight.device, dtype=torch.long)]

    def _text_embed(self, ids) -> torch.Tensor:
        embed = self.model.llm.model.embed_tokens
        return embed(torch.as_tensor(list(ids), device=embed.weight.device, dtype=torch.long))

    def _embed_item(self, rid: str, item: InputItem) -> torch.Tensor:
        if item[0] == "text":
            return self._text_embed([item[1]])
        if item[0] == "speech":
            return self._speech_embed([item[1]])
        return self._last_fed[rid]  # "repeat": the previous input again

    def _stream(self, rid: str) -> BistreamSchedule | None:
        """The stream of ``rid``, moved to the newest end (old streams are
        dropped first, and a long reply must outlive short requests)."""
        st = self._streams.pop(rid, None)
        if st is not None:
            self._streams[rid] = st
        return st

    @staticmethod
    def _sync_text(st: BistreamSchedule, info: dict) -> None:
        meta = info.get("meta") or {}
        finished = isinstance(meta, dict) and _flag(meta.get("finished", False))
        st.update_text(_text_ids(info), finished)

    def _forget(self, rid: str) -> None:
        self._streams.pop(rid, None)
        self._last_fed.pop(rid, None)

    # ------------------------------------------------------------------ inputs
    def prompt_embeds(self) -> torch.Tensor:
        return torch.cat([self._speech_embed([self.tokens.sos]), self._text_embed(self._prompt_ids)], 0)

    def preprocess(self, input_ids: torch.Tensor, input_embeds: torch.Tensor | None, **info: Any):
        info = _flatten_info(info)
        rid = info.get("req_id") or info.get("request_id")  # Model Runner V2 names it req_id
        span = int(input_ids.shape[0])
        is_prefill = info.get("_omni_is_prefill")
        if not isinstance(is_prefill, bool):
            is_prefill = span > 1
        dtype = input_embeds.dtype if input_embeds is not None else None
        if rid is None:
            embeds = self._speech_embed(input_ids.tolist()) if input_embeds is None else input_embeds
            return input_ids, embeds, {}
        if is_prefill:
            st = BistreamSchedule(self.tokens)
            self._streams[rid] = st
            if len(self._streams) > MAX_TRACKED_REQUESTS:
                for stale in list(self._streams)[: len(self._streams) - MAX_TRACKED_REQUESTS]:
                    self._forget(stale)
            self._sync_text(st, info)
            prompt = self.prompt_embeds()
            if prompt.shape[0] != span:
                logger.warning_once(
                    "Sophon-Realtime Tag talker: prefill span %d != prompt length %d "
                    "(set hf_overrides.async_chunk_prewarm_prompt_len=%d)",
                    span,
                    prompt.shape[0],
                    prompt.shape[0],
                )
                pad = self._speech_embed([self.tokens.sos]).expand(max(0, span - prompt.shape[0]), -1)
                prompt = torch.cat([pad, prompt], 0)[-span:]
            self._last_fed[rid] = prompt[-1:]
            return input_ids, prompt if dtype is None else prompt.to(dtype), {}
        st = self._stream(rid)
        if st is None:  # not started here: behave like the stock decode step
            return input_ids, self._speech_embed(input_ids.tolist()), {}
        self._sync_text(st, info)
        item = st.next_input()
        if item is not None:
            embeds = self._embed_item(rid, item)
        else:
            last = st.last_token if st.last_token is not None else int(input_ids.reshape(-1)[-1].item())
            embeds = self._speech_embed([last])
        self._last_fed[rid] = embeds
        return input_ids, embeds if dtype is None else embeds.to(dtype), {}

    # ------------------------------------------------------------------ logits / sampling
    def compute_logits(self, hidden_states):
        if self.model_stage != "cosyvoice3_talker":
            return super().compute_logits(hidden_states)
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        fill = self.tokens.fill
        raw_fill = self.model.llm_decoder(hidden_states)[..., fill].clone()
        logits = super().compute_logits(hidden_states)
        logits[..., fill] = raw_fill.to(logits.dtype)  # the stock head folds fill into EOS
        return logits

    def set_batch_req_ids(self, req_ids) -> None:
        self._batch_req_ids = list(req_ids)

    @contextmanager
    def mrv2_sampling_context(self, *, req_ids, num_output_tokens) -> Iterator[None]:
        """Model Runner V2 names the batch's requests (in logits row order) around sampling."""
        self.set_batch_req_ids(req_ids)
        yield

    def on_requests_finished(self, request_ids: set[str]) -> None:
        super().on_requests_finished(request_ids)
        for rid in request_ids:
            self._forget(rid)

    # ------------------------------------------------------------------ Model Runner V2 sampling
    def _bistream_mask(self, logits: torch.Tensor, plans: list) -> torch.Tensor:
        """Rows with a bistream plan may emit speech tokens, ``fill`` (bistream phase) or ``eos`` (final phase) only."""
        t = self.tokens
        x = logits.clone()
        rows = torch.tensor([p is not None for p in plans], device=x.device)
        final = torch.tensor([bool(p is not None and p.final) for p in plans], device=x.device)
        keep_fill = x[:, t.fill].clone()
        keep_eos = x[:, t.eos].clone()
        masked = x[:, t.speech_vocab :].clone()
        masked.fill_(float("-inf"))
        x[:, t.speech_vocab :] = torch.where(rows.unsqueeze(1), masked, x[:, t.speech_vocab :])
        neg = torch.full_like(keep_fill, float("-inf"))
        x[:, t.fill] = torch.where(rows & ~final, keep_fill, torch.where(rows, neg, keep_fill))
        x[:, t.eos] = torch.where(rows & final, keep_eos, torch.where(rows, neg, keep_eos))
        return x

    def _bistream_accept(self, streams: list, plans: list, sampled: list[int]) -> bool:
        """Apply forced tokens and advance each stream; True if any sampled token was replaced."""
        changed = False
        for i, (st, plan) in enumerate(zip(streams, plans)):
            if plan is None:
                continue
            if plan.forced is not None and sampled[i] != plan.forced:
                sampled[i] = plan.forced
                changed = True
            if plan.discard:
                continue
            starved = st.starved
            st.accept(int(sampled[i]))
            if st.starved != starved and st.starved in (1, 10, 100):
                logger.warning("Sophon-Realtime Tag talker: waiting for text (thinker still writing), x%d", st.starved)
        return changed

    def mrv2_custom_sampler(self, sampler: Any):
        """Model Runner V2: the CosyVoice3 sampler (RAS) with the bistream schedule around it.

        Before sampling, rows of a reply in progress are restricted to what the schedule allows; after it, forced
        tokens replace the draws and each reply's schedule advances. The schedule lives on the host, so this step
        reads the sampled tokens back (as the V1 path did).
        """
        result = super().mrv2_custom_sampler(sampler)
        if self.model_stage != "cosyvoice3_talker":
            return result
        inner = sampler.sample
        model = self

        def sample(_sampler, logits, *args, **kwargs):
            rows = int(logits.shape[0])
            rids = model._batch_req_ids if len(model._batch_req_ids) == rows else [None] * rows
            streams = [model._streams.get(r) if r is not None else None for r in rids]
            if not any(st is not None for st in streams):
                return inner(logits, *args, **kwargs)
            plans = [st.plan() if st is not None else None for st in streams]
            out = inner(model._bistream_mask(logits, plans), *args, **kwargs)
            sampled, rest = (out[0], out[1:]) if isinstance(out, tuple) else (out, ())
            tokens = sampled.reshape(-1).tolist()
            if model._bistream_accept(streams, plans, tokens):
                sampled = torch.tensor(tokens, device=sampled.device, dtype=sampled.dtype).view_as(sampled)
            return (sampled, *rest) if isinstance(out, tuple) else sampled

        sampler.sample = MethodType(sample, sampler)
        return result

    def sample(self, logits, sampling_metadata, **kwargs):
        if logits is None or logits.numel() == 0 or self.model_stage != "cosyvoice3_talker":
            return super().sample(logits, sampling_metadata, **kwargs)
        rows = int(logits.shape[0])
        rids = self._batch_req_ids if len(self._batch_req_ids) == rows else [None] * rows
        streams = [self._streams.get(r) if r is not None else None for r in rids]
        if not any(st is not None for st in streams):
            return super().sample(logits, sampling_metadata, **kwargs)
        dev = logits.device
        t = self.tokens
        cfg = dict(self.config.llm.get("sampling", {}))
        win = int(cfg.get("win_size", 10))
        tau = float(cfg.get("tau_r", 0.1))
        plans = [st.plan() if st is not None else None for st in streams]

        x = logits.to(torch.float32)
        keep_fill = x[:, t.fill].clone()
        keep_eos = x[:, t.eos].clone()
        x[:, t.speech_vocab :] = float("-inf")
        in_final = torch.tensor([bool(p and p.final) for p in plans], device=dev)
        x[:, t.fill] = torch.where(in_final, torch.full_like(keep_fill, float("-inf")), keep_fill)
        x[:, t.eos] = torch.where(in_final, keep_eos, torch.full_like(keep_eos, float("-inf")))

        temp = sampling_metadata.temperature
        if isinstance(temp, torch.Tensor) and temp.numel() >= rows:
            temp = temp.to(dev, torch.float32).reshape(-1)[:rows]
        else:
            temp = torch.ones(rows, device=dev)
        greedy = temp < self._sampling_eps
        logp = torch.log_softmax(x / temp.clamp(min=self._sampling_eps).unsqueeze(1), dim=1)
        top_p = sampling_metadata.top_p
        top_k = sampling_metadata.top_k
        if isinstance(top_p, torch.Tensor) and top_p.numel() >= rows:
            top_p = top_p.to(dev, torch.float32).reshape(-1)[:rows]
        else:
            top_p = torch.full((rows,), float(cfg.get("top_p", 0.8)), device=dev)
        if isinstance(top_k, torch.Tensor) and top_k.numel() >= rows:
            top_k = top_k.to(dev).reshape(-1)[:rows]
        else:
            top_k = torch.full((rows,), int(cfg.get("top_k", 25)), device=dev)
        probs = logp.softmax(dim=1)
        sorted_p, sorted_i = probs.sort(dim=1, descending=True, stable=True)
        before = sorted_p.cumsum(dim=1) - sorted_p
        rank = torch.arange(sorted_p.shape[1], device=dev).unsqueeze(0)
        keep = (before < top_p.unsqueeze(1)) & ((top_k.unsqueeze(1) <= 0) | (rank < top_k.unsqueeze(1)))
        generators = sampling_metadata.generators or {}
        pick = random_sample(sorted_p * keep.to(sorted_p.dtype), generators)
        top_id = sorted_i.gather(1, pick.long().unsqueeze(1)).squeeze(1)
        # RAS: resample from the full distribution (without top_id) when top_id
        # is already frequent in the recent window
        recent = torch.full((rows, win), -1, dtype=torch.long)
        for i, st in enumerate(streams):
            if st is not None and st.out:
                tail = st.out[-win:]
                recent[i, win - len(tail) :] = torch.as_tensor(tail)
        recent = recent.to(dev, non_blocking=True)
        repeats = (recent == top_id.unsqueeze(1)).sum(1)
        alt = random_sample(logp.scatter(1, top_id.unsqueeze(1), float("-inf")).softmax(dim=1), generators)
        tokens = torch.where(repeats >= win * tau, alt.long(), top_id)
        tokens = torch.where(greedy, logp.argmax(dim=1), tokens)
        out = tokens.tolist()

        for i, (st, plan) in enumerate(zip(streams, plans)):
            if plan is None:
                continue
            if plan.forced is not None:
                out[i] = plan.forced
            if plan.discard:
                continue
            starved = st.starved
            st.accept(int(out[i]))
            if st.starved != starved and st.starved in (1, 10, 100):
                logger.warning("Sophon-Realtime Tag talker: waiting for text (thinker still writing), x%d", st.starved)
        sampled = torch.tensor(out, device=dev, dtype=torch.int32)
        return SamplerOutput(sampled_token_ids=sampled.unsqueeze(-1), logprobs_tensors=None)
