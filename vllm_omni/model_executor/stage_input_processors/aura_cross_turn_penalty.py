# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-turn repetition penalty for AURA streaming sessions."""

from __future__ import annotations

from collections import Counter
from typing import Any

from vllm_omni.model_executor.stage_input_processors.aura_session_history import (
    is_punctuation_only_text,
)


class CrossTurnPenalty:
    """Cross-turn repetition penalty using logit_bias and bad_words."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        window: int = 2,
        logit_penalty: float = 2.0,
        ngram_sizes: list[int] | None = None,
        max_bad_ngrams: int = 200,
        max_bias_tokens: int = 500,
    ) -> None:
        self.tokenizer = tokenizer
        self.window = window
        self.logit_penalty = logit_penalty
        self.ngram_sizes = ngram_sizes if ngram_sizes is not None else [3, 4, 5]
        self.max_bad_ngrams = max_bad_ngrams
        self.max_bias_tokens = max_bias_tokens
        self._history: list[str | None] = []
        self._special_ids = set(self.tokenizer.all_special_ids)
        self._penalizable_cache: dict[int, bool] = {}

    def _is_penalizable(self, token_id: int) -> bool:
        cached = self._penalizable_cache.get(token_id)
        if cached is not None:
            return cached
        if token_id in self._special_ids:
            self._penalizable_cache[token_id] = False
            return False
        decoded = self.tokenizer.decode([token_id]).strip()
        if not decoded or is_punctuation_only_text(decoded) or decoded.isdigit():
            self._penalizable_cache[token_id] = False
            return False
        self._penalizable_cache[token_id] = True
        return True

    def _spoken_history(self) -> list[str]:
        return [text for text in self._history if text is not None]

    def _build_logit_bias(self) -> dict[int, float]:
        spoken = self._spoken_history()
        if len(spoken) < 2:
            return {}
        n = len(spoken)

        token_presence: dict[int, int] = {}
        for text in spoken:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            for tid in set(ids):
                token_presence[tid] = token_presence.get(tid, 0) + 1
        cross_turn_tids = {tid for tid, cnt in token_presence.items() if cnt >= 2}
        if not cross_turn_tids:
            return {}

        bias: dict[int, float] = {}
        for idx, text in enumerate(spoken):
            recency = (idx + 1) / n
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            freq = Counter(ids)
            for tid, cnt in freq.items():
                if tid not in cross_turn_tids or not self._is_penalizable(tid):
                    continue
                p = self.logit_penalty * min(cnt, 3) * recency
                bias[tid] = bias.get(tid, 0.0) + p

        if len(bias) > self.max_bias_tokens:
            items = sorted(bias.items(), key=lambda kv: kv[1], reverse=True)
            bias = dict(items[: self.max_bias_tokens])
        return {k: min(v, 100.0) for k, v in bias.items()}

    def _build_bad_ngram_map(self) -> dict[tuple, set]:
        spoken = self._spoken_history()
        if not spoken:
            return {}
        prefix_map: dict[tuple, set] = {}
        seen: set[tuple] = set()
        count = 0
        for text in reversed(spoken):
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            for ng_size in self.ngram_sizes:
                if len(ids) < ng_size:
                    continue
                for i in range(len(ids) - ng_size + 1):
                    ngram = tuple(ids[i : i + ng_size])
                    if ngram in seen:
                        continue
                    phrase = self.tokenizer.decode(list(ngram)).strip()
                    if not phrase or is_punctuation_only_text(phrase):
                        continue
                    seen.add(ngram)
                    prefix = ngram[:-1]
                    prefix_map.setdefault(prefix, set()).add(ngram[-1])
                    count += 1
                    if count >= self.max_bad_ngrams:
                        return prefix_map
        return prefix_map

    def build_sampling_kwargs(self) -> dict[str, Any]:
        raw_bias = self._build_logit_bias()
        bad_ngram_map = self._build_bad_ngram_map()
        if not raw_bias and not bad_ngram_map:
            return {}

        kwargs: dict[str, Any] = {}
        if raw_bias:
            kwargs["logit_bias"] = {tid: -val for tid, val in raw_bias.items()}
        if bad_ngram_map:
            bad_words: list[str] = []
            seen: set[tuple] = set()
            for prefix, blocked_set in bad_ngram_map.items():
                for last_tok in blocked_set:
                    ngram = prefix + (last_tok,)
                    if ngram in seen:
                        continue
                    seen.add(ngram)
                    phrase = self.tokenizer.decode(list(ngram))
                    if phrase.strip():
                        bad_words.append(phrase)
            if bad_words:
                kwargs["bad_words"] = bad_words
        return kwargs

    def record(self, response_text: str | None = None) -> None:
        if response_text and response_text.strip():
            self._history.append(response_text)
        else:
            self._history.append(None)
        if len(self._history) > self.window:
            self._history.pop(0)

    def reset(self) -> None:
        self._history.clear()


def merge_penalty_sampling_params(
    sampling_params_list: list[dict[str, Any]] | None,
    penalty_kwargs: dict[str, Any],
    *,
    stage_index: int = 1,
    num_stages: int = 4,
) -> list[dict[str, Any]]:
    """Merge cross-turn penalty kwargs into per-stage sampling params."""
    if not penalty_kwargs:
        return list(sampling_params_list or [])
    params = [dict(stage) for stage in (sampling_params_list or [])]
    while len(params) < num_stages:
        params.append({})
    stage_params = dict(params[stage_index])
    if "logit_bias" in penalty_kwargs:
        merged_bias = dict(stage_params.get("logit_bias") or {})
        merged_bias.update(penalty_kwargs["logit_bias"])
        stage_params["logit_bias"] = merged_bias
    if "bad_words" in penalty_kwargs:
        merged_bad = list(stage_params.get("bad_words") or [])
        merged_bad.extend(penalty_kwargs["bad_words"])
        stage_params["bad_words"] = merged_bad
    params[stage_index] = stage_params
    return params
