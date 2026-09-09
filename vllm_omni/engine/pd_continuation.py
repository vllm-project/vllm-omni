# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The logical generation state handed from a PD producer to its consumer."""

from __future__ import annotations

import msgspec

PD_RESUME_KEY = "pd_resume_from_prefill"
PD_PREFILL_KEY = "pd_prefill_one_token"


class PDContinuation(msgspec.Struct):
    prompt_len: int
    token_ids: list[int]
    # Filled by the output processor before admission, using its tokenizer.
    stop_string: str | None = None

    def validate(self, prompt_ids: list[int] | None) -> None:
        if not prompt_ids or len(prompt_ids) != self.prompt_len:
            raise ValueError("PD continuation must retain the complete original prompt")
        if len(self.token_ids) != 1 or type(self.token_ids[0]) is not int or self.token_ids[0] < 0:
            raise ValueError("PD continuation requires exactly one producer output token")

    @classmethod
    def from_output(cls, output, prompt_ids: list[int]) -> PDContinuation:
        completions = getattr(output, "outputs", None) or []
        if len(completions) != 1:
            raise ValueError("PD continuation requires one prefill completion")
        completion = completions[0]
        if getattr(completion, "finish_reason", None) != "length":
            raise ValueError("PD producer did not complete the KV handoff")
        token_ids = getattr(completion, "cumulative_token_ids", None)
        if token_ids is None:
            token_ids = completion.token_ids
        result = cls(len(prompt_ids), list(token_ids))
        result.validate(prompt_ids)
        return result


def validate_pd_sampling(params) -> None:
    """Phase one supports greedy, single-completion AR continuation."""
    if params.temperature != 0 or params.n != 1:
        raise ValueError("PD first-token continuation currently requires temperature=0 and n=1")
    if params.logprobs is not None or params.prompt_logprobs is not None:
        raise ValueError("PD first-token continuation does not yet transfer logprobs")
    if getattr(params, "structured_outputs", None) is not None:
        raise ValueError("PD first-token continuation does not yet transfer grammar state")


def initial_output_tokens(request) -> list[int]:
    """Only continuation requests can arrive at a worker with output history."""
    if getattr(request, "pd_continuation", None) is None:
        return []
    return list(request.output_token_ids)


def prepend_initial_output(request, token_ids: list[int], stopped: bool) -> list[int]:
    """Emit imported output once, without appending it to model history twice."""
    if getattr(request, "pd_output_prefix_pending", False) and (token_ids or stopped):
        request.pd_output_prefix_pending = False
        return list(request.pd_continuation.token_ids) + list(token_ids)
    return token_ids
