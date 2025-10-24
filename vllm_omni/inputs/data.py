from typing import Any, NotRequired, Optional

import torch
from vllm.inputs.data import EmbedsPrompt, TokenInputs, TokensPrompt


class OmniTokensPrompt(TokensPrompt):
    prompt_embeds: NotRequired[torch.Tensor]
    """The embeddings of the prompt."""

    # New: optional additional information dictionary
    # Values may be torch.Tensor or list
    additional_information: NotRequired[dict[str, Any]]


class OmniTokenInputs(TokenInputs):
    # New: optional prompt embeddings aligned with token ids
    prompt_embeds: NotRequired[torch.Tensor]

    # New: optional additional information dictionary
    # Values may be torch.Tensor or list
    additional_information: NotRequired[dict[str, Any]]


class OmniEmbedsPrompt(EmbedsPrompt):
    # New: optional prompt embeddings aligned with token ids
    prompt_embeds: NotRequired[torch.Tensor]

    # New: optional additional information dictionary
    # Values may be torch.Tensor or list
    additional_information: NotRequired[dict[str, Any]]


def token_inputs_omni(
    prompt_token_ids: list[int],
    token_type_ids: Optional[list[int]] = None,
    prompt: Optional[str] = None,
    cache_salt: Optional[str] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    additional_information: Optional[dict[str, Any]] = None,
) -> OmniTokenInputs:
    """Construct [`TokenInputs`][vllm.inputs.data.TokenInputs] from optional
    values."""
    inputs = OmniTokenInputs(type="token", prompt_token_ids=prompt_token_ids)

    if prompt is not None:
        inputs["prompt"] = prompt
    if token_type_ids is not None:
        inputs["token_type_ids"] = token_type_ids
    if cache_salt is not None:
        inputs["cache_salt"] = cache_salt
    if prompt_embeds is not None:
        inputs["prompt_embeds"] = prompt_embeds
    if additional_information is not None:
        inputs["additional_information"] = additional_information

    return inputs
