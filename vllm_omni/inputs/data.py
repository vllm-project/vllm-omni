from typing import Any, Optional
from dataclasses import dataclass, field

try:
    from typing import NotRequired
except ImportError:
    # Python < 3.11: use typing_extensions
    from typing_extensions import NotRequired

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


@dataclass
class OmniDiffusionRequest:
    request_id: str | None
    prompt: str | list[str] | None = None
    negative_prompt: str | list[str] | None = None
    height: list[int] | int | None = None
    width: list[int] | int | None = None
    num_inference_steps: int = 50
    true_cfg_scale: float = 4.0
    generator: torch.Generator | list[torch.Generator] | None = None

    prompt_embeds: list[torch.Tensor] | torch.Tensor = field(default_factory=list)
    negative_prompt_embeds: list[torch.Tensor] | None = None
    prompt_embeds_mask: list[torch.Tensor] | torch.Tensor | None = None
    prompt_attention_mask: list[torch.Tensor] | None = None
    negative_attention_mask: list[torch.Tensor] | None = None


def token_inputs_omni(
    prompt_token_ids: list[int],
    prompt: Optional[str] = None,
    cache_salt: Optional[str] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    additional_information: Optional[dict[str, Any]] = None,
) -> OmniTokenInputs:
    """Construct token inputs with optional embeddings and metadata."""
    inputs = OmniTokenInputs(type="token", prompt_token_ids=prompt_token_ids)

    if prompt is not None:
        inputs["prompt"] = prompt
    if cache_salt is not None:
        inputs["cache_salt"] = cache_salt
    if prompt_embeds is not None:
        inputs["prompt_embeds"] = prompt_embeds
    if additional_information is not None:
        inputs["additional_information"] = additional_information

    return inputs
