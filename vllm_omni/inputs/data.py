from vllm.inputs.data import TokensPrompt
from typing import Any, NotRequired
import torch


class OmniTokensPrompt(TokensPrompt):
    prompt_embeds: NotRequired[torch.Tensor]
    """The embeddings of the prompt."""

    # New: optional additional information dictionary
    # Values may be torch.Tensor or list
    additional_information: NotRequired[dict[str, Any]]