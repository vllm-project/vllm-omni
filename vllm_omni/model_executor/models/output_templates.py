from typing import NamedTuple

import torch
from vllm.sequence import IntermediateTensors


class OmniOutput(NamedTuple):
    """Output from the merged Omni model containing both text and audio."""

    text_hidden_states: torch.Tensor
    multimodal_outputs: dict | None = None
    intermediate_tensors: IntermediateTensors | None = None
    next_token_id: torch.Tensor | None = None
