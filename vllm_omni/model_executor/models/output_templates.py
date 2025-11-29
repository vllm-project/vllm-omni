from typing import NamedTuple, Optional

import torch
from vllm.sequence import IntermediateTensors


class OmniOutput(NamedTuple):
    """Output from the merged Omni model containing both text and audio."""

    text_hidden_states: torch.Tensor
    multimodal_outputs: Optional[dict] = None
    intermediate_tensors: Optional[IntermediateTensors] = None
    next_token_id: Optional[torch.Tensor] = None
