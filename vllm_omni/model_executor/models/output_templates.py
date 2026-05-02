from typing import NamedTuple

import torch
from vllm.sequence import IntermediateTensors

from vllm_omni.data_entry_keys import OmniPayload


class OmniOutput(NamedTuple):
    """Output from the merged Omni model containing both text and audio."""

    text_hidden_states: torch.Tensor
    multimodal_outputs: OmniPayload | None = None
    intermediate_tensors: IntermediateTensors | None = None
    next_token_id: torch.Tensor | None = None
