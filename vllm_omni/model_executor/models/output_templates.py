# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import NamedTuple

import torch
from vllm.sequence import IntermediateTensors

from vllm_omni.data_entry_keys import OmniPayload


class OwnedBatchTensor(NamedTuple):
    """A batch tensor whose producer transfers ownership to the buffer.

    The producer must have freshly allocated the tensor (e.g. an index_select
    result) and must not write it afterwards. The buffer keeps row views
    instead of taking a second snapshot. Tensors without this marker are
    treated as borrowed (graph output / shared scratch) and still get a clone.
    """

    tensor: torch.Tensor


class OmniOutput(NamedTuple):
    """Output from the merged Omni model containing both text and audio."""

    text_hidden_states: torch.Tensor
    multimodal_outputs: OmniPayload | None = None
    intermediate_tensors: IntermediateTensors | None = None
    next_token_id: torch.Tensor | None = None
