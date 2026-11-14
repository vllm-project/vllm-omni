# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Vendored from vLLM 4ba787ec9959e566d3b587a796dff07673324da5;
# baseline V1 proposer behavior retained, removed imports use v1_compat.

import torch
from vllm.config import VllmConfig

from vllm_omni.worker.v1_compat.llm_base_proposer import SpecDecodeBaseProposer


class EagleProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config,
            device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )
