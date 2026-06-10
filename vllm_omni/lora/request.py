# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# for now, it suffices to use vLLM's implementation directly
# as this is a user-facing variable, defined here to so that user can directly import LoRARequest from vllm_omni
from vllm.lora.request import LoRARequest

__all__ = ["LoRARequest"]
