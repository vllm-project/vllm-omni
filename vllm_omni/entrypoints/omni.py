# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.diffusion.omni_diffusion import OmniDiffusion
from vllm_omni.diffusion.utils.hf_utils import is_diffusion_model
from vllm_omni.entrypoints.omni_llm import OmniLLM


class Omni:
    """Unified entrypoint for both LLM and Diffusion models for better usability."""

    def __init__(self, *args, **kwargs):
        model = args[0] if args else kwargs.get("model", "")
        if is_diffusion_model(model):
            self.instance: OmniLLM | OmniDiffusion = OmniDiffusion(*args, **kwargs)
        else:
            self.instance: OmniLLM | OmniDiffusion = OmniLLM(*args, **kwargs)

    def __getattr__(self, name):
        """Delegate attribute access to the chosen backend instance."""
        return getattr(self.instance, name)

    def generate(self, *args, **kwargs):
        """Convenience wrapper to call `generate` on the backend if available."""
        if hasattr(self.instance, "generate"):
            return getattr(self.instance, "generate")(*args, **kwargs)
        raise AttributeError(f"'{self.instance.__class__.__name__}' has no attribute 'generate'")
