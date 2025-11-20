from vllm_omni.utils.diffusers_utils import is_diffusion_model


class Omni:
    def __init__(self, model: str, **kwargs):
        """Wrapper that selects the proper implementation (LLM or diffusion).

        Stores the concrete implementation on `self._impl` and delegates
        attribute access to it. This avoids returning from __init__ which is
        invalid in Python.
        """
        self.is_diffusion_model = is_diffusion_model(model)
        if not self.is_diffusion_model:
            from vllm_omni.entrypoints.omni_llm import OmniLLM

            self._impl = OmniLLM(model, **kwargs)
        else:
            from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion

            self._impl = OmniDiffusion(model, **kwargs)

    def __getattr__(self, name: str):
        # Delegate attribute access to the concrete implementation.
        return getattr(self._impl, name)
