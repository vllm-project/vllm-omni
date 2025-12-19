# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

from vllm_omni.diffusion.utils.hf_utils import is_diffusion_model
from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion
from vllm_omni.entrypoints.omni_llm import OmniLLM


def _dummy_snapshot_download(model_id):
    return model_id


def omni_snapshot_download(model_id) -> str:
    # TODO: this is just a workaround for quickly use modelscope, we should support
    # modelscope in weight loading feature instead of using `snapshot_download`
    if os.environ.get("VLLM_USE_MODELSCOPE", False):
        from modelscope.hub.snapshot_download import snapshot_download

        return snapshot_download(model_id)
    else:
        return _dummy_snapshot_download(model_id)


class Omni:
    """Unified entrypoint for both LLM and Diffusion models for better usability."""

    def __init__(self, *args, **kwargs):
        model = args[0] if args else kwargs.get("model", "")
        assert model != "", "Null model id detected, please specify a model id."
        model = omni_snapshot_download(model)
        if args:
            args[0] = model
        elif kwargs.get("model", "") != "":
            kwargs["model"] = model
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

    def close(self) -> None:
        close_method = getattr(self.instance, "close", None)
        if callable(close_method):
            close_method()

    def __del__(self):  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass
