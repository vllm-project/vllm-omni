# ruff: noqa
from __future__ import annotations

import sys

from vllm_omni.model_executor.models.omni_diffusion.third_party.glm4voice import cosyvoice as _cosyvoice
from vllm_omni.model_executor.models.omni_diffusion.third_party.glm4voice import matcha as _matcha


def register_official_module_aliases() -> None:
    """Expose vendored GLM-4-Voice runtime modules under official names.

    The GLM-4-Voice decoder config instantiates classes by import paths such as
    ``cosyvoice.flow...`` and ``matcha.models.components...``. Register those
    names without requiring users to provide an external GLM-4-Voice checkout.
    """

    sys.modules["cosyvoice"] = _cosyvoice
    sys.modules["matcha"] = _matcha


register_official_module_aliases()

__all__ = ["register_official_module_aliases"]
