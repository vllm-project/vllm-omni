# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registry mapping diffusion model class names to runtime_v2 adapters.

``multiproc_worker.py`` imports this lazily (so importing the worker module
does not require the adapters to exist), then calls
``get_runtime_v2_adapter(model_class_name)`` to obtain the per-model
``RuntimeV2Adapter`` that builds the task compiler + step-exec executors.

PR1 registers only ``QwenImagePipeline``.
"""

from __future__ import annotations

from collections.abc import Callable

from vllm_omni.diffusion.runtime_v2.interfaces import RuntimeV2Adapter


def _build_qwen_image_adapter() -> RuntimeV2Adapter:
    # Imported inside the factory so importing the registry stays cheap and does
    # not eagerly pull in the adapter module's (lazy) torch/worker dependencies.
    from vllm_omni.diffusion.runtime_v2.adapters.qwen_image import QwenImageRuntimeV2Adapter

    return QwenImageRuntimeV2Adapter()


# model_class_name -> zero-arg adapter factory.
_ADAPTER_FACTORIES: dict[str, Callable[[], RuntimeV2Adapter]] = {
    "QwenImagePipeline": _build_qwen_image_adapter,
}


def supports_runtime_v2_model(model_class_name: str | None) -> bool:
    """Return True iff a runtime_v2 adapter is registered for the model class."""
    return model_class_name in _ADAPTER_FACTORIES


def get_runtime_v2_adapter(model_class_name: str | None) -> RuntimeV2Adapter:
    """Resolve the runtime_v2 adapter for ``model_class_name``.

    Raises ``KeyError`` (with the known names) when the model is unsupported,
    so callers fail loudly rather than silently falling back to a wrong path.
    """
    factory = _ADAPTER_FACTORIES.get(model_class_name) if model_class_name is not None else None
    if factory is None:
        raise KeyError(
            f"no runtime_v2 adapter registered for model_class_name={model_class_name!r}; "
            f"known models: {sorted(_ADAPTER_FACTORIES)}"
        )
    return factory()
