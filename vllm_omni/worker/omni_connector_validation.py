"""Validation for Omni connector support on selected LLM workers."""

import importlib
from typing import Any


def validate_worker_omni_connector(
    worker_cls_path: str | type[Any] | None,
    required: bool,
) -> None:
    """Require the selected worker's runner to implement the connector mixin."""
    if not required:
        return
    if worker_cls_path is None:
        raise ValueError("Omni connector support requires a resolved worker_cls.")

    worker_cls = _resolve_worker_cls(worker_cls_path)
    model_runner_cls = getattr(worker_cls, "model_runner_cls", None)

    from vllm_omni.worker.omni_connector_model_runner_mixin import (
        OmniConnectorModelRunnerMixin,
    )

    if not isinstance(model_runner_cls, type) or not issubclass(
        model_runner_cls,
        OmniConnectorModelRunnerMixin,
    ):
        worker_name = getattr(worker_cls, "__name__", str(worker_cls_path))
        raise ValueError(f"Worker {worker_name!r} does not provide an Omni connector model runner.")


def _resolve_worker_cls(worker_cls_path: str | type[Any]) -> type[Any]:
    if not isinstance(worker_cls_path, str):
        return worker_cls_path
    module_path, separator, class_name = worker_cls_path.rpartition(".")
    if not separator:
        raise ValueError(f"worker_cls must be a fully qualified class path, got {worker_cls_path!r}.")
    try:
        worker_module = importlib.import_module(module_path)
        return getattr(worker_module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Unable to resolve worker_cls {worker_cls_path!r} for connector validation.") from exc
