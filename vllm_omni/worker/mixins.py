from __future__ import annotations

import hashlib
from typing import Any

import torch


class OmniWorkerMixin:
    """Mixin to ensure Omni plugins are loaded in worker processes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        from vllm_omni.plugins import load_omni_general_plugins

        load_omni_general_plugins()

    def get_weights_checksum(self, component: str | None = None) -> dict[str, Any]:
        """Hash this worker's local model shard for post-update verification."""
        if component == "draft":
            target = self.get_draft_model()
        else:
            target = self.get_model()
            if component not in (None, "", "model", "target"):
                get_submodule = getattr(target, "get_submodule", None)
                target = get_submodule(component) if callable(get_submodule) else getattr(target, component, None)
        if target is None:
            raise ValueError(f"Unknown or unavailable model component: {component}")

        digest = hashlib.sha256()
        parameter_count = 0
        for name, parameter in sorted(target.named_parameters(), key=lambda item: item[0]):
            digest.update(name.encode())
            digest.update(str(parameter.dtype).encode())
            digest.update(str(tuple(parameter.shape)).encode())
            raw = parameter.detach().contiguous().view(torch.uint8).cpu().numpy()
            digest.update(raw.tobytes())
            parameter_count += 1
        return {
            "supported": True,
            "rank": self.rank,
            "component": component or "model",
            "algorithm": "sha256",
            "checksum": digest.hexdigest(),
            "parameter_count": parameter_count,
        }
