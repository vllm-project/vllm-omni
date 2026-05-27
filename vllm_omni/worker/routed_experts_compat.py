from __future__ import annotations

from typing import Any


try:
    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
        extract_routed_experts_for_current_batch,
        get_global_experts_capturer,
        issue_routing_d2h_copy,
    )
except ImportError:

    def get_global_experts_capturer() -> Any | None:
        return None

    def issue_routing_d2h_copy(*args: Any, **kwargs: Any) -> None:
        return None

    def extract_routed_experts_for_current_batch(*args: Any, **kwargs: Any) -> None:
        return None
