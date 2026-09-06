"""OmniModelStatePlugin — abstract extension point for model-specific behavior.

Plugins are registered on OmniModelState at model-load time and receive
callbacks at well-defined points in the request/step lifecycle.  The base
class provides default no-op implementations for optional hooks so that
concrete plugins only need to override what they use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.states import RequestState


class OmniModelStatePlugin(ABC):
    """Extension point for model-specific behavior in OmniModelState."""

    @abstractmethod
    def on_add_request(self, req_index: int, new_req_data: NewRequestData) -> None: ...

    @abstractmethod
    def on_remove_request(self, req_index: int) -> None: ...

    def prepare_extra_inputs(self, input_batch: InputBatch, req_states: RequestState) -> dict[str, Any]:
        return {}

    def postprocess(
        self,
        text_hidden: Any,
        multimodal_outputs: dict,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> tuple[Any, dict]:
        """Post-process model output.  Default is pass-through."""
        return text_hidden, multimodal_outputs

    def dummy_run(self, state: Any, num_tokens: int) -> None:
        pass
