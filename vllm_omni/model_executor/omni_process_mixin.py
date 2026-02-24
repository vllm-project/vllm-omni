"""Omni model interface and mixin for stage-dependent processing.

This module provides:
- OmniModelCapability: Enum of capabilities models can declare
- OmniProcessingInterface: ABC defining the interface for omni models
- OmniProcessingMixin: Mixin implementing the interface with sensible defaults
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    pass


class OmniModelCapability(Enum):
    """Capabilities that an omni model can declare.

    Models declare their capabilities by implementing `get_capabilities()`.
    The runner queries these to determine which hooks to call.
    """

    PREPROCESS = auto()  # Model has custom preprocess logic
    POSTPROCESS = auto()  # Model has custom postprocess logic

    # TALKER_MTP = auto()  # Model has a talker MTP (multi-token prediction) module

    # Future capabilities can be added here


class OmniProcessingInterface(ABC):
    """Abstract base class defining the interface for omni models.

    Models implementing this interface can declare capabilities and
    provide hooks for different runtime stages. Use `OmniProcessingMixin`
    for a concrete implementation with sensible defaults.
    """

    @abstractmethod
    def get_capabilities(self) -> set[OmniModelCapability]:
        """Return the set of capabilities this model has.

        Returns:
            Set of OmniModelCapability values this model supports.
        """
        ...

    @abstractmethod
    def has_capability(self, capability: OmniModelCapability) -> bool:
        """Check if the model has a specific capability.

        Args:
            capability: The capability to check for.

        Returns:
            True if the model has the capability, False otherwise.
        """
        ...

    @abstractmethod
    def preprocess(
        self, input_ids: torch.Tensor, input_embeds: torch.Tensor, **input_dict: object
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Preprocess inputs before the main model forward pass.

        Args:
            input_ids: Token IDs for this request slice.
            input_embeds: Input embeddings for this request slice.
            **input_dict: Additional per-request information.

        Returns:
            Tuple of (processed_input_ids, processed_input_embeds, update_dict).
            update_dict contains any state updates to merge back to request.
        """
        ...

    @abstractmethod
    def postprocess(self, model_output: Any, **info_dict: object) -> dict:
        """Postprocess model outputs after the main forward pass.

        Args:
            model_output: The model's output (hidden states, etc.).
            **info_dict: Additional per-request information.

        Returns:
            Dictionary of updates to merge back to request state.
        """
        ...


class OmniProcessingMixin(OmniProcessingInterface):
    """Mixin class implementing OmniModelInterface with sensible defaults.

    Models should inherit from this mixin and override methods as needed.
    Capabilities are automatically tracked based on what's registered.

    This mixin provides:
    - Automatic capability tracking via register_capability() / unregister_capability()
    - set_custom_preprocess() / set_custom_postprocess() for easy hook registration
    - Default implementations that raise NotImplementedError when not configured

    Example:
        class MyModel(nn.Module, CustomProcessMixin):
            def __init__(self, ...):
                super().__init__()
                # Register preprocessing - this also registers the capability
                self.set_custom_preprocess(self.my_preprocess)

            def my_preprocess(self, input_ids, input_embeds, **kwargs):
                # Custom logic
                return input_ids, input_embeds, {}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._capabilities = set[OmniModelCapability]()

    def register_capability(self, capability: OmniModelCapability) -> None:
        self._capabilities.add(capability)

    def unregister_capability(self, capability: OmniModelCapability) -> None:
        self._capabilities.discard(capability)

    def get_capabilities(self) -> set[OmniModelCapability]:
        return self._capabilities.copy()

    def has_capability(self, capability: OmniModelCapability) -> bool:
        return capability in self._capabilities

    def set_custom_preprocess(self, preprocess_fn: Callable) -> None:
        assert preprocess_fn is not None
        self._preprocess_fn = preprocess_fn
        self.register_capability(OmniModelCapability.PREPROCESS)

    def set_custom_postprocess(self, postprocess_fn: Callable) -> None:
        assert postprocess_fn is not None
        self._postprocess_fn = postprocess_fn
        self.register_capability(OmniModelCapability.POSTPROCESS)

    def preprocess(
        self, input_ids: torch.Tensor, input_embeds: torch.Tensor, **input_dict: object
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        if self.has_capability(OmniModelCapability.PREPROCESS):
            return self._preprocess_fn(input_ids, input_embeds, **input_dict)
        raise NotImplementedError("Preprocess is not implemented for this model.")

    def postprocess(self, model_output: Any, **info_dict: object) -> dict:
        if self.has_capability(OmniModelCapability.POSTPROCESS):
            return self._postprocess_fn(model_output, **info_dict)
        raise NotImplementedError("Postprocess is not implemented for this model.")


# Backwards compatibility helpers
# These allow existing runner code to work with both old and new models


def has_preprocess(model: Any) -> bool:
    if isinstance(model, OmniProcessingInterface):
        return model.has_capability(OmniModelCapability.PREPROCESS)
    return getattr(model, "has_preprocess", False)


def has_postprocess(model: Any) -> bool:
    if isinstance(model, OmniProcessingInterface):
        return model.has_capability(OmniModelCapability.POSTPROCESS)
    return getattr(model, "has_postprocess", False)


# def _has_talker_mtp(model: Any) -> bool:
#     if isinstance(model, OmniModelInterface):
#         return model.has_capability(OmniModelCapability.TALKER_MTP)
#     # Fallback to old hasattr check
#     return hasattr(model, "talker_mtp") and getattr(model, "talker", None) is not None
