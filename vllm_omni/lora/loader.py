# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Protocol

import torch
from vllm.logger import init_logger
from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path

logger = init_logger(__name__)


class LoRALoader(Protocol):
    """Load an adapter without changing process-global manager behavior.

    Implementations may keep adapter source state, such as in-memory tensors,
    in the loader instance. The manager remains responsible for validation,
    layer binding, scaling, and adapter lifecycle.

    The caller retains ownership of an injected loader and its resources.
    LoRAManager never closes it. Pipeline-supplied loaders follow the owning
    pipeline/worker lifecycle and may be reused across manager reinitialization.
    """

    def load(self, lora_request: LoRARequest) -> tuple[LoRAModel, PEFTHelper]:
        """Return the adapter model and its validated PEFT configuration."""
        ...


class FilesystemLoRALoader:
    """Default adapter loader preserving the existing filesystem behavior."""

    def __init__(
        self,
        *,
        pipeline: object,
        expected_lora_modules: set[str],
        dtype: torch.dtype,
    ) -> None:
        self._pipeline = pipeline
        self._expected_lora_modules = expected_lora_modules
        self._dtype = dtype

    def load(self, lora_request: LoRARequest) -> tuple[LoRAModel, PEFTHelper]:
        lora_path = get_adapter_absolute_path(lora_request.lora_path)
        logger.debug("Resolved LoRA path: %s", lora_path)
        model_loader = getattr(self._pipeline, "_load_diffusion_lora_adapter", None)
        loaded = None
        if callable(model_loader):
            loaded = model_loader(
                lora_request=lora_request,
                lora_path=lora_path,
                dtype=self._dtype,
            )
        if loaded is not None:
            return loaded

        peft_helper = PEFTHelper.from_local_dir(
            lora_path,
            max_position_embeddings=None,  # no need in diffusion
        )
        lora_model = LoRAModel.from_local_checkpoint(
            lora_path,
            expected_lora_modules=self._expected_lora_modules,
            peft_helper=peft_helper,
            lora_model_id=lora_request.lora_int_id,
            device="cpu",  # consistent with vLLM's behavior
            dtype=self._dtype,
            model_vocab_size=None,
            weights_mapper=None,
        )
        return lora_model, peft_helper


__all__ = ["FilesystemLoRALoader", "LoRALoader"]
