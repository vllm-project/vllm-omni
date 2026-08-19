# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IndexTTS adapter for BigVGAN's official fused CUDA activation."""

from __future__ import annotations

from importlib import import_module

import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.models.common.alias_free_activation import (
    AliasFreeActivation1d,
)

logger = init_logger(__name__)


class OfficialFusedAliasFreeActivation1d(AliasFreeActivation1d):
    """Strict adapter for BigVGAN's official fused CUDA activation.

    The upstream kernel hard-codes ratio 2, 12-tap filters, and replicate
    padding. Contract mismatches and unavailable extensions fall back to the
    portable implementation.
    """

    _extension = None
    _extension_unavailable = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_fused_active = False

    @property
    def fused_activation_requested(self) -> bool:
        return True

    @property
    def fused_activation_loaded(self) -> bool:
        return type(self)._extension is not None

    @property
    def fused_activation_active(self) -> bool:
        return self._last_fused_active

    @classmethod
    def _load_extension(cls):
        if cls._extension is not None:
            return cls._extension
        if cls._extension_unavailable:
            return None
        try:
            from .load import load

            cls._extension = load()
        except (ImportError, OSError, RuntimeError):
            try:
                cls._extension = import_module("anti_alias_activation_cuda")
            except ImportError:
                try:
                    module = import_module("indextts.s2mel.modules.bigvgan.alias_free_activation.cuda.activation1d")
                    cls._extension = module.anti_alias_activation_cuda
                except ImportError:
                    cls._extension_unavailable = True
                    logger.warning_once(
                        "Official BigVGAN fused alias-free CUDA extension is unavailable; using eager activation"
                    )
                    return None
        return cls._extension

    @classmethod
    def preload_extension(cls) -> bool:
        """Build/load the optional CUDA extension during model warmup."""
        return cls._load_extension() is not None

    def _strict_contract_matches(self) -> bool:
        return (
            self.upsample.ratio == 2
            and self.downsample.stride == 2
            and self.upsample.kernel_size == 12
            and int(self.downsample.filter.shape[-1]) == 12
            and self.upsample.pad == 5
            and self.downsample.pad_left == 5
            and self.downsample.pad_right == 6
        )

    @staticmethod
    def _is_fatal_cuda_error(error: BaseException) -> bool:
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return True
        if not isinstance(error, RuntimeError):
            return False
        message = str(error).lower()
        cuda_markers = (
            "cuda error",
            "device-side assert",
            "illegal memory access",
            "out of memory",
            "cublas",
            "cudnn",
            "cufft",
        )
        return any(marker in message for marker in cuda_markers)

    @classmethod
    def _handle_extension_failure(cls, error: BaseException) -> None:
        if cls._is_fatal_cuda_error(error):
            raise error
        logger.warning_once(
            "Official BigVGAN fused alias-free activation failed (%s); disabling the extension",
            str(error),
        )
        cls._extension_unavailable = True
        cls._extension = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self._last_fused_active = False
        if hidden_states.device.type != "cuda" or torch.is_grad_enabled() or not self._strict_contract_matches():
            return super().forward(hidden_states)
        if (
            self.upsample.filter.device != hidden_states.device
            or self.downsample.filter.device != hidden_states.device
            or self.upsample.filter.dtype != hidden_states.dtype
            or self.downsample.filter.dtype != hidden_states.dtype
            or self.act.alpha.dtype != hidden_states.dtype
        ):
            return super().forward(hidden_states)
        extension = self._load_extension()
        if extension is None:
            return super().forward(hidden_states)
        beta = self.act.alpha if self.act.__class__.__name__ == "Snake" else self.act.beta
        alpha = self.act.alpha
        if not self.act.alpha_logscale:
            alpha = torch.log(alpha)
            beta = torch.log(beta)
        try:
            output = extension.forward(
                hidden_states.contiguous(),
                self.upsample.filter,
                self.downsample.filter,
                alpha,
                beta,
            )
            self._last_fused_active = True
            return output
        except Exception as error:
            type(self)._handle_extension_failure(error)
            return super().forward(hidden_states)
