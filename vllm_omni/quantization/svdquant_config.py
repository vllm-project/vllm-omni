# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SVDQuant W4A4 config + LinearMethod for diffusion transformers.

SVDQuant (https://arxiv.org/abs/2411.05007) is a 4-bit weight, 4-bit
activation quantization scheme paired with a low-rank residual that
absorbs the quantization error. It is the dominant practical
quantization method for diffusion transformers, delivering >2x
speedup vs BF16 with minimal quality loss.

This module owns the on-disk parameter layout (canonical row-major
NVFP4 / INT4-nibble) and the vLLM `LinearMethodBase` plumbing.
Backend-specific kernel calls and weight prep live in sibling modules
(`svdquant_nunchaku.py`, future `svdquant_flashinfer.py`); the active
backend is selected at `__init__` via `svdquant_dispatch.select_backend`.

Diffusion-specific weight key remapping (e.g. diffusers naming
conventions) is not handled here; downstream pipelines remap before
loading. Checkpoints are expected to already store gated-activation
halves in `[gate; hidden]` order — produced at quantization time, not
at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Parameter
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .svdquant_dispatch import (
    SVDQuantPrecision,
    assert_svdquant_supported,
    select_backend,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods

logger = init_logger(__name__)

# Group sizes are dictated by the kernel's scaled-MMA tile:
#   * NVFP4 uses tcgen05's 16-element scale block.
#   * INT4 uses Nunchaku's 64-element block.
_GROUP_SIZE_BY_PRECISION: dict[str, int] = {"int4": 64, "nvfp4": 16}


class DiffusionSVDQuantConfig(QuantizationConfig):
    """Configuration for SVDQuant W4A4 quantization.

    Parameters mirror what's on disk in a SVDQuant-produced checkpoint:

    Args:
        rank: SVD low-rank correction dimension. Typical values are
            16, 32, or 64; the checkpoint dictates the value.
        precision: 4-bit format, either "int4" or "nvfp4".
        act_unsigned: Whether activations are quantized as unsigned
            (saves the sign bit at a small accuracy cost). Per
            checkpoint config.
        modules_to_not_convert: Layer names (or substring patterns)
            that should keep their unquantized weight, e.g. embedders
            and adaLN-modulation projections in diffusion models.
    """

    def __init__(
        self,
        rank: int = 32,
        precision: SVDQuantPrecision = "int4",
        act_unsigned: bool = False,
        modules_to_not_convert: list[str] | None = None,
    ) -> None:
        super().__init__()
        if precision not in _GROUP_SIZE_BY_PRECISION:
            raise ValueError(f"SVDQuant precision must be one of {set(_GROUP_SIZE_BY_PRECISION)}; got {precision!r}")
        self.rank = rank
        self.precision = precision
        self.group_size = _GROUP_SIZE_BY_PRECISION[precision]
        self.act_unsigned = act_unsigned
        self.modules_to_not_convert = modules_to_not_convert or []

    def __repr__(self) -> str:
        return (
            f"DiffusionSVDQuantConfig(rank={self.rank}, precision={self.precision!r}, act_unsigned={self.act_unsigned})"
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "svdquant"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # SM_75 (Turing) is the floor; the dispatcher rejects SM_90 and
        # routes SM_100+ separately.
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DiffusionSVDQuantConfig:
        return cls(
            rank=config.get("rank", 32),
            precision=config.get("precision", "int4"),
            act_unsigned=config.get("act_unsigned", False),
            modules_to_not_convert=config.get("modules_to_not_convert"),
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        if is_layer_skipped(
            prefix,
            self.modules_to_not_convert,
            self.packed_modules_mapping,
            skip_with_substr=True,
        ):
            return UnquantizedLinearMethod()
        return DiffusionSVDQuantLinearMethod(self)


class DiffusionSVDQuantLinearMethod(LinearMethodBase):
    """Backend-agnostic LinearMethod for SVDQuant W4A4.

    The same parameter layout serves both the int4 and nvfp4 paths;
    only the dtypes of `wscales` and the LoRA matrices differ. The
    active platform is checked at `__init__` time and an unsupported
    GPU raises here, before any weights are allocated.

    All backend-specific behavior (weight prep, GEMM call) is
    delegated to the module returned by
    `svdquant_dispatch.select_backend`. The on-disk layout is fixed
    and shared across backends.
    """

    _hardware_logged = False

    def __init__(self, quant_config: DiffusionSVDQuantConfig) -> None:
        self.quant_config = quant_config
        assert_svdquant_supported(quant_config.precision)
        self._backend = select_backend(quant_config.precision)
        if not DiffusionSVDQuantLinearMethod._hardware_logged:
            logger.info(
                "SVDQuant backend selected: %s (precision=%s)",
                self._backend.__name__,
                quant_config.precision,
            )
            DiffusionSVDQuantLinearMethod._hardware_logged = True

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del extra_weight_attrs  # weight_loader is set explicitly per-param.
        output_size_per_partition = sum(output_partition_sizes)

        config = self.quant_config
        rank = config.rank
        group_size = config.group_size
        precision = config.precision

        # The LoRA matrices and the smooth factor must be in the same
        # dtype as the kernel's accumulator. Nunchaku's nvfp4 path
        # locks this to bf16 regardless of the model's params_dtype;
        # the int4 path inherits params_dtype.
        lora_dtype = torch.bfloat16 if precision == "nvfp4" else params_dtype

        wscales_dtype = torch.float8_e4m3fn if precision == "nvfp4" else params_dtype

        # qweight: 4-bit weights packed two-per-byte along the input
        # axis. Shape (out_per_partition, in_per_partition // 2).
        qweight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        _set_attrs(
            qweight,
            input_dim=1,
            output_dim=0,
            weight_loader=default_weight_loader,
        )

        # wscales: per-(group_size) input-column scale,
        # shape (in_per_partition // group_size, out_per_partition).
        wscales = Parameter(
            torch.empty(
                input_size_per_partition // group_size,
                output_size_per_partition,
                dtype=wscales_dtype,
            ),
            requires_grad=False,
        )
        _set_attrs(
            wscales,
            input_dim=0,
            output_dim=1,
            weight_loader=default_weight_loader,
        )

        # SVD low-rank correction matrices.
        proj_down = Parameter(
            torch.empty(input_size_per_partition, rank, dtype=lora_dtype),
            requires_grad=False,
        )
        _set_attrs(
            proj_down,
            input_dim=0,
            output_dim=1,
            weight_loader=default_weight_loader,
        )

        proj_up = Parameter(
            torch.empty(output_size_per_partition, rank, dtype=lora_dtype),
            requires_grad=False,
        )
        _set_attrs(
            proj_up,
            input_dim=1,
            output_dim=0,
            weight_loader=default_weight_loader,
        )

        # Smooth-quant factors. Live on the input axis: replicated for
        # column-parallel layers, sharded for row-parallel.
        smooth_factor = Parameter(
            torch.empty(input_size_per_partition, dtype=lora_dtype),
            requires_grad=False,
        )
        _set_attrs(
            smooth_factor,
            input_dim=0,
            weight_loader=default_weight_loader,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("wscales", wscales)
        layer.register_parameter("proj_down", proj_down)
        layer.register_parameter("proj_up", proj_up)
        layer.register_parameter("smooth_factor", smooth_factor)

        if precision == "nvfp4":
            # Per-output-channel BF16 scale; sharded with the output dim.
            wcscales = Parameter(
                torch.ones(output_size_per_partition, dtype=lora_dtype),
                requires_grad=False,
            )
            _set_attrs(
                wcscales,
                output_dim=0,
                weight_loader=default_weight_loader,
            )
            # Per-tensor global scale (shape (1,) on disk).
            wtscale = Parameter(
                torch.ones(1, dtype=lora_dtype),
                requires_grad=False,
            )
            _set_attrs(wtscale, weight_loader=default_weight_loader)
            layer.register_parameter("wcscales", wcscales)
            layer.register_parameter("wtscale", wtscale)
        else:
            # Keep the attributes present so backend.apply() can branch
            # uniformly without `hasattr` checks.
            layer.wcscales = None
            layer.wtscale = None

        # Stashed for backend.apply() to consume.
        layer.in_features = input_size
        layer.out_features = output_size
        layer.out_features_per_partition = output_size_per_partition
        layer.precision = precision
        layer.act_unsigned = config.act_unsigned

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Delegate post-load weight prep to the active backend.

        All parameters are produced by our quantization pipeline and
        must be loaded by the time we get here; a meta tensor at this
        point is a checkpoint bug, not a missing-shard case to paper
        over.
        """
        self._backend.prepare_weights(layer, self.quant_config.precision)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._backend.apply(layer, x, bias)


def _set_attrs(param: torch.nn.Parameter, **attrs: Any) -> None:
    for key, value in attrs.items():
        setattr(param, key, value)


__all__ = ["DiffusionSVDQuantConfig", "DiffusionSVDQuantLinearMethod"]
