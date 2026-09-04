# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""INT8 quantization config for diffusion transformers.

Supports both online (dynamic) and offline (checkpoint) INT8 quantization
on CUDA and NPU platforms.
"""

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch.nn import Module
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    init_int8_linear_kernel,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.model_loader.reload.meta import (
    CopyCounter as CopyNumelCounter,
)
from vllm.model_executor.model_loader.weight_utils import initialize_single_dummy_weight
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
)
from vllm.model_executor.utils import replace_parameter

from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization._copy_missing_attrs import (
    copy_missing_attrs as _copy_missing_attrs,
)

if current_omni_platform.is_npu():
    import torch_npu
else:
    torch_npu = None

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

# Dynamic quantization is supported first.
ACTIVATION_SCHEMES = ["dynamic"]

# Ascend's npu_quant_matmul (QuantBatchMatmulV3) refuses a weight whose last
# dimension exceeds this, and it fails at the first forward rather than at load
# time. Layers still wider than the limit after TP sharding stay unquantized.
NPU_QUANT_MATMUL_MAX_OUT_FEATURES = 65535

logger = init_logger(__name__)

# Set by the diffusion loader while it constructs a model whose weights will be
# offloaded back to host memory after online quantization (DLO). Over-wide
# layers that npu_quant_matmul cannot run then load straight into host memory
# instead of being built on the accelerator and moved off at the end of
# loading — on MiniMax H3 the 50 fallback adaln layers are ~24 GiB of bf16,
# which is the dominant startup memory peak.
_LOAD_UNQUANTIZABLE_FALLBACK_ON_CPU: ContextVar[bool] = ContextVar(
    "int8_load_unquantizable_fallback_on_cpu", default=False
)


@contextmanager
def load_unquantizable_fallback_on_cpu():
    """Create npu_quant_matmul-incompatible fallback weights on meta and load
    them straight into host memory.

    Only valid when the whole model returns to the host after loading (the
    loader's offload-after-quant path); otherwise the fallback weights would
    stay on CPU while the rest of the model runs on the accelerator.
    """
    token = _LOAD_UNQUANTIZABLE_FALLBACK_ON_CPU.set(True)
    try:
        yield
    finally:
        _LOAD_UNQUANTIZABLE_FALLBACK_ON_CPU.reset(token)


def _fell_back_to_unquantized_npu(
    layer: torch.nn.Module,
    input_size_per_partition: int,
    output_partition_sizes: list[int],
    input_size: int,
    output_size: int,
    params_dtype: torch.dtype,
    **extra_weight_attrs,
) -> bool:
    """Swap a layer to unquantized weights when npu_quant_matmul cannot run its shape.

    The check uses the per-partition output size because that is what the kernel
    actually sees; a layer over the limit on one rank can be within it at a
    higher TP degree. Returns True when the layer was swapped and its weights
    created, in which case the caller must not create its own.
    """
    output_size_per_partition = sum(output_partition_sizes)
    if output_size_per_partition <= NPU_QUANT_MATMUL_MAX_OUT_FEATURES:
        return False

    logger.warning_once(
        "Keeping a %d-wide linear unquantized: npu_quant_matmul rejects an output dimension past "
        "%d. Tensor parallelism shrinks this per-rank dimension, so a higher TP degree brings such "
        "layers back into range.",
        output_size_per_partition,
        NPU_QUANT_MATMUL_MAX_OUT_FEATURES,
    )
    if _LOAD_UNQUANTIZABLE_FALLBACK_ON_CPU.get():
        logger.info_once(
            "Loading over-wide unquantized fallback weights straight into host memory "
            "(offload-after-quant is active); they only reach the accelerator via "
            "the offload backend's runtime prefetch."
        )
        fallback: LinearMethodBase = UnquantizedHostLinearMethod()
    else:
        fallback = UnquantizedLinearMethod()
    layer.quant_method = fallback
    fallback.create_weights(
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    )
    return True


def create_weight_parameter(
    output_size_per_partition: int,
    input_size_per_partition: int,
    weight_loader: Callable | None,
    params_dtype: torch.dtype,
) -> torch.nn.Parameter:
    """
    Create int8 weight parameter.
    """
    return ModelWeightParameter(
        data=torch.empty(
            output_size_per_partition,
            input_size_per_partition,
            dtype=params_dtype,
        ),
        input_dim=1,
        output_dim=0,
        weight_loader=weight_loader,
    )


class DiffusionInt8Config(QuantizationConfig):
    """INT8 quantization config for diffusion transformers.

    Supports online (dynamic) quantization from BF16/FP16 checkpoints
    and offline quantization from serialized INT8 checkpoints.
    Works on both CUDA and NPU platforms.
    """

    def __init__(
        self,
        is_checkpoint_int8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.is_checkpoint_int8_serialized = is_checkpoint_int8_serialized

        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "int8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Have verified on A100 and H20, but not on oldest versions.
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.ignored_layers is not None:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DiffusionInt8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_int8_serialized = "int8" in quant_method
        activation_scheme = cls.get_from_keys_or(config, ["activation_scheme"], "dynamic")
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)

        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(config, ["modules_to_not_convert"], None)
        return cls(
            is_checkpoint_int8_serialized=is_checkpoint_int8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            if not self.is_checkpoint_int8_serialized:
                if current_omni_platform.is_cuda():
                    online_method = Int8OnlineLinearMethod(self)
                elif current_omni_platform.is_npu():
                    online_method = NPUInt8OnlineLinearMethod(self)
                else:
                    raise NotImplementedError("The current platform is not supported int8 online quant.")
                return online_method
            else:
                if current_omni_platform.is_cuda():
                    offline_method = Int8LinearMethod(self)
                elif current_omni_platform.is_npu():
                    offline_method = NPUInt8LinearMethod(self)
                else:
                    raise NotImplementedError("The current platform is not supported int8 offline quant.")
                return offline_method
        return None


class BaseInt8LinearMethod(LinearMethodBase):
    """
    Linear method for Int8
    Supports loading Int8 checkpoints with static weight scale and dynamic activation scale.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: DiffusionInt8Config):
        self.quant_config = quant_config
        self.out_dtype = torch.get_default_dtype()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        params_dtype = torch.int8 if self.quant_config.is_checkpoint_int8_serialized else params_dtype
        weight = create_weight_parameter(
            output_size_per_partition=output_size_per_partition,
            input_size_per_partition=input_size_per_partition,
            weight_loader=weight_loader,
            params_dtype=params_dtype,
        )
        layer.register_parameter("weight", weight)

        if self.quant_config.is_checkpoint_int8_serialized:
            scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_scale", scale)

    def process_weights_after_loading(self, layer: Module) -> None:
        raise NotImplementedError("No BaseInt8LinearMethod process_weights_after_loading implementation.")

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("No BaseInt8LinearMethod apply implementation.")


class LazyWeightMixin:
    """
    Mixin for lazy weight loading with meta device.
    weighs are created on meta device and materialized just-in-time during loadding.
    """

    uses_meta_device: bool = True

    # This mixin knows when a layer's weight is final, so it can hand the layer
    # back to the host right there. Loaders that intend to offload the whole
    # model after loading opt in per layer via ``enable_offload_after_quant``.
    supports_offload_after_quant: bool = True
    _offload_after_quant: bool = False

    def enable_offload_after_quant(self) -> None:
        """Return each layer to host memory as soon as it has been quantized.

        Caps the load-time device footprint at one layer instead of the whole
        model. A quant method instance belongs to a single layer, so this is not
        a global switch.
        """
        self._offload_after_quant = True

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        # WEIGHT
        def patched_weight_loader(param, loaded_weight, *args, **kwargs):
            # track how many elements we have updated
            if not hasattr(layer, "_loaded_numel"):
                layer._loaded_numel = 0

                # when the first `loaded_weight` is about to be
                # loaded to `param`, materialize `param` just-in-time
                weight = ModelWeightParameter(
                    data=torch.empty_like(layer.weight, device=layer._load_device),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=patched_weight_loader,
                )
                _copy_missing_attrs(layer.weight, weight)
                layer.register_parameter("weight", weight)
                del layer._load_device

            # refresh the reference to `param` to reflect just-in-time
            # materialization
            param = layer.weight

            # load the current weight chunk
            copy_numel_counter = CopyNumelCounter()
            with copy_numel_counter:
                res = weight_loader(param, loaded_weight, *args, **kwargs)  # type: ignore[misc]
            layer._loaded_numel += copy_numel_counter.copied_numel

            # if we have loaded all of the elements, call
            # process_weights_after_loading
            target_loaded_numel = layer.weight.numel()
            if layer._loaded_numel == target_loaded_numel:
                self.process_weights_after_loading(layer)

                # Prevent the usual `process_weights_after_loading` call from doing
                # anything
                layer._already_called_process_weights_after_loading = True

                # This layer's weight is final, so nothing needs it on the
                # accelerator until inference.
                if self._offload_after_quant:
                    layer.to("cpu")

                # Note that we keep `layer._loaded_numel` around just in case
                # there is logic added to vllm in the future which calls a
                # weight loader twice - we do not want to re-initialize in
                # that case.

            return res

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                # materialized just-in-time in `patched_weight_loader`
                device="meta",
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=patched_weight_loader,
        )
        # stash the correct device for `patched_weight_loader`
        layer._load_device = torch.get_default_device()
        layer.register_parameter("weight", weight)


class UnquantizedHostLinearMethod(UnquantizedLinearMethod):
    """Unquantized linear method whose weight loads straight into host memory.

    Layers wider than ``NPU_QUANT_MATMUL_MAX_OUT_FEATURES`` cannot be quantized,
    so under the ordinary path their bf16 weights materialize on the accelerator
    at construction time and stay there until the whole model is moved off after
    loading. When the loader has entered ``load_unquantizable_fallback_on_cpu()``
    (offload-after-quant, i.e. DLO), that round trip is pure startup peak — the
    weights end up pinned on the host either way and only visit the accelerator
    through the offload backend's runtime prefetch.

    This method mirrors ``LazyWeightMixin``'s meta + just-in-time materialize
    pattern, but materializes on CPU and skips quantization entirely.
    """

    # The weight really is deferred on meta until loading, like the online
    # quant methods, so the loader's meta-aware bookkeeping treats the layer
    # correctly. Offload-after-quant marking is deliberately not advertised:
    # the weight never visits the accelerator, so there is nothing to return.
    uses_meta_device: bool = True

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        def patched_weight_loader(param, loaded_weight, *args, **kwargs):
            # Materialize on host just-in-time: checkpoint chunks are CPU
            # tensors, so loading stays CPU->CPU with no accelerator round trip.
            if layer.weight.device.type == "meta":
                weight = ModelWeightParameter(
                    data=torch.empty_like(layer.weight, device="cpu"),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=patched_weight_loader,
                )
                _copy_missing_attrs(layer.weight, weight)
                layer.register_parameter("weight", weight)

            # refresh the reference to `param` to reflect just-in-time
            # materialization
            param = layer.weight

            copy_numel_counter = CopyNumelCounter()
            with copy_numel_counter:
                res = weight_loader(param, loaded_weight, *args, **kwargs)  # type: ignore[misc]
            layer._loaded_numel = getattr(layer, "_loaded_numel", 0) + copy_numel_counter.copied_numel

            if layer._loaded_numel == layer.weight.numel():
                # process_weights_after_loading is a no-op for unquantized
                # weights; the flag keeps the post-load sweep from bouncing the
                # host tensor to the accelerator and back just to call it.
                layer._already_called_process_weights_after_loading = True

            return res

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                # materialized just-in-time in `patched_weight_loader`
                device="meta",
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=patched_weight_loader,
        )
        layer.register_parameter("weight", weight)


class Int8LinearMethod(BaseInt8LinearMethod):
    """
    Linear method for Int8
    Supports loading Int8 checkpoints.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: DiffusionInt8Config):
        super().__init__(quant_config)

        self.int8_linear = init_int8_linear_kernel(
            is_channelwise=False,
            is_static_input_scheme=False,
            input_symmetric=True,
            module_name=self.__class__.__name__,
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        self.int8_linear.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.int8_linear.apply_weights(layer, x, bias)


class NPUInt8LinearMethod(BaseInt8LinearMethod):
    """
    NPU Linear method for Int8
    Supports loading Int8 checkpoints.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: DiffusionInt8Config):
        super().__init__(quant_config)

    def create_weights(self, layer: torch.nn.Module, *args, **kwargs) -> None:
        if _fell_back_to_unquantized_npu(layer, *args, **kwargs):
            return
        super().create_weights(layer, *args, **kwargs)

    def process_weights_after_loading(self, layer: Module) -> None:
        layer.weight.data = layer.weight.data.t().contiguous()
        layer.weight_scale.data = layer.weight_scale.data.squeeze()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ori_shape = x.shape
        ori_dtype = x.dtype

        x = x.reshape(-1, ori_shape[-1])
        quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x)

        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            bias=bias,
            pertoken_scale=pertoken_scale,
            output_dtype=ori_dtype,
        )
        output = output.reshape(*ori_shape[:-1], -1)
        return output


class Int8OnlineLinearMethod(LazyWeightMixin, Int8LinearMethod):
    """
    Online version of Int8LinearMethod, loads the fp16/bf16 checkpoint
    and quantized the weights during loading.
    """

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        if layer.weight.device == torch.device("meta"):
            weight = ModelWeightParameter(
                data=torch.empty_like(layer.weight, device=layer._load_device),
                input_dim=1,
                output_dim=0,
                weight_loader=layer.weight.weight_loader,
            )
            _copy_missing_attrs(layer.weight, weight)
            layer.register_parameter("weight", weight)
            initialize_single_dummy_weight(layer.weight)

        w_q_name, w_s_name, i_s_name, i_zp_name, azp_adj_name = self.int8_linear.layer_param_names
        qweight, weight_scale, _ = ops.scaled_int8_quant(layer.weight, scale=None)

        # Update layer with new values.
        replace_parameter(layer, w_q_name, torch.nn.Parameter(qweight.t().data, requires_grad=False))
        replace_parameter(layer, w_s_name, torch.nn.Parameter(weight_scale.data, requires_grad=False))

        setattr(layer, i_s_name, None)
        setattr(layer, i_zp_name, None)
        setattr(layer, azp_adj_name, None)

        # Prevent duplicate processing (e.g., during weight reload)
        layer._already_called_process_weights_after_loading = True


class NPUInt8OnlineLinearMethod(LazyWeightMixin, NPUInt8LinearMethod):
    """
    NPU Online version of Int8LinearMethod, loads the fp16/bf16 checkpoint
    and quantized the weights during loading.
    """

    def create_weights(self, layer: torch.nn.Module, *args, **kwargs) -> None:
        # NPUInt8LinearMethod's override is unreachable from here: LazyWeightMixin
        # comes first in the MRO and does not call super().
        if _fell_back_to_unquantized_npu(layer, *args, **kwargs):
            return
        super().create_weights(layer, *args, **kwargs)

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        if layer.weight.device == torch.device("meta"):
            weight = ModelWeightParameter(
                data=torch.empty_like(layer.weight, device=layer._load_device),
                input_dim=1,
                output_dim=0,
                weight_loader=layer.weight.weight_loader,
            )
            _copy_missing_attrs(layer.weight, weight)
            layer.register_parameter("weight", weight)
            initialize_single_dummy_weight(layer.weight)

        weight = layer.weight
        qweight, weight_scale = torch_npu.npu_dynamic_quant(weight)

        qweight = qweight.t().contiguous()

        # Update layer with new values.
        replace_parameter(layer, "weight", qweight)
        replace_parameter(layer, "weight_scale", weight_scale)

        # Prevent duplicate processing (e.g., during weight reload)
        layer._already_called_process_weights_after_loading = True
