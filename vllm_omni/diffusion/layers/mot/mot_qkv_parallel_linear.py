from __future__ import annotations

import torch
from torch.nn.parameter import Parameter
from vllm.distributed import (
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    WEIGHT_LOADER_V2_SUPPORTED,
    QKVParallelLinear,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

from vllm_omni.diffusion.layers.mot.ops.mot_gemm import invoke_mot_gemm

logger = init_logger(__name__)


class MoTQKVParallelLinear(QKVParallelLinear):
    """QKVParallelLinear with Mixture-of-Tokens routing.

    Text weights: stored directly on self (self.weight, self.weight_scale, ...),
              created through the standard QKVParallelLinear.__init__ process.

    VAE weights: stored in the permanent submodule self.gen_exp
             (self.gen_exp.weight, ...),
             created via quant_method.create_weights(self.gen_exp, ...).
             gen_exp.quant_method points to the same quant_method, so that
             the vLLM framework’s process_weights_after_loading
             can automatically detect and process it.

    Forward behavior:
        - und mode (text_indices is None): fully reuse super().forward()
        - gen mode: call the MoT fused GEMM kernel
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = True,
        vae_bias: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        v_head_size: int | None = None,
    ):
        # ---- 1) Parent class creates text weights ----
        # super().__init__  will do the following:
        # quant_method.create_weights(self, ...) → self.weight, self.weight_scale, ...
        # QKVParallelLinear hardcodes gather_output=False
        super().__init__(
            hidden_size,
            head_size,
            total_num_heads,
            total_num_kv_heads,
            bias,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
            v_head_size=v_head_size,
        )

        # ---- 2) Create vae weights (permanent submodule) ----
        assert self.quant_method is not None

        self.gen_exp = torch.nn.Module()

        # Use the same weight_loader as text
        vae_weight_loader = (
            self.weight_loader_v2
            if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
            else self.weight_loader
        )
        self.quant_method.create_weights(
            layer=self.gen_exp,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=vae_weight_loader,
        )

        # Make gen_exp discoverable by vLLM framework's process_weights_after_loading
        self.gen_exp.quant_method = self.quant_method

        # ---- 3) vae bias ----
        if vae_bias:
            self.gen_exp.bias = Parameter(torch.empty(self.output_size_per_partition, dtype=self.params_dtype))
            set_weight_attrs(
                self.gen_exp.bias,
                {"output_dim": 0, "weight_loader": self.weight_loader},
            )
        else:
            self.gen_exp.register_parameter("bias", None)

        self.update_param_tp_status()

    # ==================================================================
    #  Forward
    # ==================================================================
    def forward(
        self,
        input_: torch.Tensor,
        text_indices: torch.Tensor | None = None,
        vae_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        # ---- und mode: fully reuse parent class (only text path) ----
        if text_indices is None:
            return super().forward(input_)

        # ---- gen mode: fuse MoT GEMM ----
        output_parallel = self._mot_gemm_dispatch(input_, text_indices, vae_indices)

        # QKVParallelLinear hardcodes gather_output=False, this branch never enters;
        # retained for future subclass changes
        if self.gather_output and self.tp_size > 1:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel

        if not self.return_bias:
            return output

        if self.skip_bias_add and text_indices is not None and (self.bias is not None or self.gen_exp.bias is not None):
            merged_bias = torch.zeros(
                output.size(0),
                self.output_size_per_partition,
                dtype=output.dtype,
                device=output.device,
            )
            if self.bias is not None:
                merged_bias[text_indices] = self.bias
            if self.gen_exp.bias is not None:
                merged_bias[vae_indices] = self.gen_exp.bias
            return output, merged_bias

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    # ==================================================================
    #  MoT GEMM dispatcher
    # ==================================================================
    def _mot_gemm_dispatch(
        self,
        x: torch.Tensor,
        text_indices: torch.Tensor,
        vae_indices: torch.Tensor,
    ) -> torch.Tensor:
        # Any other backend (ROCm, XPU, TPU, CPU) uses the safe vllm fallback.
        if not current_platform.is_cuda():
            return self._mot_fallback(x, text_indices, vae_indices)

        """Dispatch to different MoT kernel paths based on weight dtype / quant attributes."""
        bias_text = self.bias if not self.skip_bias_add else None
        bias_vae = self.gen_exp.bias if not self.skip_bias_add else None

        w_text = self.weight
        w_vae = self.gen_exp.weight
        assert w_text.dtype.is_floating_point == w_vae.dtype.is_floating_point, (
            "weight of text expert and image expert should be the same dtype."
        )

        # w_text.dtype.itemsize >= 2 means bytes_per_element >= 2 (16bits or 32bits)
        if w_text.dtype.is_floating_point and w_text.dtype.itemsize >= 2:
            return self._mot_gemm_unquantized(
                x,
                text_indices,
                vae_indices,
                bias_text,
                bias_vae,
            )
        elif w_text.dtype == torch.float8_e4m3fn:
            return self._mot_gemm_fp8_w8a8(
                x,
                text_indices,
                vae_indices,
                bias_text,
                bias_vae,
            )
        elif w_text.dtype == torch.int8:
            return self._mot_gemm_weight_only(
                x,
                text_indices,
                vae_indices,
                bias_text,
                bias_vae,
            )
        else:
            return self._mot_fallback(
                x,
                text_indices,
                vae_indices,
            )

    # ==================================================================
    #  Implementation of each quantization path
    # ==================================================================
    def _mot_gemm_unquantized(self, x, text_idx, vae_idx, bias_t, bias_v):
        """BF16/FP16 path."""
        N = self.output_size_per_partition
        C = torch.zeros(x.size(0), N, dtype=x.dtype, device=x.device)
        invoke_mot_gemm(
            A=x,
            B_text=self.weight.data.t(),
            B_vae=self.gen_exp.weight.data.t(),
            C=C,
            bias_text=bias_t,
            bias_vae=bias_v,
            text_indices=text_idx,
            vae_indices=vae_idx,
            A_scale=None,
            B_text_scale=None,
            B_vae_scale=None,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            A_per_channel_quant=False,
            B_per_channel_quant=False,
        )
        return C

    def _mot_gemm_fp8_w8a8(self, x, text_idx, vae_idx, bias_t, bias_v):
        """FP8 W8A8 path.

        1) Activation quantization: reuse vllm's ops.scaled_fp8_quant
        2) MoT GEMM: text/vae use different fp8 weights and weight_scale
        3) De-quantization: done by MoT kernel's epilogue
        """
        from vllm import _custom_ops as ops

        x_2d = x.view(-1, x.shape[-1])
        input_scale = getattr(self, "input_scale", None)
        x_fp8, x_scale = ops.scaled_fp8_quant(
            x_2d,
            input_scale,
            use_per_token_if_dynamic=True,
        )

        N = self.output_size_per_partition
        C = torch.zeros(x.size(0), N, dtype=x.dtype, device=x.device)

        invoke_mot_gemm(
            A=x_fp8,
            B_text=self.weight.data,
            B_vae=self.gen_exp.weight.data,
            C=C,
            bias_text=bias_t,
            bias_vae=bias_v,
            text_indices=text_idx,
            vae_indices=vae_idx,
            A_scale=x_scale,
            B_text_scale=self.weight_scale.data,
            B_vae_scale=self.gen_exp.weight_scale.data,
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            A_per_channel_quant=True,
            B_per_channel_quant=False,
        )
        return C

    def _mot_gemm_weight_only(self, x, text_idx, vae_idx, bias_t, bias_v):
        """Weight-Only W8A16 path.

        Activation values kept as bf16/fp16, weights are int8 + per-channel scale.
        De-quantization done by MoT kernel's epilogue.
        """
        N = self.output_size_per_partition
        C = torch.zeros(x.size(0), N, dtype=x.dtype, device=x.device)
        invoke_mot_gemm(
            A=x,
            B_text=self.weight.data.t(),
            B_vae=self.gen_exp.weight.data.t(),
            C=C,
            bias_text=bias_t,
            bias_vae=bias_v,
            text_indices=text_idx,
            vae_indices=vae_idx,
            A_scale=None,
            B_text_scale=self.weight_scale.data,
            B_vae_scale=self.gen_exp.weight_scale.data,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=True,
            use_int4_w4a16=False,
            A_per_channel_quant=False,
            B_per_channel_quant=True,
        )
        return C

    def _mot_fallback(self, x, text_idx, vae_idx):
        """Fallback: fall back to gather/scatter + quant_method.apply.

        For unsupported quantization types, call standard forward for text/vae tokens.
        """
        assert self.quant_method is not None

        bias_text = self.bias if not self.skip_bias_add else None
        bias_vae = self.gen_exp.bias if not self.skip_bias_add else None

        output = torch.zeros(
            x.size(0),
            self.output_size_per_partition,
            dtype=x.dtype,
            device=x.device,
        )
        output_text = self.quant_method.apply(self, x[text_idx], bias_text)

        output_vae = self.quant_method.apply(
            self.gen_exp,
            x[vae_idx],
            bias_vae,
        )
        output[text_idx] = output_text
        output[vae_idx] = output_vae
        return output
