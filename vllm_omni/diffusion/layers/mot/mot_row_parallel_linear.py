from __future__ import annotations

import torch
from torch.nn.parameter import Parameter
from vllm.distributed import (
    split_tensor_along_last_dim,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.linear import (
    WEIGHT_LOADER_V2_SUPPORTED,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

from vllm_omni.diffusion.layers.mot.ops.mot_gemm import invoke_mot_gemm


class MoTRowParallelLinear(RowParallelLinear):
    """RowParallelLinear with Mixture-of-Tokens routing.

    text weights: directly on self (self.weight, self.weight_scale, ...),
                   created by RowParallelLinear.__init__ standard process.
    vae  weights: on permanent submodule self.gen_exp (self.gen_exp.weight, ...),
                   created by quant_method.create_weights(self.gen_exp, ...).
                   gen_exp.quant_method points to the same quant_method, enabling
                   vLLM framework's process_weights_after_loading to automatically
                   discover and process it.

    Forward behavior:
        - und mode (text_indices is None): fully reuse super().forward()
        - gen mode: call MoT fused GEMM kernel, then execute TP all-reduce
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        vae_bias: bool = False,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        # ---- Step 1: Parent class creates text weights ----
        # super().__init__ internally calls:
        #   quant_method.create_weights(self, ...) → self.weight, self.weight_scale, ...
        super().__init__(
            input_size,
            output_size,
            bias,
            input_is_parallel,
            skip_bias_add,
            params_dtype,
            reduce_results,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        # ---- Step 2: Create vae weights (permanent submodule) ----
        assert self.quant_method is not None

        self.gen_exp = torch.nn.Module()

        # Select weight_loader consistent with text
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

        # Enable gen_exp to be automatically discovered by vLLM framework's process_weights_after_loading
        self.gen_exp.quant_method = self.quant_method

        # ---- Step 3: vae bias ----
        # RowParallelLinear's bias is full output_size (not sharded)
        if vae_bias:
            self.gen_exp.bias = Parameter(torch.empty(self.output_size, dtype=self.params_dtype))
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

        # ---- gen mode ----
        # Handle input_is_parallel (consistent with parent class logic)
        if self.input_is_parallel:
            input_parallel = input_
        else:
            split_input = split_tensor_along_last_dim(input_, num_partitions=self.tp_size)
            input_parallel = split_input[self.tp_rank].contiguous()

        # Fused MoT GEMM
        output_parallel = self._mot_gemm_dispatch(input_parallel, text_indices, vae_indices)

        # ---- TP communication: all-reduce (consistent with parent class) ----
        if self.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        if not self.return_bias:
            return output

        if self.skip_bias_add and text_indices is not None and (self.bias is not None or self.gen_exp.bias is not None):
            # Construct per-token mixed bias
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
    #  MoT GEMM Dispatcher
    # ==================================================================
    def _mot_gemm_dispatch(
        self,
        x: torch.Tensor,
        text_indices: torch.Tensor,
        vae_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch to different MoT kernel paths based on weight dtype / quant attributes."""

        # Any other backend (ROCm, XPU, TPU, CPU) uses the safe vllm fallback.
        if not current_platform.is_cuda():
            return self._mot_fallback(x, text_indices, vae_indices)

        # RowParallelLinear: bias only fused into GEMM at rank 0,
        # other ranks pass None to avoid duplicate accumulation
        bias_text = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        bias_vae = None if (self.tp_rank > 0 or self.skip_bias_add) else self.gen_exp.bias

        # Determine quantization type by weight dtype, avoid isinstance coupling to specific quant_method
        # TODO: Currently does not support online quantization fp8, does not support quantization types involving int4
        w_text = self.weight
        w_vae = self.gen_exp.weight
        assert w_text.dtype.is_floating_point == w_vae.dtype.is_floating_point, (
            "weight of text expert and image expert should be the same dtype."
        )

        # w_text.dtype.itemsize >= 2 means bytes_per_element >= 2 (16bits or 32bits)
        if w_text.dtype.is_floating_point and w_text.dtype.itemsize >= 2:
            # ---- Path 0: BF16 / FP16 (unquantized) ----
            return self._mot_gemm_unquantized(
                x,
                text_indices,
                vae_indices,
                bias_text,
                bias_vae,
            )
        elif w_text.dtype == torch.float8_e4m3fn:
            # ---- Path 1: FP8 W8A8 ----
            return self._mot_gemm_fp8_w8a8(
                x,
                text_indices,
                vae_indices,
                bias_text,
                bias_vae,
            )
        elif w_text.dtype == torch.int8:
            # ---- Path 2: Weight-Only INT8 W8A16 ----
            return self._mot_gemm_weight_only(
                x,
                text_indices,
                vae_indices,
                bias_text,
                bias_vae,
            )
        else:
            # ---- Fallback: gather/scatter + quant_method.apply ----
            return self._mot_fallback(
                x,
                text_indices,
                vae_indices,
            )

    # ==================================================================
    #  Specific implementations for each quantization path
    # ==================================================================
    def _mot_gemm_unquantized(self, x, text_idx, vae_idx, bias_t, bias_v):
        """BF16/FP16 path."""
        N = self.output_size_per_partition
        C = torch.empty(x.size(0), N, dtype=x.dtype, device=x.device)
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
        2) MoT GEMM: text/vae each use different fp8 weights and weight_scale
        3) Dequantization: completed by MoT kernel internal epilogue
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
        C = torch.empty(x.size(0), N, dtype=x.dtype, device=x.device)

        # weight has been transposed to (K, N) in process_weights_after_loading
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

        Activations remain bf16/fp16, weights are int8 + per-channel scale.
        MoT kernel internally performs immediate dequantization.
        """
        N = self.output_size_per_partition
        C = torch.empty(x.size(0), N, dtype=x.dtype, device=x.device)
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
        """Fallback: degrade to gather/scatter + quant_method.apply.

        For unsupported quantization types, call standard forward separately for text/vae tokens.
        """
        assert self.quant_method is not None

        # RowParallelLinear: bias only fused at rank 0
        bias_text = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        bias_vae = None if (self.tp_rank > 0 or self.skip_bias_add) else self.gen_exp.bias

        output = torch.empty(
            x.size(0),
            self.output_size_per_partition,
            dtype=x.dtype,
            device=x.device,
        )
        # text tokens → standard quant_method (operate on weights on self)
        output[text_idx] = self.quant_method.apply(self, x[text_idx], bias_text)
        # vae tokens → same quant_method (operate on weights on gen_exp)
        output[vae_idx] = self.quant_method.apply(
            self.gen_exp,
            x[vae_idx],
            bias_vae,
        )
        return output
