import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.layers.custom_op import CustomOp

logger = init_logger(__name__)


class AdaLayerNorm(CustomOp):
    """
    AdaLayerNorm:
        out = layernorm(x) * (1 + scale) + shift
    """

    def __init__(self, hidden_size: int, elementwise_affine: bool = False, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.hidden_size = hidden_size
        self.layernorm = nn.LayerNorm(self.hidden_size, elementwise_affine=self.elementwise_affine, eps=self.eps)

    def preprocess(
        self,
        mod_params: torch.Tensor,
        index: torch.Tensor = None,
    ) -> torch.Tensor:
        # shift: b d, scale: b d, gate: b d
        shift, scale, gate = mod_params.chunk(3, dim=-1)

        if index is not None:
            # Assuming mod_params batch dim is 2*actual_batch (chunked into 2 parts)
            # So shift, scale, gate have shape [2*actual_batch, d]
            actual_batch = shift.size(0) // 2
            shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]  # each: [actual_batch, d]
            scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
            gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]

            # index: [b, l] where b is actual batch size
            # Expand to [b, l, 1] to match feature dimension
            index_expanded = index.unsqueeze(-1)  # [b, l, 1]

            # Expand chunks to [b, 1, d] then broadcast to [b, l, d]
            shift_0_exp = shift_0.unsqueeze(1)  # [b, 1, d]
            shift_1_exp = shift_1.unsqueeze(1)  # [b, 1, d]
            scale_0_exp = scale_0.unsqueeze(1)
            scale_1_exp = scale_1.unsqueeze(1)
            gate_0_exp = gate_0.unsqueeze(1)
            gate_1_exp = gate_1.unsqueeze(1)

            # Use torch.where to select based on index
            shift_result = torch.where(index_expanded == 0, shift_0_exp, shift_1_exp)
            scale_result = torch.where(index_expanded == 0, scale_0_exp, scale_1_exp)
            gate_result = torch.where(index_expanded == 0, gate_0_exp, gate_1_exp)
        else:
            shift_result = shift.unsqueeze(1)
            scale_result = scale.unsqueeze(1)
            gate_result = gate.unsqueeze(1)

        return shift_result, scale_result, gate_result

    def forward_cuda(
        self,
        x: torch.Tensor,
        mod_params: torch.Tensor,
        index: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.forward_native(x, mod_params, index)

    def forward_hip(
        self,
        x: torch.Tensor,
        mod_params: torch.Tensor,
        index: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.forward_native(x, mod_params, index)

    def forward_npu(
        self,
        x: torch.Tensor,
        mod_params: torch.Tensor,
        index: torch.Tensor = None,
    ) -> torch.Tensor:
        shift_result, scale_result, gate_result = self.preprocess(mod_params, index)

        import torch_npu

        output = torch_npu.npu_layer_norm_eval(
            x, normalized_shape=[self.hidden_size], weight=(1 + scale_result), bias=shift_result, eps=self.eps
        )

        return output, gate_result

    def forward_xpu(
        self,
        x: torch.Tensor,
        mod_params: torch.Tensor,
        index: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.forward_native(x, mod_params, index)

    def forward_native(
        self,
        x: torch.Tensor,
        mod_params: torch.Tensor,
        index: torch.Tensor = None,
    ) -> torch.Tensor:
        shift_result, scale_result, gate_result = self.preprocess(mod_params, index)

        return self.layernorm(x) * (1 + scale_result) + shift_result, gate_result
