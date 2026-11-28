# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Adapter for the paper's LSE-output SpargeAttn CUDA provider.

The validated optional build is thu-ml/SpargeAttn return_lse@bdad543f plus
tools/sparsp/spargeattn_lse_output.patch from this repository. Upstream's
integer return_lse flag is a different API and cannot implement this contract.
"""

import torch

from vllm_omni.diffusion.attention.sparsp.config import SparSPSpec


class SpargeProvider:
    q_block = 128
    k_block = 64

    @staticmethod
    def validate_available() -> None:
        try:
            from spas_sparsp import _qattn
        except ImportError as exc:
            raise ImportError(
                "SPARSP_ATTN requires the LSE-output SpargeAttn CUDA build; see the SparSP guide"
            ) from exc
        entry = getattr(_qattn, "qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold", None)
        if (
            entry is None
            or "arg14: torch.Tensor" not in (entry.__doc__ or "")
            or getattr(_qattn, "sparsp_lse_current_stream", None) != 1
        ):
            raise ImportError(
                "SPARSP_ATTN requires the supplied SpargeAttn patch: Tensor LSE output and current-stream CUDA launch"
            )

    def __init__(self, config: SparSPSpec, heads: int, device: torch.device) -> None:
        self.validate_available()
        from spas_sparsp import _qattn, utils

        self.kernel = _qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold
        self.utils = utils
        self.sim = torch.full((heads,), config.sim_threshold, device=device, dtype=torch.float32)
        self.pv = torch.full((heads,), config.pv_threshold, device=device, dtype=torch.float32)

    def pool(self, tensor: torch.Tensor, block: int) -> tuple[torch.Tensor, torch.Tensor]:
        mean, similar = self.utils.get_pool_sim_triton_simmean(tensor, block, self.sim)
        if block == self.q_block and tensor.shape[-2] % block:
            # Match the ZIP's _block_map: the native similarity check includes
            # invalid rows of the trailing Q128 window and always returns False.
            # Its pooled mean uses the valid tokens. Trust that partial mean;
            # protecting the whole row instead forces every KV block into the
            # demand union, even when the remaining queries are sparse.
            similar[..., -1] = True
        return mean, similar

    def quantize(self, tensor: torch.Tensor, block: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.utils.get_quant(tensor, None, block)

    def attend(
        self,
        query: torch.Tensor,
        query_scale: torch.Tensor,
        key: torch.Tensor,
        key_scale: torch.Tensor,
        value: torch.Tensor,
        selected: torch.Tensor,
        output: torch.Tensor,
        lse: torch.Tensor,
        scale: float,
    ) -> None:
        lut, counts = self.utils.block_map_lut_triton(selected.contiguous())
        self.kernel(
            query,
            key,
            value,
            output,
            lut,
            counts,
            self.pv,
            query_scale,
            key_scale,
            1,
            False,
            1,
            scale,
            0,
            lse,
        )
