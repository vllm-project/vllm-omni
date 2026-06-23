# Licensed under the License Terms of SongGeneration (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/tencent-ailab/SongGeneration/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. The License Terms of SongGeneration permit
# academic / research / educational use only and prohibit commercial use.
# ==============================================================================
#
# Copyright (C) 2025 Tencent. All rights reserved.
# Modifications Copyright contributors to the vLLM-Omni project.
#
# Vendored from tencent-ailab/SongGeneration (codeclm/modules/conditioners.py),
# adapted for inference-only use in vLLM-Omni.
"""ConditionFuser for SongGeneration v2 LeLM Stage 0.

Vendored from upstream ``codeclm/modules/conditioners.py``.
Prepends/sums condition embeddings into the input sequence.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

# (input1 cond emb, input2 cond emb, mask). Reference uses the same shape.
ConditionType = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class ConditionFuser(nn.Module):
    """Fuses pre-computed condition embeddings into the two LM input streams.

    The fuser is called inside ``LeLM.forward``. For inference we always
    operate in "first step" mode: the prepend conditions are concatenated
    along the sequence dim once, then the AR loop reuses the KV cache so
    subsequent steps skip prepending.
    """

    FUSING_METHODS = ("sum", "prepend")

    def __init__(self, fuse2cond: dict[str, list[str]]):
        super().__init__()
        for k in fuse2cond:
            assert k in self.FUSING_METHODS, f"unknown fuse op: {k}"
        self.fuse2cond: dict[str, list[str]] = fuse2cond
        self.cond2fuse: dict[str, str] = {}
        for fuse_method, conditions in fuse2cond.items():
            for condition in conditions:
                self.cond2fuse[condition] = fuse_method

    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
        conditions: dict[str, ConditionType],
        *,
        first_step: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepend / sum conditions onto both transformer inputs.

        Args:
            input1: ``[B, T, dim]`` token embedding for transformer 1 (stream 0).
            input2: ``[B, T, dim]`` summed embedding for transformer 2 (streams 1+2).
            conditions: ``{name: (e1, e2, mask)}`` where each ``e*`` is
                ``[B, T_cond, dim]``. Keys must be a subset of ``cond2fuse``.
            first_step: whether to actually prepend (only true on the AR
                prefill / first decode step; later steps reuse KV cache).

        Returns:
            Pair of fused tensors with the same shape pattern as the inputs.
            If ``prepend`` fires, the sequence dim grows by ``sum(T_cond)``.
        """
        assert set(conditions.keys()).issubset(set(self.cond2fuse.keys())), (
            f"unknown conditions: expected subset of {set(self.cond2fuse.keys())}, got {set(conditions.keys())}"
        )

        fused_input_1 = input1
        fused_input_2 = input2

        for fuse_op in self.fuse2cond:
            fuse_op_conditions = self.fuse2cond[fuse_op]
            if not fuse_op_conditions:
                continue
            if fuse_op == "sum":
                for cond in fuse_op_conditions:
                    if cond not in conditions:
                        continue
                    e1, e2, _ = conditions[cond]
                    fused_input_1 = fused_input_1 + e1
                    fused_input_2 = fused_input_2 + e2
            elif fuse_op == "prepend":
                if not first_step:
                    continue
                # The fuser config order is the desired concat order; the
                # upstream loops in reverse and concats each in front, which
                # yields exactly that order. We replicate the same loop so
                # the result is bit-identical to the reference.
                reverse_list = deepcopy(fuse_op_conditions)
                reverse_list.reverse()
                for cond in reverse_list:
                    if cond not in conditions:
                        continue
                    e1, e2, _ = conditions[cond]
                    fused_input_1 = torch.cat((e1, fused_input_1), dim=1)
                    fused_input_2 = torch.cat((e2, fused_input_2), dim=1)

        return fused_input_1, fused_input_2


__all__ = ["ConditionFuser", "ConditionType"]
