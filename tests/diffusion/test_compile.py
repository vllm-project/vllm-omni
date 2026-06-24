# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from vllm_omni.diffusion.compile import regionally_compile


class _WrappedBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.compile_called = False

    def compile(self, *args, **kwargs):
        self.compile_called = True
        return self


class _ModelWithWrappedRepeatedBlocks(nn.Module):
    _repeated_blocks = ["OriginalBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks"]

    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_WrappedBlock(), _WrappedBlock()])
        self.other_blocks = nn.ModuleList([_WrappedBlock()])


def test_regionally_compile_matches_wrapped_blocks_by_declared_container_attr():
    model = _ModelWithWrappedRepeatedBlocks()

    regionally_compile(model)

    assert all(block.compile_called for block in model.transformer_blocks)
    assert not model.other_blocks[0].compile_called
