# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock, call

import pytest
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.ovis_image.pipeline_ovis_image import OvisImagePipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("fail", [False, True])
def test_request_residency_reshards_before_leaving_denoising(fail: bool) -> None:
    pipeline = OvisImagePipeline.__new__(OvisImagePipeline)
    nn.Module.__init__(pipeline)
    pipeline.od_config = Mock(spec=OmniDiffusionConfig, additional_config={"hsdp_reshard_after_forward": False})
    root = Mock(spec=FSDPModule)
    block = Mock(spec=FSDPModule)
    root.modules = Mock(return_value=[root, block])
    pipeline.transformer = root
    events = Mock()
    events.attach_mock(root, "root")
    events.attach_mock(block, "block")

    def denoise() -> None:
        with pipeline._hsdp_denoising_context():
            root.set_reshard_after_forward.assert_called_once_with(False)
            if fail:
                raise RuntimeError("denoising failed")

    if fail:
        with pytest.raises(RuntimeError, match="denoising failed"):
            denoise()
    else:
        denoise()
    assert events.mock_calls[-3:] == [
        call.block.reshard(),
        call.root.reshard(),
        call.root.set_reshard_after_forward(True),
    ]


def test_default_keeps_block_resharding() -> None:
    pipeline = OvisImagePipeline.__new__(OvisImagePipeline)
    nn.Module.__init__(pipeline)
    pipeline.od_config = Mock(spec=OmniDiffusionConfig, additional_config={})
    pipeline.transformer = Mock(spec=FSDPModule)
    with pipeline._hsdp_denoising_context():
        pass
    assert pipeline.transformer.mock_calls == []
