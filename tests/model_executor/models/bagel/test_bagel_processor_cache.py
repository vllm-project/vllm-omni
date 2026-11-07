# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The AR-stage processor must not be rebuilt per output size.

The image endpoints put the requested size (``target_h`` / ``target_w``) and the
output ``modalities`` into ``mm_processor_kwargs``. vLLM keeps only the kwargs
the processor's ``__call__`` declares out of the processor cache key, so these
request values would make every new size build a new ``OmniBagelProcessor``.
The BAGEL processor has no size-dependent construction, so they are dropped.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from vllm_omni.model_executor.models.bagel.bagel import OmniBagelProcessingInfo, OmniBagelProcessor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_processor_is_shared_across_output_sizes():
    info = object.__new__(OmniBagelProcessingInfo)
    info.ctx = Mock()

    for height, width in ((512, 512), (1024, 576), (688, 512)):
        info.get_hf_processor(target_h=height, target_w=width, modalities=["image"])

    assert info.ctx.get_hf_processor.call_count == 3
    for call in info.ctx.get_hf_processor.call_args_list:
        assert call.args == (OmniBagelProcessor,)
        assert call.kwargs == {}


def test_other_processor_kwargs_are_kept():
    info = object.__new__(OmniBagelProcessingInfo)
    info.ctx = Mock()

    info.get_hf_processor(target_h=512, use_fast=True)

    info.ctx.get_hf_processor.assert_called_once_with(OmniBagelProcessor, use_fast=True)
