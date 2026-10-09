# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts_nano.modeling_moss_tts_nano import (
    _configure_attention_implementation,
    _validate_max_num_seqs,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_non_cuda_attention_uses_sdpa():
    class FakeModel:
        attention_implementation = None

        def _set_attention_implementation(self, implementation):
            self.attention_implementation = implementation

    model = FakeModel()
    _configure_attention_implementation(model, torch.device("cpu"))

    assert model.attention_implementation == "sdpa"


def test_moss_tts_nano_rejects_concurrent_active_sequences():
    _validate_max_num_seqs(1)

    with pytest.raises(ValueError, match="requires max_num_seqs=1"):
        _validate_max_num_seqs(2)
