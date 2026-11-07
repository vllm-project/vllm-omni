# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Weight-free regression of the realtime reference codec contract."""

from types import SimpleNamespace

import pytest
import torch

from tests.model_executor.models.moss_tts.test_realtime_batched_gpu_history import load_functions

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_parameterless_realtime_codec_has_actionable_error():
    encode = load_functions("reference_encoder.py", {"_encode_prepared"}, "MossRealtimeReferenceEncoder")[
        "_encode_prepared"
    ]

    class ParameterlessCodec(torch.nn.Module):
        def batch_encode(self, waveforms, num_quantizers):
            pytest.fail("A codec without parameters must be rejected before encoding")

    encoder = SimpleNamespace(_processor=ParameterlessCodec(), _n_vq=16)
    with pytest.raises(ValueError, match="realtime reference codec.*parameter.*device and dtype"):
        encode(encoder, [torch.zeros(1, 8)], graphs=None)
