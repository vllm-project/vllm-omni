# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.common.ming.audio_projector import Ming3AudioProjector
from vllm_omni.model_executor.models.common.ming.whisper_encoder import WhisperAudioEncoder
from vllm_omni.model_executor.models.ming3_audio.audio_encoder_graphs import Ming3AudioEncoderGraphs

pytestmark = [pytest.mark.core_model]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("normalize", [False, True])
@torch.inference_mode()
def test_audio_encoder_graphs_match_unpadded_pass(normalize):
    torch.manual_seed(0)
    encoder = WhisperAudioEncoder(
        n_mels=16, n_ctx=1500, n_state=128, n_head=2, n_layer=2, learned_positional_embedding=True
    )
    torch.nn.init.normal_(encoder.positional_embedding, std=0.02)
    projector = Ming3AudioProjector(128, 96, kernels=[3, 3, 3], strides=[2, 2, 2], paddings=[1, 1, 1])
    encoder, projector = encoder.cuda().bfloat16().eval(), projector.cuda().bfloat16().eval()
    graphs = Ming3AudioEncoderGraphs(encoder, projector, max_frames=600, normalize=normalize)
    graphs.capture()
    for frames in (37, 100, 101, 250, 599):
        features = torch.randn(frames, 16, device="cuda")
        encoded, lengths = encoder.forward_padded(features.unsqueeze(0), [frames])
        expected = projector(encoded)[0, : projector.output_length(lengths[0])]
        if normalize:
            expected = F.normalize(expected, dim=-1)
        actual = graphs.encode(features)
        assert actual.shape == expected.shape
        torch.testing.assert_close(actual.float(), expected.float(), atol=3e-2, rtol=3e-2)
    assert graphs.encode(torch.randn(601, 16, device="cuda")) is None
