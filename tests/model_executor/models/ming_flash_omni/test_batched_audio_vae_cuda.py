# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real CUDA decoder composition across backlog, final flush and row reuse."""

from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.common.ming.audio_vae import Decoder
from vllm_omni.model_executor.models.ming_flash_omni.talker_batch import BatchedAudioVAEDecoder

pytestmark = [pytest.mark.core_model]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs require CUDA")
@pytest.mark.parametrize("holdback", [False, True])
@torch.inference_mode()
def test_graph_decoder_matches_full_waveform_across_backlog_and_row_reuse(holdback):
    torch.manual_seed(42)
    core = (
        Decoder(
            dict(
                vocab_size=32,
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=4096,
                use_sliding_window=True,
                sliding_window=128,
                layer_types=["sliding_attention"] * 2,
                attn_implementation="sdpa",
            ),
            output_dim=16,
            latent_dim=8,
            patch_size=4,
        )
        .eval()
        .to(device="cuda", dtype=torch.bfloat16)
    )
    core.upsampling.frame_holdback = holdback
    decoder = BatchedAudioVAEDecoder(SimpleNamespace(decoder=core), rows=2, max_frames=4096, use_cuda_graphs=True)
    latents = torch.randn(1, 701, 8, device="cuda", dtype=torch.bfloat16)
    expected, _, _ = core.low_level_reconstruct(latents, use_cache=False)
    state: dict = {}
    first, peer = decoder.decode([(state, latents[:, :31], False), ({}, latents[:, :11], True)])
    final = decoder.decode([(state, latents[:, 31:], True)])[0]
    actual = torch.cat([first, final], dim=-1)
    # Packed attention changes BF16 reduction order. Compare the complete
    # waveform, including boundaries, and bound relative RMS error.
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    assert ((actual - expected).float().norm() / expected.float().norm()).item() < 0.02
    assert sorted(decoder.free_rows) == [0, 1]
    again = decoder.decode([({}, latents[:, :11], True)])[0]
    torch.testing.assert_close(again, peer, atol=1e-5, rtol=1e-5)

    bad: dict = {}
    good: dict = {}
    first = decoder.decode([(bad, latents[:, :11], False), (good, latents[:, :11], False)])[1]
    second = decoder.decode([(bad, latents[:, 11:31], False), (good, latents[:, 11:31], False)])[1]
    bad["istft"][1].fill_(-10)
    rejected, final = decoder.decode([(bad, latents[:, 31:], False), (good, latents[:, 31:], True)])
    assert isinstance(rejected, ValueError) and "window envelope underflowed" in str(rejected)
    actual = torch.cat([first, second, final], dim=-1)
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    assert ((actual - expected).float().norm() / expected.float().norm()).item() < 0.02
    assert "row" not in bad and "row" not in good
    assert sorted(decoder.free_rows) == [0, 1]
    torch.testing.assert_close(decoder.decode([({}, latents[:, :11], True)])[0], peer, atol=1e-5, rtol=1e-5)
