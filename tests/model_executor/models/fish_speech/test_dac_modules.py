# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the vendored DAC codec modules (dac_modules).

Guards the checkpoint contract: the module tree built by build_dac_codec()
must keep the exact parameter/buffer names and shapes that codec.pth was
saved with (verified against the upstream fish-speech 0.1.0 package), and
the encode/decode surfaces used by dac_encoder.py / fish_speech_dac_decoder.py
must keep their shapes.
"""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Spot checks covering every submodule family and both weight-norm styles.
# Derived from the upstream fish-speech 0.1.0 state dict (541 entries total);
# legacy weight_norm (weight_g/weight_v) comes from the descript quantizer
# projections, parametrizations.weight.original0/1 from the causal convs.
EXPECTED_KEY_SAMPLES = [
    # encoder: first conv + deepest encoder-block transformer
    "encoder.block.0.conv.parametrizations.weight.original0",
    "encoder.block.4.block.5.layers.3.attention.wqkv.weight",
    "encoder.block.4.block.5.freqs_cis",
    "encoder.block.1.block.0.block.0.alpha",
    # quantizer: semantic + residual VQ (legacy weight_norm keys)
    "quantizer.semantic_quantizer.quantizers.0.codebook.weight",
    "quantizer.quantizer.quantizers.8.in_proj.weight_g",
    "quantizer.quantizer.quantizers.8.in_proj.weight_v",
    # quantizer pre/post transformers + down/upsample convnext
    "quantizer.pre_module.layers.7.feed_forward.w1.weight",
    "quantizer.post_module.norm.weight",
    "quantizer.downsample.1.1.dwconv.conv.weight",
    "quantizer.upsample.0.0.conv.weight",
    # decoder: transposed conv + final conv
    "decoder.model.1.block.1.conv.parametrizations.weight.original1",
    "decoder.model.6.conv.parametrizations.weight.original0",
]

NUM_STATE_DICT_ENTRIES = 541
DAC_NUM_CODEBOOKS = 10  # 1 semantic + 9 residual


@pytest.fixture(scope="module")
def codec():
    from vllm_omni.model_executor.models.fish_speech.dac_utils import build_dac_codec

    model = build_dac_codec()
    model.eval()
    return model


def test_state_dict_contract(codec):
    sd = codec.state_dict()
    assert len(sd) == NUM_STATE_DICT_ENTRIES
    for key in EXPECTED_KEY_SAMPLES:
        assert key in sd, f"checkpoint-contract key missing: {key}"


def test_no_fish_speech_import_required(codec):
    import sys

    assert "fish_speech" not in sys.modules, "vendored codec must not import the fish-speech package"


@torch.no_grad()
def test_encode_shapes(codec):
    frame_length = codec.frame_length  # 2048 samples per code frame
    wav = torch.randn(1, 1, frame_length * 3 + 100)
    lengths = torch.tensor([wav.shape[-1]], dtype=torch.long)
    codes, code_lengths = codec.encode(wav, lengths)
    assert codes.shape[0] == 1
    assert codes.shape[1] == DAC_NUM_CODEBOOKS
    assert codes.shape[2] == 4  # ceil(3.x frames)
    assert code_lengths.tolist() == [4]
    assert codes.dtype in (torch.int64, torch.int32)


@torch.no_grad()
def test_decode_shapes(codec):
    frames = 8
    codes = torch.randint(0, 1024, (1, DAC_NUM_CODEBOOKS, frames), dtype=torch.long)
    feature_lengths = torch.tensor([frames], dtype=torch.long)
    wav, audio_lengths = codec.decode(codes, feature_lengths)
    assert wav.shape[0] == 1 and wav.shape[1] == 1
    assert audio_lengths.tolist() == [frames * codec.frame_length]
    assert wav.shape[-1] >= audio_lengths[0] - codec.frame_length


@torch.no_grad()
def test_decoder_runner_mutations(codec):
    """The stage-1 runner prunes encode-only parts and bakes weight norm;
    decode must still work afterwards (mirrors fish_speech_dac_decoder)."""
    from torch.nn.utils.parametrize import remove_parametrizations

    codec.encoder = None
    codec.quantizer.pre_module = None
    codec.quantizer.downsample = None
    for module in codec.modules():
        parametrizations = getattr(module, "parametrizations", None)
        if not parametrizations:
            continue
        for name in list(parametrizations.keys()):
            remove_parametrizations(module, name, leave_parametrized=True)

    codes = torch.randint(0, 1024, (2, DAC_NUM_CODEBOOKS, 5), dtype=torch.long)
    wav, audio_lengths = codec.decode(codes, torch.tensor([5, 3], dtype=torch.long))
    assert wav.shape[0] == 2
    assert audio_lengths.tolist() == [5 * codec.frame_length, 3 * codec.frame_length]
