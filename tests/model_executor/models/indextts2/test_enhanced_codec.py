# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_omni.model_executor.models.indextts2 import codec as codec_module
from vllm_omni.model_executor.models.indextts2 import preprocess_utils
from vllm_omni.model_executor.models.indextts2.codec import EnhancedCodec


def test_enhanced_codec_decode_restores_hidden_size_and_doubles_time():
    codec = EnhancedCodec(
        codebook_size=8,
        hidden_size=4,
        codebook_dim=2,
        vocos_dim=4,
        vocos_intermediate_dim=8,
        vocos_num_layers=1,
        downsample_scale=2,
    ).eval()

    decoded = codec.decode(torch.tensor([[0, 1, 2]]))

    assert decoded.shape == (1, 6, 4)


def test_enhanced_codec_inference_module_does_not_allocate_unused_encoder():
    codec = EnhancedCodec(
        codebook_size=8,
        hidden_size=4,
        codebook_dim=2,
        vocos_dim=4,
        vocos_intermediate_dim=8,
        vocos_num_layers=1,
    )

    parameter_names = set(dict(codec.named_parameters()))
    assert not any(name.startswith("encoder.") for name in parameter_names)
    assert not any(name.startswith("down.") for name in parameter_names)
    assert any(name.startswith("decoder.") for name in parameter_names)
    assert any(name.startswith("quantizer.") for name in parameter_names)


def test_semantic_codec_loader_selects_enhanced_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "codec.pth"
    checkpoint.touch()
    seen = {}

    class FakeCodec:
        def load_checkpoint(self, path):
            seen["path"] = path

        def to(self, **kwargs):
            seen["to"] = kwargs
            return self

        def eval(self):
            return self

        def parameters(self):
            return ()

    monkeypatch.setattr(codec_module, "EnhancedCodec", lambda **kwargs: FakeCodec())
    preprocess_utils._semantic_codec_cache.clear()

    codec = preprocess_utils.load_semantic_codec(
        str(tmp_path),
        {"codebook_size": 8192},
        torch.device("cpu"),
        codec_type="enhanced",
        checkpoint_name="codec.pth",
    )

    assert isinstance(codec, FakeCodec)
    assert seen["path"] == str(checkpoint)
    assert seen["to"] == {"device": torch.device("cpu"), "dtype": torch.float32}


def test_model_asset_resolver_supports_native_checkpoints_layout(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "codec.pth"
    checkpoint.parent.mkdir()
    checkpoint.touch()

    assert preprocess_utils.resolve_model_file(str(tmp_path), "codec.pth") == str(checkpoint)
