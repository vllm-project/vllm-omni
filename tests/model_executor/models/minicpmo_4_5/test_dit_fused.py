# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused Code2Wav DiT body matches the ragged reference."""

import pytest
import torch
import torch.nn as nn

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import BatchedToken2Wav
from vllm_omni.model_executor.models.minicpmo_4_5.dit_fused import (
    blocks_forward_chunk_fused,
    dit_modulation,
    supports_fused_body,
    tiled_attention_supported,
)

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)]

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _upstream_dit(device: str) -> nn.Module:
    for name in ("cosyvoice2.flow.decoder_dit", "stepaudio2.cosyvoice2.flow.decoder_dit"):
        try:
            import importlib

            decoder_dit = importlib.import_module(name)
            break
        except ImportError:
            pass
    else:
        decoder_dit = pytest.importorskip("cosyvoice2.flow.decoder_dit")
    torch.manual_seed(0)
    estimator = decoder_dit.DiT(in_channels=320, out_channels=80, depth=2, num_heads=4, head_dim=16, hidden_size=64)
    with torch.no_grad():
        for parameter in estimator.parameters():
            parameter.normal_(0.0, 0.08)
    return estimator.eval().to(device)


def _run(body, estimator, *, rows: int, frames: int, cached: int, lengths):
    torch.manual_seed(1)
    depth = len(estimator.blocks)
    channels = estimator.blocks[0].conv.block[1].in_channels
    attn = estimator.blocks[0].attn
    device = next(estimator.parameters()).device
    dtype = next(estimator.parameters()).dtype
    estimator_input = torch.randn(rows, 320, frames, device=device, dtype=dtype)
    time_embedding = estimator.t_embedder(torch.rand((), device=device, dtype=dtype).expand(rows)).unsqueeze(1)
    mask = torch.rand(rows, frames, cached + frames, device=device) > 0.3
    mask[..., 0] = True
    cnn = [torch.randn(rows, 2 * channels, 2, device=device, dtype=dtype) for _ in range(depth)]
    att = [
        torch.randn(rows, attn.num_heads, cached, 2 * attn.head_dim, device=device, dtype=dtype) for _ in range(depth)
    ]
    cnn_out = torch.zeros((depth, rows, 2 * channels, 2), device=device, dtype=dtype)
    att_out = torch.zeros((depth, rows, attn.num_heads, cached + frames, 2 * attn.head_dim), device=device, dtype=dtype)
    with torch.no_grad():
        out = body(estimator, estimator_input, time_embedding, mask, cnn, att, cnn_out, att_out, lengths)
    return out, cnn_out, att_out


@pytest.mark.parametrize("device", _DEVICES)
def test_fused_body_matches_ragged_body(device: str) -> None:
    estimator = _upstream_dit(device)
    assert supports_fused_body(estimator)
    expected = _run(
        BatchedToken2Wav._blocks_forward_chunk_ragged, estimator, rows=4, frames=6, cached=5, lengths=[6, 4]
    )
    actual = _run(blocks_forward_chunk_fused, estimator, rows=4, frames=6, cached=5, lengths=[6, 4])
    for want, got in zip(expected, actual, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=2e-4)


def test_unsupported_estimator_is_refused() -> None:
    estimator = _upstream_dit("cpu")
    estimator.blocks[0].attn.q_norm = nn.Identity()
    assert not supports_fused_body(estimator)


@pytest.mark.parametrize("cached", [0, 5])
def test_ragged_body_reuses_modulation_without_recomputing_adaln(monkeypatch, cached):
    estimator = _upstream_dit("cpu")
    reference = BatchedToken2Wav._blocks_forward_chunk_ragged
    expected = _run(reference, estimator, rows=4, frames=6, cached=cached, lengths=[6, 4])

    def unexpected_forward(*args):
        raise AssertionError("cached modulation must bypass adaLN projections")

    def precomputed(estimator, estimator_input, time_embedding, *args):
        table = dit_modulation(estimator, time_embedding[:1])
        for layer in (*estimator.blocks, estimator.final_layer):
            monkeypatch.setattr(layer.adaLN_modulation, "forward", unexpected_forward)
        return reference(estimator, estimator_input, time_embedding, *args, modulation=table)

    actual = _run(precomputed, estimator, rows=4, frames=6, cached=cached, lengths=[6, 4])
    for want, got in zip(expected, actual, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp16_fused_body_matches_fp16_ragged_body(monkeypatch):
    from vllm_omni.model_executor.models.minicpmo_4_5.cfm_attention import cfm_attention

    estimator = _upstream_dit("cuda").half()
    driver = BatchedToken2Wav.__new__(BatchedToken2Wav)
    nn.Module.__init__(driver)
    driver._ragged_fused_body = True
    calls = []
    attention_calls = []

    def tiled(*args, **kwargs):
        attention_calls.append(True)
        return cfm_attention(*args, **kwargs)

    monkeypatch.setattr(
        "vllm_omni.model_executor.models.minicpmo_4_5.cfm_attention.cfm_attention",
        tiled,
    )

    def fused(*args, **kwargs):
        calls.append(True)
        return blocks_forward_chunk_fused(*args, **kwargs)

    monkeypatch.setattr(
        "vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav.blocks_forward_chunk_fused",
        fused,
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        expected = _run(
            BatchedToken2Wav._blocks_forward_chunk_ragged,
            estimator,
            rows=4,
            frames=6,
            cached=5,
            lengths=[6, 4],
        )
        actual = _run(driver._ragged_body, estimator, rows=4, frames=6, cached=5, lengths=[6, 4])
    assert calls, "FP16 service dispatch must use the fused body"
    assert bool(attention_calls) == tiled_attention_supported(torch.accelerator.current_device_index())
    for want, got in zip(expected, actual, strict=True):
        torch.testing.assert_close(got, want, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cfm_only_fp16_preserves_encoder_precision_and_returns_half_caches():
    from tests.model_executor.models.minicpmo_4_5.test_code2wav_batching import _FakeToken2Wav

    token2wav = _FakeToken2Wav()
    token2wav.flow.encoder.register_parameter("anchor", nn.Parameter(torch.ones(1, device="cuda")))
    token2wav.flow.decoder.estimator = _upstream_dit("cuda")
    token2wav.flow.decoder.rand_noise = torch.randn(1, 80, 100, device="cuda")
    backend = BatchedToken2Wav(token2wav, cfm_float16=True)
    assert not backend.float16
    assert token2wav.flow.encoder.anchor.dtype == torch.float32
    assert next(token2wav.flow.decoder.estimator.parameters()).dtype == torch.float16
    matrix = torch.randn(8, 8, device="cuda")
    with backend._autocast(matrix.device):
        assert (matrix @ matrix).dtype == torch.float32
    with backend._cfm_autocast(matrix.device):
        assert (matrix @ matrix).dtype == torch.float16
    assert not torch.is_autocast_enabled("cuda")
    with torch.inference_mode():
        mel, cnn, att = backend._decode_cfm(
            torch.randn(1, 80, 6, device="cuda"),
            torch.randn(1, 80, device="cuda"),
            torch.randn(1, 80, 6, device="cuda"),
            cnn_cache=None,
            att_cache=None,
        )
    assert mel.dtype == cnn.dtype == att.dtype == torch.float16
    assert token2wav.flow.encoder.anchor.dtype == torch.float32


@pytest.mark.skipif("cuda" not in _DEVICES, reason="requires CUDA")
@pytest.mark.parametrize("precision_flag", ["cfm_float16", "token2wav_float16"])
@pytest.mark.parametrize("tf32_source", ["extra", "env"])
def test_tf32_disabled_fp16_default_initializes_fused_body(tmp_path, monkeypatch, caplog, precision_flag, tf32_source):
    from tests.model_executor.models.minicpmo_4_5.test_code2wav_batching import _config, _FakeToken2Wav
    from vllm_omni.model_executor.models.minicpmo_4_5 import (
        batched_token2wav,
        minicpmo_4_5_token2wav,
    )
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import MiniCPMO45Code2Wav

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.delenv("MINICPMO_CODE2WAV_TF32", raising=False)
    config = _config()
    config.model_config.model = str(tmp_path)
    extra = config.model_config.stage_connector_config["extra"]
    extra.pop("prompt_wav")
    extra.update({precision_flag: True, "enable_cfm_graph": True, "token2wav_device_spk": False})
    if tf32_source == "env":
        monkeypatch.setenv("MINICPMO_CODE2WAV_TF32", "off")
    else:
        extra["token2wav_allow_tf32"] = False
    (tmp_path / "assets" / "token2wav").mkdir(parents=True)
    (tmp_path / "assets" / "HT_ref_audio.wav").touch()

    def load_assets(path, **kwargs):
        token2wav = _FakeToken2Wav()
        token2wav.float16 = kwargs["float16"]
        token2wav.flow.encoder.register_parameter("anchor", nn.Parameter(torch.ones(1, device="cuda")))
        token2wav.flow.decoder.estimator = _upstream_dit("cuda")
        token2wav.flow.decoder.rand_noise = torch.randn(1, 80, 100, device="cuda")
        # Match the asset loader's full-flow conversion. CFM-only conversion
        # must instead happen in the real BatchedToken2Wav constructor below.
        if token2wav.float16:
            token2wav.flow.half()
        return token2wav

    monkeypatch.setattr(minicpmo_4_5_token2wav, "MiniCPMO45Token2wav", load_assets)
    monkeypatch.setattr(batched_token2wav.logger, "propagate", True)
    model = MiniCPMO45Code2Wav(vllm_config=config)
    model._build_backend()
    backend = model.backend
    assert isinstance(backend, BatchedToken2Wav)
    assert model._tf32_mode == "off"
    assert not backend._cfm_tf32
    assert backend._ragged_fused_body
    assert "cfm_fused_body requested but unsupported here" not in caplog.text
    assert backend._estimator_att_compute_dtype == backend._estimator_att_cache_dtype == torch.float16
    estimator = backend.flow.decoder.estimator
    assert all(p.dtype == torch.float16 for p in estimator.parameters())
    encoder_dtype = torch.float32 if precision_flag == "cfm_float16" else torch.float16
    assert backend.flow.encoder.anchor.dtype == encoder_dtype

    fused_calls = []

    def fused(*args, **kwargs):
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        fused_calls.append(True)
        return blocks_forward_chunk_fused(*args, **kwargs)

    monkeypatch.setattr(batched_token2wav, "blocks_forward_chunk_fused", fused)
    with torch.inference_mode(), backend._cfm_autocast(next(estimator.parameters()).device):
        expected = _run(
            BatchedToken2Wav._blocks_forward_chunk_ragged,
            estimator,
            rows=4,
            frames=6,
            cached=5,
            lengths=[6, 4],
        )
        actual = _run(backend._ragged_body, estimator, rows=4, frames=6, cached=5, lengths=[6, 4])
    assert fused_calls
    for want, got in zip(expected, actual, strict=True):
        torch.testing.assert_close(got, want, rtol=2e-3, atol=2e-3)
