# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Upstream alignment of the CosyVoice3 streaming flow: chunk-causal mask on
every estimator (torch and TensorRT), block-aligned hops, bidirectional finalize."""

import logging
import types

import pytest
import torch
from omegaconf import DictConfig
from torch import nn

from vllm_omni.diffusion.models.cosyvoice3_audio.cosyvoice3_dit import DiT
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import CausalConditionalCFM
from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav
from vllm_omni.model_executor.models.cosyvoice3.utils import build_dit_attention_mask, subsequent_chunk_mask
from vllm_omni.model_executor.stage_input_processors import cosyvoice3 as processor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CFM_PARAMS = DictConfig(
    {
        "sigma_min": 1e-06,
        "solver": "euler",
        "t_scheduler": "cosine",
        "training_cfg_rate": 0.2,
        "inference_cfg_rate": 0.7,
    }
)
BLOCK = 50


def _tiny_dit() -> DiT:
    torch.manual_seed(0)
    dit = DiT(dim=64, depth=2, heads=2, dim_head=32, dropout=0.0, mel_dim=80, mu_dim=80, spk_dim=80)
    dit.static_chunk_size = BLOCK
    return dit.eval()


def _inputs(frames: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 80, frames, generator=g)
    mask = torch.ones(1, 1, frames)
    mu = torch.randn(1, 80, frames, generator=g)
    t = torch.full((1,), 0.3)
    spks = torch.randn(1, 80, generator=g)
    cond = torch.randn(1, 80, frames, generator=g)
    return x, mask, mu, t, spks, cond


def _prefix(inputs, frames: int):
    """The same request truncated to its first ``frames`` mel frames."""
    return tuple(t[..., :frames].contiguous() if t.dim() == 3 else t for t in inputs)


class TestMaskBuilder:
    def test_streaming_map_is_upstream_block_causal(self):
        pad = torch.ones(1, 1, 120, dtype=torch.bool)
        got = build_dit_attention_mask(pad, streaming=True, static_chunk_size=BLOCK)
        assert got.shape == (1, 1, 120, 120) and got.dtype == torch.bool
        assert torch.equal(got[0, 0], subsequent_chunk_mask(120, BLOCK))
        # frame 60 (block 1) sees blocks 0-1 fully and nothing of block 2
        assert got[0, 0, 60, :100].all() and not got[0, 0, 60, 100:].any()

    def test_padding_is_applied_and_empty_rows_filled(self):
        pad = torch.tensor([[[True] * 60 + [False] * 40], [[False] * 100]])
        got = build_dit_attention_mask(pad, streaming=True, static_chunk_size=BLOCK)
        assert not got[0, 0, 10, 60:].any()  # padded keys are never attended
        assert got[0, 0, 70, :60].all() and not got[0, 0, 70, 60:].any()  # padded query, valid keys only
        assert got[1, 0].all()  # a row with no valid key is filled so softmax stays finite

    def test_non_streaming_is_padding_only(self):
        pad = torch.tensor([[[True] * 60 + [False] * 40]])
        got = build_dit_attention_mask(pad, streaming=False, static_chunk_size=BLOCK)
        assert got[0, 0, 5, :60].all() and not got[0, 0, 5, 60:].any()
        assert got[0, 0, 5, 59]  # no chunk boundary in non-streaming mode


class TestBlockAlignment:
    """The property the hop alignment relies on: with the streaming mask a
    block's output does not change when the next block is appended, so what
    the next chunk re-computes is exactly what was already emitted."""

    def test_streaming_prefix_is_invariant_to_appended_blocks(self):
        dit = _tiny_dit()
        long = _inputs(3 * BLOCK)
        short = _prefix(long, 2 * BLOCK)
        with torch.inference_mode():
            out_short = dit(*short, streaming=True)
            out_long = dit(*long, streaming=True)
        torch.testing.assert_close(out_long[..., : 2 * BLOCK], out_short, rtol=0, atol=1e-5)

    def test_full_attention_prefix_changes_with_appended_blocks(self):
        dit = _tiny_dit()
        long = _inputs(3 * BLOCK)
        short = _prefix(long, 2 * BLOCK)
        with torch.inference_mode():
            out_short = dit(*short, streaming=False)
            out_long = dit(*long, streaming=False)
        assert not torch.allclose(out_long[..., : 2 * BLOCK], out_short, atol=1e-5)

    def test_explicit_mask_input_matches_streaming_flag(self):
        """The export entry point: an explicit map reproduces ``streaming=True``."""
        dit = _tiny_dit()
        inputs = _inputs(2 * BLOCK + 7)
        attn_mask = build_dit_attention_mask(inputs[1], streaming=True, static_chunk_size=BLOCK)
        with torch.inference_mode():
            via_flag = dit(*inputs, streaming=True)
            via_mask = dit(*inputs, attn_mask=attn_mask)
            via_mask_3d = dit(*inputs, attn_mask=attn_mask[:, 0])
        torch.testing.assert_close(via_mask, via_flag, rtol=0, atol=1e-6)
        torch.testing.assert_close(via_mask_3d, via_flag, rtol=0, atol=1e-6)


class _FakeTrtContext:
    def __init__(self):
        self.shapes: dict[str, tuple] = {}
        self.addresses: dict[str, int] = {}

    def set_input_shape(self, name, shape):
        self.shapes[name] = tuple(shape)

    def set_tensor_address(self, name, address):
        self.addresses[name] = address

    def execute_async_v3(self, stream):
        return True


class _FakeStream:
    cuda_stream = 0

    def wait_stream(self, other):
        pass


class _FakeTrtEstimator:
    def __init__(self, *, supports_attn_mask: bool):
        self.io_dtype = torch.float32
        self.supports_attn_mask = supports_attn_mask
        self.static_chunk_size = BLOCK
        self.context = _FakeTrtContext()

    def acquire_estimator(self):
        return [self.context, _FakeStream()], object()

    def release_estimator(self, context, stream):
        pass


def _cfm(estimator) -> CausalConditionalCFM:
    return CausalConditionalCFM(in_channels=240, cfm_params=CFM_PARAMS, n_spks=1, spk_emb_dim=80, estimator=estimator)


class TestTensorRtMaskInput:
    def _run(self, cfm, monkeypatch, *, streaming: bool, frames: int = 2 * BLOCK + 10):
        monkeypatch.setattr(torch.cuda, "current_stream", lambda device: _FakeStream())
        monkeypatch.setattr(torch.cuda, "stream", lambda stream: torch.no_grad())
        x = torch.randn(2, 80, frames)
        mask = torch.ones(2, 1, frames)
        cfm.forward_estimator(x, mask, torch.randn(2, 80, frames), torch.rand(2), torch.randn(2, 80), x, streaming)
        return cfm.estimator.context

    def test_chunk_mask_engine_receives_the_streaming_map(self, monkeypatch):
        cfm = _cfm(_FakeTrtEstimator(supports_attn_mask=True))
        ctx = self._run(cfm, monkeypatch, streaming=True)
        assert ctx.shapes["attn_mask"] == (2, 1, 2 * BLOCK + 10, 2 * BLOCK + 10)
        assert "attn_mask" in ctx.addresses and "estimator_out" in ctx.addresses
        cached = cfm._trt_attn_mask_cache[2]
        assert cached.dtype == torch.bool
        assert torch.equal(cached[0, 0], subsequent_chunk_mask(2 * BLOCK + 10, BLOCK))

    def test_non_streaming_map_is_all_true_and_cached_per_shape(self, monkeypatch):
        cfm = _cfm(_FakeTrtEstimator(supports_attn_mask=True))
        self._run(cfm, monkeypatch, streaming=False)
        first = cfm._trt_attn_mask_cache[2]
        assert first.all()
        self._run(cfm, monkeypatch, streaming=False)
        assert cfm._trt_attn_mask_cache[2] is first  # the ten Euler steps share one map

    def test_legacy_engine_gets_no_map_and_warns_once(self, monkeypatch, caplog):
        cfm = _cfm(_FakeTrtEstimator(supports_attn_mask=False))
        with caplog.at_level(logging.WARNING):
            ctx = self._run(cfm, monkeypatch, streaming=True)
            self._run(cfm, monkeypatch, streaming=True)
        assert "attn_mask" not in ctx.shapes
        assert sum("no attn_mask input" in r.message for r in caplog.records) == 1


class TestFinalizeIsBidirectional:
    def test_forward_streaming_passes_streaming_unless_finalize(self):
        calls = []
        model = object.__new__(CosyVoice3Code2Wav)
        nn.Module.__init__(model)
        model.flow_model = types.SimpleNamespace(token_mel_ratio=2, pre_lookahead_len=3)
        model._forward_mel = types.MethodType(
            lambda self, token, prompt_token, prompt_feat, embedding, **kw: (calls.append(kw), torch.zeros(1, 80, 4))[
                1
            ],
            model,
        )
        model._stream_hift_from_feat = types.MethodType(lambda self, feat, **kw: (feat, {}), model)
        common = dict(
            prompt_token=torch.zeros(1, 2, dtype=torch.int32),
            prompt_feat=torch.zeros(1, 4, 80),
            embedding=torch.zeros(1, 192),
        )
        model.forward_streaming(token=torch.zeros(1, 8, dtype=torch.int32), **common)
        model.forward_streaming(token=torch.zeros(1, 8, dtype=torch.int32), **common, finalize=True)
        assert [c["streaming"] for c in calls] == [True, False]
        assert [c["finalize"] for c in calls] == [False, True]


class TestHopAlignmentWarning:
    def test_unaligned_hop_warns_once(self, caplog):
        processor._warned_unaligned.discard("codec_chunk_frames")
        with caplog.at_level(logging.WARNING):
            processor._warn_unaligned_chunk("codec_chunk_frames", 15)
            processor._warn_unaligned_chunk("codec_chunk_frames", 15)
        assert sum("25-token attention block" in r.message for r in caplog.records) == 1
        assert processor._FLOW_CHUNK_TOKENS == 25


def test_export_has_attn_mask_input(tmp_path):
    onnx = pytest.importorskip("onnx")

    from vllm_omni.model_executor.models.cosyvoice3.flow_estimator_trt import (
        ATTN_MASK_INPUT,
        export_chunk_mask_estimator_onnx,
    )

    path = export_chunk_mask_estimator_onnx(_tiny_dit(), str(tmp_path / "est.onnx"), fp16=False)
    model = onnx.load(path)
    names = [i.name for i in model.graph.input]
    assert names == ["x", "mask", "mu", "t", "spks", "cond", ATTN_MASK_INPUT]
    mask_input = model.graph.input[-1]
    dims = [d.dim_param or d.dim_value for d in mask_input.type.tensor_type.shape.dim]
    assert dims[:2] == [2, 1] and dims[2] == dims[3] == "seq_len"
