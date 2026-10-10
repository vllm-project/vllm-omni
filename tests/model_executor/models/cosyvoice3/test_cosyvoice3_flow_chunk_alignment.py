# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Upstream alignment of the CosyVoice3 streaming flow: chunk-causal mask on
every estimator (torch and TensorRT), block-aligned hops, bidirectional finalize."""

import logging
import types
from collections import defaultdict

import pytest
import torch
from omegaconf import DictConfig
from torch import nn

from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import AttentionConfig
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
    # Pin TORCH_SDPA: on a GPU host the platform default (e.g. CUDNN_ATTN on
    # Blackwell) has no CPU kernel for the padding-only path.
    od_config = types.SimpleNamespace(
        diffusion_attention_config=AttentionConfig(default="TORCH_SDPA"),
        parallel_config=types.SimpleNamespace(ring_degree=1),
    )
    with set_current_diffusion_config(od_config):
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


class TestFlowSolveBlockConsistency:
    """The headline property end to end through the flow solve: CFG batch,
    fixed initial noise and several Euler steps on a real (tiny) DiT. The
    frames one hop emits are the frames the next hop re-computes."""

    PROMPT = BLOCK  # prompt mel frames, block-aligned as after prompt padding

    def _solve(self, frames: int, *, streaming: bool):
        torch.manual_seed(0)
        cfm = _cfm(_tiny_dit())
        g = torch.Generator().manual_seed(1)
        mu = torch.randn(1, 80, 3 * BLOCK, generator=g)[..., :frames]
        cond = torch.zeros(1, 80, frames)
        cond[..., : self.PROMPT] = torch.randn(1, 80, self.PROMPT, generator=g)
        spks = torch.randn(1, 80, generator=g)
        out, _ = cfm(
            mu.contiguous(),
            torch.ones(1, 1, frames),
            n_timesteps=4,
            spks=spks,
            cond=cond,
            streaming=streaming,
            prompt_len=self.PROMPT,
        )
        return out

    def test_aligned_hop_recomputes_the_emitted_frames(self):
        first = self._solve(2 * BLOCK, streaming=True)
        second = self._solve(3 * BLOCK, streaming=True)
        torch.testing.assert_close(second[..., : 2 * BLOCK], first, rtol=0, atol=1e-5)

    def test_full_attention_recomputes_different_frames(self):
        first = self._solve(2 * BLOCK, streaming=False)
        second = self._solve(3 * BLOCK, streaming=False)
        assert not torch.allclose(second[..., self.PROMPT : 2 * BLOCK], first[..., self.PROMPT :], atol=1e-4)

    def test_unaligned_hop_breaks_the_partial_block(self):
        """A hop ending mid-block: the emitted half of that block changes once
        the rest of the block arrives, even with the chunk mask."""
        first = self._solve(BLOCK + BLOCK // 2, streaming=True)
        second = self._solve(2 * BLOCK, streaming=True)
        torch.testing.assert_close(second[..., : self.PROMPT], first[..., : self.PROMPT], rtol=0, atol=1e-5)
        assert not torch.allclose(second[..., self.PROMPT : BLOCK + BLOCK // 2], first[..., self.PROMPT :], atol=1e-4)


class _FakeTrtContext:
    def __init__(self):
        self.shapes: dict[str, tuple] = {}
        self.addresses: dict[str, int] = {}

    def set_input_shape(self, name, shape):
        self.shapes[name] = tuple(shape)
        return True

    def set_tensor_address(self, name, address):
        self.addresses[name] = address

    def execute_async_v3(self, stream):
        return True


class _FakeStream:
    cuda_stream = 0

    def wait_stream(self, other):
        pass


_LEGACY_INPUTS = frozenset({"x", "mask", "mu", "t", "spks", "cond"})


class _FakeTrtEstimator:
    def __init__(self, *, supports_attn_mask: bool, input_names=None, context=None):
        self.io_dtype = torch.float32
        self.out_dtype = torch.float32
        self.supports_attn_mask = supports_attn_mask
        if input_names is None:
            input_names = _LEGACY_INPUTS | ({"attn_mask"} if supports_attn_mask else set())
        self.input_names = input_names
        self.static_chunk_size = BLOCK
        self.context = context if context is not None else _FakeTrtContext()
        self.released = 0

    def acquire_estimator(self):
        return [self.context, _FakeStream()], object()

    def release_estimator(self, context, stream):
        assert context is self.context
        self.released += 1


class _RejectingShapeContext(_FakeTrtContext):
    def set_input_shape(self, name, shape):
        super().set_input_shape(name, shape)
        return name != "x"  # TRT returns False for a shape outside the profile


class _FailingEnqueueContext(_FakeTrtContext):
    def execute_async_v3(self, stream):
        return False


def _cfm(estimator) -> CausalConditionalCFM:
    return CausalConditionalCFM(in_channels=240, cfm_params=CFM_PARAMS, n_spks=1, spk_emb_dim=80, estimator=estimator)


class TestTensorRtMaskInput:
    @staticmethod
    def _fake_cuda(monkeypatch):
        monkeypatch.setattr(torch.cuda, "current_stream", lambda device: _FakeStream())
        monkeypatch.setattr(torch.cuda, "stream", lambda stream: torch.no_grad())

    def _run(self, cfm, monkeypatch, *, streaming: bool, frames: int = 2 * BLOCK + 10, mask=None, attn_mask=None):
        self._fake_cuda(monkeypatch)
        x = torch.randn(2, 80, frames)
        if mask is None:
            mask = torch.ones(2, 1, frames)
        cfm.forward_estimator(
            x, mask, torch.randn(2, 80, frames), torch.rand(2), torch.randn(2, 80), x, streaming, attn_mask=attn_mask
        )
        return cfm.estimator.context

    def test_bound_address_is_the_passed_map(self, monkeypatch):
        frames = 2 * BLOCK + 10
        cfm = _cfm(_FakeTrtEstimator(supports_attn_mask=True))
        attn_mask = cfm._trt_attention_mask(torch.ones(2, 1, frames), True)
        ctx = self._run(cfm, monkeypatch, streaming=True, frames=frames, attn_mask=attn_mask)
        assert ctx.addresses["attn_mask"] == attn_mask.data_ptr()
        assert set(ctx.shapes) == _LEGACY_INPUTS | {"attn_mask"}

    def test_only_declared_inputs_are_bound(self, monkeypatch):
        """An exporter prunes inputs the graph never reads; binding one of
        those names would make TensorRT reject the call."""
        declared = frozenset({"x", "mu", "t", "spks", "cond", "attn_mask"})
        cfm = _cfm(_FakeTrtEstimator(supports_attn_mask=True, input_names=declared))
        ctx = self._run(cfm, monkeypatch, streaming=True)
        assert set(ctx.shapes) == declared
        assert set(ctx.addresses) == declared | {"estimator_out"}

    def test_legacy_engine_binds_exactly_its_six_inputs(self, monkeypatch):
        cfm = _cfm(_FakeTrtEstimator(supports_attn_mask=False, input_names=_LEGACY_INPUTS))
        ctx = self._run(cfm, monkeypatch, streaming=True)
        assert set(ctx.shapes) == _LEGACY_INPUTS
        assert set(ctx.addresses) == _LEGACY_INPUTS | {"estimator_out"}

    def test_finalize_map_is_padding_only_per_row(self):
        """Finalize runs bidirectional, but a padded batch row must still not
        attend to its padding."""
        frames = 2 * BLOCK + 10
        mask = torch.ones(2, 1, frames)
        mask[1, :, 70:] = 0
        cfm = _cfm(_FakeTrtEstimator(supports_attn_mask=True))
        got = cfm._trt_attention_mask(mask, False)
        assert got.shape == (2, 1, frames, frames)
        assert got[0].all()
        assert got[1, 0, :70, :70].all()  # no chunk boundary at frame 50
        assert not got[1, 0, :70, 70:].any()

    def test_chunk_mask_engine_receives_the_streaming_map(self, monkeypatch):
        cfm = _cfm(_FakeTrtEstimator(supports_attn_mask=True))
        ctx = self._run(cfm, monkeypatch, streaming=True)
        frames = 2 * BLOCK + 10
        assert ctx.shapes["attn_mask"] == (2, 1, frames, frames)
        assert "attn_mask" in ctx.addresses and "estimator_out" in ctx.addresses
        built = cfm._trt_attention_mask(torch.ones(2, 1, frames), True)
        assert built.dtype == torch.bool
        assert torch.equal(built[0, 0], subsequent_chunk_mask(frames, BLOCK))

    def test_non_streaming_map_is_all_true(self):
        cfm = _cfm(_FakeTrtEstimator(supports_attn_mask=True))
        assert cfm._trt_attention_mask(torch.ones(2, 1, 30), False).all()

    def test_solve_builds_the_map_once_for_all_euler_steps(self, monkeypatch):
        """The map is step-invariant; rebuilding or re-checking it per step
        would add work and, with a content check, a host sync per step."""
        self._fake_cuda(monkeypatch)
        cfm = _cfm(_FakeTrtEstimator(supports_attn_mask=True))
        built = []
        real = cfm._trt_attention_mask
        monkeypatch.setattr(
            cfm, "_trt_attention_mask", lambda mask, streaming: built.append(1) or real(mask, streaming)
        )
        frames = BLOCK + 6
        mu = torch.randn(1, 80, frames)
        cfm.solve_euler(
            torch.randn(1, 80, frames),
            t_span=torch.linspace(0, 1, 5),
            mu=mu,
            mask=torch.ones(1, 1, frames),
            spks=torch.randn(1, 80),
            cond=torch.zeros_like(mu),
            streaming=True,
        )
        assert built == [1]
        assert cfm.estimator.context.shapes["attn_mask"] == (2, 1, frames, frames)

    def test_legacy_engine_gets_no_map_and_warns_once(self, monkeypatch, caplog):
        from vllm.logger import _print_warning_once

        # warning_once dedupes per process; an earlier test may already have
        # fired this message.
        _print_warning_once.cache_clear()
        cfm = _cfm(_FakeTrtEstimator(supports_attn_mask=False))
        with caplog.at_level(logging.WARNING):
            ctx = self._run(cfm, monkeypatch, streaming=True)
            self._run(cfm, monkeypatch, streaming=True)
        assert "attn_mask" not in ctx.shapes
        assert sum("no attn_mask input" in r.message for r in caplog.records) == 1


class TestTensorRtContextIsReleased:
    """A raised error must hand the pooled context back, or the next request
    blocks on ``acquire_estimator`` forever."""

    def _run(self, cfm, monkeypatch):
        monkeypatch.setattr(torch.cuda, "current_stream", lambda device: _FakeStream())
        monkeypatch.setattr(torch.cuda, "stream", lambda stream: torch.no_grad())
        frames = BLOCK
        x = torch.randn(2, 80, frames)
        cfm.forward_estimator(x, torch.ones(2, 1, frames), x, torch.rand(2), torch.randn(2, 80), x, False)

    def test_out_of_profile_shape_raises_and_releases(self, monkeypatch):
        est = _FakeTrtEstimator(supports_attn_mask=True, context=_RejectingShapeContext())
        with pytest.raises(RuntimeError, match="optimization profile"):
            self._run(_cfm(est), monkeypatch)
        assert est.released == 1

    def test_failed_enqueue_raises_and_releases(self, monkeypatch):
        est = _FakeTrtEstimator(supports_attn_mask=True, context=_FailingEnqueueContext())
        with pytest.raises(RuntimeError, match="execute_async_v3"):
            self._run(_cfm(est), monkeypatch)
        assert est.released == 1

    def test_success_releases_once(self, monkeypatch):
        est = _FakeTrtEstimator(supports_attn_mask=True)
        self._run(_cfm(est), monkeypatch)
        assert est.released == 1


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


class TestStreamingBatch:
    """``forward_streaming_batch`` is what serves concurrency > 1; it builds
    its own flow call instead of going through ``forward_streaming``."""

    @staticmethod
    def _model(calls: list):
        model = object.__new__(CosyVoice3Code2Wav)
        nn.Module.__init__(model)
        model.flow_model = types.SimpleNamespace(token_mel_ratio=2, pre_lookahead_len=3)

        def fake_forward_mel(self, token, prompt_token, prompt_feat, embedding, **kw):
            calls.append(kw)
            return torch.zeros(token.shape[0], 80, 2 * int(token.shape[1]))

        def fake_hift(self, feat, *, cache_state=None, finalize=False):
            return feat, (None if finalize else {"mel_frames": int(feat.shape[-1])})

        model._forward_mel = types.MethodType(fake_forward_mel, model)
        model._stream_hift_from_feat = types.MethodType(fake_hift, model)
        return model

    @staticmethod
    def _item(index: int, tokens: int, *, offset: int = 0, emitted=None, finalize: bool = False):
        return {
            "index": index,
            "token": torch.zeros(1, tokens, dtype=torch.int32),
            "prompt_token": torch.zeros(1, 2, dtype=torch.int32),
            "prompt_feat": torch.zeros(1, 4, 80),
            "embedding": torch.zeros(1, 192),
            "cache_state": None if emitted is None else {"flow_emitted_tokens": emitted},
            "token_offset_tokens": offset,
            "finalize": finalize,
        }

    def test_batched_groups_stream_unless_finalize_and_track_noise_per_row(self):
        calls: list = []
        model = self._model(calls)
        items = [
            self._item(0, 28),  # first hop: 25 new + 3 lookahead
            self._item(1, 28, offset=25, emitted=45, finalize=True),
            self._item(2, 53, offset=25, emitted=45),  # bounded window: resends 25, first resent is abs 20
            self._item(3, 30, offset=0, emitted=25, finalize=True),
        ]
        results = model.forward_streaming_batch(items, n_timesteps=2)

        by_finalize = {c["finalize"]: c for c in calls}
        assert len(calls) == 2
        assert by_finalize[False]["streaming"] is True
        assert by_finalize[True]["streaming"] is False
        assert by_finalize[False]["noise_offset_tokens"].tolist() == [0, 20]
        assert by_finalize[True]["noise_offset_tokens"].tolist() == [20, 25]
        assert by_finalize[False]["token_lens"].tolist() == [28, 53]

        # Results come back in item order; only live streams carry state.
        assert results[1][1] is None and results[3][1] is None
        assert results[0][1]["flow_emitted_tokens"] == 25
        assert results[2][1]["flow_emitted_tokens"] == 45 + (53 - 3 - 25)
        # Row 2 drops its lookahead and its 25 resent tokens: 25 new tokens.
        assert results[2][1]["mel_frames"] == 2 * 25
        assert results[0][1]["mel_frames"] == 2 * 25


class TestHopAlignmentWarning:
    def test_unaligned_hop_warns_once(self, caplog):
        with caplog.at_level(logging.WARNING):
            processor._warn_if_unaligned("codec_chunk_frames", 15)
            processor._warn_if_unaligned("codec_chunk_frames", 15)
            processor._warn_if_unaligned("codec_chunk_frames", 50)
        assert sum("25-token attention block" in r.message for r in caplog.records) == 1
        assert processor._FLOW_CHUNK_TOKENS == 25

    @staticmethod
    def _call_processor(chunk_frames: int, max_chunk_frames: int):
        manager = types.SimpleNamespace(
            code_prompt_token_ids=defaultdict(list),
            request_payload={},
            connector=types.SimpleNamespace(
                config={
                    "extra": {
                        "codec_chunk_frames": chunk_frames,
                        "codec_max_chunk_frames": max_chunk_frames,
                        "codec_vocab_size": 6561,
                    }
                }
            ),
        )
        request = types.SimpleNamespace(
            external_req_id="rid-hop",
            output_token_ids=[],
            additional_information=None,
            is_finished=lambda: False,
        )
        processor.talker2code2wav_async_chunk(manager, None, request, is_finished=False)

    @pytest.mark.parametrize(
        ("chunk_frames", "max_chunk_frames", "warned"),
        [
            (15, 60, {"codec_chunk_frames", "codec_max_chunk_frames"}),
            (25, 60, {"codec_max_chunk_frames"}),
            (25, 100, set()),
        ],
    )
    def test_processor_checks_the_configured_hops(self, monkeypatch, chunk_frames, max_chunk_frames, warned):
        seen: list[str] = []
        monkeypatch.setattr(processor.logger, "warning_once", lambda msg, name, *args: seen.append(name))
        self._call_processor(chunk_frames, max_chunk_frames)
        assert set(seen) == warned

    def test_shipped_deploy_config_is_block_aligned(self):
        from pathlib import Path

        import yaml

        import vllm_omni

        deploy = Path(vllm_omni.__file__).parent / "deploy" / "cosyvoice3.yaml"
        extras = []

        def walk(node):
            if isinstance(node, dict):
                if "codec_chunk_frames" in node:
                    extras.append(node)
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(yaml.safe_load(deploy.read_text()))
        assert extras, "cosyvoice3.yaml no longer sets codec_chunk_frames"
        for extra in extras:
            chunk = int(extra["codec_chunk_frames"])
            assert chunk % processor._FLOW_CHUNK_TOKENS == 0
            assert int(extra.get("codec_max_chunk_frames", 4 * chunk)) % processor._FLOW_CHUNK_TOKENS == 0


class TestTensorRtEstimatorSelection:
    """``_maybe_enable_code2wav_trt``: which engine ends up in the flow."""

    class _Wrapper:
        def __init__(self, supports_attn_mask: bool):
            self.supports_attn_mask = supports_attn_mask

    @staticmethod
    def _model(tmp_path):
        from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import CosyVoice3Model

        model = object.__new__(CosyVoice3Model)
        decoder = nn.Module()
        decoder.estimator = _tiny_dit()
        model.code2wav = types.SimpleNamespace(flow_model=types.SimpleNamespace(decoder=decoder))
        model.model_dir = str(tmp_path)
        model._resolve_flow_estimator_onnx = lambda: str(tmp_path / "flow.decoder.estimator.fp32.onnx")
        return model, decoder

    def _patch(self, monkeypatch, *, chunk_mask_error: Exception | None = None):
        from vllm_omni.model_executor.models.cosyvoice3 import flow_estimator_trt

        built: list[str] = []

        def chunk_mask_builder(estimator, onnx_dir, device, cache_key=None):
            built.append("chunk_mask")
            if chunk_mask_error is not None:
                raise chunk_mask_error
            return self._Wrapper(supports_attn_mask=True)

        def legacy_builder(onnx_path, device):
            built.append("legacy")
            return self._Wrapper(supports_attn_mask=False)

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(flow_estimator_trt, "build_chunk_mask_flow_estimator_trt", chunk_mask_builder)
        monkeypatch.setattr(flow_estimator_trt, "build_flow_estimator_trt", legacy_builder)
        return built

    def test_default_installs_the_chunk_mask_engine(self, tmp_path, monkeypatch):
        monkeypatch.delenv("COSYVOICE3_TRT_CHUNK_MASK", raising=False)
        monkeypatch.delenv("COSYVOICE3_TRT", raising=False)
        built = self._patch(monkeypatch)
        model, decoder = self._model(tmp_path)
        model._maybe_enable_code2wav_trt()
        assert built == ["chunk_mask"]
        assert decoder.estimator.supports_attn_mask is True

    def test_export_failure_falls_back_to_the_bundled_engine(self, tmp_path, monkeypatch):
        monkeypatch.delenv("COSYVOICE3_TRT_CHUNK_MASK", raising=False)
        monkeypatch.delenv("COSYVOICE3_TRT", raising=False)
        built = self._patch(monkeypatch, chunk_mask_error=ImportError("no onnx"))
        model, decoder = self._model(tmp_path)
        model._maybe_enable_code2wav_trt()
        assert built == ["chunk_mask", "legacy"]
        assert decoder.estimator.supports_attn_mask is False

    @pytest.mark.parametrize("value", ["0", "false"])
    def test_env_off_skips_the_export(self, tmp_path, monkeypatch, value):
        monkeypatch.setenv("COSYVOICE3_TRT_CHUNK_MASK", value)
        monkeypatch.delenv("COSYVOICE3_TRT", raising=False)
        built = self._patch(monkeypatch)
        model, decoder = self._model(tmp_path)
        model._maybe_enable_code2wav_trt()
        assert built == ["legacy"]
        assert decoder.estimator.supports_attn_mask is False

    def test_onnx_dir_falls_back_to_the_plan_cache_when_model_dir_is_read_only(self, tmp_path, monkeypatch):
        model, _ = self._model(tmp_path)
        assert model._flow_estimator_onnx_dir() == str(tmp_path)
        monkeypatch.setenv("COSYVOICE3_TRT_CACHE", str(tmp_path / "cache"))
        monkeypatch.setattr("os.access", lambda path, mode: False)
        assert model._flow_estimator_onnx_dir() == str(tmp_path / "cache" / "exported")


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


def test_fp32_export_attention_is_scoped_to_the_estimator():
    from vllm_omni.model_executor.models.cosyvoice3.flow_estimator_trt import _fp32_attention_for_export

    dit, other = _tiny_dit(), _tiny_dit()
    layers = [m for m in dit.modules() if hasattr(m, "fp32_masked_attention")]
    assert layers
    with _fp32_attention_for_export(dit):
        assert all(m.fp32_masked_attention for m in layers)
        assert not any(getattr(m, "fp32_masked_attention", False) for m in other.modules())
    assert not any(m.fp32_masked_attention for m in layers)


def test_fused_export_emits_one_attention_node_per_layer(tmp_path):
    """Each masked SDPA becomes one opset-23 ``Attention`` node, which TensorRT
    runs as a fused MHA kernel, instead of matmuls + softmax it cannot fuse."""
    onnx = pytest.importorskip("onnx")

    from vllm_omni.model_executor.models.cosyvoice3.flow_estimator_trt import export_chunk_mask_estimator_onnx

    path = export_chunk_mask_estimator_onnx(_tiny_dit(), str(tmp_path / "est.onnx"), fp16=True, fused_attention=True)
    model = onnx.load(path)
    nodes = [n for n in model.graph.node if n.op_type == "Attention"]
    assert len(nodes) == 2 and all(n.domain == "" for n in nodes)
    assert not any(n.op_type == "Softmax" for n in model.graph.node)
    assert {o.domain: o.version for o in model.opset_import}.get("") == 23
    assert [i.name for i in model.graph.input] == ["x", "mask", "mu", "t", "spks", "cond", "attn_mask"]


@pytest.mark.parametrize("streaming", [True, False])
def test_fused_export_matches_the_torch_dit(tmp_path, streaming):
    """The ``Attention`` node must read the bool map with the polarity SDPA
    uses (True attends) and honour padding, or the chunk mask is inverted."""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    from vllm_omni.model_executor.models.cosyvoice3.flow_estimator_trt import export_chunk_mask_estimator_onnx

    dit = _tiny_dit()
    path = export_chunk_mask_estimator_onnx(dit, str(tmp_path / "est.onnx"), fp16=True, fused_attention=True)
    frames = 2 * BLOCK + 7
    g = torch.Generator().manual_seed(5)
    x, mu, cond = (torch.randn(2, 80, frames, generator=g) for _ in range(3))
    mask = torch.ones(2, 1, frames)
    mask[1, :, frames - 20 :] = 0
    t = torch.rand(2, generator=g)
    spks = torch.randn(2, 80, generator=g)
    attn_mask = build_dit_attention_mask(mask, streaming=streaming, static_chunk_size=BLOCK)
    with torch.inference_mode():
        expected = dit(x, mask, mu, t, spks, cond, attn_mask=attn_mask) * mask
        other = (
            dit(
                x,
                mask,
                mu,
                t,
                spks,
                cond,
                attn_mask=build_dit_attention_mask(mask, streaming=not streaming, static_chunk_size=BLOCK),
            )
            * mask
        )
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feeds = {"x": x, "mask": mask, "mu": mu, "t": t, "spks": spks, "cond": cond, "attn_mask": attn_mask}
    (got,) = session.run(None, {k: v.numpy() for k, v in feeds.items()})
    got = torch.from_numpy(got).float()
    err = (got - expected).abs().mean().item()
    # fp16 elsewhere in the graph; the other mask must be clearly further away.
    assert err < 0.02
    assert err < 0.2 * (other - expected).abs().mean().item()


@pytest.mark.parametrize("frames", [2 * BLOCK + 7, 3 * BLOCK])
@pytest.mark.parametrize("streaming", [True, False])
def test_exported_onnx_matches_the_torch_dit(tmp_path, frames, streaming):
    """The export is traced at 64 frames; run it at other lengths so a
    sequence length baked into the graph (rope, positions) shows up as a
    numeric mismatch, not only at TensorRT build time on a GPU box."""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    from vllm_omni.model_executor.models.cosyvoice3.flow_estimator_trt import export_chunk_mask_estimator_onnx

    dit = _tiny_dit()
    path = export_chunk_mask_estimator_onnx(dit, str(tmp_path / "est.onnx"), fp16=False)

    g = torch.Generator().manual_seed(3)
    x = torch.randn(2, 80, frames, generator=g)
    mask = torch.ones(2, 1, frames)
    mask[1, :, frames - 20 :] = 0  # one padded row, as in a flow batch
    mu = torch.randn(2, 80, frames, generator=g)
    t = torch.rand(2, generator=g)
    spks = torch.randn(2, 80, generator=g)
    cond = torch.randn(2, 80, frames, generator=g)
    attn_mask = build_dit_attention_mask(mask, streaming=streaming, static_chunk_size=BLOCK)

    with torch.inference_mode():
        expected = dit(x, mask, mu, t, spks, cond, attn_mask=attn_mask) * mask

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feeds = {"x": x, "mask": mask, "mu": mu, "t": t, "spks": spks, "cond": cond, "attn_mask": attn_mask}
    declared = {i.name for i in session.get_inputs()}
    (got,) = session.run(None, {k: v.numpy() for k, v in feeds.items() if k in declared})
    torch.testing.assert_close(torch.from_numpy(got), expected, rtol=1e-4, atol=1e-4)
