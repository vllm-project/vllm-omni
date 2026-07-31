# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends import trtllm_attn as tg
from vllm_omni.diffusion.attention.backends.trtllm_attn import TrtllmAttentionImpl

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cuda]


def _has_trtllm_attn() -> bool:
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] < 10:
        return False
    try:
        import flashinfer  # noqa: F401
    except Exception:
        return False
    return tg.HAS_FLASHINFER


requires_trtllm_attn = pytest.mark.skipif(
    not _has_trtllm_attn(), reason="requires Blackwell SM100+ GPU with flashinfer"
)


def _impl(**bk):
    return TrtllmAttentionImpl(8, 128, 1.0 / math.sqrt(128), causal=False, num_kv_heads=8, backend_kwargs=bk)


def test_skip_config_pure_resolution():
    from vllm_omni.diffusion.attention.backends.trtllm_attn import SkipSoftmaxConfig

    assert not SkipSoftmaxConfig().enabled
    assert SkipSoftmaxConfig().resolve_factor(4096, timestep=0.3) is None

    cfg = SkipSoftmaxConfig.from_backend_kwargs({"skip_softmax_threshold": 0.01, "disabled_until_timestep": 0.6})
    assert cfg.enabled and cfg.gated
    assert cfg.resolve_factor(4096, timestep=0.9) is None
    assert cfg.resolve_factor(4096, timestep=0.3) == pytest.approx(0.01 * 4096)
    assert cfg.resolve_factor(4096, timestep=None) == pytest.approx(0.01 * 4096)


def test_skip_factor_none_without_curve():
    assert _impl(target_sparsity=0.5)._resolve_skip_factor(4096) is None


def _calibrated(a, b, **bk):
    impl = _impl(**bk)
    impl.set_layer_calibration(a, b)
    return impl


def test_skip_factor_from_curve():
    assert _calibrated(100.0, 0.0, target_sparsity=0.5)._resolve_skip_factor(4096) == pytest.approx(100.0)
    f2 = _calibrated(2.0, 1.0, target_sparsity=0.5)._resolve_skip_factor(4096)
    assert f2 == pytest.approx(2.0 * math.exp(0.5))


def test_skip_factor_none_until_stamped():
    assert _impl(target_sparsity=0.5)._resolve_skip_factor(4096) is None


def test_skip_factor_from_threshold_no_calibration():
    impl = _impl(skip_softmax_threshold=0.01)
    assert impl._resolve_skip_factor(4096) == pytest.approx(0.01 * 4096)
    assert impl._resolve_skip_factor(8192) == pytest.approx(0.01 * 8192)

    impl2 = _calibrated(100.0, 0.0, skip_softmax_threshold=0.02, target_sparsity=0.5)
    assert impl2._resolve_skip_factor(4096) == pytest.approx(0.02 * 4096)


def test_skip_timestep_gate(monkeypatch):
    from vllm_omni.diffusion import forward_context as fc

    impl = _calibrated(100.0, 0.0, target_sparsity=0.5, disabled_until_timestep=0.6)
    ctx = fc.ForwardContext(denoise_timestep=0.9)
    monkeypatch.setattr(fc, "_forward_context", ctx)
    assert impl._resolve_skip_factor(4096) is None
    ctx.denoise_timestep = 0.3
    assert impl._resolve_skip_factor(4096) == pytest.approx(100.0)


def test_denoise_progress_mixin(monkeypatch):
    import types

    from vllm_omni.diffusion import forward_context as fc

    class _Pipe(fc.DenoiseProgressMixin):
        scheduler = types.SimpleNamespace(config=types.SimpleNamespace(num_train_timesteps=1000))

    ctx = fc.ForwardContext()
    monkeypatch.setattr(fc, "_forward_context", ctx)
    _Pipe().record_denoise_step(3, 250)
    assert ctx.denoise_step_idx == 3
    assert ctx.denoise_timestep == pytest.approx(0.25)

    _Pipe().record_denoise_step(4)
    assert ctx.denoise_step_idx == 4 and ctx.denoise_timestep == pytest.approx(0.25)


@pytest.mark.parametrize(
    "bk, field",
    [
        ({"target_sparsity": 1.5}, "target_sparsity"),
        ({"target_sparsity": -0.1}, "target_sparsity"),
        ({"disabled_until_timestep": 1.2}, "disabled_until_timestep"),
        ({"skip_softmax_threshold": -1.0}, "skip_softmax_threshold"),
        ({"target_sparsity": float("nan")}, "target_sparsity"),
        ({"skip_softmax_threshold": float("inf")}, "skip_softmax_threshold"),
    ],
)
def test_skip_control_validation_rejects_out_of_range(bk, field):
    from vllm_omni.diffusion.attention.backends.trtllm_attn import SkipSoftmaxConfig

    with pytest.raises(ValueError, match=field):
        SkipSoftmaxConfig.from_backend_kwargs(bk)


def test_skip_control_validation_accepts_valid():
    from vllm_omni.diffusion.attention.backends.trtllm_attn import SkipSoftmaxConfig

    cfg = SkipSoftmaxConfig.from_backend_kwargs(
        {"target_sparsity": 0.65, "disabled_until_timestep": 0.86, "skip_softmax_threshold": 0.01}
    )
    assert cfg.target_sparsity == pytest.approx(0.65)
    assert cfg.disabled_until_timestep == pytest.approx(0.86)
    assert cfg.threshold == pytest.approx(0.01)


def test_ring_sp_with_skip_softmax_rejected():
    import types

    from vllm_omni.diffusion.attention.layer import Attention

    fake = types.SimpleNamespace(attention=_impl(target_sparsity=0.5), ring_runner=None)
    with pytest.raises(NotImplementedError, match="ring sequence parallelism"):
        Attention._run_ring_attention(fake, None, None, None, None)


def test_mask_free_allowlist_gates_trtllm_default():
    from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata as meta

    assert meta("WanPipeline").attention_mask_free is True
    assert meta("WanVACEPipeline").attention_mask_free is True
    assert meta("FluxPipeline").attention_mask_free is False
    assert meta(None).attention_mask_free is False


def test_masked_layer_raises():
    b, s, h, d = 1, 8, 8, 128
    q, k, v = (torch.zeros(b, s, h, d) for _ in range(3))
    mask = torch.ones(b, s, dtype=torch.bool)
    mask[:, s // 2 :] = False

    class _Meta:
        attn_mask = mask

    with pytest.raises(ValueError, match="does not support attn_mask"):
        _impl().forward_cuda(q, k, v, _Meta())


def test_propagate_calibration_045_schema(monkeypatch):
    import types

    from vllm_omni.diffusion.data import (
        AttentionConfig,
        AttentionSpec,
        OmniDiffusionConfig,
        TransformerConfig,
    )

    cfg = AttentionConfig(default=AttentionSpec(backend="TRTLLM_ATTN"))
    tf = TransformerConfig.from_dict(
        {
            "sparse_attention_config": {
                "config_groups": {
                    "group_0": {
                        "algorithm": "skip_softmax",
                        "targets": ["WanAttention"],
                        "ignore": ["blocks.0.attn2", "blocks.1.attn2"],
                        "threshold_scale_factor": {
                            "formula": "a * exp(b * target_sparsity)",
                            "coefficients": {"a": 2142.7, "b": 4.28},
                        },
                        "target_sparsity": 0.5,
                    }
                }
            }
        }
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.attention.backends.trtllm_calibration.parse_secondary_expert_calibration",
        lambda *a, **k: None,
    )
    OmniDiffusionConfig._propagate_skip_softmax_calibration(
        types.SimpleNamespace(
            diffusion_attention_config=cfg,
            model="/x",
        ),
        tf,
    )
    spec, _ = cfg.resolve_with_source("self")
    cal = spec.skip_calibration["by_expert"]["transformer"]
    assert cal["a"] == pytest.approx(2142.7)
    assert cal["b"] == pytest.approx(4.28)
    assert cal["ignore"] == ["blocks.0.attn2", "blocks.1.attn2"]
    assert "transformer_2" not in spec.skip_calibration["by_expert"]


def test_target_sparsity_without_calibration_raises():
    import types

    from vllm_omni.diffusion.data import (
        AttentionConfig,
        AttentionSpec,
        OmniDiffusionConfig,
        TransformerConfig,
    )

    def _cfg(skip_softmax):
        return types.SimpleNamespace(
            diffusion_attention_config=AttentionConfig(
                default=AttentionSpec(backend="TRTLLM_ATTN", skip_softmax=skip_softmax)
            ),
            model="/models/no-such-calibration",
        )

    with pytest.raises(ValueError, match="no skip-softmax calibration"):
        OmniDiffusionConfig._propagate_skip_softmax_calibration(
            _cfg({"target_sparsity": 0.5}), TransformerConfig.from_dict({})
        )

    OmniDiffusionConfig._propagate_skip_softmax_calibration(_cfg({"threshold": 0.02}), TransformerConfig.from_dict({}))


def _sdpa_ref(q, k, v, scale):
    qp, kp, vp = (x.permute(0, 2, 1, 3) for x in (q, k, v))
    return F.scaled_dot_product_attention(qp, kp, vp, is_causal=False, scale=scale).permute(0, 2, 1, 3).float()


@requires_trtllm_attn
def test_bf16_dense_matches_sdpa():
    torch.manual_seed(0)
    B, S, H, D = 2, 256, 8, 128
    scale = 1.0 / math.sqrt(D)
    q, k, v = (torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    out = _impl().forward_cuda(q, k, v, None).float()
    ref = _sdpa_ref(q, k, v, scale)
    rel = (out - ref).abs().mean() / ref.abs().mean()
    assert rel < 0.01, f"BF16 dense rel err {rel:.4f} too high"


@requires_trtllm_attn
def test_skip_tiny_threshold_matches_dense():
    torch.manual_seed(0)
    B, S, H, D = 2, 512, 8, 128
    scale = 1.0 / math.sqrt(D)
    q, k, v = (torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))

    out = _impl(skip_softmax_threshold=1e-9).forward_cuda(q, k, v, None).float()
    ref = _sdpa_ref(q, k, v, scale)
    rel = (out - ref).abs().mean() / ref.abs().mean()
    assert rel < 0.02, f"skip(tiny) rel err {rel:.4f} should be ~dense"


@requires_trtllm_attn
def test_skip_threshold_engages():
    torch.manual_seed(0)
    B, S, H, D = 2, 512, 8, 128
    q, k, v = (torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    dense = _impl().forward_cuda(q, k, v, None).float()
    skipped = _impl(skip_softmax_threshold=2.0).forward_cuda(q, k, v, None).float()
    assert torch.isfinite(skipped).all()
    assert (skipped - dense).abs().mean() / dense.abs().mean() > 0.02


@requires_trtllm_attn
def test_gqa_matches_sdpa():
    torch.manual_seed(0)
    B, S, Hq, Hkv, D = 1, 256, 64, 8, 128
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, S, Hq, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.bfloat16)
    out = (
        TrtllmAttentionImpl(Hq, D, scale, causal=False, num_kv_heads=Hkv, backend_kwargs={})
        .forward_cuda(q, k, v, None)
        .float()
    )
    qp, kp, vp = (x.permute(0, 2, 1, 3) for x in (q, k, v))
    ref = (
        F.scaled_dot_product_attention(qp, kp, vp, is_causal=False, scale=scale, enable_gqa=True)
        .permute(0, 2, 1, 3)
        .float()
    )
    rel = (out - ref).abs().mean() / ref.abs().mean()
    assert rel < 0.01, f"GQA rel err {rel:.4f} vs SDPA(GQA) too high"


@requires_trtllm_attn
def test_unsupported_head_dim_raises():
    q = k = v = torch.randn(1, 32, 8, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(Exception, match="[Uu]nsupported head dim"):
        _impl().forward_cuda(q, k, v, None)


def test_skip_end_to_end_config_path(monkeypatch):
    import types

    from vllm_omni.diffusion import forward_context as fc
    from vllm_omni.diffusion.data import (
        AttentionConfig,
        AttentionSpec,
        OmniDiffusionConfig,
        TransformerConfig,
    )

    cfg = AttentionConfig(
        default=AttentionSpec(
            backend="TRTLLM_ATTN",
            skip_softmax={"target_sparsity": 0.65, "disabled_until_timestep": 0.86},
        )
    )

    tf = TransformerConfig.from_dict(
        {"sparse_attention_config": {"threshold_scale_factor": {"prefill": {"a": 7.93, "b": 8.61}}}}
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.attention.backends.trtllm_calibration.parse_secondary_expert_calibration",
        lambda *a, **k: None,
    )
    OmniDiffusionConfig._propagate_skip_softmax_calibration(
        types.SimpleNamespace(
            diffusion_attention_config=cfg,
            model="/x",
        ),
        tf,
    )
    spec, _ = cfg.resolve_with_source("self")
    cal = spec.skip_calibration["by_expert"]["transformer"]
    assert cal["a"] == 7.93 and cal["b"] == 8.61

    impl = TrtllmAttentionImpl(
        8, 128, 1.0 / math.sqrt(128), causal=False, num_kv_heads=8, backend_kwargs=spec.backend_kwargs()
    )
    impl.set_layer_calibration(cal["a"], cal["b"])
    expected = 7.93 * math.exp(8.61 * 0.65)

    ctx = fc.ForwardContext(denoise_timestep=0.3)
    monkeypatch.setattr(fc, "_forward_context", ctx)
    assert impl._resolve_skip_factor(4096) == pytest.approx(expected)
