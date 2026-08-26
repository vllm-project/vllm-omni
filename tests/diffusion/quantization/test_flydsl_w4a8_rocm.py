# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""gfx950 numerics for W4A8, plain and SVDQuant, through a real vLLM linear layer.

Shapes are the production Wan2.2 TI2V-5B and T2V-A14B linear dimensions at the
diffusion prefill token count (M ~ 4.7k). M is deliberately not a multiple of
tile_m: the FlyDSL kernel does not mask ragged rows, so Quark pads M internally
and slices back, and that padding is a correctness requirement rather than an
optimization.

Config routing and the capability gate are covered on CPU in
test_quark_w4a8_config.py.
"""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.rocm, pytest.mark.MI355]


def _gcn_arch() -> str:
    if not (torch.cuda.is_available() and torch.version.hip):
        return ""
    return torch.cuda.get_device_properties(torch.accelerator.current_device_index()).gcnArchName


# Runtime gate, not just the MI355 marker: the marker only filters CI lanes, and
# a local run on gfx942 would otherwise fail inside the MLIR backend.
requires_gfx950 = pytest.mark.skipif(
    "gfx950" not in _gcn_arch(),
    reason=f"W4A8 requires CDNA4 scaled MFMA (gfx950); detected {_gcn_arch() or 'no ROCm device'}",
)

# Wan diffusion prefill is ~4.7k tokens. 4680 is not a multiple of 32 or 64, so
# every case exercises the ragged-M padding path.
M_RAGGED = 4680
M_ALIGNED = 4736  # 74 * 64

SHAPES_5B = [(3072, 3072), (3072, 14336), (14336, 3072), (4096, 3072)]
SHAPES_A14B = [(5120, 5120), (5120, 13824), (13824, 5120), (4096, 5120)]

pytestmark.append(requires_gfx950)


@pytest.fixture(autouse=True)
def _cuda_default_device():
    """vLLM sets the accelerator as the default device before building a model.
    _LazyWeightMixin captures torch.get_default_device() to materialise the
    weight, so without this the BF16 copy lands on CPU and the kernel rejects it.
    """
    torch.set_default_device("cuda")
    yield
    torch.set_default_device("cpu")


@pytest.fixture(autouse=True)
def _patch_tp_state(monkeypatch):
    """ModelWeightParameter reads the global TP rank in __init__. Patching it is
    cheaper than standing up a process group for a single-device test."""
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def _make_layer(cfg, in_features: int, out_features: int, bias: bool = True):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.linear import ReplicatedLinear

    with set_current_vllm_config(VllmConfig()):
        return ReplicatedLinear(
            input_size=in_features,
            output_size=out_features,
            bias=bias,
            params_dtype=torch.bfloat16,
            quant_config=cfg,
            prefix="transformer.blocks.0.attn.to_q",
            disable_tp=True,
        )


def _spectral_weight(out_features: int, in_features: int, rank: int, decay: float = 64.0):
    """A weight with a decaying singular spectrum, plus its exact rank-R factors.

    SVDQuant only helps when the top-R subspace carries real energy, which holds
    for trained transformer weights and not for i.i.d. Gaussian noise. Building
    ``U diag(s) V.T`` directly keeps the factors exact and avoids a second SVD.

    Returns ``(W, L2, L1)`` with ``W ~= L2 @ L1 + residual``.
    """
    r = min(out_features, in_features)
    u, _ = torch.linalg.qr(torch.randn(out_features, r, device="cuda", dtype=torch.float32))
    v, _ = torch.linalg.qr(torch.randn(in_features, r, device="cuda", dtype=torch.float32))
    s = torch.exp(-torch.arange(r, device="cuda", dtype=torch.float32) / decay)
    w = (u * s) @ v.T
    scale = 0.1 / w.std()
    return w * scale, u[:, :rank] * s[:rank] * scale, v[:, :rank].T


# ---------------------------------------------------------------------------
# Plain W4A8
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES_5B + SHAPES_A14B, ids=lambda s: f"{s[1]}x{s[0]}")
@pytest.mark.parametrize("m", [M_RAGGED, M_ALIGNED], ids=["ragged", "aligned"])
def test_plain_w4a8_matches_bf16(shape, m):
    from vllm_omni.quantization.quark_w4a8_config import DiffusionQuarkW4A8Config

    in_features, out_features = shape
    torch.manual_seed(0)
    weight = (torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    bias = (torch.randn(out_features, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    x = torch.randn(m, in_features, device="cuda", dtype=torch.bfloat16) * 0.5

    layer = _make_layer(DiffusionQuarkW4A8Config(), in_features, out_features)
    layer.weight.weight_loader(layer.weight, weight)
    layer.bias.data.copy_(bias)

    out, _ = layer(x)
    reference = torch.nn.functional.linear(x.float(), weight.float(), bias.float())

    assert out.shape == (m, out_features)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
    assert _cos(out, reference) > 0.98


def test_plain_w4a8_applies_bias():
    """The bias must reach the GEMM epilogue, not just the call signature.

    Quark's entrypoint gates the bias operand on ``bias is not None and
    epilogue != "none"``, so handing it a bias without also asking for the bias
    epilogue drops it silently. A small bias hides inside the cosine tolerance
    of the shape sweep above -- this compares the two calls directly instead,
    with a bias large enough to dominate, which is what a diffusion
    transformer's modulation projections actually look like.
    """
    from vllm_omni.quantization import flydsl_w4a8

    flydsl_w4a8.register_ops()
    in_features, out_features, m = 3072, 3072, M_ALIGNED
    torch.manual_seed(0)
    weight = (torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    bias = torch.randn(out_features, device="cuda", dtype=torch.bfloat16).contiguous()
    x = torch.randn(m, in_features, device="cuda", dtype=torch.bfloat16) * 0.5
    packed, scale = flydsl_w4a8.pack_weight(weight)

    with_bias = torch.ops.vllm_omni.flydsl_w4a8_gemm(x, packed, scale, bias, out_features)
    without_bias = torch.ops.vllm_omni.flydsl_w4a8_gemm(x, packed, scale, None, out_features)

    # The only difference between the two calls is the bias row-broadcast, so
    # their difference must be the bias itself. Both outputs are separately
    # rounded to bf16 (8 mantissa bits), so allow two ULPs at the output scale.
    delta = (with_bias.float() - without_bias.float()) - bias.float()
    tolerance = with_bias.abs().max().item() * 2**-7
    assert delta.abs().max().item() < tolerance

    reference = torch.nn.functional.linear(x.float(), weight.float(), bias.float())
    assert _cos(with_bias, reference) > 0.98
    assert _cos(without_bias, reference) < 0.95


def test_plain_w4a8_frees_bf16_weight():
    """A14B holds both experts resident, so the BF16 copy must go at load time,
    and the kernel buffers must stay out of the state dict."""
    from vllm_omni.quantization.quark_w4a8_config import DiffusionQuarkW4A8Config

    layer = _make_layer(DiffusionQuarkW4A8Config(), 5120, 5120)
    assert layer.weight.device.type == "meta"

    layer.weight.weight_loader(layer.weight, torch.randn(5120, 5120, device="cuda", dtype=torch.bfloat16) * 0.1)

    assert layer.weight is None
    assert layer._kernel_weight.dtype == torch.uint8
    assert layer._kernel_scale.dtype == torch.uint8
    assert "_kernel_weight" not in layer.state_dict()
    assert "_kernel_scale" not in layer.state_dict()


def test_plain_w4a8_reshapes_3d_activations():
    """Diffusion activations arrive as (B, T, K); the reshape lives in
    MXFPLinearMethodBase.apply() and must not change the result."""
    from vllm_omni.quantization.quark_w4a8_config import DiffusionQuarkW4A8Config

    torch.manual_seed(0)
    layer = _make_layer(DiffusionQuarkW4A8Config(), 3072, 3072)
    layer.weight.weight_loader(layer.weight, torch.randn(3072, 3072, device="cuda", dtype=torch.bfloat16) * 0.1)
    x = torch.randn(M_RAGGED, 3072, device="cuda", dtype=torch.bfloat16) * 0.5

    out_2d, _ = layer(x)
    out_3d, _ = layer(x.view(6, M_RAGGED // 6, 3072))

    assert out_3d.shape == (6, M_RAGGED // 6, 3072)
    assert torch.equal(out_3d.reshape(M_RAGGED, 3072), out_2d)


def test_plain_w4a8_is_deterministic():
    from vllm_omni.quantization.quark_w4a8_config import DiffusionQuarkW4A8Config

    torch.manual_seed(0)
    layer = _make_layer(DiffusionQuarkW4A8Config(), 5120, 5120)
    layer.weight.weight_loader(layer.weight, torch.randn(5120, 5120, device="cuda", dtype=torch.bfloat16) * 0.1)
    x = torch.randn(M_RAGGED, 5120, device="cuda", dtype=torch.bfloat16) * 0.5

    first, _ = layer(x)
    for _ in range(2):
        again, _ = layer(x)
        assert torch.equal(first, again)


# ---------------------------------------------------------------------------
# SVDQuant W4A8
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(3072, 3072), (5120, 5120), (5120, 13824)], ids=lambda s: f"{s[1]}x{s[0]}")
@pytest.mark.parametrize("rank", [16, 32])
def test_svdquant_beats_plain_w4a8(shape, rank):
    """The low-rank branch exists to absorb the weight's dominant directions, so
    it must be at least as accurate as quantizing the full weight outright."""
    from vllm_omni.quantization.quark_w4a8_config import DiffusionQuarkW4A8Config

    in_features, out_features = shape
    torch.manual_seed(0)
    w32, _, _ = _spectral_weight(out_features, in_features, rank)
    weight = w32.to(torch.bfloat16).contiguous()
    bias = (torch.randn(out_features, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    x = torch.randn(M_RAGGED, in_features, device="cuda", dtype=torch.bfloat16) * 0.5

    # Both variants are handed the same full BF16 weight; the SVD variant splits
    # off its own low-rank term at load, exactly as it does from a checkpoint.
    svd_layer = _make_layer(DiffusionQuarkW4A8Config(svd_rank=rank), in_features, out_features)
    svd_layer.weight.weight_loader(svd_layer.weight, weight)
    svd_layer.bias.data.copy_(bias)
    assert svd_layer.proj_down.shape == (rank, in_features)
    assert svd_layer.proj_up.shape == (out_features, rank)

    plain_layer = _make_layer(DiffusionQuarkW4A8Config(), in_features, out_features)
    plain_layer.weight.weight_loader(plain_layer.weight, weight)
    plain_layer.bias.data.copy_(bias)

    reference = torch.nn.functional.linear(x.float(), weight.float(), bias.float())
    svd_out, _ = svd_layer(x)
    plain_out, _ = plain_layer(x)

    assert torch.isfinite(svd_out).all()
    svd_cos, plain_cos = _cos(svd_out, reference), _cos(plain_out, reference)
    assert svd_cos > 0.98
    assert svd_cos >= plain_cos, f"svd={svd_cos:.5f} plain={plain_cos:.5f}"


def test_svdquant_keeps_projections_after_packing():
    from vllm_omni.quantization.quark_w4a8_config import DiffusionQuarkW4A8Config

    torch.manual_seed(0)
    weight = (torch.randn(5120, 5120, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    layer = _make_layer(DiffusionQuarkW4A8Config(svd_rank=32), 5120, 5120)
    layer.weight.weight_loader(layer.weight, weight)

    assert layer.weight is None
    assert layer.proj_down is not None
    assert layer.proj_up is not None
    assert layer.quant_method.quant_config.svd_rank == 32
    # Derived at load from a BF16 checkpoint, so they are not checkpoint keys.
    assert "proj_down" not in layer.state_dict()
    assert "proj_up" not in layer.state_dict()


def test_svdquant_fused_matches_unfused():
    """QUARK_SVDQUANT_FUSED is not consulted here; instead compare the fused
    epilogue against plain-GEMM-plus-low-rank-add computed in torch."""
    from vllm_omni.quantization import flydsl_w4a8
    from vllm_omni.quantization.quark_w4a8_config import DiffusionQuarkW4A8Config

    in_features = out_features = 5120
    rank = 32
    torch.manual_seed(0)
    weight, _, _ = _spectral_weight(out_features, in_features, rank)
    x = torch.randn(M_RAGGED, in_features, device="cuda", dtype=torch.bfloat16) * 0.5

    layer = _make_layer(DiffusionQuarkW4A8Config(svd_rank=rank), in_features, out_features, bias=False)
    layer.weight.weight_loader(layer.weight, weight.to(torch.bfloat16).contiguous())

    fused, _ = layer(x)
    flydsl_w4a8.register_ops()
    unfused = torch.ops.vllm_omni.flydsl_w4a8_gemm(
        x, layer._kernel_weight, layer._kernel_scale, None, out_features
    ).float() + (torch.nn.functional.linear(x, layer.proj_down).float() @ layer.proj_up.float().T)

    assert _cos(fused, unfused) > 0.999


# ---------------------------------------------------------------------------
# Serialized checkpoints (offline calibrated export)
# ---------------------------------------------------------------------------


def test_plain_serialized_checkpoint_matches_bf16():
    """A plain serialized checkpoint loads a real BF16 weight from disk and packs
    it at load, exactly like the online path (which it subclasses)."""
    from vllm_omni.quantization.quark_w4a8_config import (
        DiffusionQuarkW4A8Config,
        QuarkW4A8LinearMethod,
    )

    in_features = out_features = 5120
    torch.manual_seed(0)
    weight = (torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    x = torch.randn(M_RAGGED, in_features, device="cuda", dtype=torch.bfloat16) * 0.5

    layer = _make_layer(
        DiffusionQuarkW4A8Config(is_checkpoint_w4a8_serialized=True), in_features, out_features, bias=False
    )
    assert type(layer.quant_method) is QuarkW4A8LinearMethod
    assert layer.quant_method.storage.name == "bf16"
    layer.weight.weight_loader(layer.weight, weight)  # lazy pack triggers here

    out, _ = layer(x)
    reference = torch.nn.functional.linear(x.float(), weight.float())
    assert torch.isfinite(out).all()
    assert _cos(out, reference) > 0.98


def test_svdquant_serialized_checkpoint_matches_bf16():
    """The SVD checkpoint method loads on-disk proj_down/proj_up (does NOT derive
    them) and its fused output must track the full-precision weight."""
    from vllm_omni.quantization.quark_w4a8_config import (
        DiffusionQuarkW4A8Config,
        QuarkW4A8SVDLinearMethod,
    )

    in_features = out_features = 5120
    rank = 32
    torch.manual_seed(0)
    weight, proj_up, proj_down = _spectral_weight(out_features, in_features, rank)
    residual = (weight - proj_up @ proj_down).to(torch.bfloat16).contiguous()
    x = torch.randn(M_RAGGED, in_features, device="cuda", dtype=torch.bfloat16) * 0.5

    cfg = DiffusionQuarkW4A8Config(svd_rank=rank, is_checkpoint_w4a8_serialized=True)
    layer = _make_layer(cfg, in_features, out_features, bias=False)
    assert isinstance(layer.quant_method, QuarkW4A8SVDLinearMethod)
    assert layer.quant_method.storage.name == "bf16"

    # Serialized create_weights registers real (non-meta) params; fill from "disk".
    layer.weight.data.copy_(residual)
    layer.proj_down.data.copy_(proj_down.to(torch.bfloat16))
    layer.proj_up.data.copy_(proj_up.to(torch.bfloat16))
    layer.quant_method.process_weights_after_loading(layer)

    assert layer.weight is None  # residual packed and dropped
    assert layer.proj_down.shape == (rank, in_features)
    assert layer.proj_up.shape == (out_features, rank)

    out, _ = layer(x)
    reference = torch.nn.functional.linear(x.float(), weight.float())
    assert torch.isfinite(out).all()
    assert _cos(out, reference) > 0.98


def test_plain_packed_matches_bf16_at_load():
    """A pre-packed checkpoint stores exactly what pack_weight emits, and the
    bf16-at-load path calls the same pack_weight at load, so the two are
    bit-identical."""
    from vllm_omni.quantization import flydsl_w4a8
    from vllm_omni.quantization.quark_w4a8_config import (
        DiffusionQuarkW4A8Config,
        QuarkW4A8LinearMethod,
    )

    in_features = out_features = 5120
    torch.manual_seed(0)
    weight = (torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    x = torch.randn(M_RAGGED, in_features, device="cuda", dtype=torch.bfloat16) * 0.5

    ref_layer = _make_layer(
        DiffusionQuarkW4A8Config(is_checkpoint_w4a8_serialized=True), in_features, out_features, bias=False
    )
    ref_layer.weight.weight_loader(ref_layer.weight, weight)  # packs at load
    ref_out, _ = ref_layer(x)

    packed, scale = flydsl_w4a8.pack_weight(weight)
    cfg = DiffusionQuarkW4A8Config(is_checkpoint_w4a8_serialized=True, quark_export_format="mxfp4_packed")
    layer = _make_layer(cfg, in_features, out_features, bias=False)
    assert type(layer.quant_method) is QuarkW4A8LinearMethod
    assert layer.quant_method.storage.name == "mxfp4_packed"
    layer.weight_shuffle.data.copy_(packed)
    layer.weight_scale.data.copy_(scale)
    layer.quant_method.process_weights_after_loading(layer)

    out, _ = layer(x)
    assert torch.equal(out, ref_out)


def test_svdquant_packed_matches_bf16_at_load():
    """The packed SVD path loads the packed residual + BF16 factors and must equal
    the bf16-at-load serialized SVD path on the same tensors."""
    from vllm_omni.quantization import flydsl_w4a8
    from vllm_omni.quantization.quark_w4a8_config import (
        DiffusionQuarkW4A8Config,
        QuarkW4A8SVDLinearMethod,
    )

    in_features = out_features = 5120
    rank = 32
    torch.manual_seed(0)
    weight, proj_up, proj_down = _spectral_weight(out_features, in_features, rank)
    residual = (weight - proj_up @ proj_down).to(torch.bfloat16).contiguous()
    proj_down = proj_down.to(torch.bfloat16).contiguous()
    proj_up = proj_up.to(torch.bfloat16).contiguous()
    x = torch.randn(M_RAGGED, in_features, device="cuda", dtype=torch.bfloat16) * 0.5

    ref = _make_layer(
        DiffusionQuarkW4A8Config(svd_rank=rank, is_checkpoint_w4a8_serialized=True),
        in_features,
        out_features,
        bias=False,
    )
    ref.weight.data.copy_(residual)
    ref.proj_down.data.copy_(proj_down)
    ref.proj_up.data.copy_(proj_up)
    ref.quant_method.process_weights_after_loading(ref)
    ref_out, _ = ref(x)

    packed, scale = flydsl_w4a8.pack_weight(residual)
    cfg = DiffusionQuarkW4A8Config(
        svd_rank=rank, is_checkpoint_w4a8_serialized=True, quark_export_format="mxfp4_packed"
    )
    layer = _make_layer(cfg, in_features, out_features, bias=False)
    assert isinstance(layer.quant_method, QuarkW4A8SVDLinearMethod)
    assert layer.quant_method.storage.name == "mxfp4_packed"
    layer.weight_shuffle.data.copy_(packed)
    layer.weight_scale.data.copy_(scale)
    layer.proj_down.data.copy_(proj_down)
    layer.proj_up.data.copy_(proj_up)
    layer.quant_method.process_weights_after_loading(layer)

    out, _ = layer(x)
    assert torch.equal(out, ref_out)


# ---------------------------------------------------------------------------
# Unshuffled serialized checkpoints (TP-capable format; TP=1 equivalence here)
# ---------------------------------------------------------------------------


def test_pack_unshuffled_then_shuffle_equals_packed():
    """The provider split must reproduce the whole-weight pack: shuffle_for_kernel
    of the unshuffled quant == pack_weight, byte-for-byte."""
    from vllm_omni.quantization import flydsl_w4a8

    torch.manual_seed(0)
    w = (torch.randn(5120, 5120, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    kw_p, ks_p = flydsl_w4a8.pack_weight(w)
    wq, ws = flydsl_w4a8.pack_weight_unshuffled(w)
    assert wq.shape == (5120, 2560) and ws.shape == (5120, 160)
    kw_u, ks_u = flydsl_w4a8.shuffle_for_kernel(wq, ws)
    assert torch.equal(kw_u, kw_p) and torch.equal(ks_u, ks_p)


def test_plain_unshuffled_matches_packed_at_tp1():
    from vllm_omni.quantization import flydsl_w4a8
    from vllm_omni.quantization.quark_w4a8_config import (
        DiffusionQuarkW4A8Config,
        QuarkW4A8LinearMethod,
    )

    in_features = out_features = 5120
    torch.manual_seed(0)
    weight = (torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    x = torch.randn(M_RAGGED, in_features, device="cuda", dtype=torch.bfloat16) * 0.5

    ref = _make_layer(
        DiffusionQuarkW4A8Config(is_checkpoint_w4a8_serialized=True), in_features, out_features, bias=False
    )
    ref.weight.weight_loader(ref.weight, weight)
    ref_out, _ = ref(x)

    wq, ws = flydsl_w4a8.pack_weight_unshuffled(weight)
    cfg = DiffusionQuarkW4A8Config(is_checkpoint_w4a8_serialized=True, quark_export_format="mxfp4_unshuffled")
    layer = _make_layer(cfg, in_features, out_features, bias=False)
    assert type(layer.quant_method) is QuarkW4A8LinearMethod
    assert layer.quant_method.storage.name == "mxfp4_unshuffled"
    layer.weight_packed.data.copy_(wq)
    layer.weight_scale.data.copy_(ws)
    layer.quant_method.process_weights_after_loading(layer)

    out, _ = layer(x)
    assert torch.equal(out, ref_out)


def test_svdquant_unshuffled_matches_packed_at_tp1():
    from vllm_omni.quantization import flydsl_w4a8
    from vllm_omni.quantization.quark_w4a8_config import (
        DiffusionQuarkW4A8Config,
        QuarkW4A8SVDLinearMethod,
    )

    in_features = out_features = 5120
    rank = 32
    torch.manual_seed(0)
    weight, proj_up, proj_down = _spectral_weight(out_features, in_features, rank)
    residual = (weight - proj_up @ proj_down).to(torch.bfloat16).contiguous()
    proj_down = proj_down.to(torch.bfloat16).contiguous()
    proj_up = proj_up.to(torch.bfloat16).contiguous()
    x = torch.randn(M_RAGGED, in_features, device="cuda", dtype=torch.bfloat16) * 0.5

    ref = _make_layer(
        DiffusionQuarkW4A8Config(svd_rank=rank, is_checkpoint_w4a8_serialized=True),
        in_features,
        out_features,
        bias=False,
    )
    ref.weight.data.copy_(residual)
    ref.proj_down.data.copy_(proj_down)
    ref.proj_up.data.copy_(proj_up)
    ref.quant_method.process_weights_after_loading(ref)
    ref_out, _ = ref(x)

    wq, ws = flydsl_w4a8.pack_weight_unshuffled(residual)
    cfg = DiffusionQuarkW4A8Config(
        svd_rank=rank, is_checkpoint_w4a8_serialized=True, quark_export_format="mxfp4_unshuffled"
    )
    layer = _make_layer(cfg, in_features, out_features, bias=False)
    assert isinstance(layer.quant_method, QuarkW4A8SVDLinearMethod)
    assert layer.quant_method.storage.name == "mxfp4_unshuffled"
    layer.weight_packed.data.copy_(wq)
    layer.weight_scale.data.copy_(ws)
    layer.proj_down.data.copy_(proj_down)
    layer.proj_up.data.copy_(proj_up)
    layer.quant_method.process_weights_after_loading(layer)

    out, _ = layer(x)
    assert torch.equal(out, ref_out)
