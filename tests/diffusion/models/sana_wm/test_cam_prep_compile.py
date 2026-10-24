# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the instance-bound dispatch of the Sana-WM camera-prep compile.

The opt-in selects a strict-lossless double-leaf compiled implementation: two
elementwise leaves (input prepare, rope+output) run compiled while the ray
projections and the inflation chain stay eager. Dispatch is bound to each
attention instance, never to the shared compiled singleton alone, and falls
back to the native function for unsupported inputs or gradient runs. These
tests pin down: dispatch routing (including a prepopulated cache), singleton
reuse, transformer-wide propagation without cache invalidation, the
env/enforce_eager/option gate, the full supported-range metadata gate, gradient
fallback across all seven tensor inputs, broadcastable-but-forbidden shapes,
and output parity on a small CPU shape. CPU-only; CUDA routing is exercised by
the review experiments.
"""

import pytest
import torch

import vllm_omni.diffusion.models.sana_wm.ucpe as ucpe
from vllm_omni.diffusion.models.sana_wm import sana_wm_transformer as tr_module
from vllm_omni.diffusion.models.sana_wm.pipeline_sana_wm import _camprep_compile_requested
from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import SanaWmSelfAttention, SanaWmTransformer3DModel
from vllm_omni.diffusion.models.sana_wm.ucpe import cam_prep_func, get_compiled_cam_prep

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

GRAD_INPUTS = ("q_normed", "k_normed", "v_raw", "proj_q", "proj_kv", "rope_cos", "rope_sin")


@pytest.fixture
def restore_cam_prep_state():
    saved = ucpe._compiled_cam_prep
    yield
    ucpe._compiled_cam_prep = saved


def _bare_attention() -> SanaWmSelfAttention:
    attn = SanaWmSelfAttention.__new__(SanaWmSelfAttention)
    torch.nn.Module.__init__(attn)
    attn._cam_prep_use_compiled = False
    return attn


def _bare_transformer_with(attns: list[SanaWmSelfAttention]) -> SanaWmTransformer3DModel:
    container = SanaWmTransformer3DModel.__new__(SanaWmTransformer3DModel)
    torch.nn.Module.__init__(container)
    for i, attn in enumerate(attns):
        container.register_module(f"attn{i}", attn)
    return container


def _small_inputs(device: str = "cpu") -> dict:
    batch, tokens, heads, head_dim = 1, 12, 2, 8
    half = head_dim // 2
    gen = torch.Generator(device=device).manual_seed(11)
    return {
        "q_normed": torch.randn(batch, tokens, heads, head_dim, generator=gen, device=device, dtype=torch.bfloat16),
        "k_normed": torch.randn(batch, tokens, heads, head_dim, generator=gen, device=device, dtype=torch.bfloat16),
        "v_raw": torch.randn(batch, tokens, heads, head_dim, generator=gen, device=device, dtype=torch.bfloat16),
        "proj_q": torch.randn(batch, tokens, 4, 4, generator=gen, device=device),
        "proj_kv": torch.randn(batch, tokens, 4, 4, generator=gen, device=device),
        "rope_cos": torch.randn(tokens, half, generator=gen, device=device),
        "rope_sin": torch.randn(tokens, half, generator=gen, device=device),
        "k_scale": 0.05,
    }


@pytest.mark.usefixtures("restore_cam_prep_state")
def test_dispatch_binds_to_instance_not_global_cache():
    # Real dispatch with a non-empty cache: pre-fill the global cache with a sentinel.
    sentinel = object()
    ucpe._compiled_cam_prep = sentinel

    a1, a2 = _bare_attention(), _bare_attention()
    t1 = _bare_transformer_with([a1])
    t2 = _bare_transformer_with([a2])
    t1.set_cam_prep_use_compiled(True)
    t2.set_cam_prep_use_compiled(True)

    # The enabled instance must dispatch to the compiled dispatch (bound-method
    # underlying-function identity).
    enabled_fn = a1._cam_prep_callable()
    assert getattr(enabled_fn, "__func__", None) is SanaWmSelfAttention._cam_prep_compiled_dispatch
    t2.set_cam_prep_use_compiled(False)
    assert ucpe._compiled_cam_prep is sentinel  # disabling never clears the cache
    enabled_fn2 = a1._cam_prep_callable()
    assert getattr(enabled_fn2, "__func__", None) is SanaWmSelfAttention._cam_prep_compiled_dispatch
    assert a2._cam_prep_callable() is cam_prep_func


@pytest.mark.usefixtures("restore_cam_prep_state")
def test_prepopulated_cache_routes_enabled_instance_and_spares_disabled(monkeypatch):
    # Pre-filled cache (no mock getter): a callable sentinel returning the marker tensor.
    marker_out = torch.zeros(1)

    def sentinel_out(*args, **kwargs):
        return marker_out

    ucpe._compiled_cam_prep = sentinel_out

    a1, a2 = _bare_attention(), _bare_attention()
    t1 = _bare_transformer_with([a1])
    t2 = _bare_transformer_with([a2])
    t1.set_cam_prep_use_compiled(True)
    t2.set_cam_prep_use_compiled(True)

    # get_compiled_cam_prep returns the cached callable; with the sentinel
    # pre-filled, dispatch must resolve to it.
    monkeypatch.setattr(tr_module, "get_compiled_cam_prep", lambda: ucpe._compiled_cam_prep)
    monkeypatch.setattr(SanaWmSelfAttention, "_cam_prep_supported", lambda self, q, k, v: True)
    monkeypatch.setattr(torch, "is_grad_enabled", lambda: False)

    kwargs = _small_inputs()
    out_enabled = a1._cam_prep_callable()(**kwargs)
    assert out_enabled is marker_out

    t2.set_cam_prep_use_compiled(False)
    assert ucpe._compiled_cam_prep is sentinel_out  # disabling never clears the cache
    assert a1._cam_prep_callable()(**kwargs) is marker_out  # the other enabled instance
    out_disabled = a2._cam_prep_callable()(**kwargs)  # the disabled instance falls back
    assert out_disabled is not marker_out
    assert a2._cam_prep_callable() is cam_prep_func


@pytest.mark.usefixtures("restore_cam_prep_state")
def test_compiled_coordinator_is_reused_across_instances():
    first = get_compiled_cam_prep()
    second = get_compiled_cam_prep()
    assert first is second  # lazily created once, no re-wrapping


@pytest.mark.usefixtures("restore_cam_prep_state")
def test_transformer_propagation_sets_all_attention_instances():
    attns = [_bare_attention() for _ in range(3)]
    container = _bare_transformer_with(attns)
    container.set_cam_prep_use_compiled(True)
    assert all(attn._cam_prep_use_compiled for attn in attns)
    saved_global = ucpe._compiled_cam_prep
    container.set_cam_prep_use_compiled(False)
    assert all(not attn._cam_prep_use_compiled for attn in attns)
    assert ucpe._compiled_cam_prep is saved_global  # disabling never clears the cache


@pytest.mark.usefixtures("restore_cam_prep_state")
def test_compiled_output_matches_native_on_small_shape():
    compiled = get_compiled_cam_prep()
    assert get_compiled_cam_prep() is compiled

    kwargs = _small_inputs()
    assert all(kwargs[n].dtype is torch.bfloat16 for n in ("q_normed", "k_normed", "v_raw"))
    with torch.inference_mode():
        native = cam_prep_func(**kwargs)
        out = compiled(**kwargs)
    assert len(native) == len(out) == 4
    for name, ref, got in zip(("q_out", "k_out", "v_out", "inflation_sq"), native, out, strict=True):
        assert got.dtype == ref.dtype, name
        assert got.shape == ref.shape, name
        assert got.is_contiguous() == ref.is_contiguous(), name
        view = torch.int16 if ref.dtype == torch.bfloat16 else torch.int32
        diff = int((got.view(view) != ref.view(view)).sum())
        assert diff == 0, (name, diff)


@pytest.mark.usefixtures("restore_cam_prep_state")
def test_enabled_branch_rejects_contract_violating_inputs():
    # Enabled instance with contract-violating inputs (odd head_dim, matching
    # shapes): the metadata gate only checks the even/tail contract, so odd D
    # reaches the native validation and raises ValueError.
    a = _bare_attention()
    a.set_cam_prep_use_compiled(True)
    kwargs = _small_inputs()
    for name in ("q_normed", "k_normed", "v_raw"):
        kwargs[name] = torch.randn(1, 12, 2, 7, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="head_dim"):
        a._cam_prep_callable()(**kwargs)


@pytest.mark.usefixtures("restore_cam_prep_state")
def test_broadcastable_but_native_forbidden_shapes_raise(monkeypatch):
    # Broadcastable-but-forbidden RoPE/projection shapes: both the enabled
    # dispatch and the native function must reject them.
    monkeypatch.setattr(SanaWmSelfAttention, "_cam_prep_supported", lambda self, q, k, v: True)
    monkeypatch.setattr(torch, "is_grad_enabled", lambda: False)
    a = _bare_attention()
    a.set_cam_prep_use_compiled(True)
    get_compiled_cam_prep()  # build the coordinator up front

    kwargs = _small_inputs()
    kwargs["rope_cos"] = torch.randn(1, kwargs["rope_cos"].shape[-1])  # (1, D/2): broadcastable
    kwargs["rope_sin"] = torch.randn(1, kwargs["rope_sin"].shape[-1])
    with pytest.raises(ValueError, match="rope table shapes"):
        a._cam_prep_callable()(**kwargs)
    with pytest.raises(ValueError, match="rope table shapes"):
        cam_prep_func(**kwargs)

    kwargs2 = _small_inputs()
    kwargs2["proj_q"] = torch.randn(1, 1, 4, 4, dtype=torch.bfloat16)  # (1,1,4,4): broadcastable
    kwargs2["proj_kv"] = torch.randn(1, 1, 4, 4, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="proj_q shape"):
        a._cam_prep_callable()(**kwargs2)
    with pytest.raises(ValueError, match="proj_q shape"):
        cam_prep_func(**kwargs2)


@pytest.mark.usefixtures("restore_cam_prep_state")
def test_unsupported_inputs_fall_back_to_native(monkeypatch):
    compiled_calls = []
    monkeypatch.setattr(
        tr_module,
        "get_compiled_cam_prep",
        lambda: (compiled_calls.append(1) or cam_prep_func),
    )
    a = _bare_attention()
    a.set_cam_prep_use_compiled(True)

    kwargs = _small_inputs()
    kwargs["v_raw"] = kwargs["v_raw"].to(torch.float32)  # non-BF16 -> unsupported
    with torch.inference_mode():
        ref = cam_prep_func(**kwargs)
        out = a._cam_prep_callable()(**kwargs)
    assert compiled_calls == []  # the compiled path was never reached
    for r, o in zip(ref, out, strict=True):
        assert torch.equal(r, o)


@pytest.mark.usefixtures("restore_cam_prep_state")
def test_supported_gate_checks_each_condition():
    # Per-condition supported-gate checks (require_cuda=False lifts only the
    # device check so the remaining conditions fail/pass independently).
    base = _small_inputs()

    def variant(**over):
        kw = dict(base)
        kw.update(over)
        return kw

    def supported(kw):
        return SanaWmSelfAttention._cam_prep_supported(kw["q_normed"], kw["k_normed"], kw["v_raw"], require_cuda=False)

    assert supported(base) is True  # baseline satisfies the gate
    assert supported(variant(v_raw=base["v_raw"].to(torch.float32))) is False  # dtype
    single = base["q_normed"][0]
    assert supported(variant(q_normed=single, k_normed=single, v_raw=single)) is False  # ndim
    mismatched = base["k_normed"][..., :6].contiguous()
    assert supported(variant(k_normed=mismatched)) is False  # q/k/v shape mismatch
    transposed = base["q_normed"].transpose(-1, -2)
    assert supported(variant(q_normed=transposed)) is False  # non-contiguous layout
    bad_dim = torch.randn(1, 12, 2, 6, dtype=torch.bfloat16)
    assert supported(variant(q_normed=bad_dim, k_normed=bad_dim, v_raw=bad_dim)) is False  # (D/2)%4!=0


@pytest.mark.usefixtures("restore_cam_prep_state")
def test_gradient_runs_fall_back_to_native(monkeypatch):
    # Lift the other metadata conditions first, then cover each of the seven
    # tensor inputs for gradient requirements.
    compiled_calls = []
    marker_out = torch.zeros(1)

    def sentinel(*args, **kwargs):
        return marker_out

    monkeypatch.setattr(tr_module, "get_compiled_cam_prep", lambda: sentinel)
    monkeypatch.setattr(SanaWmSelfAttention, "_cam_prep_supported", lambda self, q, k, v: True)
    a = _bare_attention()
    a.set_cam_prep_use_compiled(True)

    for name in GRAD_INPUTS:
        kwargs = _small_inputs()
        for other in GRAD_INPUTS:
            kwargs[other].requires_grad_(False)
        kwargs[name].requires_grad_(True)
        with torch.enable_grad():
            out_fallback = a._cam_prep_callable()(**kwargs)
        assert compiled_calls == [], f"{name} with requires_grad must not reach the compiled leaves"
        assert out_fallback is not marker_out  # falls back to native, not the sentinel
        # The same inputs without grad requirements go through the compiled path.
        for other in GRAD_INPUTS:
            kwargs[other].requires_grad_(False)
        with torch.enable_grad():
            out = a._cam_prep_callable()(**kwargs)
        assert out is marker_out
    assert compiled_calls == []  # the compiled path was never reached


def test_enable_gate_requires_env_and_non_eager_config(monkeypatch):
    class _Cfg:
        def __init__(self, eager: bool) -> None:
            self.enforce_eager = eager

    import vllm_omni.diffusion.models.sana_wm.pipeline_sana_wm as pipeline_module

    monkeypatch.delenv("VLLM_OMNI_SANA_WM_CAMPREP_COMPILE", raising=False)
    monkeypatch.setattr(pipeline_module, "_inductor_options_available", lambda: True)
    monkeypatch.setattr(pipeline_module.current_omni_platform, "is_available", lambda: True)
    monkeypatch.setattr(pipeline_module.current_omni_platform, "supports_torch_inductor", lambda: True)

    assert _camprep_compile_requested(None) is False
    assert _camprep_compile_requested(_Cfg(eager=True)) is False
    assert _camprep_compile_requested(_Cfg(eager=False)) is False

    monkeypatch.setenv("VLLM_OMNI_SANA_WM_CAMPREP_COMPILE", "1")
    assert _camprep_compile_requested(_Cfg(eager=True)) is False
    assert _camprep_compile_requested(_Cfg(eager=False)) is True

    monkeypatch.setattr(pipeline_module, "_inductor_options_available", lambda: False)
    assert _camprep_compile_requested(_Cfg(eager=False)) is False

    monkeypatch.setattr(pipeline_module.current_omni_platform, "supports_torch_inductor", lambda: False)
    assert _camprep_compile_requested(_Cfg(eager=False)) is False


@pytest.mark.usefixtures("restore_cam_prep_state")
def test_inductor_options_available_requires_pinned_keys(monkeypatch):
    real = ucpe.torch._inductor.list_options

    def fake_options(missing):
        return [o for o in real() if o not in missing]

    monkeypatch.setattr(ucpe.torch._inductor, "list_options", lambda: fake_options({"emulate_precision_casts"}))
    assert ucpe._inductor_options_available() is False
    monkeypatch.setattr(ucpe.torch._inductor, "list_options", real)
    assert ucpe._inductor_options_available() is True
