# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for the MiniMax-H3 VAE decoder fast-path gating.

The three env switches (``H3_VAE_XEFMHA``, ``H3_VAE_COMPILE``,
``H3_VAE_FP16_WEIGHTS``) are XPU-only, and the compile switch additionally
requires the pipeline's eager/inductor decision (``allow_decoder_compile``).
Non-XPU construction must be a quiet no-op: no wraps, no casts, no ERROR.
"""

import json
import sys
import types

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module
from vllm_omni.diffusion.models.minimax_h3 import vae_xefmha as xefmha_module
from vllm_omni.diffusion.models.minimax_h3.ops import vae as vae_ops
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE, vae_compile_allowed

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _FakeAttn:
    def __init__(self):
        self.training = False
        self.spatial_parallel = False
        self.calls: list[tuple] = []

        def original(query, key, value, pack_info):
            self.calls.append((query, key, value, pack_info))
            return query

        self.perform_attention = original
        self._original = original


class _FakeBlock:
    def __init__(self):
        self.attn = _FakeAttn()
        self.forward = lambda *a, **k: None


class _FakeDecoder(nn.Module):
    def __init__(self, num_blocks=2):
        super().__init__()
        self.transformer_blocks = [_FakeBlock() for _ in range(num_blocks)]
        self.t_causal = False


class _FakeRemote(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = types.SimpleNamespace(
            decoder=_FakeDecoder(),
            _all_gather_tiled_results=lambda *a, **k: None,
        )

    def to(self, *a, **k):  # never move: CI has no XPU
        return self

    def eval(self):
        return self

    @classmethod
    def from_pretrained(cls, _path):
        return cls()


@pytest.fixture
def component_dir(tmp_path):
    cfg = {
        "latent_channels": 4,
        "latents_mean": [0.0] * 4,
        "latents_std": [1.0] * 4,
        "auto_map": {"AutoModel": "modeling.FakeVAE"},
    }
    d = tmp_path / "video_vae"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg))
    return str(d)


@pytest.fixture
def compile_spy(monkeypatch):
    calls = []

    def fake_compile(fn, **kwargs):
        calls.append(fn)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    return calls


@pytest.fixture
def linears(monkeypatch):
    block_linears = [nn.Linear(4, 4), nn.Linear(4, 4)]
    monkeypatch.setattr(vae_ops, "_decoder_block_linears", lambda _decoder: block_linears)
    return block_linears


@pytest.fixture
def fake_checkpoint(monkeypatch):
    monkeypatch.setattr(vae_module, "get_class_from_dynamic_module", lambda _ref, _path: _FakeRemote)
    monkeypatch.setattr(vae_module, "install_h3_vae_optimizations", lambda *a, **k: False)
    monkeypatch.setitem(
        sys.modules,
        "vllm._xpu_ops",
        types.SimpleNamespace(xpu_ops=types.SimpleNamespace(flash_attn_varlen_func=lambda **kw: kw["q"])),
    )


def _build(component_dir, device, **kwargs):
    return MiniMaxH3VideoVAE(
        component_dir,
        device=torch.device(device),
        trust_remote_code=True,
        **kwargs,
    )


def test_xpu_allowed_installs_all_three(component_dir, compile_spy, linears, fake_checkpoint, monkeypatch):
    for var in ("H3_VAE_XEFMHA", "H3_VAE_COMPILE", "H3_VAE_FP16_WEIGHTS"):
        monkeypatch.delenv(var, raising=False)
    vae = _build(component_dir, "xpu", allow_decoder_compile=True)
    decoder = vae.remote.model.decoder

    assert len(compile_spy) == len(decoder.transformer_blocks)
    assert all(lin.weight.dtype == torch.float16 for lin in linears)
    assert all(b.attn.perform_attention is not b.attn._original for b in decoder.transformer_blocks)
    assert getattr(decoder, "_h3_xefmha_stats", None) is not None


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_non_xpu_is_a_quiet_noop(component_dir, compile_spy, linears, fake_checkpoint, monkeypatch, device):
    for var in ("H3_VAE_XEFMHA", "H3_VAE_COMPILE", "H3_VAE_FP16_WEIGHTS"):
        monkeypatch.delenv(var, raising=False)
    vae = _build(component_dir, device, allow_decoder_compile=True)
    decoder = vae.remote.model.decoder

    assert not compile_spy
    assert all(lin.weight.dtype == torch.float32 for lin in linears)
    assert all(b.attn.perform_attention is b.attn._original for b in decoder.transformer_blocks)
    assert getattr(decoder, "_h3_xefmha_stats", None) is None


def test_xpu_compile_denied_keeps_other_two(component_dir, compile_spy, linears, fake_checkpoint, monkeypatch):
    for var in ("H3_VAE_XEFMHA", "H3_VAE_COMPILE", "H3_VAE_FP16_WEIGHTS"):
        monkeypatch.delenv(var, raising=False)
    vae = _build(component_dir, "xpu", allow_decoder_compile=False)
    decoder = vae.remote.model.decoder

    assert not compile_spy
    assert all(lin.weight.dtype == torch.float16 for lin in linears)
    assert all(b.attn.perform_attention is not b.attn._original for b in decoder.transformer_blocks)


def test_xpu_env_zero_disables_each_switch(component_dir, compile_spy, linears, fake_checkpoint, monkeypatch):
    for var in ("H3_VAE_XEFMHA", "H3_VAE_COMPILE", "H3_VAE_FP16_WEIGHTS"):
        monkeypatch.setenv(var, "0")
    vae = _build(component_dir, "xpu", allow_decoder_compile=True)
    decoder = vae.remote.model.decoder

    assert not compile_spy
    assert all(lin.weight.dtype == torch.float32 for lin in linears)
    assert all(b.attn.perform_attention is b.attn._original for b in decoder.transformer_blocks)


@pytest.mark.parametrize(
    ("enforce_eager", "supports_inductor", "expected"),
    [
        (True, True, False),  # eager veto beats platform capability
        (True, False, False),
        (False, True, True),
        (False, False, False),
    ],
)
def test_vae_compile_allowed_priority(enforce_eager, supports_inductor, expected):
    platform = types.SimpleNamespace(supports_torch_inductor=lambda: supports_inductor)
    assert vae_compile_allowed(enforce_eager, platform) is expected


def test_pipeline_passes_the_compile_decision_to_the_vae():
    """The ctor default is False, so the gate only means anything if the
    pipeline actually passes it. Pin the one production construction site:
    ``allow_decoder_compile=vae_compile_allowed(bool(od_config.enforce_eager),
    current_omni_platform)``. Deleting the keyword or the call turns this red.
    Source-level: importing the pipeline module needs the full serving stack.
    """
    import ast

    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3

    tree = ast.parse(open(pipeline_minimax_h3.__file__.rstrip("c"), encoding="utf-8").read())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (getattr(node.func, "id", None) or getattr(node.func, "attr", None)) == "MiniMaxH3VideoVAE"
    ]
    assert len(calls) == 1, f"expected exactly one video-VAE construction site, found {len(calls)}"
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    assert "allow_decoder_compile" in keywords, "pipeline no longer passes the compile decision"
    value = keywords["allow_decoder_compile"]
    assert isinstance(value, ast.Call), ast.unparse(value)
    assert (getattr(value.func, "id", None) or getattr(value.func, "attr", None)) == "vae_compile_allowed"
    assert [ast.unparse(a) for a in value.args] == ["bool(od_config.enforce_eager)", "current_omni_platform"]


def test_non_xpu_xefmha_installer_is_silent():
    """install_vae_xefmha off-XPU is a quiet no-op, not a raise."""
    decoder = _FakeDecoder()
    xefmha_module.install_vae_xefmha(decoder, torch.device("cuda"), vae_module.logger)
    assert getattr(decoder, "_h3_xefmha_stats", None) is None
    assert all(b.attn.perform_attention is b.attn._original for b in decoder.transformer_blocks)


class _LoggerSpy:
    def __init__(self):
        self.lines = []

    def info(self, msg, *args):
        self.lines.append(msg % args if args else str(msg))


def test_xefmha_fallback_logs_reason_once_and_periodic_stats(monkeypatch):
    """A guarded call says WHY it fell back (once per reason), and the wrapper
    reports calls/fallbacks periodically instead of counting silently."""
    monkeypatch.setitem(
        sys.modules,
        "vllm._xpu_ops",
        types.SimpleNamespace(xpu_ops=types.SimpleNamespace(flash_attn_varlen_func=lambda **kw: kw["q"])),
    )
    log = _LoggerSpy()
    decoder = _FakeDecoder(num_blocks=1)
    xefmha_module.install_vae_xefmha(decoder, torch.device("xpu"), log)
    attn = decoder.transformer_blocks[0].attn
    attn.training = True
    q = torch.randn(1, 4, 32, 64, dtype=torch.float16)

    for _ in range(xefmha_module._STATS_LOG_INTERVAL):
        attn.perform_attention(q, q, q, None)

    fallback_lines = [ln for ln in log.lines if "falling back" in ln]
    assert len(fallback_lines) == 1
    assert "mode" in fallback_lines[0]
    assert any("calls=0" in ln and "fallbacks=1000" in ln for ln in log.lines)
    assert decoder._h3_xefmha_stats == {"calls": 0, "fallbacks": 1000}


def test_xefmha_module_header_is_factual():
    """The module is deployed code, not a review proposal, and makes no claim
    that the kernel still needs trace confirmation."""
    src = open(xefmha_module.__file__.rstrip("c"), encoding="utf-8").read()
    assert "Read-only review proposal" not in src
    assert "trace confirmation" not in src
    assert src.startswith("# SPDX-License-Identifier")


class _XpuTensorProxy:
    """Wraps a CPU tensor but reports ``device.type == 'xpu'`` (CI has no XPU);
    everything else passes through to the wrapped tensor."""

    _device = types.SimpleNamespace(type="xpu")

    def __init__(self, t):
        object.__setattr__(self, "_t", t)

    @property
    def device(self):
        return type(self)._device

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_t"), name)


def _installed_attn(monkeypatch, ops_calls):
    monkeypatch.setitem(
        sys.modules,
        "vllm._xpu_ops",
        types.SimpleNamespace(
            xpu_ops=types.SimpleNamespace(flash_attn_varlen_func=lambda **kw: ops_calls.append(kw) or kw["q"])
        ),
    )
    log = _LoggerSpy()
    decoder = _FakeDecoder(num_blocks=1)
    xefmha_module.install_vae_xefmha(decoder, torch.device("xpu"), log)
    return decoder.transformer_blocks[0].attn, log


def _cpu_arange_shim(monkeypatch):
    """Force ``torch.arange`` allocations onto the CPU (CI has no XPU).

    The fast path allocates its ``cu_seqlens`` on ``query.device``, which the
    proxy reports as a fake XPU. With this shim in place a call that wrongly
    passes the guards *executes* the fast path against the stubbed kernel
    instead of dying in ``torch.arange`` -- so the fallback assertions below
    stay behavioral: deleting a guard makes them fail on the missing fallback
    line and original-implementation call, not on an infrastructure TypeError.
    """
    real_arange = torch.arange
    monkeypatch.setattr(torch, "arange", lambda *a, **k: real_arange(*a, **{**k, "device": "cpu"}))


def test_xefmha_device_mismatch_falls_back_with_a_device_reason(monkeypatch):
    """Q claims XPU but K/V live elsewhere: decline with the device reason."""
    ops_calls: list[dict] = []
    attn, log = _installed_attn(monkeypatch, ops_calls)
    _cpu_arange_shim(monkeypatch)
    q = _XpuTensorProxy(torch.randn(1, 4, 32, 64, dtype=torch.float16))
    kv = torch.randn(1, 4, 32, 64, dtype=torch.float16)  # a real CPU tensor

    attn.perform_attention(q, kv, kv, None)

    fallback_lines = [ln for ln in log.lines if "falling back" in ln]
    assert len(fallback_lines) == 1 and "device" in fallback_lines[0].lower(), log.lines
    assert not ops_calls
    assert len(attn.calls) == 1, "the call must reach the original implementation"


def test_xefmha_misaligned_strides_fall_back_with_the_alignment_reason(monkeypatch):
    """Everything else in the envelope (xpu / fp16 / (1, L, 32, 64)) but the
    row strides break the kernel's 64-byte alignment contract: a [..., :64]
    slice of a 65-wide tensor keeps stride(-2) == 65 and 65 * 2 % 64 != 0."""
    ops_calls: list[dict] = []
    attn, log = _installed_attn(monkeypatch, ops_calls)
    _cpu_arange_shim(monkeypatch)
    q = _XpuTensorProxy(torch.randn(1, 8, 32, 65, dtype=torch.float16)[..., :64])

    attn.perform_attention(q, q, q, None)

    fallback_lines = [ln for ln in log.lines if "falling back" in ln]
    assert len(fallback_lines) == 1 and "alignment" in fallback_lines[0].lower(), log.lines
    assert not ops_calls, "the misaligned call must not reach the kernel"
    assert len(attn.calls) == 1, "the call must reach the original implementation"
