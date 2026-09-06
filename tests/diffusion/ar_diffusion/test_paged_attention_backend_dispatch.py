# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Backend-dispatch tests for AR-Diffusion paged self-attention.

``ar_diffusion_paged_attention`` used to select its backend with
``Tensor.is_cuda``, which is ``False`` on XPU and therefore routed every call to
the dense Python reference. These cover the replacement contract:

1. An accelerator with a paged kernel uses it, not ``_reference_paged_attention``.
2. The kernel receives the original ``block_table`` and KV pools (no dense gather).
3. CUDA still resolves an ``fa_version``; XPU does not need one.
4. A missing kernel on such a device raises instead of degrading silently, and
   the opt-in env switch restores the reference path tagged as ``"reference"``.
5. CPU keeps the reference path.

Everything is CPU/mock based, so it needs no accelerator: ``query.device.type``
is faked and vLLM's ``fa_utils`` entry points are stubbed. ``torch.version.hip``
is pinned to ``None`` so the ROCm branch is not selected by the host build.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from vllm_omni.experimental.ar_diffusion.kv_cache import paged_attention as pa

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

BLOCK = 16
N_HEADS = 4
HEAD_DIM = 64
SCALE = HEAD_DIM**-0.5

# fa_utils only binds flash_attn_varlen_func on CUDA/XPU/ROCm, so on a CPU
# runner the attribute does not exist and patch() must be allowed to create it.
_FA_FUNC = "vllm.v1.attention.backends.fa_utils.flash_attn_varlen_func"
_FA_AVAILABLE = "vllm.v1.attention.backends.fa_utils.is_flash_attn_varlen_func_available"


def _make_paged_inputs(kv_len=BLOCK, q_len=BLOCK):
    """Minimal (query, KV pools, block-table metadata) for one attention call."""
    num_blocks = (kv_len + BLOCK - 1) // BLOCK + 1  # +1 spare (padding) block
    query = torch.randn(q_len, N_HEADS, HEAD_DIM)
    key_cache = torch.randn(num_blocks, BLOCK, N_HEADS, HEAD_DIM)
    value_cache = torch.randn_like(key_cache)
    n_used = (kv_len + BLOCK - 1) // BLOCK
    used = torch.arange(n_used, dtype=torch.int32).flip(0)
    pad = torch.full((num_blocks - n_used,), num_blocks - 1, dtype=torch.int32)
    block_table = torch.cat([used, pad]).view(1, num_blocks)
    query_start_loc = torch.tensor([0, q_len], dtype=torch.int32)
    seq_lens = torch.tensor([kv_len], dtype=torch.int32)
    return query, key_cache, value_cache, block_table, query_start_loc, seq_lens


class _FakeDeviceTensor(torch.Tensor):
    """A real CPU tensor that reports an arbitrary ``device.type``.

    A Tensor subclass so ``torch.empty_like`` / ``reshape`` keep working on real
    CPU storage; only ``device`` is overridden, which is enough to make the
    dispatch believe it is on cuda/xpu with no accelerator present.
    """

    _device_type = "cpu"

    @staticmethod
    def make(t: torch.Tensor, device_type: str) -> _FakeDeviceTensor:
        obj = t.as_subclass(_FakeDeviceTensor)
        obj._device_type = device_type
        return obj

    @property
    def device(self):  # type: ignore[override]
        class _D:
            type = self._device_type

        return _D()


def _call(query, key_cache, value_cache, block_table, query_start_loc, seq_lens):
    return pa.ar_diffusion_paged_attention(
        query,
        key_cache,
        value_cache,
        block_table=block_table,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_query_len=BLOCK,
        max_seq_len=BLOCK,
        softmax_scale=SCALE,
    )


@pytest.mark.parametrize("device_type", ["xpu", "cuda"])
def test_accelerator_uses_platform_kernel_not_reference(device_type):
    """(1)+(2) An available kernel is used, and the paged inputs reach it as-is."""
    q, kc, vc, bt, qsl, sl = _make_paged_inputs()
    seen: dict[str, object] = {}

    def fake_fa(**kwargs):
        seen.update(kwargs)
        return torch.zeros_like(kwargs["q"])

    with (
        patch("torch.version.hip", None),
        patch(_FA_FUNC, side_effect=fake_fa, create=True),
        patch(_FA_AVAILABLE, return_value=True),
        patch.object(pa, "_reference_paged_attention") as ref_mock,
    ):
        out = _call(_FakeDeviceTensor.make(q, device_type), kc, vc, bt, qsl, sl)

    ref_mock.assert_not_called()
    assert pa.ar_diffusion_paged_attention_backend == device_type
    assert out.shape == q.shape
    # Paged pools and block table are forwarded unchanged -- no dense gather.
    assert seen["block_table"] is bt
    assert seen["k"] is kc
    assert seen["v"] is vc
    assert seen["seqused_k"] is sl


def test_cuda_resolves_fa_version_xpu_does_not():
    """(3) fa_version is a CUDA concern; the XPU kernel is FA2 and ignores it."""
    q, kc, vc, bt, qsl, sl = _make_paged_inputs()
    versions: dict[str, object] = {}

    def fake_fa(**kwargs):
        versions[str(kwargs["q"].device.type)] = kwargs["fa_version"]
        return torch.zeros_like(kwargs["q"])

    for device_type in ("cuda", "xpu"):
        with (
            patch("torch.version.hip", None),
            patch(_FA_FUNC, side_effect=fake_fa, create=True),
            patch(_FA_AVAILABLE, return_value=True),
            patch.object(pa, "_resolve_fa_version", return_value=3) as ver_mock,
        ):
            _call(_FakeDeviceTensor.make(q, device_type), kc, vc, bt, qsl, sl)
            assert ver_mock.call_count == (1 if device_type == "cuda" else 0)

    assert versions == {"cuda": 3, "xpu": 2}


@pytest.mark.parametrize("device_type", ["xpu", "cuda"])
def test_missing_kernel_fails_fast(device_type, monkeypatch):
    """(4) No silent reference fallback on a device that should have a kernel."""
    q, kc, vc, bt, qsl, sl = _make_paged_inputs()
    monkeypatch.delenv(pa._ALLOW_REFERENCE_ATTN_ENV, raising=False)

    with (
        patch("torch.version.hip", None),
        patch(_FA_AVAILABLE, return_value=False),
        patch.object(pa, "_reference_paged_attention") as ref_mock,
    ):
        with pytest.raises(RuntimeError, match="refusing to fall back"):
            _call(_FakeDeviceTensor.make(q, device_type), kc, vc, bt, qsl, sl)

    ref_mock.assert_not_called()


def test_missing_kernel_with_env_switch_uses_reference(monkeypatch):
    """(4b) The opt-in switch restores the reference path, tagged as such."""
    q, kc, vc, bt, qsl, sl = _make_paged_inputs()
    monkeypatch.setenv(pa._ALLOW_REFERENCE_ATTN_ENV, "1")

    with (
        patch("torch.version.hip", None),
        patch(_FA_AVAILABLE, return_value=False),
        patch.object(pa, "_reference_paged_attention", return_value=torch.zeros_like(q)) as ref_mock,
    ):
        _call(_FakeDeviceTensor.make(q, "xpu"), kc, vc, bt, qsl, sl)

    ref_mock.assert_called_once()
    assert pa.ar_diffusion_paged_attention_backend == "reference"


def test_cpu_uses_reference_backend():
    """(5) CPU has no paged kernel: dense reference, tagged 'reference'."""
    q, kc, vc, bt, qsl, sl = _make_paged_inputs()
    out = _call(q, kc, vc, bt, qsl, sl)
    assert out.shape == q.shape
    assert pa.ar_diffusion_paged_attention_backend == "reference"


def test_xpu_page_size_refusal_falls_back_and_is_remembered(monkeypatch):
    """A kernel that refuses the page size must degrade loudly, not abort.

    ``is_flash_attn_varlen_func_available()`` reports True on XPU unconditionally --
    it answers "is an entry point bound", not "can it service this geometry". The
    page-size check lives in the kernel's C++ and raises from there, so without this
    path a build that refuses DreamZero's frame-length page turns a working (if slow)
    XPU run into a hard abort, and the env-var escape hatch never even runs.

    The refusal is recorded, so later calls skip the failed dispatch entirely rather
    than paying it once per layer per step.
    """
    q, kc, vc, bt, qsl, sl = _make_paged_inputs()
    monkeypatch.setattr(pa, "_XPU_REJECTED_PAGE_SIZES", set())
    page_size = kc.shape[1]

    calls = {"kernel": 0}

    def refusing_fa(**kwargs):
        calls["kernel"] += 1
        raise RuntimeError(f"chunk_prefill: unsupported block_size={page_size} (supported: 16, 32, ...)")

    with (
        patch("torch.version.hip", None),
        patch(_FA_FUNC, side_effect=refusing_fa, create=True),
        patch(_FA_AVAILABLE, return_value=True),
    ):
        first = _call(_FakeDeviceTensor.make(q, "xpu"), kc, vc, bt, qsl, sl)
        assert pa.ar_diffusion_paged_attention_backend == "reference"
        assert first.shape == q.shape
        assert page_size in pa._XPU_REJECTED_PAGE_SIZES

        second = _call(_FakeDeviceTensor.make(q, "xpu"), kc, vc, bt, qsl, sl)

    assert calls["kernel"] == 1, "the refused page size should not be retried per call"
    assert second.shape == q.shape
    assert pa.ar_diffusion_paged_attention_backend == "reference"


def test_unrelated_kernel_errors_still_propagate(monkeypatch):
    """Only the page-size complaint is absorbed; real failures must not be hidden."""
    q, kc, vc, bt, qsl, sl = _make_paged_inputs()
    monkeypatch.setattr(pa, "_XPU_REJECTED_PAGE_SIZES", set())

    def exploding_fa(**kwargs):
        raise RuntimeError("XPU out of memory")

    with (
        patch("torch.version.hip", None),
        patch(_FA_FUNC, side_effect=exploding_fa, create=True),
        patch(_FA_AVAILABLE, return_value=True),
    ):
        with pytest.raises(RuntimeError, match="out of memory"):
            _call(_FakeDeviceTensor.make(q, "xpu"), kc, vc, bt, qsl, sl)

    assert pa._XPU_REJECTED_PAGE_SIZES == set()


def test_cuda_page_size_errors_are_not_absorbed(monkeypatch):
    """The refusal path is XPU-only; CUDA has no such page-size constraint."""
    q, kc, vc, bt, qsl, sl = _make_paged_inputs()
    monkeypatch.setattr(pa, "_XPU_REJECTED_PAGE_SIZES", set())

    def refusing_fa(**kwargs):
        raise RuntimeError("chunk_prefill: unsupported block_size=880")

    with (
        patch("torch.version.hip", None),
        patch(_FA_FUNC, side_effect=refusing_fa, create=True),
        patch(_FA_AVAILABLE, return_value=True),
    ):
        with pytest.raises(RuntimeError, match="unsupported block_size"):
            _call(_FakeDeviceTensor.make(q, "cuda"), kc, vc, bt, qsl, sl)

    assert pa._XPU_REJECTED_PAGE_SIZES == set()
