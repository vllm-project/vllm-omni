# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Host-side bias contract of the fused MoT GEMM.

``MoTQKVParallelLinear`` and ``MoTRowParallelLinear`` expose ``bias`` and
``vae_bias`` as independent constructor flags, so the text and VAE experts can
each have or lack a bias. These tests pin the four combinations down at the
``invoke_mot_gemm`` boundary, which is where a one-sided bias used to be
rejected outright.

CPU-only: everything asserted here is host-side argument resolution, so no GPU
or Triton compilation is involved.
"""

from __future__ import annotations

import types

import pytest
import torch

from vllm_omni.diffusion.layers.mot.ops import mot_gemm

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_N = 32
_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": _N,
    "BLOCK_SIZE_K": 16,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 2,
}


@pytest.fixture
def biases() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.randn(_N), torch.randn(_N)


# ---------------------------------------------------------------------------
# _resolve_bias_pointers
# ---------------------------------------------------------------------------


def test_both_biases_pass_through(biases):
    bias_text, bias_vae = biases
    p_text, p_vae, has_text, has_vae = mot_gemm._resolve_bias_pointers(bias_text, bias_vae)

    assert p_text is bias_text
    assert p_vae is bias_vae
    assert (has_text, has_vae) == (True, True)


def test_no_bias_yields_null_pointers():
    p_text, p_vae, has_text, has_vae = mot_gemm._resolve_bias_pointers(None, None)

    assert (p_text, p_vae) == (0, 0)
    assert (has_text, has_vae) == (False, False)


@pytest.mark.parametrize("side", ["text", "vae"])
def test_one_sided_bias_is_accepted(biases, side):
    """A one-sided bias must resolve, not raise.

    ``bias=True, vae_bias=False`` is the constructor default of both MoT linear
    layers, so this is the common case rather than an exotic one.
    """
    bias, _ = biases
    bias_text = bias if side == "text" else None
    bias_vae = bias if side == "vae" else None

    p_text, p_vae, has_text, has_vae = mot_gemm._resolve_bias_pointers(bias_text, bias_vae)

    assert (has_text, has_vae) == (side == "text", side == "vae")
    # The bias-less expert aliases the present tensor so the kernel's runtime
    # pointer select stays typeable and its load stays in bounds.
    assert p_text is bias
    assert p_vae is bias


@pytest.mark.parametrize(
    ("bias_text_present", "bias_vae_present"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_pointers_are_never_mixed_int_and_tensor(biases, bias_text_present, bias_vae_present):
    """Both pointers must be the same kind.

    The kernel picks one of the two bias pointers at runtime, so a ``0``
    placeholder next to a real tensor would not be typeable in Triton.
    """
    bias_text, bias_vae = biases
    p_text, p_vae, _, _ = mot_gemm._resolve_bias_pointers(
        bias_text if bias_text_present else None,
        bias_vae if bias_vae_present else None,
    )

    assert isinstance(p_text, torch.Tensor) == isinstance(p_vae, torch.Tensor)


def test_mismatched_bias_dtypes_are_rejected():
    with pytest.raises(AssertionError, match="dtype"):
        mot_gemm._resolve_bias_pointers(torch.randn(_N, dtype=torch.float32), torch.randn(_N, dtype=torch.bfloat16))


# ---------------------------------------------------------------------------
# invoke_mot_gemm wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def triton_dtypes(monkeypatch):
    """Make ``mot_gemm.tl`` expose the dtypes ``invoke_mot_gemm`` reads.

    ``invoke_mot_gemm`` maps torch dtypes onto ``tl`` dtypes while assembling
    launch arguments. Where Triton is absent, ``vllm.triton_utils`` supplies a
    placeholder without those attributes, so stand in with sentinels -- nothing
    here is compiled, the values only have to be distinguishable.
    """
    if getattr(mot_gemm.tl, "float32", None) is not None:
        return
    monkeypatch.setattr(
        mot_gemm,
        "tl",
        types.SimpleNamespace(
            float32="float32",
            float16="float16",
            bfloat16="bfloat16",
            int32="int32",
            int8="int8",
            float8e4nv="float8e4nv",
            float8e5="float8e5",
            float8e4m3fn="float8e4m3fn",
        ),
    )


@pytest.mark.parametrize(
    ("bias_text_present", "bias_vae_present"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_invoke_mot_gemm_forwards_per_expert_flags(
    monkeypatch, biases, triton_dtypes, bias_text_present, bias_vae_present
):
    """Every bias combination reaches the kernel with matching flags.

    A one-sided bias used to raise ``AssertionError: Bias must be provided for
    both Text and VAE simultaneously, or neither.`` before the launch.
    """
    captured: dict[str, object] = {}

    class _Recorder:
        def __getitem__(self, grid):
            def _launch(*args, **kwargs):
                captured.update(kwargs)

            return _launch

    monkeypatch.setattr(mot_gemm, "mot_unified_gemm_kernel", _Recorder())

    m_text, m_vae, k = 4, 4, 16
    m = m_text + m_vae
    bias_text, bias_vae = biases
    mot_gemm.invoke_mot_gemm(
        A=torch.randn(m, k, dtype=torch.bfloat16),
        B_text=torch.randn(k, _N, dtype=torch.bfloat16),
        B_vae=torch.randn(k, _N, dtype=torch.bfloat16),
        C=torch.zeros(m, _N, dtype=torch.bfloat16),
        bias_text=bias_text.to(torch.bfloat16) if bias_text_present else None,
        bias_vae=bias_vae.to(torch.bfloat16) if bias_vae_present else None,
        text_indices=torch.arange(0, m_text, dtype=torch.int32),
        vae_indices=torch.arange(m_text, m, dtype=torch.int32),
        A_scale=None,
        B_text_scale=None,
        B_vae_scale=None,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        A_per_channel_quant=False,
        B_per_channel_quant=False,
        config=dict(_CONFIG),
    )

    assert captured["HAS_BIAS_TEXT"] is bias_text_present
    assert captured["HAS_BIAS_VAE"] is bias_vae_present
