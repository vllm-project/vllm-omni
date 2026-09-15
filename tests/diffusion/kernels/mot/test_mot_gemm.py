# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.layers.mot.ops import mot_gemm

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 2,
}


class _RecordingKernel:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __getitem__(self, grid):
        del grid
        return self._launch

    def _launch(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


@pytest.mark.parametrize(
    ("bias_text", "bias_vae"),
    [(None, None), ("text", None), (None, "vae"), ("text", "vae")],
    ids=["no-bias", "text-only", "vae-only", "both"],
)
def test_invoke_mot_gemm_accepts_independent_biases(
    bias_text: str | None,
    bias_vae: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _RecordingKernel()
    monkeypatch.setattr(mot_gemm, "mot_unified_gemm_kernel", kernel)

    A = torch.ones((3, 4), dtype=torch.float16)
    B_text = torch.ones((4, 5), dtype=torch.float16)
    B_vae = torch.ones((4, 5), dtype=torch.float16)
    C = torch.empty((3, 5), dtype=torch.float16)
    text_indices = torch.tensor([0, 2], dtype=torch.long)
    vae_indices = torch.tensor([1], dtype=torch.long)
    text_bias = torch.ones(5, dtype=torch.float16) if bias_text else None
    vae_bias = torch.ones(5, dtype=torch.float16) if bias_vae else None

    mot_gemm.invoke_mot_gemm(
        A=A,
        B_text=B_text,
        B_vae=B_vae,
        C=C,
        bias_text=text_bias,
        bias_vae=vae_bias,
        text_indices=text_indices,
        vae_indices=vae_indices,
        A_scale=None,
        B_text_scale=None,
        B_vae_scale=None,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        A_per_channel_quant=False,
        B_per_channel_quant=False,
        config=_CONFIG,
    )

    assert len(kernel.calls) == 1
    args, kwargs = kernel.calls[0]
    assert kwargs["HAS_BIAS_TEXT"] is (text_bias is not None)
    assert kwargs["HAS_BIAS_VAE"] is (vae_bias is not None)
    assert "HAS_BIAS" not in kwargs

    launched_text_bias, launched_vae_bias = args[4:6]
    if text_bias is None and vae_bias is None:
        assert launched_text_bias == launched_vae_bias == 0
    elif text_bias is None:
        assert launched_text_bias is vae_bias
        assert launched_vae_bias is vae_bias
    elif vae_bias is None:
        assert launched_text_bias is text_bias
        assert launched_vae_bias is text_bias
    else:
        assert launched_text_bias is text_bias
        assert launched_vae_bias is vae_bias
