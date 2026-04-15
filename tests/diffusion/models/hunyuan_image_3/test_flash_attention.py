# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Functional correctness tests for FlashAttention dispatch paths
in the HunyuanImage3 denoising transformer.

Verifies numerical consistency between each FlashAttention path
and the SDPA reference (path 6 / fallback) across all testable
dispatch branches:

  Path 1: Step 2-50, non-SP, varlen FA
  Path 2: Step 2-50, non-SP, split two-call FA
  Path 4: Step 1,    non-SP, causal+full FA
  Path 6: Fallback   SDPA with 4D mask (reference baseline)

Paths 3 and 5 (SP mode) require multi-GPU distributed setup and are
covered separately in integration tests.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.platforms import current_omni_platform

is_gpu = current_omni_platform.is_cuda_alike()

pytestmark = [pytest.mark.core_model]

try:
    from vllm_omni.diffusion.attention.backends.utils.fa import (
        flash_attn_func,
        flash_attn_varlen_func,
    )
except ImportError:
    flash_attn_func = None
    flash_attn_varlen_func = None

HAS_FA = flash_attn_func is not None
HAS_VARLEN = flash_attn_varlen_func is not None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEVICE = torch.device(current_omni_platform.device_type) if is_gpu else torch.device("cpu")
DTYPE = torch.bfloat16

NUM_HEADS = 8
NUM_KV_HEADS = 4
HEAD_DIM = 64
SCALING = 1.0 / (HEAD_DIM**0.5)

TEXT_LEN = 12
IMAGE_LEN = 32
Q_IMAGE_LEN = IMAGE_LEN


def _unwrap(out):
    return out[0] if isinstance(out, tuple) else out


def _make_qkv(bs: int, q_len: int, kv_len: int | None = None, *, seed: int = 42):
    """Generate random Q/K/V tensors. KV uses NUM_KV_HEADS; Q uses NUM_HEADS."""
    if kv_len is None:
        kv_len = q_len
    rng = torch.Generator(device=DEVICE).manual_seed(seed)
    q = torch.randn(bs, q_len, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE, generator=rng)
    k = torch.randn(bs, kv_len, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE, generator=rng)
    v = torch.randn(bs, kv_len, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE, generator=rng)
    return q, k, v


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    bs, slen, n_kv, hd = x.shape
    return x[:, :, :, None, :].expand(bs, slen, n_kv, n_rep, hd).reshape(bs, slen, n_kv * n_rep, hd)


def _build_step1_mask(bs: int, seq_len: int, text_len: int) -> torch.Tensor:
    """
    Build the 4D attention mask for HunyuanImage3 step-1.

    Sequence layout: [text(text_len) | timestamp(1) | image(image_len) | eoi(1)]
    - text: causal within text
    - timestamp: attends to all text + timestamp
    - image: attends to all text + timestamp + all image
    - eoi: attends to everything
    """
    ts_pos = text_len
    img_start = text_len + 1
    img_end = seq_len - 1  # eoi is the last token
    mask = torch.zeros(bs, 1, seq_len, seq_len, device=DEVICE, dtype=DTYPE)
    mask.fill_(float("-inf"))
    for i in range(seq_len):
        for j in range(seq_len):
            if i < text_len:
                if j <= i:
                    mask[:, :, i, j] = 0.0
            elif i == ts_pos:
                if j <= ts_pos:
                    mask[:, :, i, j] = 0.0
            elif img_start <= i < img_end:
                if j < img_end:
                    mask[:, :, i, j] = 0.0
            else:  # eoi
                mask[:, :, i, j] = 0.0
    return mask


def _build_step2_mask(bs: int, q_len: int, kv_len: int, text_len: int) -> torch.Tensor:
    """
    Build the 4D mask for steps 2-50.

    Q layout: [timestamp(1) | image(q_len-1)]
    KV layout: [text(text_len) | timestamp(1) | image(q_len-1)]
    - timestamp Q attends to text + timestamp only
    - image Q attends to all KV
    """
    mask = torch.zeros(bs, 1, q_len, kv_len, device=DEVICE, dtype=DTYPE)
    mask.fill_(float("-inf"))
    ts_kv_end = text_len + 1
    for qi in range(q_len):
        for kj in range(kv_len):
            if qi == 0:
                if kj < ts_kv_end:
                    mask[:, :, qi, kj] = 0.0
            else:
                mask[:, :, qi, kj] = 0.0
    return mask


def _sdpa_reference(q, k, v, mask_4d):
    """SDPA reference with GQA expansion and 4D mask."""
    rep = NUM_HEADS // NUM_KV_HEADS
    k_exp = _repeat_kv(k, rep)
    v_exp = _repeat_kv(v, rep)
    q_t = q.permute(0, 2, 1, 3)
    k_t = k_exp.permute(0, 2, 1, 3)
    v_t = v_exp.permute(0, 2, 1, 3)
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=mask_4d, scale=SCALING)
    return out.permute(0, 2, 1, 3)


# ---------------------------------------------------------------------------
# Path 6 (fallback / SDPA reference) — always available
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_gpu, reason="Requires CUDA")
class TestPath6_SDPABaseline:
    """Smoke test for the SDPA reference used in other comparisons."""

    def test_step1_sdpa_runs(self):
        seq_len = TEXT_LEN + 1 + IMAGE_LEN + 1
        q, k, v = _make_qkv(1, seq_len)
        mask = _build_step1_mask(1, seq_len, TEXT_LEN)
        out = _sdpa_reference(q, k, v, mask)
        assert out.shape == (1, seq_len, NUM_HEADS, HEAD_DIM)

    def test_step2_sdpa_runs(self):
        q_len = 1 + Q_IMAGE_LEN
        kv_len = TEXT_LEN + 1 + Q_IMAGE_LEN
        q, k, v = _make_qkv(1, q_len, kv_len)
        mask = _build_step2_mask(1, q_len, kv_len, TEXT_LEN)
        out = _sdpa_reference(q, k, v, mask)
        assert out.shape == (1, q_len, NUM_HEADS, HEAD_DIM)


# ---------------------------------------------------------------------------
# Path 4: Step 1, non-SP, causal + full FA
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_gpu, reason="Requires CUDA")
@pytest.mark.skipif(not HAS_FA, reason="FlashAttention not available")
class TestPath4_Step1_NonSP:
    """Step 1 path: causal for text+timestamp, full for image, skip eoi."""

    @pytest.mark.parametrize("bs", [1, 2])
    def test_vs_sdpa(self, bs: int):
        seq_len = TEXT_LEN + 1 + IMAGE_LEN + 1
        q, k, v = _make_qkv(bs, seq_len)

        # SDPA reference
        mask = _build_step1_mask(bs, seq_len, TEXT_LEN)
        ref = _sdpa_reference(q, k, v, mask)

        # Path 4: causal(text+ts) + full(image) + zero(eoi)
        ts_end = TEXT_LEN + 1
        img_end = seq_len - 1  # before eoi

        out_causal = _unwrap(
            flash_attn_func(
                q[:, :ts_end],
                k[:, :ts_end],
                v[:, :ts_end],
                causal=True,
                softmax_scale=SCALING,
            )
        )
        out_image = _unwrap(
            flash_attn_func(
                q[:, ts_end:img_end],
                k[:, :img_end],
                v[:, :img_end],
                causal=False,
                softmax_scale=SCALING,
            )
        )
        out_eoi = torch.zeros_like(q[:, img_end:])
        fa_out = torch.cat([out_causal, out_image, out_eoi], dim=1)

        # Compare only causal+image region (eoi is zeros in FA, non-zero in SDPA)
        ref_trimmed = ref[:, :img_end]
        fa_trimmed = fa_out[:, :img_end]

        max_diff = (ref_trimmed - fa_trimmed).abs().max().item()
        mean_diff = (ref_trimmed - fa_trimmed).abs().mean().item()
        assert max_diff < 0.05, f"Path 4 max diff {max_diff:.6f} exceeds 0.05"
        assert mean_diff < 0.005, f"Path 4 mean diff {mean_diff:.6f} exceeds 0.005"


# ---------------------------------------------------------------------------
# Path 2: Steps 2-50, non-SP, two-call split fallback
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_gpu, reason="Requires CUDA")
@pytest.mark.skipif(not HAS_FA, reason="FlashAttention not available")
class TestPath2_Step2_NonSP_Split:
    """Steps 2-50, non-SP, split into timestamp + image calls."""

    @pytest.mark.parametrize("bs", [1, 2])
    def test_vs_sdpa(self, bs: int):
        q_len = 1 + Q_IMAGE_LEN
        kv_len = TEXT_LEN + 1 + Q_IMAGE_LEN  # text + timestamp + image (no eoi)
        q, k, v = _make_qkv(bs, q_len, kv_len)

        # SDPA reference
        mask = _build_step2_mask(bs, q_len, kv_len, TEXT_LEN)
        ref = _sdpa_reference(q, k, v, mask)

        # Path 2: split timestamp and image
        ts_kv_len = TEXT_LEN + 1

        # timestamp(1) -> text+ts only
        out_ts = _unwrap(
            flash_attn_func(
                q[:, :1],
                k[:, :ts_kv_len],
                v[:, :ts_kv_len],
                causal=False,
                softmax_scale=SCALING,
            )
        )
        # image(q_len-1) -> all KV
        out_img = _unwrap(
            flash_attn_func(
                q[:, 1:],
                k,
                v,
                causal=False,
                softmax_scale=SCALING,
            )
        )
        fa_out = torch.cat([out_ts, out_img], dim=1)

        max_diff = (ref - fa_out).abs().max().item()
        mean_diff = (ref - fa_out).abs().mean().item()
        assert max_diff < 0.05, f"Path 2 max diff {max_diff:.6f} exceeds 0.05"
        assert mean_diff < 0.005, f"Path 2 mean diff {mean_diff:.6f} exceeds 0.005"


# ---------------------------------------------------------------------------
# Path 1: Steps 2-50, non-SP, varlen packed FA
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_gpu, reason="Requires CUDA")
@pytest.mark.skipif(not HAS_VARLEN, reason="flash_attn_varlen_func not available")
class TestPath1_Step2_NonSP_Varlen:
    """Steps 2-50, non-SP, packed varlen — one kernel call."""

    @pytest.mark.parametrize("bs", [1, 2])
    def test_vs_sdpa(self, bs: int):
        q_len = 1 + Q_IMAGE_LEN
        kv_len = TEXT_LEN + 1 + Q_IMAGE_LEN
        q, k, v = _make_qkv(bs, q_len, kv_len)

        # SDPA reference
        mask = _build_step2_mask(bs, q_len, kv_len, TEXT_LEN)
        ref = _sdpa_reference(q, k, v, mask)

        # Path 1: varlen packed. Process per-batch because varlen
        # concatenates sub-sequences along dim 0.
        ts_kv_len = TEXT_LEN + 1

        outputs = []
        for b in range(bs):
            q_packed = q[b]  # (q_len, H, D)
            k_packed = torch.cat([k[b, :ts_kv_len], k[b]], dim=0)
            v_packed = torch.cat([v[b, :ts_kv_len], v[b]], dim=0)

            cu_q = torch.tensor([0, 1, q_len], dtype=torch.int32, device=DEVICE)
            cu_k = torch.tensor([0, ts_kv_len, ts_kv_len + kv_len], dtype=torch.int32, device=DEVICE)

            out = _unwrap(
                flash_attn_varlen_func(
                    q_packed.contiguous(),
                    k_packed.contiguous(),
                    v_packed.contiguous(),
                    cu_seqlens_q=cu_q,
                    cu_seqlens_k=cu_k,
                    max_seqlen_q=q_len - 1,
                    max_seqlen_k=kv_len,
                    causal=False,
                )
            )
            outputs.append(out)

        fa_out = torch.stack(outputs, dim=0)

        max_diff = (ref - fa_out).abs().max().item()
        mean_diff = (ref - fa_out).abs().mean().item()
        assert max_diff < 0.05, f"Path 1 max diff {max_diff:.6f} exceeds 0.05"
        assert mean_diff < 0.005, f"Path 1 mean diff {mean_diff:.6f} exceeds 0.005"

    def test_varlen_vs_split_path(self):
        """Path 1 (varlen) should match Path 2 (split) exactly."""
        bs = 1
        q_len = 1 + Q_IMAGE_LEN
        kv_len = TEXT_LEN + 1 + Q_IMAGE_LEN
        q, k, v = _make_qkv(bs, q_len, kv_len, seed=99)

        ts_kv_len = TEXT_LEN + 1

        # Path 2: split
        out_ts = _unwrap(
            flash_attn_func(
                q[:, :1],
                k[:, :ts_kv_len],
                v[:, :ts_kv_len],
                causal=False,
                softmax_scale=SCALING,
            )
        )
        out_img = _unwrap(
            flash_attn_func(
                q[:, 1:],
                k,
                v,
                causal=False,
                softmax_scale=SCALING,
            )
        )
        split_out = torch.cat([out_ts, out_img], dim=1)

        # Path 1: varlen
        q_packed = q[0]
        k_packed = torch.cat([k[0, :ts_kv_len], k[0]], dim=0)
        v_packed = torch.cat([v[0, :ts_kv_len], v[0]], dim=0)
        cu_q = torch.tensor([0, 1, q_len], dtype=torch.int32, device=DEVICE)
        cu_k = torch.tensor([0, ts_kv_len, ts_kv_len + kv_len], dtype=torch.int32, device=DEVICE)

        varlen_out = _unwrap(
            flash_attn_varlen_func(
                q_packed.contiguous(),
                k_packed.contiguous(),
                v_packed.contiguous(),
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=q_len - 1,
                max_seqlen_k=kv_len,
                causal=False,
            )
        ).unsqueeze(0)

        max_diff = (split_out - varlen_out).abs().max().item()
        assert max_diff < 0.01, f"Varlen vs split max diff {max_diff:.6f} exceeds 0.01"


# ---------------------------------------------------------------------------
# EOI handling: verify eoi exclusion doesn't corrupt non-eoi outputs
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_gpu, reason="Requires CUDA")
@pytest.mark.skipif(not HAS_FA, reason="FlashAttention not available")
class TestEOIHandling:
    """Verify that dropping eoi from KV cache doesn't affect text/image outputs."""

    def test_step1_eoi_output_is_unused(self):
        """Path 4 emits zeros for eoi; verify remaining tokens match SDPA."""
        bs = 1
        seq_len = TEXT_LEN + 1 + IMAGE_LEN + 1
        q, k, v = _make_qkv(bs, seq_len)

        mask = _build_step1_mask(bs, seq_len, TEXT_LEN)
        ref = _sdpa_reference(q, k, v, mask)

        ts_end = TEXT_LEN + 1
        img_end = seq_len - 1

        out_causal = _unwrap(
            flash_attn_func(
                q[:, :ts_end],
                k[:, :ts_end],
                v[:, :ts_end],
                causal=True,
                softmax_scale=SCALING,
            )
        )
        out_image = _unwrap(
            flash_attn_func(
                q[:, ts_end:img_end],
                k[:, :img_end],
                v[:, :img_end],
                causal=False,
                softmax_scale=SCALING,
            )
        )

        # Text+ts region should match
        max_diff_text = (ref[:, :ts_end] - out_causal).abs().max().item()
        assert max_diff_text < 0.05, f"Text region diff {max_diff_text:.6f}"

        # Image region should match
        max_diff_img = (ref[:, ts_end:img_end] - out_image).abs().max().item()
        assert max_diff_img < 0.05, f"Image region diff {max_diff_img:.6f}"

    def test_step2_kv_without_eoi_matches_with_eoi(self):
        """Removing eoi from KV should match SDPA with eoi masked out."""
        bs = 1
        q_len = 1 + Q_IMAGE_LEN
        kv_len_with_eoi = TEXT_LEN + 1 + Q_IMAGE_LEN + 1
        kv_len_no_eoi = kv_len_with_eoi - 1

        q, _, _ = _make_qkv(bs, q_len, seed=77)
        _, k_full, v_full = _make_qkv(bs, kv_len_with_eoi, seed=78)

        k_no_eoi = k_full[:, :-1]
        v_no_eoi = v_full[:, :-1]

        # SDPA with eoi: mask out the eoi column
        mask_with_eoi = _build_step2_mask(bs, q_len, kv_len_with_eoi, TEXT_LEN)
        mask_with_eoi[:, :, :, -1] = float("-inf")  # block eoi
        ref = _sdpa_reference(q, k_full, v_full, mask_with_eoi)

        # FA without eoi (no mask needed for image region)
        mask_no_eoi = _build_step2_mask(bs, q_len, kv_len_no_eoi, TEXT_LEN)
        out = _sdpa_reference(q, k_no_eoi, v_no_eoi, mask_no_eoi)

        max_diff = (ref - out).abs().max().item()
        assert max_diff < 0.01, f"EOI removal diff {max_diff:.6f} exceeds 0.01"


# ---------------------------------------------------------------------------
# Cross-path consistency
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_gpu, reason="Requires CUDA")
@pytest.mark.skipif(not HAS_FA, reason="FlashAttention not available")
class TestCrossPathConsistency:
    """All non-SP paths should produce equivalent results for identical inputs."""

    def test_all_step2_paths_agree(self):
        """Path 1 (varlen), Path 2 (split), Path 6 (SDPA) should all agree."""
        bs = 1
        q_len = 1 + Q_IMAGE_LEN
        kv_len = TEXT_LEN + 1 + Q_IMAGE_LEN
        q, k, v = _make_qkv(bs, q_len, kv_len, seed=123)

        ts_kv_len = TEXT_LEN + 1

        # Path 6: SDPA
        mask = _build_step2_mask(bs, q_len, kv_len, TEXT_LEN)
        sdpa_out = _sdpa_reference(q, k, v, mask)

        # Path 2: split
        out_ts = _unwrap(
            flash_attn_func(
                q[:, :1],
                k[:, :ts_kv_len],
                v[:, :ts_kv_len],
                causal=False,
                softmax_scale=SCALING,
            )
        )
        out_img = _unwrap(
            flash_attn_func(
                q[:, 1:],
                k,
                v,
                causal=False,
                softmax_scale=SCALING,
            )
        )
        split_out = torch.cat([out_ts, out_img], dim=1)

        results = {"SDPA vs Split": (sdpa_out, split_out)}

        # Path 1: varlen (if available)
        if HAS_VARLEN:
            q_packed = q[0]
            k_packed = torch.cat([k[0, :ts_kv_len], k[0]], dim=0)
            v_packed = torch.cat([v[0, :ts_kv_len], v[0]], dim=0)
            cu_q = torch.tensor([0, 1, q_len], dtype=torch.int32, device=DEVICE)
            cu_k = torch.tensor([0, ts_kv_len, ts_kv_len + kv_len], dtype=torch.int32, device=DEVICE)
            varlen_out = _unwrap(
                flash_attn_varlen_func(
                    q_packed.contiguous(),
                    k_packed.contiguous(),
                    v_packed.contiguous(),
                    cu_seqlens_q=cu_q,
                    cu_seqlens_k=cu_k,
                    max_seqlen_q=q_len - 1,
                    max_seqlen_k=kv_len,
                    causal=False,
                )
            ).unsqueeze(0)
            results["SDPA vs Varlen"] = (sdpa_out, varlen_out)
            results["Split vs Varlen"] = (split_out, varlen_out)

        for name, (a, b) in results.items():
            max_diff = (a - b).abs().max().item()
            assert max_diff < 0.05, f"{name}: max diff {max_diff:.6f} exceeds 0.05"
