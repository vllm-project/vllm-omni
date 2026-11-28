# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.models.hy_nomas.attention import RangeAttention, action_ranges
from vllm_omni.diffusion.models.hy_nomas.hy_nomas_transformer import HyNomasTransformer

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_heads", [2, 4])
@torch.no_grad()
def test_compiled_ranges_match_fp32_sdpa(dtype, kv_heads):
    generator = torch.Generator(device="cuda").manual_seed(41)
    q = torch.randn(1, 19, 4, 128, device="cuda", dtype=dtype, generator=generator)
    k, v = [torch.randn(1, 23, kv_heads, 128, device="cuda", dtype=dtype, generator=generator) for _ in range(2)]
    qm, km = torch.ones(1, 19, device="cuda", dtype=torch.bool), torch.ones(1, 23, device="cuda", dtype=torch.bool)
    qm[:, 2], km[:, 5] = False, False
    ranges = [[(0, 10, 0, 13), (10, 19, 8, 23)]]
    actual = RangeAttention(128**-0.5)(q, k, v, ranges=ranges, query_mask=qm, key_mask=km)
    pair_mask = torch.zeros(1, 1, 19, 23, device="cuda", dtype=torch.bool)
    pair_mask[:, :, :10, :13], pair_mask[:, :, 10:, 8:] = True, True
    pair_mask &= qm[:, None, :, None] & km[:, None, None, :]
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(
            q.float().transpose(1, 2),
            k.float().repeat_interleave(4 // kv_heads, 2).transpose(1, 2),
            v.float().repeat_interleave(4 // kv_heads, 2).transpose(1, 2),
            attn_mask=pair_mask,
        ).transpose(1, 2)
    torch.testing.assert_close(actual.float(), expected, rtol=1e-2, atol=1e-2)
    assert torch.count_nonzero(actual[:, 2]) == 0
    assert torch.isfinite(actual).all()


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@torch.no_grad()
def test_fa3_joint_attention_preserves_visibility_and_falls_back_for_ranges():
    from vllm.vllm_flash_attn.flash_attn_interface import fa_version_unsupported_reason

    if (reason := fa_version_unsupported_reason(3)) is not None:
        pytest.skip(reason)

    generator = torch.Generator(device="cuda").manual_seed(43)
    q = torch.randn(3, 19, 4, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    k, v = [torch.randn(3, 23, 2, 128, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(2)]
    qm = torch.ones(3, 19, dtype=torch.bool, device="cuda")
    km = torch.ones(3, 23, dtype=torch.bool, device="cuda")
    qm[0, 16:], qm[1, 2], km[1, 5], km[2, :] = False, False, False, False
    masks = {"query_mask": qm, "key_mask": km}
    baseline = RangeAttention(128**-0.5)
    candidate = RangeAttention(128**-0.5, backend="fa3")

    plan = candidate.resolve_masks(19, 23, q.device, batch_size=3, **masks)
    reused = candidate.resolve_masks(19, 23, q.device, batch_size=3, **masks)
    assert all(first is second for first, second in zip(plan, reused, strict=True))
    actual = candidate(q, k, v, block_masks=plan, phase="denoise")
    expected = baseline(q, k, v, **masks)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    assert torch.count_nonzero(actual[~qm]) == 0
    assert torch.count_nonzero(actual[2]) == 0
    assert candidate.backend_hits["denoise"] == {"fa3": 2, "empty": 1}

    qm[0, 0] = False
    updated = candidate.resolve_masks(19, 23, q.device, batch_size=3, **masks)
    assert updated[0] is not plan[0] and updated[1] is plan[1]
    assert torch.count_nonzero(candidate(q, k, v, block_masks=updated)[0, 0]) == 0

    ranges = [[(0, 10, 0, 13), (10, 19, 8, 23)]] * 3
    actual = candidate(q, k, v, ranges=ranges, phase="kv_commit", **masks)
    expected = baseline(q, k, v, ranges=ranges, **masks)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert candidate.backend_hits["kv_commit"] == {"flex": 3}


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@torch.no_grad()
def test_fa3_lse_preserves_action_visibility():
    from vllm.vllm_flash_attn.flash_attn_interface import fa_version_unsupported_reason

    if (reason := fa_version_unsupported_reason(3)) is not None:
        pytest.skip(reason)

    generator = torch.Generator(device="cuda").manual_seed(44)
    q = torch.randn(1, 21, 4, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    k, v = [torch.randn(1, 28, 2, 128, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(2)]
    history = torch.tensor([[True, False, True, True, False, True, True]], device="cuda")
    text = torch.tensor([[True, False, True, True, True, True, True, True, True]], device="cuda")
    query_mask = torch.cat((torch.ones(1, 12, dtype=torch.bool, device="cuda"), text), dim=1)
    key_mask = torch.cat((history, query_mask), dim=1)
    masks = {"query_mask": query_mask, "key_mask": key_mask}
    ranges = action_ranges(history, text, 12, [[2, 5, 8]], [(0, 1, 0), (1, 2, -1), (2, 3, 1)], 4)
    baseline = RangeAttention(128**-0.5)
    candidate = RangeAttention(128**-0.5, backend="fa3_lse")

    actual = candidate(q, k, v, ranges=ranges, phase="denoise", **masks)
    expected = baseline(q, k, v, ranges=ranges, **masks)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    assert actual[~query_mask].count_nonzero() == 0
    assert candidate.backend_hits["denoise"] == {"fa3_lse": 1}

    overlapping_queries = [[(0, 12, 0, 7), (4, 8, 7, 11)]]
    actual = candidate(q, k, v, ranges=overlapping_queries, phase="kv_commit", **masks)
    expected = baseline(q, k, v, ranges=overlapping_queries, **masks)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert candidate.backend_hits["kv_commit"] == {"flex": 1}


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.no_grad()
def test_prebuilt_mask_preserves_static_kernel_numerics():
    generator = torch.Generator(device="cuda").manual_seed(42)
    q = torch.randn(1, 9600, 32, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    k, v = [torch.randn(1, 9600, 8, 128, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(2)]

    def mask_mod(batch, head, query, key):
        return (query // 640 - key // 640).abs() <= 1

    mask = create_block_mask(mask_mod, None, None, 9600, 9600, device="cuda", _compile=True)
    actual = RangeAttention(128**-0.5)(q, k, v, block_mask=mask)
    expected = torch.compile(flex_attention, dynamic=False)(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=mask, enable_gqa=True
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("global_frame_counter", [0, 2])
@torch.no_grad()
def test_hy_nomas_transformer_warm_step_has_zero_cuda_syncs(global_frame_counter):
    torch._dynamo.reset()
    config = {
        "in_channels": 4,
        "out_channels": 4,
        "num_layers": 2,
        "num_attention_heads": 4,
        "attention_head_dim": 64,
        "text_embed_dim": 16,
        "rope_axes_dim": [16, 24, 24],
        "camera_embed_dim": 16,
        "camera_attention_dim": 256,
        "camera_num_attention_heads": 4,
        "camera_rope_axes_dim": [16, 24, 24],
        "mlp_ratio": 2.0,
        "rope_theta": 256.0,
    }
    model = HyNomasTransformer(config, sp_size=1, device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    base_inputs = dict(
        hidden_states=torch.randn(1, 4, 2, 1, 1, device="cuda", dtype=torch.bfloat16),
        timestep=torch.tensor([300.0], device="cuda", dtype=torch.float32),
        text_states=torch.randn(1, 3, 16, device="cuda", dtype=torch.bfloat16),
        text_mask=torch.tensor([[True, True, False]], device="cuda"),
        cache_valid_len=4,
        update_cache=False,
        global_frame_counter=global_frame_counter,
        cache_txt=True,
        camera_embedding=torch.randn(1, 2, 64, device="cuda", dtype=torch.bfloat16),
        intra_seg_txt_offsets=[[1, 2]],
        intra_seg_vid_windows=[(0, 1, 0), (1, 2, 1)],
    )

    def _fresh_state():
        kv = [
            {
                "k": torch.zeros(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16),
                "v": torch.zeros(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16),
            }
            for _ in range(2)
        ]
        mask = torch.zeros(1, 32, device="cuda", dtype=torch.bool)
        mask[:, :4] = True
        return {**base_inputs, "kv_cache_list": kv, "cache_mask_list": [mask] * 2}

    def _run_zero_sync_sequence(inputs):
        torch.accelerator.synchronize()
        prev_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            warm_out, _ = model.forward_with_cache(**inputs)
            commit_out, written = model.forward_with_cache(**{**inputs, "update_cache": True})
            post_out, _ = model.forward_with_cache(**inputs)
        finally:
            torch.cuda.set_sync_debug_mode(prev_mode)
        return warm_out, commit_out, post_out, written

    eager_in = _fresh_state()
    ref_cold, _ = model.forward_with_cache(**eager_in)
    ref_warm, ref_commit, ref_post, written_e = _run_zero_sync_sequence(eager_in)
    for out in (ref_warm, ref_commit, ref_post):
        torch.testing.assert_close(out, ref_cold, rtol=0, atol=0)

    model.clear_attention_cache()
    torch._dynamo.reset()
    try:
        regionally_compile(model, dynamic=True, options={"triton.cudagraphs": False, "triton.cudagraph_trees": False})
        wrapped = {n for n, m in model.named_modules() if "forward" in m.__dict__}
        assert wrapped == {"double_blocks.0", "double_blocks.1", "action_modules.0", "action_modules.1"}
        comp_in = _fresh_state()
        comp_cold, _ = model.forward_with_cache(**comp_in)
        comp_warmup_commit, _ = model.forward_with_cache(**{**comp_in, "update_cache": True})
        comp_warm, comp_commit, comp_post, written_c = _run_zero_sync_sequence(comp_in)
    finally:
        torch._dynamo.reset()

    assert written_e == written_c == 5 and len(model.attention.masks) == 1 and len(model.camera_attention.masks) == 1
    for out in (comp_cold, comp_warmup_commit, comp_warm, comp_commit, comp_post):
        torch.testing.assert_close(out, ref_cold, rtol=1e-2, atol=1e-2)

    from vllm_omni.diffusion.models.hy_nomas.pipeline_hy_nomas import _as_versioned_tensor

    model.clear_attention_cache()
    with torch.inference_mode():
        inf_in = {**_fresh_state(), "text_mask": base_inputs["text_mask"].clone()}
        assert inf_in["text_mask"].is_inference() and inf_in["cache_mask_list"][0].is_inference()
        inf_cold, _ = model.forward_with_cache(**inf_in)
        torch.testing.assert_close(model.forward_with_cache(**inf_in)[0], inf_cold, rtol=0, atol=0)
        assert len(model.attention.masks) == 1
        inf_in["cache_mask_list"][0][0, 0] = False
        assert not torch.equal(model.forward_with_cache(**inf_in)[0], inf_cold)

        model.clear_attention_cache()
        ver_mask = _as_versioned_tensor(inf_in["cache_mask_list"][0])
        ver_in = {
            **_fresh_state(),
            "text_mask": _as_versioned_tensor(inf_in["text_mask"]),
            "cache_mask_list": [ver_mask] * 2,
        }
        ver_cold, _ = model.forward_with_cache(**ver_in)
        ver_warm, ver_commit, ver_post, ver_written = _run_zero_sync_sequence(ver_in)
        assert ver_written == 5
        for out in (ver_warm, ver_commit, ver_post):
            torch.testing.assert_close(out, ver_cold, rtol=0, atol=0)
