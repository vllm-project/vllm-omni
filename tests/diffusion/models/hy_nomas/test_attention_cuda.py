# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.models.hy_nomas.attention import RangeAttention
from vllm_omni.diffusion.models.hy_nomas.hy_nomas_transformer import HyNomasTransformer
from vllm_omni.diffusion.models.hy_nomas.layers import rotary_tables

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
@pytest.mark.parametrize("regional_compile", [False, True])
@pytest.mark.parametrize("global_frame_counter", [0, 2])
@torch.no_grad()
def test_hy_nomas_transformer_warm_step_has_zero_cuda_syncs(global_frame_counter, regional_compile):
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
    cache = [
        {
            "k": torch.zeros(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16),
            "v": torch.zeros(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16),
        }
        for _ in range(2)
    ]
    shared_mask = torch.zeros(1, 32, device="cuda", dtype=torch.bool)
    shared_mask[:, :4] = True
    inputs = dict(
        hidden_states=torch.randn(1, 4, 2, 1, 1, device="cuda", dtype=torch.bfloat16),
        timestep=torch.tensor([300.0], device="cuda", dtype=torch.float32),
        text_states=torch.randn(1, 3, 16, device="cuda", dtype=torch.bfloat16),
        text_mask=torch.tensor([[True, True, False]], device="cuda"),
        kv_cache_list=cache,
        cache_valid_len=4,
        update_cache=False,
        global_frame_counter=global_frame_counter,
        cache_txt=True,
        cache_mask_list=[shared_mask] * 2,
        camera_embedding=torch.randn(1, 2, 64, device="cuda", dtype=torch.bfloat16),
        intra_seg_txt_offsets=[[1, 2]],
        intra_seg_vid_windows=[(0, 1, 0), (1, 2, 1)],
    )
    batch, _, frames, height, width = inputs["hidden_states"].shape
    tables = rotary_tables(
        (frames, height, width),
        model.config.rope_axes_dim,
        model.config.rope_theta,
        time_offset=global_frame_counter,
        device=inputs["hidden_states"].device if global_frame_counter == 0 else None,
    )
    img = model.img_in(inputs["hidden_states"])
    vec = model.time_in(inputs["timestep"])
    txt = model.txt_in(inputs["text_states"], inputs["timestep"], inputs["text_mask"])
    ref_cache = [{"k": layer["k"].clone(), "v": layer["v"].clone()} for layer in cache]
    ref_mask = shared_mask.clone()
    for index, block in enumerate(model.double_blocks):
        img, txt = block(
            img,
            txt,
            vec,
            tables,
            inputs["text_mask"],
            ref_cache[index],
            inputs["cache_valid_len"],
            False,
            True,
            ref_mask,
            index,
            model.attention,
            model.sp_size,
            inputs["intra_seg_txt_offsets"],
            inputs["intra_seg_vid_windows"],
            height * width,
        )
        img = model.action_modules[index](
            img,
            frames,
            height,
            width,
            inputs["camera_embedding"],
            model.camera_attention,
            model.sp_size,
            None,
        )
    ref_out = (
        model.final_layer(img, vec).reshape(batch, frames, height, width, model.out_channels).permute(0, 4, 1, 2, 3)
    )

    model.clear_attention_cache()
    tol = 1e-2 if regional_compile else 0
    if regional_compile:
        torch._dynamo.reset()
    try:
        if regional_compile:
            opts = {"triton.cudagraphs": False, "triton.cudagraph_trees": False}
            regionally_compile(model, dynamic=True, options=opts)
            wrapped = {n for n, m in model.named_modules() if "forward" in m.__dict__}
            assert wrapped == {"double_blocks.0", "double_blocks.1", "action_modules.0", "action_modules.1"}
        cold_out, _ = model.forward_with_cache(**inputs)
        torch.testing.assert_close(cold_out, ref_out, rtol=tol, atol=tol)
        if regional_compile:
            w_cache = [{"k": layer["k"].clone(), "v": layer["v"].clone()} for layer in cache]
            w_in = {
                **inputs,
                "kv_cache_list": w_cache,
                "cache_mask_list": [shared_mask.clone()] * 2,
                "update_cache": True,
            }
            w_out, _ = model.forward_with_cache(**w_in)
            torch.testing.assert_close(w_out, ref_out, rtol=tol, atol=tol)
        torch.accelerator.synchronize()
        prev_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            warm_out, _ = model.forward_with_cache(**inputs)
            commit_out, written = model.forward_with_cache(**{**inputs, "update_cache": True})
            post_commit_out, _ = model.forward_with_cache(**inputs)
        finally:
            torch.cuda.set_sync_debug_mode(prev_mode)
    finally:
        if regional_compile:
            torch._dynamo.reset()
    assert written == 5 and len(model.attention.masks) == 1 and len(model.camera_attention.masks) == 1
    for out in (warm_out, commit_out, post_commit_out):
        torch.testing.assert_close(out, ref_out, rtol=tol, atol=tol)
