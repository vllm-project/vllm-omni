# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.minimax_h3.vae_compile import install_compiled_rope_output_guard

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _checkpoint_decoder(compiled):
    # The checkpoint imports one helper backed by a module-level compile cache.
    namespace = {"nn": nn, "_COMPILED_APPLY_ROTARY_POS_EMB": compiled, "disabled": False}
    exec(
        "def _get_apply_rotary_pos_emb_impl():\n"
        "    if disabled or _COMPILED_APPLY_ROTARY_POS_EMB is None:\n"
        "        return lambda value, rope: value\n"
        "    return _COMPILED_APPLY_ROTARY_POS_EMB\n"
        "def apply_rotary_pos_emb(value, rope):\n"
        "    return _get_apply_rotary_pos_emb_impl()(value, rope)\n"
        "class Attention(nn.Module):\n"
        "    def forward(self, query, key, rope=None):\n"
        "        query = apply_rotary_pos_emb(query, rope)\n"
        "        key = apply_rotary_pos_emb(key, rope)\n"
        "        return query, key\n",
        namespace,
    )
    decoder, block = nn.Module(), nn.Module()
    block.attn = namespace["Attention"]()
    decoder.transformer_blocks = nn.ModuleList([block])
    return decoder, namespace


@pytest.mark.cpu
def test_compiled_outputs_survive_the_next_call():
    buffer = torch.empty(4)

    def compiled(value, rope):
        return buffer.copy_(value)

    decoder, _ = _checkpoint_decoder(compiled)
    query, key = torch.arange(4.0), torch.arange(4.0) + 10
    unguarded_q, _ = decoder.transformer_blocks[0].attn(query, key)
    assert torch.equal(unguarded_q, key)

    install_compiled_rope_output_guard(decoder)
    actual_q, actual_k = decoder.transformer_blocks[0].attn(query, key)
    compiled(key + 20, None)
    assert torch.equal(actual_q, query)
    assert torch.equal(actual_k, key)
    assert actual_q.data_ptr() != actual_k.data_ptr()


@pytest.mark.cpu
@pytest.mark.parametrize("mode", ["eager", "disabled", "fallback"])
def test_eager_and_fallback_outputs_preserve_identity(mode):
    decoder, namespace = _checkpoint_decoder(lambda value, rope: value)
    if mode == "eager":
        namespace["_COMPILED_APPLY_ROTARY_POS_EMB"] = None
    elif mode == "disabled":
        namespace["disabled"] = True
    else:
        exec(
            "def apply_rotary_pos_emb(value, rope):\n"
            "    global _COMPILED_APPLY_ROTARY_POS_EMB\n"
            "    _COMPILED_APPLY_ROTARY_POS_EMB = None\n"
            "    return value\n",
            namespace,
        )
    install_compiled_rope_output_guard(decoder)
    query, key = torch.randn(4), torch.randn(4)
    actual_q, actual_k = decoder.transformer_blocks[0].attn(query, key)
    assert actual_q is query
    assert actual_k is key


@pytest.mark.cpu
def test_guard_is_idempotent_and_unknown_contract_is_untouched():
    decoder, namespace = _checkpoint_decoder(None)
    install_compiled_rope_output_guard(decoder)
    guarded = namespace["apply_rotary_pos_emb"]
    install_compiled_rope_output_guard(decoder)
    assert namespace["apply_rotary_pos_emb"] is guarded

    unknown, namespace = _checkpoint_decoder(None)
    original = namespace["apply_rotary_pos_emb"]
    del namespace["_COMPILED_APPLY_ROTARY_POS_EMB"]
    install_compiled_rope_output_guard(unknown)
    assert namespace["apply_rotary_pos_emb"] is original
    install_compiled_rope_output_guard(nn.Module())


def _rope(value, rope):
    cos, sin = rope
    rotated, passed = value[..., : cos.shape[-1]], value[..., cos.shape[-1] :]
    first, second = rotated.chunk(2, dim=-1)
    rotated = rotated * cos + torch.cat((-second, first), dim=-1) * sin
    return torch.cat((rotated, passed), dim=-1)


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph regression requires CUDA")
def test_reduce_overhead_rope_outputs_survive_qk_and_later_calls():
    compiled = torch.compile(_rope, mode="reduce-overhead", fullgraph=True)
    decoder, _ = _checkpoint_decoder(compiled)
    install_compiled_rope_output_guard(decoder)
    attention = decoder.transformer_blocks[0].attn
    torch.manual_seed(3100)
    query = torch.randn(1, 195, 32, 64, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    cos = torch.randn(1, 195, 1, 48, device="cuda", dtype=torch.float16)
    rope = cos, torch.randn_like(cos)
    retained = []
    with torch.inference_mode():
        for scale in (1.0, 0.99, 1.01, 0.98):
            current_q, current_k = query * scale, key * scale
            actual_q, actual_k = attention(current_q, current_k, rope)
            retained.append((actual_q, actual_k, _rope(current_q, rope), _rope(current_k, rope)))
        for actual_q, actual_k, expected_q, expected_k in retained:
            torch.testing.assert_close(actual_q, expected_q, atol=4e-3, rtol=2e-3)
            torch.testing.assert_close(actual_k, expected_k, atol=4e-3, rtol=2e-3)
            assert actual_q.data_ptr() != actual_k.data_ptr()
