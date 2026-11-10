# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packed Talker/AudioVAE graphs preserve sequence isolation and sliding KV."""

import pytest
import torch
from transformers import Qwen2Config, Qwen2Model

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.ming_flash_omni.talker_batch import PackedQwen2

pytestmark = [pytest.mark.core_model]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs require CUDA")
@pytest.mark.parametrize("window", [None, 128])
@pytest.mark.parametrize("compile_layers", [False, True])
@torch.inference_mode()
def test_packed_graphs_match_independent_sequences_across_kv_wrap_and_row_reuse(window, compile_layers):
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        use_sliding_window=window is not None,
        sliding_window=window,
        layer_types=["sliding_attention" if window else "full_attention"] * 2,
    )
    config._attn_implementation = "sdpa"
    model = Qwen2Model(config).eval().to(device="cuda", dtype=torch.bfloat16)
    packed = PackedQwen2(
        model, rows=4, max_len=1024, window=window, ring_blocks=2 if window else None, compile_layers=compile_layers
    )
    packed.capture_graphs([32, 128, 256])
    histories: dict[int, torch.Tensor] = {}
    batches = [
        [(3, 7), (0, 13)],
        [(0, 89), (3, 50)],
        [(3, 170), (0, 80)],
        [(0, 170), (3, 60)],
        [(3, 150), (0, 80)],
        [(3, 100), (0, 20)],  # Row 3 wraps its 512-position physical KV ring.
    ]
    for batch in batches:
        rows, lengths = zip(*batch, strict=True)
        starts = [histories[row].shape[0] if row in histories else 0 for row in rows]
        embeds = [torch.randn(n, 128, device="cuda", dtype=torch.bfloat16) for n in lengths]
        actual = packed.run(embeds, list(rows), starts).split(lengths)
        for row, x, out in zip(rows, embeds, actual, strict=True):
            histories[row] = torch.cat([histories[row], x]) if row in histories else x
            expected = model(inputs_embeds=histories[row].unsqueeze(0), use_cache=False).last_hidden_state[
                0, -x.shape[0] :
            ]
            torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)

    # A completed request's row can start a shorter prompt without reading old KV.
    fresh = torch.randn(9, 128, device="cuda", dtype=torch.bfloat16)
    actual = packed.run([fresh], [3], [0])
    expected = model(inputs_embeds=fresh.unsqueeze(0), use_cache=False).last_hidden_state[0]
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
