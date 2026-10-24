# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("graph", [False, True])
def test_incremental_query_metadata_packs_actual_lengths(graph):
    module = pytest.importorskip(
        "vllm_omni.model_executor.models.cosyvoice3.code2wav_core.incremental", exc_type=ImportError
    )
    dit = SimpleNamespace(
        static_chunk_size=50,
        dim=4,
        input_embed=SimpleNamespace(conv_pos_embed=SimpleNamespace(kernel_size=3)),
        transformer_blocks=[SimpleNamespace(attn=SimpleNamespace(heads=1, dim_head=4))],
        rotary_embed=SimpleNamespace(forward_from_seq_len=lambda n: (torch.zeros(1, n, 4), None)),
    )
    runner = module.IncrementalFlowRunner(SimpleNamespace(dit=dit))
    noise = [torch.randn(1, length, 4) for length in (13, 101)]
    speakers = [torch.randn(1, 4) for _ in noise]
    span = torch.linspace(0, 1, 3)
    layout = runner._layout([13, 101], [0, 0], graph=graph)
    state = runner._allocate(layout, noise[0], speakers[0], span)
    for view in (state.segment_kv_slots, state.cache_seqlens, state.cu_seqlens_q):
        assert view.dtype == torch.int32
        assert view.is_contiguous()
        assert view.untyped_storage().data_ptr() == state.segment_metadata.untyped_storage().data_ptr()
    runner._load(state, noise, span, noise, speakers, noise, [None, None])
    torch.testing.assert_close(state.noise[:, :114], torch.cat(noise, dim=1))
    assert state.lengths.tolist() == [13, 101, 0]
    # The short request ends at 13, without padding it to the long request.
    assert state.cu_seqlens_q[:5].tolist() == [0, 13, 63, 113, 114]
    assert state.cache_seqlens[:4].tolist() == [13, 50, 100, 101]
    assert state.cu_seqlens_q[-1].item() == 2 * layout[1]
    assert state.noise.shape[1] == (128 if graph else 114)

    # Same allocation can carry another distribution on graph replay.
    changed = [torch.randn(1, length, 4) for length in (70, 44)]
    runner._load(state, changed, span, changed, speakers, changed, [None, None])
    torch.testing.assert_close(state.noise[:, :114], torch.cat(changed, dim=1))
    assert state.cu_seqlens_q[:5].tolist() == [0, 50, 70, 114, 128 if graph else 114]
