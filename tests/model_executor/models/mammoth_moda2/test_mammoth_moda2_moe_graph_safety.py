# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regressions for MammothModa2 AR MoE graph-safe routing.

Covers the cudagraph_mode FULL fix from #7318: the graph-safe `torch.where`
routing must be numerically identical to the eager dispatch path, and the
graph-safe embedding blend must produce the same result as eager lookup.
"""

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.mammoth_moda2 import mammoth_moda2 as m

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _und(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0


def _gen(x: torch.Tensor) -> torch.Tensor:
    return x * 3.0


@pytest.mark.parametrize("shape", [(2, 3, 4), (6, 4)])
def test_moe_blend_matches_eager_mixed_routing(shape: tuple[int, ...]) -> None:
    """`_moe_blend` must be numerically identical to the eager mixed dispatch path."""
    hidden = torch.randn(*shape)
    mask_2d = torch.tensor([[False, True, False], [True, True, False]])
    if len(shape) == 2:
        mask = mask_2d.reshape(-1)
    else:
        mask = mask_2d

    eager = m.moe_forward(hidden, _und, _gen, mask)
    blended = m._moe_blend(hidden, _und, _gen, mask)

    torch.testing.assert_close(blended, eager)
    assert blended.shape == eager.shape


def test_moe_blend_routes_each_token_to_the_right_expert() -> None:
    """Sanity: blend selects gen vs und per token, not per whole tensor."""
    hidden = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    mask = torch.tensor([False, True, False, True])

    out = m._moe_blend(hidden, _und, _gen, mask)

    torch.testing.assert_close(out[0], _und(hidden)[0])
    torch.testing.assert_close(out[1], _gen(hidden)[1])
    torch.testing.assert_close(out[2], _und(hidden)[2])
    torch.testing.assert_close(out[3], _gen(hidden)[3])


def test_moe_forward_dispatch_to_blend_when_capturing(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the stream is capturing, `moe_forward` must take the graph-safe path."""
    hidden = torch.randn(2, 3, 4)
    mask = torch.tensor([[False, True, False], [True, True, False]])

    monkeypatch.setattr(m, "_is_graph_capturing", lambda device: True)

    blended = m.moe_forward(hidden, _und, _gen, mask)
    reference = m._moe_blend(hidden, _und, _gen, mask)

    torch.testing.assert_close(blended, reference)


def test_moe_forward_eager_path_unchanged_when_not_capturing() -> None:
    """Non-capturing path keeps the original all-und / all-gen fast branches."""
    hidden = torch.randn(2, 3, 4)
    all_false = torch.zeros(2, 3, dtype=torch.bool)
    all_true = torch.ones(2, 3, dtype=torch.bool)

    torch.testing.assert_close(m.moe_forward(hidden, _und, _gen, all_false), _und(hidden))
    torch.testing.assert_close(m.moe_forward(hidden, _und, _gen, all_true), _gen(hidden))


def test_is_graph_capturing_cpu_is_false() -> None:
    assert m._is_graph_capturing(torch.device("cpu")) is False


def test_is_graph_capturing_missing_backend_api_is_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Devices without is_current_stream_capturing (e.g. no torch.npu) return False."""
    monkeypatch.delattr(torch, "npu", raising=False)

    from types import SimpleNamespace

    fake_npu = SimpleNamespace(type="npu")
    assert m._is_graph_capturing(fake_npu) is False


def test_is_graph_capturing_uses_cuda_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA path delegates to torch.cuda.is_current_stream_capturing."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert m._is_graph_capturing(torch.device("cuda")) is True
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert m._is_graph_capturing(torch.device("cuda")) is False


def _make_lm() -> m.MammothModa2Qwen2ForCausalLM:
    """Minimal language model with predictable base/gen embedding tables."""
    lm = m.MammothModa2Qwen2ForCausalLM.__new__(m.MammothModa2Qwen2ForCausalLM)
    nn.Module.__init__(lm)
    lm.extra_gen_vocab = True
    lm.gen_embed_tokens = nn.Embedding(4, 3)
    lm.gen_embed_tokens.weight.data = torch.full((4, 3), 2.0)
    lm.embed_tokens = nn.Embedding(8, 3)
    lm.embed_tokens.weight.data = torch.full((8, 3), 1.0)
    lm.gen_vocab_start_index = 8
    lm.base_vocab_size = 8
    lm.config = type("Cfg", (), {"hidden_size": 3})()
    return lm


def test_get_input_embeddings_graph_safe_blend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Graph-safe embedding path blends base/gen tables per token."""
    lm = _make_lm()
    monkeypatch.setattr(m, "_is_graph_capturing", lambda device: True)

    input_ids = torch.tensor([[0, 8, 5], [9, 10, 7]])  # base=8: ids >= 8 are gen
    out = lm.get_input_embeddings(input_ids)

    # base positions -> 1.0, gen positions -> 2.0
    expected = torch.where((input_ids >= 8).unsqueeze(-1), torch.full((2, 3, 3), 2.0), torch.full((2, 3, 3), 1.0))
    torch.testing.assert_close(out, expected)


def test_get_input_embeddings_eager_path_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-capturing eager path still uses the original dispatch branches."""
    lm = _make_lm()
    monkeypatch.setattr(m, "_is_graph_capturing", lambda device: False)

    all_base = torch.tensor([[1, 2], [3, 4]])
    torch.testing.assert_close(lm.get_input_embeddings(all_base), lm.embed_tokens(all_base))

    all_gen = torch.tensor([[8, 9], [10, 11]])
    torch.testing.assert_close(lm.get_input_embeddings(all_gen), lm.gen_embed_tokens(all_gen - 8))
