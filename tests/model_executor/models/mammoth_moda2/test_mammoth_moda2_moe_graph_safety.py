# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regressions for MammothModa2 AR MoE routing under graph capture.

Host-side routing baked into a graph cannot be reused for real inputs: during
capture the masks are all zeros, so any `.any()` / `.all()` branch recorded
there replays the wrong expert. These tests pin the graph-safe `torch.where`
routing to the eager result, keep both experts in the captured graph, and make
capture without a mask fail closed instead of silently routing to the
understanding expert.
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


def test_moe_forward_capturing_none_mask_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture with no mask must raise, not silently bake the understanding expert."""
    monkeypatch.setattr(m, "_is_graph_capturing", lambda device: True)
    hidden = torch.randn(2, 3, 4)

    # FULL_DECODE_ONLY / PIECEWISE still capture MoE when the stream is
    # capturing, so the only honest workaround this guard can offer is eager.
    match = r"requires gen_token_mask.*enforce_eager"
    for mask in (None, torch.zeros(0, dtype=torch.bool)):
        with pytest.raises(RuntimeError, match=match) as exc_info:
            m.moe_forward(hidden, _und, _gen, mask)
        message = str(exc_info.value)
        assert "FULL_DECODE_ONLY" not in message
        assert "enforce_eager" in message
        assert "cudagraph_mode: PIECEWISE" not in message


def test_moe_forward_capturing_all_false_mask_still_runs_both_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default BLEND specialty: dummy all-false mask still records both experts."""
    monkeypatch.setattr(m, "_is_graph_capturing", lambda device: True)
    hidden = torch.randn(4, 2)
    calls: list[str] = []

    def und(x: torch.Tensor) -> torch.Tensor:
        calls.append("und")
        return x * 2.0

    def gen(x: torch.Tensor) -> torch.Tensor:
        calls.append("gen")
        return x * 3.0

    out = m.moe_forward(hidden, und, gen, torch.zeros(4, dtype=torch.bool))

    assert calls == ["und", "gen"]
    torch.testing.assert_close(out, hidden * 2.0)


@pytest.mark.parametrize(
    ("key", "expected_calls", "scale"),
    [
        ("ALL_UND", ["und"], 2.0),
        ("ALL_GEN", ["gen"], 3.0),
        ("BLEND", ["und", "gen"], 2.0),
    ],
)
def test_moe_forward_capturing_respects_routing_specialty(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    expected_calls: list[str],
    scale: float,
) -> None:
    """#9865: capture specialty selects which experts are recorded."""
    from vllm_omni.model_executor.models.mammoth_moda2.moe_cudagraph import (
        MoERoutingKey,
        moe_routing_key_scope,
    )

    specialty = MoERoutingKey[key]

    monkeypatch.setattr(m, "_is_graph_capturing", lambda device: True)
    hidden = torch.randn(4, 2)
    calls: list[str] = []

    def und(x: torch.Tensor) -> torch.Tensor:
        calls.append("und")
        return x * 2.0

    def gen(x: torch.Tensor) -> torch.Tensor:
        calls.append("gen")
        return x * 3.0

    # BLEND with all-false mask still yields und outputs; ALL_* ignore mask.
    mask = torch.zeros(4, dtype=torch.bool)
    with moe_routing_key_scope(specialty):
        out = m.moe_forward(hidden, und, gen, mask)

    assert calls == expected_calls
    torch.testing.assert_close(out, hidden * scale)


def test_moe_routing_key_from_mask_classifies_batches() -> None:
    from vllm_omni.model_executor.models.mammoth_moda2.moe_cudagraph import (
        MoERoutingKey,
        moe_routing_key_from_mask,
    )

    assert moe_routing_key_from_mask(None) is MoERoutingKey.BLEND
    assert moe_routing_key_from_mask(torch.zeros(0, dtype=torch.bool)) is MoERoutingKey.BLEND
    assert moe_routing_key_from_mask(torch.zeros(4, dtype=torch.bool)) is MoERoutingKey.ALL_UND
    assert moe_routing_key_from_mask(torch.ones(4, dtype=torch.bool)) is MoERoutingKey.ALL_GEN
    assert moe_routing_key_from_mask(torch.tensor([False, True, False])) is MoERoutingKey.BLEND


def test_ar_wrappers_keep_raw_input_tokens() -> None:
    """The runner decides whether to drop input_ids from the top-level module."""
    assert m.MammothModa2ARForConditionalGeneration.requires_raw_input_tokens is True
    assert m.MammothModa2Qwen3ARForConditionalGeneration.requires_raw_input_tokens is True
    # Qwen3 LM must keep the Qwen2 MoE host's forward / embedding paths so
    # requires_raw_input_tokens still feeds gen_token_mask under capture.
    assert issubclass(m.MammothModa2Qwen3ForCausalLM, m.MammothModa2Qwen2ForCausalLM)
    assert m.MammothModa2Qwen3ForCausalLM.forward is m.MammothModa2Qwen2ForCausalLM.forward
    assert m.MammothModa2Qwen3ForCausalLM.get_input_embeddings is m.MammothModa2Qwen2ForCausalLM.get_input_embeddings


@pytest.mark.parametrize(
    ("is_first_rank", "is_last_rank"),
    [
        (True, True),
        (True, False),  # non-last PP rank: factory still seeds gen_token_mask
    ],
)
def test_empty_intermediate_tensors_seed_gen_token_mask(
    monkeypatch: pytest.MonkeyPatch,
    is_first_rank: bool,
    is_last_rank: bool,
) -> None:
    """PP empty tensors must come from the factory the LM constructor installs.

    Stub only what ``__init__`` needs without a distributed runtime:
    ``VocabParallelEmbedding``, ``ParallelLMHead``, ``LogitsProcessor``,
    ``RMSNorm``, ``make_layers``, and ``get_pp_group``. Leave
    ``make_empty_intermediate_tensors`` as the closure ``__init__`` assigns.
    Non-last ranks use real ``PPMissingLayer`` for head/norm.
    """
    from types import SimpleNamespace

    batch_size, hidden = 3, 8
    text = SimpleNamespace(
        hidden_size=hidden,
        num_hidden_layers=1,
        vocab_size=32,
        tie_word_embeddings=False,
        extra_gen_vocab=False,
        rms_norm_eps=1e-6,
        layer_types=None,
    )
    text.get_text_config = lambda: text
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=text, hf_text_config=text),
        cache_config=None,
        quant_config=None,
    )
    monkeypatch.setattr(m, "VocabParallelEmbedding", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(m, "ParallelLMHead", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(m, "LogitsProcessor", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(m, "RMSNorm", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(m, "make_layers", lambda *args, **kwargs: (0, 0, nn.ModuleList()))
    monkeypatch.setattr(
        m,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=is_first_rank, is_last_rank=is_last_rank),
    )

    real_helper = m.make_empty_intermediate_tensors_with_gen_mask
    spy_calls: list[tuple[int, int]] = []

    def _spy(
        batch_size: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        spy_calls.append((batch_size, hidden_size))
        return real_helper(batch_size, hidden_size, dtype, device)

    monkeypatch.setattr(m, "make_empty_intermediate_tensors_with_gen_mask", _spy)

    lm = m.MammothModa2Qwen2ForCausalLM(vllm_config=vllm_config)
    tensors = lm.make_empty_intermediate_tensors(batch_size, torch.float32, torch.device("cpu"))

    assert spy_calls == [(batch_size, hidden)]
    assert "gen_token_mask" in tensors.tensors
    assert tensors.tensors["gen_token_mask"].dtype == torch.bool
    assert tuple(tensors.tensors["gen_token_mask"].shape) == (batch_size,)
    assert tuple(tensors.tensors["hidden_states"].shape) == (batch_size, hidden)
    assert tuple(tensors.tensors["residual"].shape) == (batch_size, hidden)


def test_is_graph_capturing_warns_once_without_backend_probe(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A backend with no capture probe must surface once, not fail silently."""
    monkeypatch.delattr(torch, "npu", raising=False)
    monkeypatch.setattr(m, "_WARNED_CAPTURE_PROBES", set())

    from types import SimpleNamespace

    fake_npu = SimpleNamespace(type="npu")
    assert m._is_graph_capturing(fake_npu) is False
    assert m._is_graph_capturing(fake_npu) is False

    hits = [r for r in caplog.records if "no is_current_stream_capturing" in r.getMessage()]
    assert len(hits) == 1


def test_get_input_embeddings_graph_safe_clamps_oob_gen_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """The discarded side of the blend must not index out of range."""
    lm = _make_lm()
    monkeypatch.setattr(m, "_is_graph_capturing", lambda device: True)

    # 100 is far past both tables; the gen lookup must clamp instead of raising.
    out = lm.get_input_embeddings(torch.tensor([0, 100]))
    torch.testing.assert_close(out[0], torch.full((3,), 1.0))
    torch.testing.assert_close(out[1], torch.full((3,), 2.0))


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
