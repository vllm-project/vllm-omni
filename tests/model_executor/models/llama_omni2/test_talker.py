# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import re
from types import SimpleNamespace

import pytest
import torch
from vllm.distributed import parallel_state
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)

from vllm_omni.model_executor.models.llama_omni2.llama_omni2_talker import (
    TALKER_WEIGHTS_MAPPER,
    LlamaOmni2TalkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def init_fake_tp_group(mocker):
    """Provide a fake TP group so vLLM linear layers can be instantiated."""
    mock_tp = mocker.MagicMock()
    mock_tp.world_size = 1
    mock_tp.rank_in_group = 0
    old_tp = parallel_state._TP
    parallel_state._TP = mock_tp
    try:
        yield
    finally:
        parallel_state._TP = old_tp


def test_talker_projection_and_gate_match_checkpoint_shapes(init_fake_tp_group):
    model = object.__new__(LlamaOmni2TalkerForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model._init_fusion_layers(thinker_hidden_size=896, talker_hidden_size=896)

    assert model.input_proj[0].input_size == 896
    assert model.input_proj[0].output_size == 1792
    assert model.input_proj[2].input_size == 1792
    assert model.input_proj[2].output_size == 896
    assert model.gate[0].input_size == 1792
    assert model.gate[0].output_size == 896
    assert isinstance(model.input_proj[0], ColumnParallelLinear)
    assert isinstance(model.input_proj[2], RowParallelLinear)
    assert isinstance(model.gate[0], ColumnParallelLinear)


def test_talker_fusion_uses_sigmoid_gate():
    model = object.__new__(LlamaOmni2TalkerForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.gate = torch.nn.Sequential(
        torch.nn.Linear(4, 2, bias=False),
        torch.nn.Sigmoid(),
    )
    torch.nn.init.zeros_(model.gate[0].weight)
    representation = torch.tensor([[2.0, 4.0]])
    token_embedding = torch.tensor([[6.0, 8.0]])

    fused = model.fusion(representation, token_embedding)

    assert torch.equal(fused, torch.tensor([[4.0, 6.0]]))


def test_talker_rejects_misaligned_hidden_states_and_token_ids():
    model = object.__new__(LlamaOmni2TalkerForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.input_proj = torch.nn.Sequential(
        torch.nn.Identity(),
        torch.nn.Identity(),
        torch.nn.Identity(),
    )
    model.language_model = SimpleNamespace(embed_input_ids=lambda ids: torch.zeros(ids.shape[0], 3))

    with pytest.raises(ValueError, match="same number of rows"):
        model.prepare_talker_embeddings(
            torch.zeros(2, 3),
            torch.tensor([1, 2, 3]),
        )


@pytest.mark.parametrize(
    ("token_ids", "invalid_ids"),
    [
        ([-1, 2], "[-1]"),
        ([2, 17], "[17]"),
    ],
)
def test_talker_rejects_token_ids_outside_embedding_vocab(
    token_ids,
    invalid_ids,
):
    model = object.__new__(LlamaOmni2TalkerForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.input_proj = torch.nn.Sequential(
        torch.nn.Identity(),
        torch.nn.Identity(),
        torch.nn.Identity(),
    )
    model.language_model = SimpleNamespace(
        config=SimpleNamespace(vocab_size=17),
        embed_input_ids=lambda ids: pytest.fail("invalid token IDs must be rejected before embedding lookup"),
    )

    with pytest.raises(
        ValueError,
        match=rf"Talker embedding vocabulary range \[0, 17\).*{re.escape(invalid_ids)}",
    ):
        model.prepare_talker_embeddings(
            torch.zeros(2, 3),
            torch.tensor(token_ids),
        )


def test_talker_composes_registered_vllm_qwen2_and_delegates_logits(
    monkeypatch,
    init_fake_tp_group,
):
    import vllm_omni.model_executor.models.llama_omni2.llama_omni2_talker as talker_module

    talker_config = SimpleNamespace(hidden_size=8, vocab_size=17)
    root_config = SimpleNamespace(
        hidden_size=8,
        thinker_config=SimpleNamespace(hidden_size=8),
        talker_config=talker_config,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=root_config,
            multimodal_config=None,
        ),
        quant_config=None,
    )
    calls = {}

    class FakeLanguageModel(torch.nn.Module):
        make_empty_intermediate_tensors = object()

        def forward(
            self,
            input_ids,
            positions,
            intermediate_tensors=None,
            inputs_embeds=None,
        ):
            calls["forward"] = (
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
            )
            return torch.ones(2, 8)

        def compute_logits(self, hidden_states):
            calls["logits"] = hidden_states
            return torch.ones(hidden_states.shape[0], 17)

        def embed_input_ids(self, input_ids):
            return torch.zeros(input_ids.shape[0], 8)

    language_model = FakeLanguageModel()

    def fake_init_vllm_registered_model(**kwargs):
        calls["init"] = kwargs
        return language_model

    monkeypatch.setattr(
        talker_module,
        "init_vllm_registered_model",
        fake_init_vllm_registered_model,
    )
    model = LlamaOmni2TalkerForConditionalGeneration(
        vllm_config=vllm_config,
        prefix="talker",
    )

    assert calls["init"]["hf_config"] is talker_config
    assert calls["init"]["architectures"] == ["Qwen2ForCausalLM"]
    assert calls["init"]["prefix"] == "talker.language_model"

    hidden_states = model(
        input_ids=torch.tensor([1, 2]),
        positions=torch.tensor([0, 1]),
    )
    logits = model.compute_logits(hidden_states)
    embeddings = model.embed_input_ids(torch.tensor([3, 4]))

    assert hidden_states.shape == (2, 8)
    assert logits.shape == (2, 17)
    assert embeddings.shape == (2, 8)


def test_talker_sampler_masks_non_codec_tokens():
    model = object.__new__(LlamaOmni2TalkerForConditionalGeneration)
    torch.nn.Module.__init__(model)
    captured = {}

    class CapturingSampler:
        def __call__(self, *, logits, sampling_metadata):
            captured["logits"] = logits.detach().clone()
            captured["sampling_metadata"] = sampling_metadata
            return SimpleNamespace(
                sampled_token_ids=torch.tensor([[151666]], dtype=torch.int32),
            )

    model._sampler = CapturingSampler()
    logits = torch.zeros((1, 158228), dtype=torch.float32)
    sampling_metadata = object()

    model.sample(logits, sampling_metadata)

    masked = captured["logits"][0]
    assert captured["sampling_metadata"] is sampling_metadata
    assert torch.isneginf(masked[6240])
    assert torch.isfinite(masked[151643])
    assert torch.isneginf(masked[151644])
    assert torch.isneginf(masked[151665])
    assert torch.isfinite(masked[151666])
    assert torch.isfinite(masked[158226])
    assert torch.isneginf(masked[158227])


def test_talker_weight_mapper_routes_real_checkpoint_prefixes():
    checkpoint_names = [
        "speech_generator.input_proj.0.weight",
        "speech_generator.gate.0.bias",
        "speech_generator.model.model.embed_tokens.weight",
        "speech_generator.model.model.layers.0.self_attn.q_proj.weight",
        "speech_generator.model.lm_head.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ]

    assert TALKER_WEIGHTS_MAPPER.apply_list(checkpoint_names) == [
        "input_proj.0.weight",
        "gate.0.bias",
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.self_attn.qkv_proj.weight",
        "language_model.lm_head.weight",
    ]
