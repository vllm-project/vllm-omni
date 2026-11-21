# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.confucius4_tts import (
    confucius4_tts_talker as talker_module,
)
from vllm_omni.model_executor.models.confucius4_tts.confucius4_tts_talker import (
    Confucius4TTSTalkerForConditionalGeneration,
)
from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _PositionEmbeddingStub(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.shape[1], device=x.device)
        return x + self.embedding(positions).unsqueeze(0)


def _talker() -> Confucius4TTSTalkerForConditionalGeneration:
    talker = object.__new__(Confucius4TTSTalkerForConditionalGeneration)
    nn.Module.__init__(talker)
    return talker


@dataclass
class _SemanticConfig:
    bos_token_id: int = 8192


def test_talker_keeps_full_cuda_graphs_under_mrv2() -> None:
    # forward() returns a tensor, so the MRv2 runner must not treat the talker
    # as a structured-output model and drop FULL graph capture.
    from vllm_omni.worker_v2.omni_model_runner import _needs_capture_tensor_unwrap

    talker = _talker()
    talker.have_multimodal_outputs = True
    assert _needs_capture_tensor_unwrap(talker) is False


def test_gpt2_layer_factory_accepts_keyword_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_block(config, cache_config, quant_config, *, prefix):
        seen.update(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        return nn.Identity()

    monkeypatch.setattr(talker_module, "GPT2Block", fake_block)
    factory = talker_module._gpt2_block_factory(
        "config",
        "cache",
        "quant",
    )

    assert isinstance(factory(prefix="transformer.h.0"), nn.Identity)
    assert seen == {
        "config": "config",
        "cache_config": "cache",
        "quant_config": "quant",
        "prefix": "transformer.h.0",
    }


def test_compute_logits_normalizes_hidden_and_leaves_bos_unmasked() -> None:
    talker = _talker()
    talker.config = _SemanticConfig()
    talker.final_norm = nn.LayerNorm(2)
    talker.semantic_head = nn.Linear(2, 8194, bias=False)
    captured: dict[str, torch.Tensor] = {}

    def logits_processor(
        _head: nn.Module,
        hidden_states: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        captured["hidden_states"] = hidden_states
        return torch.zeros(hidden_states.shape[0], 8194)

    talker.logits_processor = logits_processor  # type: ignore[assignment]
    predictor_hidden = torch.tensor([[1.0, 3.0]])

    logits = talker.compute_logits(predictor_hidden)

    assert logits is not None
    assert torch.allclose(
        captured["hidden_states"],
        talker.final_norm(predictor_hidden),
    )
    # Upstream samples BOS and strips it afterwards; the stage processor does
    # the same, so the sampling distribution must stay untouched here.
    assert torch.equal(logits, torch.zeros(1, 8194))


def test_rows_without_request_state_do_not_emit_payload() -> None:
    talker = _talker()
    output = talker.make_omni_output(
        torch.zeros(3, 4),
        model_intermediate_buffer=[{}],
    )
    assert output.multimodal_outputs == {}


def test_conditioning_is_emitted_once_on_the_prefill_step() -> None:
    # The prefill step samples the first semantic token, so a request may
    # finish there (max_tokens=1). The conditioning must already be published
    # by then, or Stage 1 never receives its input.
    talker = _talker()
    conditioning = {
        "text_token_ids": torch.tensor([10, 11]),
        "condition_vector": torch.ones(1, 2, 3),
        "ref_mel": torch.ones(1, 3, 80),
        "style": torch.ones(1, 192),
    }
    request_meta = {"semantic_start_offset": 3, **conditioning}

    prefill = talker.make_omni_output(
        torch.zeros(4, 4),
        model_intermediate_buffer=[{"meta": {**request_meta, "semantic_code_count": 0}}],
    )
    decode = talker.make_omni_output(
        torch.zeros(1, 4),
        model_intermediate_buffer=[{"meta": {**request_meta, "semantic_code_count": 1}}],
    )

    prefill_meta = prefill.multimodal_outputs["meta"][0]
    assert set(conditioning) <= set(prefill_meta)
    assert prefill_meta["semantic_code_count"] == 0
    assert decode.multimodal_outputs["meta"][0] == {"semantic_code_count": 1}
    assert "codes" not in prefill.multimodal_outputs
    assert "hidden_states" not in prefill.multimodal_outputs


def test_decode_preprocess_increments_semantic_code_count() -> None:
    talker = _talker()
    talker.semantic_embedding = nn.Embedding(32, 4)
    talker.semantic_position_embedding = _PositionEmbeddingStub(16, 4)

    token = torch.tensor([7])
    output_ids, embeds, updates = talker.preprocess(
        token,
        None,
        meta={
            "semantic_start_offset": 3,
            "semantic_code_count": 4,
        },
    )

    assert torch.equal(output_ids, token)
    assert embeds.shape == (1, 4)
    assert set(updates) == {"meta"}
    assert updates["meta"] == {"semantic_code_count": 5}


def test_prefill_caches_inline_reference_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    talker = _talker()
    talker.model_path = "model"
    talker.config = SimpleNamespace(
        max_text_tokens=520,
        max_reference_audio_seconds=30.0,
        w2v_stat="stats.pt",
        bos_token_id=8192,
        hidden_size=4,
    )
    talker._w2v_stats = torch.ones(2, 3)
    talker._speaker_cache = SpeakerEmbeddingCache(max_bytes=1024 * 1024)
    talker.semantic_embedding = nn.Embedding(8194, 4)
    talker.text_position_embedding = _PositionEmbeddingStub(520, 4)
    talker.semantic_position_embedding = _PositionEmbeddingStub(1520, 4)

    class _Projector(nn.Module):
        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            return torch.zeros(*token_ids.shape, 4)

    class _SpeakerEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1))

        def forward(self, condition: torch.Tensor) -> torch.Tensor:
            return torch.ones(condition.shape[0], 4)

    talker.text_projector = _Projector()
    talker.speaker_encoder = _SpeakerEncoder()
    extract = Mock(
        return_value={
            "condition_vector": torch.ones(1, 2, 3),
            "style": torch.ones(1, 192),
            "ref_mel": torch.ones(1, 3, 80),
        }
    )
    monkeypatch.setattr(talker_module, "extract_reference_features", extract)

    info = {
        "text": ["Hello."],
        "lang": ["en"],
        "text_token_ids": [[10, 11]],
        "voice": [[torch.zeros(10), 16000]],
        "ref_audio_cache_key": ["content-hash"],
    }
    input_ids = torch.zeros(4, dtype=torch.long)

    talker.preprocess(input_ids, None, **info)
    _, _, updates = talker.preprocess(input_ids, None, **info)

    extract.assert_called_once()
    assert extract.call_args.kwargs["max_audio_length_seconds"] == 30.0
    meta = updates["meta"]
    assert meta["text_token_ids"].tolist() == [10, 11]
    assert meta["condition_vector"].shape == (1, 2, 3)
    assert meta["semantic_start_offset"] == 3
    assert "hidden_states" not in updates and "codes" not in updates


def test_make_omni_output_supports_legacy_runtime_information(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    talker = _talker()
    warning = Mock(wraps=talker_module.logger.warning_once)
    monkeypatch.setattr(talker_module.logger, "warning_once", warning)

    output = talker.make_omni_output(
        torch.zeros(1, 4),
        runtime_additional_information=[{"meta": {"semantic_start_offset": 3, "semantic_code_count": 2}}],
    )

    assert output.multimodal_outputs["meta"][0] == {"semantic_code_count": 2}
    warning.assert_called_once()


@dataclass
class _WeightConfig:
    t2s_checkpoint: str = "t2s_model.safetensors"


def test_load_weights_transposes_gpt2_conv1d_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    talker = _talker()
    talker.model_path = "/model"
    talker.config = _WeightConfig()
    talker.text_projector = nn.Linear(1, 1, bias=False)
    talker.speaker_encoder = nn.Linear(1, 1, bias=False)
    talker.text_position_embedding = _PositionEmbeddingStub(1, 1)
    talker.semantic_position_embedding = _PositionEmbeddingStub(1, 1)
    talker.semantic_embedding = nn.Embedding(1, 1)
    talker.transformer = nn.Module()
    block = nn.Module()
    block.c_attn = nn.Linear(2, 3, bias=False)
    block.c_proj = nn.Linear(3, 2, bias=False)
    block.c_fc = nn.Linear(2, 4, bias=False)
    talker.transformer.h = nn.ModuleList([block])
    talker.transformer.ln_f = nn.LayerNorm(1)
    talker.final_norm = nn.LayerNorm(1)
    talker.semantic_head = nn.Linear(1, 1, bias=False)
    talker.start_layer = 0
    talker.end_layer = 1

    checkpoint = {name: torch.full_like(param, 2.0) for name, param in talker.named_parameters()}
    sources = {}
    for name in (
        "transformer.h.0.c_attn.weight",
        "transformer.h.0.c_proj.weight",
        "transformer.h.0.c_fc.weight",
    ):
        target = checkpoint[name]
        source = torch.arange(
            target.numel(),
            dtype=target.dtype,
        ).reshape(target.shape[1], target.shape[0])
        checkpoint[name] = source
        sources[name] = source

    monkeypatch.setattr(
        talker_module,
        "resolve_model_file",
        lambda *_args: "/model/t2s_model.safetensors",
    )
    monkeypatch.setattr(
        talker_module,
        "load_file",
        lambda *_args, **_kwargs: checkpoint,
    )
    monkeypatch.setattr(
        talker_module,
        "is_pp_missing_parameter",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        talker_module,
        "get_pp_group",
        lambda: Mock(is_first_rank=True, is_last_rank=True),
    )

    loaded = talker.load_weights([])

    for name, source in sources.items():
        assert name in loaded
        assert torch.equal(
            dict(talker.named_parameters())[name],
            source.t(),
        )
    # Upstream only zeroes row 0 in its vLLM copy, whose prefix embeddings
    # already include positions. The native model keeps the stored row, and
    # the BOS position embedding here is built natively too.
    position_row = talker.semantic_position_embedding.embedding.weight[0]
    assert torch.equal(position_row, torch.full_like(position_row, 2.0))


def test_load_weights_middle_pp_rank_requires_only_local_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    talker = _talker()
    talker.model_path = "/model"
    talker.config = _WeightConfig()
    talker.semantic_position_embedding = _PositionEmbeddingStub(1, 1)
    talker.transformer = nn.Module()
    talker.transformer.h = nn.ModuleList([nn.Linear(1, 1, bias=False)])
    talker.start_layer = 0
    talker.end_layer = 1

    checkpoint = {
        "transformer.h.0.weight": torch.ones(1, 1),
    }
    monkeypatch.setattr(
        talker_module,
        "resolve_model_file",
        lambda *_args: "/model/t2s_model.safetensors",
    )
    monkeypatch.setattr(
        talker_module,
        "load_file",
        lambda *_args, **_kwargs: checkpoint,
    )
    monkeypatch.setattr(
        talker_module,
        "is_pp_missing_parameter",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        talker_module,
        "get_pp_group",
        lambda: Mock(is_first_rank=False, is_last_rank=False),
    )

    assert talker.load_weights([]) == {"transformer.h.0.weight"}


def test_load_weights_rejects_missing_local_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    talker = _talker()
    talker.model_path = "/model"
    talker.config = _WeightConfig()
    talker.transformer = nn.Module()
    talker.transformer.h = nn.ModuleList([nn.Linear(1, 1)])
    talker.start_layer = 0
    talker.end_layer = 1

    monkeypatch.setattr(
        talker_module,
        "resolve_model_file",
        lambda *_args: "/model/t2s_model.safetensors",
    )
    monkeypatch.setattr(
        talker_module,
        "load_file",
        lambda *_args, **_kwargs: {
            "transformer.h.0.weight": torch.ones(1, 1),
        },
    )
    monkeypatch.setattr(
        talker_module,
        "is_pp_missing_parameter",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        talker_module,
        "get_pp_group",
        lambda: Mock(is_first_rank=False, is_last_rank=False),
    )

    with pytest.raises(
        RuntimeError,
        match=r"missing required parameters: transformer\.h\.0\.bias",
    ):
        talker.load_weights([])


def test_load_weights_accepts_loaded_parameter_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    talker = _talker()
    talker.model_path = "/model"
    talker.config = _WeightConfig()
    talker.transformer = nn.Module()
    layer = nn.Linear(1, 1, bias=False)
    layer.weight_alias = layer.weight
    talker.transformer.h = nn.ModuleList([layer])
    talker.start_layer = 0
    talker.end_layer = 1

    monkeypatch.setattr(
        talker_module,
        "resolve_model_file",
        lambda *_args: "/model/t2s_model.safetensors",
    )
    monkeypatch.setattr(
        talker_module,
        "load_file",
        lambda *_args, **_kwargs: {
            "transformer.h.0.weight": torch.ones(1, 1),
        },
    )
    monkeypatch.setattr(
        talker_module,
        "is_pp_missing_parameter",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        talker_module,
        "get_pp_group",
        lambda: Mock(is_first_rank=False, is_last_rank=False),
    )

    assert talker.load_weights([]) == {"transformer.h.0.weight"}


def test_load_weights_warns_for_unexpected_checkpoint_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    talker = _talker()
    talker.model_path = "/model"
    talker.config = _WeightConfig()
    talker.transformer = nn.Module()
    talker.transformer.h = nn.ModuleList([nn.Linear(1, 1)])
    talker.start_layer = 0
    talker.end_layer = 1
    warning = Mock()

    monkeypatch.setattr(
        talker_module,
        "resolve_model_file",
        lambda *_args: "/model/t2s_model.safetensors",
    )
    monkeypatch.setattr(
        talker_module,
        "load_file",
        lambda *_args, **_kwargs: {
            "transformer.h.0.weight": torch.ones(1, 1),
            "transformer.h.0.bias": torch.ones(1),
            "unexpected.weight": torch.ones(1),
        },
    )
    monkeypatch.setattr(
        talker_module,
        "is_pp_missing_parameter",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        talker_module,
        "get_pp_group",
        lambda: Mock(is_first_rank=False, is_last_rank=False),
    )
    monkeypatch.setattr(talker_module.logger, "warning", warning)

    assert talker.load_weights([]) == {
        "transformer.h.0.weight",
        "transformer.h.0.bias",
    }
    warning.assert_called_once_with(
        "Ignored %d unexpected Confucius4-TTS checkpoint keys: %s",
        1,
        "unexpected.weight",
    )
