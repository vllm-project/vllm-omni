# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from transformers import BatchFeature

from vllm_omni.model_executor.models.llama_omni2.llama_omni2_thinker import (
    SPEECH_TOKEN_ID,
    THINKER_WEIGHTS_MAPPER,
    EncoderProjectorConcat,
    LlamaOmni2MultiModalProcessor,
    LlamaOmni2Processor,
    LlamaOmni2ThinkerForConditionalGeneration,
    load_openai_whisper_encoder,
    projected_speech_lengths,
    speech_placeholder_token_ids,
    splice_speech_embeddings,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_encoder_projector_discards_incomplete_frame_groups():
    config = SimpleNamespace(
        speech_encoder_ds_rate=5,
        speech_encoder_hidden_size=1280,
        hidden_size=896,
    )
    projector = EncoderProjectorConcat(config)
    encoder_output = torch.randn(2, 12, 1280)

    projected = projector(encoder_output)

    assert projected.shape == (2, 2, 896)
    assert projector.linear1.in_features == 6400
    assert projector.linear1.out_features == 2048
    assert projector.linear2.in_features == 2048
    assert projector.linear2.out_features == 896


@pytest.mark.parametrize(
    ("input_lengths", "expected"),
    [
        ([100, 101], [10, 10]),
        ([9, 10, 11], [1, 1, 1]),
        ([0, 1], [0, 0]),
    ],
)
def test_projected_speech_lengths_matches_whisper_and_projector_downsampling(
    input_lengths,
    expected,
):
    lengths = torch.tensor(input_lengths)

    actual = projected_speech_lengths(lengths, projector_stride=5)

    assert actual.tolist() == expected


def test_splice_speech_embeddings_preserves_text_order_for_multiple_audio_items():
    input_ids = torch.tensor([11, 12, -200, 13, -200, 14, 15])
    speech_features = [
        torch.full((2, 3), 101.0),
        torch.full((1, 3), 202.0),
    ]

    def embed_tokens(token_ids: torch.Tensor) -> torch.Tensor:
        return token_ids.to(torch.float32).unsqueeze(-1).repeat(1, 3)

    result = splice_speech_embeddings(
        input_ids,
        speech_features,
        embed_tokens=embed_tokens,
    )

    assert result[:, 0].tolist() == [
        11.0,
        12.0,
        101.0,
        101.0,
        13.0,
        202.0,
        14.0,
        15.0,
    ]


@pytest.mark.parametrize("feature_count", [0, 2])
def test_splice_speech_embeddings_rejects_placeholder_count_mismatch(feature_count):
    input_ids = torch.tensor([11, -200, 12])
    speech_features = [torch.zeros(1, 3) for _ in range(feature_count)]

    with pytest.raises(
        ValueError,
        match=r"speech placeholder count \(1\).*speech feature count",
    ):
        splice_speech_embeddings(
            input_ids,
            speech_features,
            embed_tokens=lambda ids: torch.zeros(ids.shape[0], 3),
        )


def test_thinker_composes_registered_vllm_qwen2(monkeypatch):
    import vllm_omni.model_executor.models.llama_omni2.llama_omni2_thinker as thinker_module

    thinker_config = SimpleNamespace(hidden_size=8)
    root_config = SimpleNamespace(
        thinker_config=thinker_config,
        hidden_size=8,
        speech_encoder="large-v3",
        speech_encoder_ds_rate=5,
        speech_encoder_hidden_size=4,
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
            return torch.ones(2, 16)

        def embed_input_ids(self, input_ids):
            return torch.zeros(input_ids.shape[0], 8)

    language_model = FakeLanguageModel()

    def fake_init_vllm_registered_model(**kwargs):
        calls["init"] = kwargs
        return language_model

    monkeypatch.setattr(
        thinker_module,
        "init_vllm_registered_model",
        fake_init_vllm_registered_model,
    )
    monkeypatch.setattr(
        thinker_module,
        "load_openai_whisper_encoder",
        lambda _: torch.nn.Identity(),
    )
    model = LlamaOmni2ThinkerForConditionalGeneration(
        vllm_config=vllm_config,
        prefix="thinker",
    )

    assert calls["init"]["vllm_config"] is vllm_config
    assert calls["init"]["hf_config"] is thinker_config
    assert calls["init"]["architectures"] == ["Qwen2ForCausalLM"]
    assert calls["init"]["prefix"] == "thinker.language_model"
    assert model.language_model is language_model
    assert model.make_empty_intermediate_tensors is language_model.make_empty_intermediate_tensors

    input_ids = torch.tensor([1, 2])
    positions = torch.tensor([0, 1])
    hidden_states = model(input_ids, positions)
    logits = model.compute_logits(hidden_states)

    assert hidden_states.shape == (2, 8)
    assert logits.shape == (2, 16)
    assert calls["forward"] == (input_ids, positions, None, None)
    assert calls["logits"] is hidden_states


def test_thinker_weight_mapper_routes_real_checkpoint_prefixes():
    checkpoint_names = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "lm_head.weight",
        "model.speech_encoder.conv1.weight",
        "model.speech_projector.linear1.weight",
        "speech_generator.model.layers.0.self_attn.q_proj.weight",
    ]

    assert THINKER_WEIGHTS_MAPPER.apply_list(checkpoint_names) == [
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.self_attn.qkv_proj.weight",
        "language_model.lm_head.weight",
        "speech_encoder.conv1.weight",
        "speech_projector.linear1.weight",
    ]


def test_thinker_embed_input_ids_scatter_speech_features():
    language_model = SimpleNamespace(embed_input_ids=lambda ids: ids.to(torch.float32).unsqueeze(-1).repeat(1, 3))
    model = object.__new__(LlamaOmni2ThinkerForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.language_model = language_model
    input_ids = torch.tensor([11, 151665, 12, 151665])
    speech_features = [
        torch.tensor([[101.0, 101.0, 101.0]]),
        torch.tensor([[202.0, 202.0, 202.0]]),
    ]

    embeddings = model.embed_input_ids(
        input_ids,
        multimodal_embeddings=speech_features,
        is_multimodal=input_ids == 151665,
    )

    assert embeddings[:, 0].tolist() == [11.0, 101.0, 12.0, 202.0]


def test_thinker_processes_whisper_features_and_trims_each_audio():
    class FakeEncoder(torch.nn.Module):
        def forward(self, features):
            assert features.shape == (2, 128, 21)
            return torch.arange(2 * 11 * 4, dtype=torch.float32).view(2, 11, 4)

    class FakeProjector(torch.nn.Module):
        k = 5

        def forward(self, features):
            return features[:, :10].reshape(2, 2, 20)[..., :3]

    model = object.__new__(LlamaOmni2ThinkerForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.speech_encoder = FakeEncoder()
    model.speech_projector = FakeProjector()
    model.config = SimpleNamespace(speech_encoder_type="whisper")

    features = model._process_speech_input(
        torch.zeros(2, 128, 21),
        torch.tensor([21, 10]),
    )

    assert [feature.shape for feature in features] == [(2, 3), (1, 3)]


def test_speech_placeholder_token_ids_match_projected_lengths():
    replacements = speech_placeholder_token_ids(torch.tensor([3000, 1501]))

    assert [len(tokens) for tokens in replacements] == [300, 150]
    assert replacements[0] == [SPEECH_TOKEN_ID] * 300


def test_llama_omni2_processor_returns_checkpoint_compatible_mels(monkeypatch):
    tokenizer_calls = {}

    class FakeTokenizer:
        def __call__(self, text, **kwargs):
            tokenizer_calls["args"] = (text, kwargs)
            return {"input_ids": torch.tensor([[1, SPEECH_TOKEN_ID, 2]])}

    class FakeWhisper:
        @staticmethod
        def pad_or_trim(audio):
            assert audio.shape == (160,)
            return torch.zeros(480000)

        @staticmethod
        def log_mel_spectrogram(audio, n_mels):
            assert audio.shape == (480000,)
            assert n_mels == 128
            return torch.zeros(128, 3000)

    import vllm_omni.model_executor.models.llama_omni2.llama_omni2_thinker as thinker_module

    monkeypatch.setattr(thinker_module, "_import_openai_whisper", lambda: FakeWhisper)
    processor = LlamaOmni2Processor(FakeTokenizer())

    output = processor(
        text="listen <speech>",
        audio=[torch.zeros(160), torch.ones(160)],
        return_tensors="pt",
    )

    assert isinstance(output, BatchFeature)
    assert tokenizer_calls["args"][0] == "listen <speech>"
    assert output["speech"].shape == (2, 128, 3000)
    assert output["speech_lengths"].tolist() == [3000, 3000]


def test_whisper_encoder_is_constructed_without_external_checkpoint(monkeypatch):
    import vllm_omni.model_executor.models.llama_omni2.llama_omni2_thinker as thinker_module

    calls = {}

    class FakeLayerNorm(torch.nn.LayerNorm):
        pass

    class FakeAudioEncoder(torch.nn.Module):
        def __init__(self, n_mels, n_ctx, n_state, n_head, n_layer):
            super().__init__()
            calls["dims"] = (n_mels, n_ctx, n_state, n_head, n_layer)
            self.norm = FakeLayerNorm(n_state)

    fake_whisper = SimpleNamespace(
        model=SimpleNamespace(
            AudioEncoder=FakeAudioEncoder,
            LayerNorm=FakeLayerNorm,
        ),
        load_model=lambda *args, **kwargs: pytest.fail("Whisper checkpoint download must not be used"),
    )
    monkeypatch.setattr(
        thinker_module,
        "_import_openai_whisper",
        lambda: fake_whisper,
    )

    encoder = load_openai_whisper_encoder("models/speech_encoder/large-v3.pt")

    assert calls["dims"] == (128, 1500, 1280, 20, 32)
    assert isinstance(encoder.norm, torch.nn.LayerNorm)
    assert not isinstance(encoder.norm, FakeLayerNorm)


def test_thinker_registers_vllm_multimodal_processor():
    assert hasattr(
        LlamaOmni2ThinkerForConditionalGeneration,
        "_processor_factory",
    )


def test_thinker_exports_cumulative_postprocess_rows_to_full_payload_pooler():
    assert LlamaOmni2ThinkerForConditionalGeneration.cumulative_postprocess_output_buffer_keys == {
        ("ids", "output"),
        ("embed", "decode"),
        ("hidden_states", "output"),
    }


def test_multimodal_processor_fields_and_prompt_replacement_follow_lengths():
    processor = object.__new__(LlamaOmni2MultiModalProcessor)
    hf_inputs = {
        "speech": torch.zeros(2, 128, 3000),
        "speech_lengths": torch.tensor([3000, 1501]),
    }
    fields = processor._get_mm_fields_config(hf_inputs, {})

    assert set(fields) == {"speech", "speech_lengths"}

    class FakeOutputKwargs:
        @staticmethod
        def get_data():
            return {"speech_lengths": torch.tensor([3000, 1501])}

    updates = processor._get_prompt_updates(
        {"audio": [object(), object()]},
        {},
        FakeOutputKwargs(),
    )
    first = updates[0].resolve(0)
    second = updates[0].resolve(1)

    assert first.target == "<speech>"
    assert len(first.content.full) == 300
    assert len(second.content.full) == 150
    assert set(first.content.full) == {SPEECH_TOKEN_ID}


def test_multimodal_processor_applies_prompt_expansion_after_hf_processing():
    processor = object.__new__(LlamaOmni2MultiModalProcessor)

    assert (
        processor._hf_processor_applies_updates(
            prompt_text="listen <speech>",
            mm_items={"audio": [object()]},
            hf_processor_mm_kwargs={},
            tokenization_kwargs={},
        )
        is False
    )
