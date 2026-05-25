# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.common.nucleus_ras_sampling import ras_sample_one
from vllm_omni.model_executor.models.glm_tts import glm_tts as glm_tts_module
from vllm_omni.model_executor.models.glm_tts.glm_tts import GLMTTSForConditionalGeneration
from vllm_omni.model_executor.models.glm_tts.text_frontend import GLMTTSTextFrontend

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_glm_tts_ras_fallback_matches_official_random_resample() -> None:
    sampled = ras_sample_one(
        torch.tensor([0.0, float("-inf")]),
        decoded_tokens=[0] * 10,
        top_p=1.0,
        top_k=1,
        win_size=10,
        tau_r=0.1,
        generator=None,
    )

    assert sampled == 0


def test_glm_tts_text_frontend_applies_official_symbol_replacements() -> None:
    frontend = GLMTTSTextFrontend()

    assert frontend.pre_replace("3-2 咯，") == "3减2 喽，"
    assert frontend.post_replace("α②∈A/B~C²") == "阿尔法二属于A每B到C平方"


def test_glm_tts_call_hf_processor_builds_prompt_text_text_boa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = object.__new__(glm_tts_module.GLMTTSMultiModalProcessor)
    processor.info = SimpleNamespace(
        ctx=SimpleNamespace(
            get_hf_config=lambda *_args, **_kwargs: SimpleNamespace(),
            model_config=SimpleNamespace(model="fake-model"),
        )
    )
    processor.text_frontend = object()
    processor.special_ids = {"boa": 99}
    processor.speech_tokenizer = object()
    processor.processor_device = torch.device("cpu")
    processor.campplus_session = object()

    monkeypatch.setattr(
        processor,
        "_ensure_cached_runtime_components",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        processor,
        "_encode_text",
        lambda text: torch.tensor([[11, 12]]) if text.endswith(" ") else torch.tensor([[21, 22, 23]]),
    )
    monkeypatch.setattr(processor, "_get_audio", lambda _mm_data: (torch.zeros(8), 24000))
    monkeypatch.setattr(
        glm_tts_module,
        "_normalize_glm_tts_processor_text",
        lambda _frontend, text, add_trailing_space=False: text + (" " if add_trailing_space else ""),
    )
    monkeypatch.setattr(glm_tts_module, "extract_prompt_speech_token", lambda *_args, **_kwargs: [7, 8, 9, 10])
    monkeypatch.setattr(glm_tts_module, "extract_prompt_feat", lambda *_args, **_kwargs: torch.ones(4, 3))
    monkeypatch.setattr(glm_tts_module, "extract_spk_embedding", lambda *_args, **_kwargs: [0.1, 0.2, 0.3])

    out = processor._call_hf_processor(
        prompt="target text",
        mm_data={"audio": object()},
        mm_kwargs={"prompt_text": "reference text"},
        tok_kwargs={},
    )

    assert out["input_ids"].tolist() == [[11, 12, 21, 22, 23, 99]]
    assert out["prompt_speech_token"].tolist() == [[7, 8, 9, 10]]
    assert out["prompt_speech_token_len"][0].item() == 4
    assert out["glm_tts_text_token_len"][0].item() == 3


def test_glm_tts_prompt_updates_append_single_audio_suffix() -> None:
    processor = object.__new__(glm_tts_module.GLMTTSMultiModalProcessor)
    out_mm_kwargs = {
        "audio": [{"prompt_speech_token_len": SimpleNamespace(data=[torch.tensor([5], dtype=torch.long)])}]
    }

    updates = processor._get_prompt_updates(
        mm_items=object(),
        hf_processor_mm_kwargs={},
        out_mm_kwargs=out_mm_kwargs,
    )

    assert len(updates) == 1
    assert updates[0].modality == "audio"
    assert updates[0].insertion(0) == [1, 1, 1, 1, 1]


def test_glm_tts_dummy_inputs_forbid_dummy_wav(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = object.__new__(glm_tts_module.GLMTTSDummyInputsBuilder)
    monkeypatch.setattr(
        builder,
        "_get_dummy_audios",
        lambda *_args, **_kwargs: pytest.fail("GLM-TTS dummy inputs must not fabricate wav audio"),
    )

    assert builder.get_dummy_mm_data(seq_len=16, mm_counts={"audio": 1}) == {}


def test_glm_tts_processing_info_declares_audio_token_budget() -> None:
    info = object.__new__(glm_tts_module.GLMTTSMultiModalProcessingInfo)

    assert info.get_mm_max_tokens_per_item(seq_len=4096, mm_counts={"audio": 1}) == {"audio": 1024}
    assert info.get_mm_max_tokens_per_item(seq_len=512, mm_counts={"audio": 1}) == {"audio": 512}
    assert info.get_mm_max_tokens_per_item(seq_len=4096, mm_counts={}) == {}


def test_glm_tts_text_only_processor_skips_voice_clone(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTokenizer:
        def encode(self, text: str) -> list[int]:
            return [11, 12, 13]

    processor = object.__new__(glm_tts_module.GLMTTSMultiModalProcessor)
    processor.info = SimpleNamespace(
        ctx=SimpleNamespace(
            get_hf_config=lambda *_args, **_kwargs: SimpleNamespace(),
            model_config=SimpleNamespace(model="fake-model", tokenizer=None, trust_remote_code=True),
        )
    )

    monkeypatch.setattr(glm_tts_module, "resolve_glm_tts_model_dir", lambda *_args, **_kwargs: "/fake-root")
    monkeypatch.setattr(glm_tts_module, "load_glm_tts_tokenizer", lambda *_args, **_kwargs: FakeTokenizer())
    monkeypatch.setattr(glm_tts_module, "get_glm_tts_special_token_ids", lambda _tokenizer: {"boa": 99})
    monkeypatch.setattr(glm_tts_module, "GLMTTSTextFrontend", lambda: object())
    monkeypatch.setattr(
        glm_tts_module,
        "_normalize_glm_tts_processor_text",
        lambda _frontend, text, add_trailing_space=False: text,
    )
    monkeypatch.setattr(
        glm_tts_module,
        "load_voice_clone_frontend",
        lambda *_args, **_kwargs: pytest.fail("text-only dummy path should not load voice clone frontend"),
    )

    out = processor._call_hf_processor(prompt="target text", mm_data={}, mm_kwargs={}, tok_kwargs={})

    assert out["input_ids"].tolist() == [[11, 12, 13, 99]]
    assert out["prompt_speech_token"].shape == (1, 1024)
    assert out["prompt_speech_token_len"][0].item() == 1024
    assert out["glm_tts_prompt_text_token_len"][0].item() == 0
    assert out["glm_tts_text_token_len"][0].item() == 3

    fields = processor._get_mm_fields_config(out, {})
    assert {key: value.modality for key, value in fields.items()} == {
        "prompt_speech_token": "audio",
        "prompt_speech_token_len": "audio",
        "glm_tts_prompt_text_token_len": "audio",
        "glm_tts_text_token_len": "audio",
    }

    updates = processor._get_prompt_updates(
        mm_items=object(),
        hf_processor_mm_kwargs={},
        out_mm_kwargs={
            "audio": [{"prompt_speech_token_len": SimpleNamespace(data=[out["prompt_speech_token_len"][0]])}]
        },
    )
    assert len(updates) == 1
    assert len(updates[0].insertion(0)) == 1024


def test_glm_tts_embed_replaces_multimodal_positions_per_request() -> None:
    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()))

        def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
            return input_ids.to(torch.float32).reshape(-1, 1)

    model = object.__new__(GLMTTSForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model_stage = "glm_tts"
    model.model = FakeModel()

    input_ids = torch.tensor([101, 102, 1, 1, 201, 1, 1, 1])
    is_multimodal = torch.tensor([False, False, True, True, False, True, True, True])
    mm_embeds = [
        torch.tensor([[1001.0], [1002.0]]),
        torch.tensor([[2001.0], [2002.0], [2003.0]]),
    ]

    out = model.embed_input_ids(input_ids, multimodal_embeddings=mm_embeds, is_multimodal=is_multimodal)

    assert out.squeeze(-1).tolist() == [
        101.0,
        102.0,
        1001.0,
        1002.0,
        201.0,
        2001.0,
        2002.0,
        2003.0,
    ]


def test_glm_tts_embed_multimodal_keeps_all_batched_requests() -> None:
    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()))

        def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
            return input_ids.to(torch.float32).reshape(*input_ids.shape, 1)

    model = object.__new__(GLMTTSForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model_stage = "glm_tts"
    model.model = FakeModel()
    model._ats = 1000

    out = model.embed_multimodal(
        prompt_speech_token=[
            torch.tensor([[1, 2]]),
            torch.tensor([[3, 4, 5]]),
        ]
    )

    assert out is not None
    assert len(out) == 2
    assert out[0].squeeze(-1).tolist() == [1001.0, 1002.0]
    assert out[1].squeeze(-1).tolist() == [1003.0, 1004.0, 1005.0]


def test_glm_tts_embed_multimodal_returns_one_item_for_single_request() -> None:
    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()))

        def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
            return input_ids.to(torch.float32).reshape(*input_ids.shape, 1)

    model = object.__new__(GLMTTSForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model_stage = "glm_tts"
    model.model = FakeModel()
    model._ats = 1000

    out = model.embed_multimodal(prompt_speech_token=torch.tensor([[1, 2, 3]]))

    assert out is not None
    assert len(out) == 1
    assert out[0].squeeze(-1).tolist() == [1001.0, 1002.0, 1003.0]


def test_glm_tts_embed_multimodal_treats_scalar_tensor_list_as_one_item() -> None:
    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()))

        def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
            return input_ids.to(torch.float32).reshape(*input_ids.shape, 1)

    model = object.__new__(GLMTTSForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model_stage = "glm_tts"
    model.model = FakeModel()
    model._ats = 1000

    out = model.embed_multimodal(prompt_speech_token=[torch.tensor(1), torch.tensor(2)])

    assert out is not None
    assert len(out) == 1
    assert out[0].squeeze(-1).tolist() == [1001.0, 1002.0]


def test_glm_tts_preprocess_treats_one_token_text_tail_as_prefill() -> None:
    model = object.__new__(GLMTTSForConditionalGeneration)
    model._pad = 0
    model._ats = 1000
    model._ate = 2000
    model._eoa = 3000
    model._model_dtype = lambda: torch.float32

    input_ids, input_embeds, info = model.preprocess(
        torch.tensor([123]),
        torch.tensor([[1.0, 2.0]]),
        glm_tts_mm_prefill_done=True,
    )

    assert input_ids.tolist() == [0]
    assert input_embeds.tolist() == [[1.0, 2.0]]
    assert info["glm_tts_mm_prefill_done"] is True
    assert info["speech_tokens"].tolist() == [[-1]]


def test_glm_tts_preprocess_initializes_generation_bounds_before_sampling() -> None:
    model = object.__new__(GLMTTSForConditionalGeneration)
    model._pad = 0
    model._ats = 1000
    model._ate = 2000
    model._eoa = 3000
    model._model_dtype = lambda: torch.float32
    model.config = SimpleNamespace(min_token_text_ratio=2.0, max_token_text_ratio=20.0)

    _, _, info = model.preprocess(
        torch.tensor([11, 12]),
        torch.tensor([[1.0], [2.0]]),
        glm_tts_text_token_len=[torch.tensor([27])],
    )

    assert info["glm_tts_text_token_len"][0].item() == 27


def test_glm_tts_make_omni_output_emits_prompt_speech_len_once() -> None:
    model = object.__new__(GLMTTSForConditionalGeneration)

    out = model.make_omni_output(
        torch.randn(2, 4),
        model_intermediate_buffer=[{}],
        prompt_speech_token_len=[torch.tensor([9])],
        glm_tts_text_token_len=[torch.tensor([5])],
    )

    assert out.multimodal_outputs is not None
    assert out.multimodal_outputs["prompt_speech_token_len"][0].item() == 9
    assert out.multimodal_outputs["glm_tts_text_token_len"][0].item() == 5


def test_glm_tts_postprocess_caches_prompt_speech_token_len() -> None:
    model = object.__new__(GLMTTSForConditionalGeneration)
    model.config = SimpleNamespace(min_token_text_ratio=2.0, max_token_text_ratio=20.0)

    update = model.postprocess(
        torch.randn(2, 4),
        multimodal_outputs={
            "prompt_speech_token_len": [torch.tensor([13])],
            "glm_tts_text_token_len": [torch.tensor([6])],
        },
    )

    assert update["prompt_speech_token_len"].item() == 13
    assert update["glm_tts_text_token_len"].item() == 6
