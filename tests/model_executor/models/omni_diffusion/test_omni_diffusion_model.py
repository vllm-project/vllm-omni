# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OmniDiffusionForConditionalGeneration helper methods (no model loading)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import vllm_omni.model_executor.models.omni_diffusion.omni_diffusion as omni_module
from vllm_omni.model_executor.models.omni_diffusion.dream_compat import (
    _compute_default_dream_rope_parameters,
    ensure_dream_generation_config_fields,
    ensure_dream_rope_parameters,
)
from vllm_omni.model_executor.models.omni_diffusion.omni_diffusion import (
    OmniDiffusionAdditionalConfig,
    OmniDiffusionForConditionalGeneration,
)
from vllm_omni.model_executor.models.omni_diffusion.utils import (
    OMNI_DIFFUSION_AUDIO_START_TOKEN,
    OMNI_DIFFUSION_IMAGE_START_TOKEN,
    OmniDiffusionModelSpecialTokens,
    OmniDiffusionTokenizerBaseData,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tokenizer_base_data(token_map: dict[str, int] | None = None) -> OmniDiffusionTokenizerBaseData:
    """Build a TokenizerBaseData with deterministic token IDs."""
    tok = MagicMock()

    default_map = {
        OmniDiffusionModelSpecialTokens.AUD_TAG.value: 100,
        OmniDiffusionModelSpecialTokens.AUD_CONTEXT.value: 101,
        OmniDiffusionModelSpecialTokens.AUD_START.value: 102,
        OmniDiffusionModelSpecialTokens.AUD_END.value: 103,
        OmniDiffusionModelSpecialTokens.IMG_TAG.value: 200,
        OmniDiffusionModelSpecialTokens.IMG_START.value: 201,
        OmniDiffusionModelSpecialTokens.IMG_END.value: 202,
    }

    def _encode(tokens, add_special_tokens):
        ids = []
        for t in tokens:
            tid = None
            if token_map:
                tid = token_map.get(t)
            if tid is None:
                tid = default_map.get(t)
            if tid is None:
                tid = abs(hash(t)) % 1000 + 500
            ids.append([tid])
        return SimpleNamespace(input_ids=ids)

    tok.side_effect = _encode
    return OmniDiffusionTokenizerBaseData(tok)


def _make_stub_model(
    tokenizer_base_data: OmniDiffusionTokenizerBaseData | None = None,
    image_start_id: int = 100000,
    audio_start_id: int = 110000,
) -> OmniDiffusionForConditionalGeneration:
    """Build a minimal OmniDiffusionForConditionalGeneration with stubs for testing helpers."""

    if tokenizer_base_data is None:
        tokenizer_base_data = _make_tokenizer_base_data()

    with patch.object(OmniDiffusionForConditionalGeneration, "__init__", lambda self, **kwargs: None):
        model = object.__new__(OmniDiffusionForConditionalGeneration)
        model.tokenizer = MagicMock()
        model.tokenizer_base_data = tokenizer_base_data
        model.hidden_size = 4096
        model.dtype = torch.float32

        def _convert_tokens_to_ids(token_str):
            mapping = {
                OMNI_DIFFUSION_AUDIO_START_TOKEN: audio_start_id,
                OMNI_DIFFUSION_IMAGE_START_TOKEN: image_start_id,
                "<|im_end|>": 250,
                "<|endoftext|>": 251,
            }
            return mapping.get(token_str, abs(hash(token_str)) % 10000)

        model.tokenizer.convert_tokens_to_ids = _convert_tokens_to_ids
        model.tokenizer.eos_token_id = 1
        model.tokenizer.pad_token_id = 0
        return model


def _additional_config(task: str) -> OmniDiffusionAdditionalConfig:
    return OmniDiffusionAdditionalConfig(
        image_tokenizer_path="/fake/magvit" if task in {"T2I", "VQA", "SVQA"} else None,
        audio_tokenizer_type="sensevoice_glm4voice",
        flow_path="/fake/flow" if task == "TTS" else None,
        sensevoice_path="/fake/sensevoice" if task in {"ASR", "SVQA"} else None,
        attn_implementation="eager",
        output_text_only=task in {"VQA", "ASR", "SVQA"},
        seed=42,
        task=task,
        steps=64,
        max_new_tokens=64,
        alg="entropy",
        cfg=0.0,
        temperature=0.0,
        top_p=0.9,
        add_boa_token=0,
        max_position_penalty=2.0,
        repeat_penalty=1.0,
        top_k=None,
    )


@pytest.mark.parametrize(
    ("task", "has_image_tokenizer", "has_audio_tokenizer"),
    [
        ("T2I", True, False),
        ("VQA", True, False),
        ("ASR", False, True),
        ("TTS", False, True),
        ("SVQA", True, True),
    ],
)
def test_constructor_initializes_only_task_tokenizers(
    monkeypatch,
    task: str,
    has_image_tokenizer: bool,
    has_audio_tokenizer: bool,
) -> None:
    config = _additional_config(task)
    monkeypatch.setattr(
        omni_module.OmniDiffusionAdditionalConfig,
        "from_vllm_config",
        MagicMock(return_value=config),
    )
    tokenizer = MagicMock()
    monkeypatch.setattr(
        omni_module.AutoTokenizer,
        "from_pretrained",
        MagicMock(return_value=tokenizer),
    )
    monkeypatch.setattr(
        omni_module,
        "OmniDiffusionTokenizerBaseData",
        MagicMock(),
    )
    hf_config = SimpleNamespace(model_type="Dream", rope_parameters={"rope_type": "default"})
    monkeypatch.setattr(
        omni_module.AutoConfig,
        "from_pretrained",
        MagicMock(return_value=hf_config),
    )
    hf_model = torch.nn.Linear(1, 1)
    hf_model.config = hf_config
    monkeypatch.setattr(
        omni_module.AutoModel,
        "from_pretrained",
        MagicMock(return_value=hf_model),
    )
    for helper in (
        "ensure_dream_rope_parameters",
        "ensure_default_rope_init_function",
        "patch_remote_dream_generation_config_validate",
        "repair_default_dream_rope_buffers",
        "initialize_dream_generation_config",
    ):
        monkeypatch.setattr(omni_module, helper, MagicMock())
    image_tokenizer_cls = MagicMock()
    audio_tokenizer_cls = MagicMock()
    monkeypatch.setattr(omni_module, "OmniDiffusionImageTokenizer", image_tokenizer_cls)
    monkeypatch.setattr(omni_module, "OmniDiffusionAudioTokenizer", audio_tokenizer_cls)
    model_config = SimpleNamespace(
        model="/fake/model",
        dtype=torch.float32,
        trust_remote_code=True,
        get_hidden_size=lambda: 16,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        device_config=SimpleNamespace(device=torch.device("cpu")),
    )

    model = OmniDiffusionForConditionalGeneration(vllm_config=vllm_config)

    assert (model.image_tokenizer is not None) is has_image_tokenizer
    assert (model.audio_tokenizer is not None) is has_audio_tokenizer
    assert image_tokenizer_cls.call_count == int(has_image_tokenizer)
    assert audio_tokenizer_cls.call_count == int(has_audio_tokenizer)


# ---------------------------------------------------------------------------
# _has_image_placeholder / _has_audio_placeholder
# ---------------------------------------------------------------------------


class TestPlaceholderDetection:
    def test_detects_image_placeholder(self) -> None:
        model = _make_stub_model()
        img_tag = model.tokenizer_base_data.get_token_id(OmniDiffusionModelSpecialTokens.IMG_TAG)
        assert model._has_image_placeholder([img_tag, 10, 20]) is True

    def test_no_image_placeholder(self) -> None:
        model = _make_stub_model()
        assert model._has_image_placeholder([10, 20, 30]) is False

    def test_detects_audio_placeholder(self) -> None:
        model = _make_stub_model()
        aud_tag = model.tokenizer_base_data.get_token_id(OmniDiffusionModelSpecialTokens.AUD_TAG)
        assert model._has_audio_placeholder([aud_tag, 10, 20]) is True

    def test_no_audio_placeholder(self) -> None:
        model = _make_stub_model()
        assert model._has_audio_placeholder([10, 20, 30]) is False


class TestTaskTokenizerAccess:
    def test_returns_initialized_tokenizers(self) -> None:
        model = _make_stub_model()
        model.image_tokenizer = MagicMock()
        model.audio_tokenizer = MagicMock()

        assert model._get_image_tokenizer() is model.image_tokenizer
        assert model._get_audio_tokenizer() is model.audio_tokenizer

    def test_rejects_tokenizer_not_used_by_task(self) -> None:
        model = _make_stub_model()
        model.additional_config = SimpleNamespace(task="ASR")
        model.image_tokenizer = None
        model.audio_tokenizer = None

        with pytest.raises(RuntimeError, match="ASR.*image tokenizer"):
            model._get_image_tokenizer()
        with pytest.raises(RuntimeError, match="ASR.*audio tokenizer"):
            model._get_audio_tokenizer()


# ---------------------------------------------------------------------------
# _get_single_prompt_token_ids
# ---------------------------------------------------------------------------


class TestGetSinglePromptTokenIds:
    def test_2d_input_is_squeezed(self) -> None:
        model = _make_stub_model()
        result = model._get_single_prompt_token_ids(torch.tensor([[1, 2, 3]]))
        assert result == [1, 2, 3]

    def test_1d_input_is_wrapped_and_squeezed(self) -> None:
        model = _make_stub_model()
        result = model._get_single_prompt_token_ids(torch.tensor([4, 5, 6]))
        assert result == [4, 5, 6]

    def test_batch_size_gt_1_raises(self) -> None:
        model = _make_stub_model()
        with pytest.raises(ValueError, match="shape"):
            model._get_single_prompt_token_ids(torch.tensor([[1, 2], [3, 4]]))

    def test_3d_raises(self) -> None:
        model = _make_stub_model()
        with pytest.raises(ValueError, match="shape"):
            model._get_single_prompt_token_ids(torch.randn(2, 3, 4))


# ---------------------------------------------------------------------------
# _split_generated_token_ids
# ---------------------------------------------------------------------------


class TestSplitGeneratedTokenIds:
    def test_splits_text_audio_image_tokens(self) -> None:
        model = _make_stub_model(image_start_id=1000, audio_start_id=20000)
        # Use the model's disjoint audio and image codebook ranges.
        generated = torch.tensor([10, 20, 20000, 20001, 20002, 1000, 1001, 1002, 30])
        text, audio, image = model._split_generated_token_ids(generated)
        assert text.tolist() == [10, 20, 30]
        assert audio.tolist() == [0, 1, 2]  # offset by audio_start_id
        assert image.tolist() == [0, 1, 2]  # offset by image_start_id

    def test_filters_marker_tokens(self) -> None:
        model = _make_stub_model(image_start_id=5000, audio_start_id=20000)
        aud_start = model.tokenizer_base_data.get_token_id(OmniDiffusionModelSpecialTokens.AUD_START)
        aud_end = model.tokenizer_base_data.get_token_id(OmniDiffusionModelSpecialTokens.AUD_END)
        img_start = model.tokenizer_base_data.get_token_id(OmniDiffusionModelSpecialTokens.IMG_START)
        img_end = model.tokenizer_base_data.get_token_id(OmniDiffusionModelSpecialTokens.IMG_END)

        generated = torch.tensor([aud_start, 20000, 20001, aud_end, img_start, 5000, 5001, img_end])
        text, audio, image = model._split_generated_token_ids(generated)
        # Markers are removed, only content tokens remain.
        assert audio.tolist() == [0, 1]
        assert image.tolist() == [0, 1]

    def test_only_text(self) -> None:
        model = _make_stub_model()
        generated = torch.tensor([10, 20, 30, 40])
        text, audio, image = model._split_generated_token_ids(generated)
        assert text.tolist() == [10, 20, 30, 40]
        assert audio.numel() == 0
        assert image.numel() == 0

    def test_only_audio(self) -> None:
        model = _make_stub_model(audio_start_id=2000)
        generated = torch.tensor([2000, 2001, 2002, 2003])
        text, audio, image = model._split_generated_token_ids(generated)
        assert text.numel() == 0
        assert audio.tolist() == [0, 1, 2, 3]
        assert image.numel() == 0

    def test_only_image(self) -> None:
        model = _make_stub_model(image_start_id=1000)
        generated = torch.tensor([1000, 1001, 1002, 1003])
        text, audio, image = model._split_generated_token_ids(generated)
        assert text.numel() == 0
        assert audio.numel() == 0
        assert image.tolist() == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# _trim_generated_text_token_ids
# ---------------------------------------------------------------------------


class TestTrimGeneratedTextTokenIds:
    def test_trim_at_eos(self) -> None:
        model = _make_stub_model()
        model.tokenizer.eos_token_id = 5
        model.tokenizer.pad_token_id = 0
        tokens = torch.tensor([1, 2, 3, 5, 6, 7])
        result = model._trim_generated_text_token_ids(tokens)
        assert result.tolist() == [1, 2, 3]

    def test_trim_at_im_end(self) -> None:
        model = _make_stub_model()
        model.tokenizer.eos_token_id = None
        model.tokenizer.pad_token_id = None
        # convert_tokens_to_ids will return 250 for "<|im_end|>"
        tokens = torch.tensor([1, 2, 250, 6, 7])
        result = model._trim_generated_text_token_ids(tokens)
        assert result.tolist() == [1, 2]

    def test_no_stop_token_returns_all(self) -> None:
        model = _make_stub_model()
        model.tokenizer.eos_token_id = 9999
        model.tokenizer.pad_token_id = 0
        # Override convert_tokens_to_ids for im_end and endoftext to return -1.
        model.tokenizer.convert_tokens_to_ids = lambda x: -1
        tokens = torch.tensor([1, 2, 3])
        result = model._trim_generated_text_token_ids(tokens)
        assert result.tolist() == [1, 2, 3]

    def test_empty_tokens(self) -> None:
        model = _make_stub_model()
        tokens = torch.tensor([], dtype=torch.int64)
        result = model._trim_generated_text_token_ids(tokens)
        assert result.numel() == 0

    def test_stop_at_first_occurrence(self) -> None:
        model = _make_stub_model()
        model.tokenizer.eos_token_id = 5
        tokens = torch.tensor([1, 5, 2, 5, 3])
        result = model._trim_generated_text_token_ids(tokens)
        assert result.tolist() == [1]


# ---------------------------------------------------------------------------
# _empty_omni_output
# ---------------------------------------------------------------------------


class TestEmptyOmniOutput:
    def test_returns_zero_hidden_states(self) -> None:
        model = _make_stub_model()
        input_ids = torch.tensor([[1, 2, 3]])
        output = model._empty_omni_output(input_ids, cpu_hidden_states=True)
        assert output.text_hidden_states.shape == (3, model.hidden_size)
        assert output.multimodal_outputs == {}

    def test_cpu_hidden_states_flag(self) -> None:
        model = _make_stub_model()
        input_ids = torch.tensor([[1, 2]])
        output = model._empty_omni_output(input_ids, cpu_hidden_states=True)
        assert output.text_hidden_states.device.type == "cpu"


# ---------------------------------------------------------------------------
# embed_input_ids
# ---------------------------------------------------------------------------


class TestEmbedInputIds:
    def test_returns_zero_embeddings_of_correct_shape(self) -> None:
        model = _make_stub_model()
        input_ids = torch.tensor([1, 2, 3])
        embeddings = model.embed_input_ids(input_ids)
        assert embeddings.shape == (3, model.hidden_size)
        assert embeddings.dtype == model.dtype
        assert embeddings.sum() == 0.0

    def test_0d_input_ids_shape(self) -> None:
        model = _make_stub_model()
        input_ids = torch.tensor(5)
        embeddings = model.embed_input_ids(input_ids)
        assert embeddings.shape == (1, model.hidden_size)


# ---------------------------------------------------------------------------
# make_empty_intermediate_tensors
# ---------------------------------------------------------------------------


class TestMakeEmptyIntermediateTensors:
    def test_returns_empty_intermediate_tensors(self) -> None:
        model = _make_stub_model()
        result = model.make_empty_intermediate_tensors(
            batch_size=2,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        assert isinstance(result, dict) or hasattr(result, "items")
        # Should be empty regardless of batch_size.
        assert len(list(result.items())) == 0


# ---------------------------------------------------------------------------
# compute_logits / sample — always return None
# ---------------------------------------------------------------------------


class TestComputeLogitsAndSample:
    def test_compute_logits_returns_none(self) -> None:
        model = _make_stub_model()
        assert model.compute_logits(torch.randn(1, 10, 4096)) is None

    def test_sample_returns_none(self) -> None:
        model = _make_stub_model()
        assert model.sample(torch.randn(1, 32000), MagicMock()) is None


# ---------------------------------------------------------------------------
# get_dummy_runtime_additional_information
# ---------------------------------------------------------------------------


class TestGetDummyRuntimeAdditionalInfo:
    def test_returns_is_dummy_markers(self) -> None:
        model = _make_stub_model()
        result = model.get_dummy_runtime_additional_information(num_reqs=3)
        assert len(result) == 3
        for item in result:
            assert item == {"_is_dummy": True}


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------


class TestRoPEHelpers:
    def test_ensure_dream_rope_parameters_rejects_non_dream_config(self) -> None:
        with pytest.raises(ValueError, match="Dream"):
            ensure_dream_rope_parameters(SimpleNamespace(model_type="Llama"))

    def test_ensure_dream_rope_parameters_noops_when_present(self) -> None:
        config = SimpleNamespace(
            model_type="Dream",
            rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
        )
        # Should not raise.
        ensure_dream_rope_parameters(config)

    def test_ensure_dream_rope_parameters_migrates_legacy_fields(self) -> None:
        config = SimpleNamespace(
            model_type="Dream",
            rope_parameters=None,
            rope_theta=500000.0,
            rope_scaling={"type": "linear", "factor": 2.0},
        )

        ensure_dream_rope_parameters(config)

        assert config.rope_parameters == {
            "rope_type": "linear",
            "rope_theta": 500000.0,
            "factor": 2.0,
        }

    def test_compute_default_dream_rope_parameters(self) -> None:
        config = SimpleNamespace(
            model_type="Dream",
            hidden_size=256,
            num_attention_heads=8,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 1.0,
            },
        )
        inv_freq, attn_scaling = _compute_default_dream_rope_parameters(config)
        assert inv_freq.ndim == 1
        # head_dim = 256/8 = 32, dim = 32*1.0 = 32
        assert inv_freq.shape[0] == 16  # dim/2
        assert attn_scaling == 1.0


# ---------------------------------------------------------------------------
# Dream generation config helpers
# ---------------------------------------------------------------------------


class TestDreamGenerationConfigHelpers:
    def test_ensure_fields_fills_missing_defaults(self) -> None:
        gen_config = SimpleNamespace()
        model_config = SimpleNamespace()
        tokenizer = SimpleNamespace()

        ensure_dream_generation_config_fields(gen_config, model_config, tokenizer)
        assert gen_config.eps == 1e-3
        assert gen_config.steps == 512
        assert gen_config.alg == "origin"
        assert gen_config.num_return_sequences == 1
        assert gen_config.return_dict_in_generate is False
        assert gen_config.output_history is False

    def test_ensure_fields_preserves_existing_values(self) -> None:
        gen_config = SimpleNamespace(steps=100, alg="entropy")
        model_config = SimpleNamespace()
        tokenizer = SimpleNamespace()

        ensure_dream_generation_config_fields(gen_config, model_config, tokenizer)
        assert gen_config.steps == 100  # preserved
        assert gen_config.alg == "entropy"  # preserved

    def test_ensure_fields_fills_token_ids_from_model_config(self) -> None:
        gen_config = SimpleNamespace()
        model_config = SimpleNamespace(
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            mask_token_id=3,
        )
        tokenizer = SimpleNamespace()

        ensure_dream_generation_config_fields(gen_config, model_config, tokenizer)
        assert gen_config.bos_token_id == 1
        assert gen_config.eos_token_id == 2
        assert gen_config.pad_token_id == 0
        assert gen_config.mask_token_id == 3
