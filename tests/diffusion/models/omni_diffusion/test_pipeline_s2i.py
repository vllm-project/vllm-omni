# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for OmniDiffusionS2IPipeline without loading model weights."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

import vllm_omni.diffusion.models.omni_diffusion.pipeline_s2i as s2i_module
from vllm_omni.diffusion.models.omni_diffusion.pipeline_s2i import (
    OmniDiffusionS2IPipeline,
    OmniDiffusionS2IPipelineConfig,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.omni_diffusion.utils import (
    OMNI_DIFFUSION_IMAGE_CODEBOOK_SIZE,
    OMNI_DIFFUSION_IMAGE_START_TOKEN,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _model_config(tmp_path: Path, **overrides: object) -> dict[str, object]:
    image_path = tmp_path / "magvit"
    sensevoice_path = tmp_path / "sensevoice"
    image_path.mkdir(exist_ok=True)
    sensevoice_path.mkdir(exist_ok=True)
    return {
        "image_tokenizer_path": str(image_path),
        "sensevoice_path": str(sensevoice_path),
        **overrides,
    }


def _pipeline() -> OmniDiffusionS2IPipeline:
    pipeline = OmniDiffusionS2IPipeline.__new__(OmniDiffusionS2IPipeline)
    torch.nn.Module.__init__(pipeline)
    return pipeline


def test_pipeline_declares_only_s2i_serving_extras() -> None:
    assert OmniDiffusionS2IPipeline.support_audio_input is True
    assert OmniDiffusionS2IPipeline.dummy_run_num_frames == 1
    assert OmniDiffusionS2IPipeline.EXTRA_BODY_PARAMS == frozenset({"audio_path", "audio_url"})


def test_pipeline_config_parses_defaults(tmp_path: Path) -> None:
    config = OmniDiffusionS2IPipelineConfig.from_model_config(_model_config(tmp_path))

    assert config.task == "S2I"
    assert config.steps == 260
    assert config.max_new_tokens == 260
    assert config.alg == "entropy-penalty"
    assert config.cfg == 2.0
    assert config.temperature == 0.0
    assert config.top_p == 0.9
    assert config.top_k is None
    assert config.add_boa_token == 0
    assert config.max_position_penalty == 2.0
    assert config.repeat_penalty == 1.2
    assert config.seed is None


def test_pipeline_config_parses_overrides(tmp_path: Path) -> None:
    config = OmniDiffusionS2IPipelineConfig.from_model_config(
        _model_config(
            tmp_path,
            attn_implementation="eager",
            steps="32",
            max_new_tokens="64",
            cfg="3.0",
            top_k="20",
            seed="42",
        )
    )

    assert config.attn_implementation == "eager"
    assert config.steps == 32
    assert config.max_new_tokens == 64
    assert config.cfg == 3.0
    assert config.top_k == 20
    assert config.seed == 42


def test_pipeline_config_rejects_non_mapping() -> None:
    with pytest.raises(TypeError, match="mapping"):
        OmniDiffusionS2IPipelineConfig.from_model_config("invalid")  # type: ignore[arg-type]


def test_pipeline_config_resolves_default_components() -> None:
    resolved_paths = iter(["/cache/magvit", "/cache/sensevoice"])
    with patch.object(s2i_module, "resolve_omni_diffusion_component_path", side_effect=resolved_paths) as resolve:
        config = OmniDiffusionS2IPipelineConfig.from_model_config({})

    assert config.image_tokenizer_path == "/cache/magvit"
    assert config.sensevoice_path == "/cache/sensevoice"
    assert resolve.call_count == 2


def test_pipeline_config_rejects_missing_directory(tmp_path: Path) -> None:
    config = _model_config(tmp_path)
    config["image_tokenizer_path"] = str(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="existing local directory"):
        OmniDiffusionS2IPipelineConfig.from_model_config(config)


def test_prepare_text_audio_inputs_inserts_placeholder() -> None:
    pipeline = _pipeline()
    pipeline.device = torch.device("cpu")
    pipeline.tokenizer = MagicMock()
    pipeline.tokenizer.apply_chat_template.return_value = [10, 11]
    pipeline.tokenizer_base_data = MagicMock()
    pipeline.audio_tokenizer = MagicMock()
    pipeline.audio_tokenizer.prepare_contiguous_audio_inputs.return_value = (
        [10, 20, 11],
        [torch.zeros((4, 80))],
        [torch.zeros((2, 8), dtype=torch.long)],
    )

    input_ids, audios, audio_indices = pipeline._prepare_text_audio_inputs(
        "generate an image",
        torch.zeros(160),
        16000,
    )

    messages = pipeline.tokenizer.apply_chat_template.call_args.args[0]
    assert "<|audio|>" in messages[0]["content"]
    assert input_ids.tolist() == [[10, 20, 11]]
    assert len(audios) == 1
    assert len(audio_indices) == 1


def test_extract_image_codebook_ids_filters_and_offsets_tokens() -> None:
    pipeline = _pipeline()
    pipeline.tokenizer = MagicMock()
    pipeline.tokenizer.convert_tokens_to_ids.return_value = 1000
    generated = torch.tensor([999, 1000, 1002, 1000 + OMNI_DIFFUSION_IMAGE_CODEBOOK_SIZE, 42])

    result = pipeline._extract_image_codebook_ids(generated)

    pipeline.tokenizer.convert_tokens_to_ids.assert_called_once_with(OMNI_DIFFUSION_IMAGE_START_TOKEN)
    assert result.tolist() == [0, 2]


def test_load_weights_reports_pipeline_parameters() -> None:
    pipeline = _pipeline()
    pipeline.register_parameter("weight", torch.nn.Parameter(torch.ones(1)))
    assert pipeline.load_weights([]) == {"weight"}


def test_forward_generates_and_decodes_image(monkeypatch, tmp_path: Path) -> None:
    pipeline = _pipeline()
    pipeline.pipeline_config = OmniDiffusionS2IPipelineConfig.from_model_config(_model_config(tmp_path, seed=42))
    pipeline.model = MagicMock()
    pipeline.model.generation_config = SimpleNamespace()
    pipeline.model.config = SimpleNamespace()
    pipeline.tokenizer = MagicMock()
    pipeline.image_tokenizer = MagicMock()
    pipeline.image_tokenizer.decode.return_value = torch.zeros((1, 3, 4, 4))
    pipeline._prepare_text_audio_inputs = MagicMock(
        return_value=(
            torch.tensor([[10, 11]]),
            [torch.zeros((2, 80))],
            [torch.zeros((2, 6), dtype=torch.long)],
        )
    )
    generated = torch.tensor([[10, 11, 1000, 1002]])
    pipeline.model.generate.return_value = (generated, [])
    pipeline._extract_image_codebook_ids = MagicMock(return_value=torch.tensor([0, 2]))

    monkeypatch.setattr(s2i_module, "_get_audio_source", lambda _req: "input.wav")
    monkeypatch.setattr(
        s2i_module,
        "_decode_audio_source",
        lambda _source: (torch.zeros(160), 16000),
    )
    expected_image = Image.new("RGB", (4, 4))
    monkeypatch.setattr(
        s2i_module,
        "_image_tensor_to_pil",
        lambda _image: expected_image,
    )
    monkeypatch.setattr(s2i_module, "set_generation_seed", MagicMock())
    monkeypatch.setattr(
        s2i_module,
        "patch_legacy_dream_generation_config_validate",
        MagicMock(),
    )
    monkeypatch.setattr(
        s2i_module,
        "ensure_dream_generation_config_fields",
        MagicMock(),
    )
    request = DiffusionRequestBatch(
        [
            OmniDiffusionRequest(
                prompt="generate from speech",
                sampling_params=OmniDiffusionSamplingParams(
                    seed=1,
                    extra_args={"audio_path": "input.wav"},
                ),
                request_id="s2i-forward",
            )
        ]
    )

    result = pipeline.forward(request)

    assert result.output is expected_image
    s2i_module.set_generation_seed.assert_called_once_with(42)
    pipeline._prepare_text_audio_inputs.assert_called_once()
    pipeline.model.generate.assert_called_once()
    generate_kwargs = pipeline.model.generate.call_args.kwargs
    assert generate_kwargs["task"] == "S2I"
    assert generate_kwargs["steps"] == 260
    assert generate_kwargs["cfg"] == 2.0
    pipeline._extract_image_codebook_ids.assert_called_once()
    extracted = pipeline._extract_image_codebook_ids.call_args.args[0]
    assert torch.equal(extracted, torch.tensor([1000, 1002]))
    pipeline.image_tokenizer.decode.assert_called_once()


def test_dummy_forward_warms_model_and_image_decoder(monkeypatch, tmp_path: Path) -> None:
    pipeline = _pipeline()
    pipeline.device = torch.device("cpu")
    pipeline.pipeline_config = OmniDiffusionS2IPipelineConfig.from_model_config(_model_config(tmp_path, seed=42))
    pipeline.model = MagicMock()
    pipeline.model.generation_config = SimpleNamespace()
    pipeline.model.config = SimpleNamespace()
    pipeline.tokenizer = MagicMock()
    pipeline.image_tokenizer = MagicMock()
    pipeline.image_tokenizer.decode.return_value = torch.zeros((1, 3, 4, 4))
    pipeline._prepare_text_audio_inputs = MagicMock(
        return_value=(
            torch.tensor([[10, 11]]),
            [torch.zeros((2, 80))],
            [torch.zeros((2, 6), dtype=torch.long)],
        )
    )
    pipeline.model.generate.return_value = (torch.tensor([[10, 11, 1000]]), [])

    expected_image = Image.new("RGB", (4, 4))
    monkeypatch.setattr(s2i_module, "_image_tensor_to_pil", lambda _image: expected_image)
    monkeypatch.setattr(s2i_module, "_get_audio_source", MagicMock())
    monkeypatch.setattr(s2i_module, "_decode_audio_source", MagicMock())
    monkeypatch.setattr(s2i_module, "set_generation_seed", MagicMock())
    monkeypatch.setattr(s2i_module, "patch_legacy_dream_generation_config_validate", MagicMock())
    monkeypatch.setattr(s2i_module, "ensure_dream_generation_config_fields", MagicMock())

    request = DiffusionRequestBatch(
        [
            OmniDiffusionRequest(
                prompt={
                    "prompt": "dummy run",
                    "multi_modal_data": {"audio": torch.zeros(32000)},
                },
                sampling_params=OmniDiffusionSamplingParams(
                    seed=999,
                    num_inference_steps=1,
                    num_frames=1,
                ),
                request_id="dummy_req_id",
            )
        ]
    )

    result = pipeline.forward(request)

    assert result.output is expected_image
    s2i_module._get_audio_source.assert_not_called()
    s2i_module._decode_audio_source.assert_not_called()
    s2i_module.set_generation_seed.assert_called_once_with(42)
    pipeline._prepare_text_audio_inputs.assert_called_once()
    prepared_audio = pipeline._prepare_text_audio_inputs.call_args.args[1]
    assert prepared_audio.shape == (32000,)
    assert pipeline._prepare_text_audio_inputs.call_args.args[2] == 16000
    generate_kwargs = pipeline.model.generate.call_args.kwargs
    assert generate_kwargs["steps"] == 1
    decoded_tokens = pipeline.image_tokenizer.decode.call_args.args[0]
    assert decoded_tokens.shape == (256,)
    assert torch.count_nonzero(decoded_tokens).item() == 0
