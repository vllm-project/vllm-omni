# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Omni-Diffusion multimodal input conversion."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image as PILImage

from vllm_omni.model_executor.models.omni_diffusion.chat_template import (
    OMNI_DIFFUSION_CHAT_TEMPLATE,
    normalize_chat_template_token_ids,
)
from vllm_omni.model_executor.models.omni_diffusion.omni_diffusion import (
    OmniDiffusionDummyInputsBuilder,
    OmniDiffusionMultiModalProcessor,
)
from vllm_omni.model_executor.models.omni_diffusion.utils import (
    OMNI_DIFFUSION_INPUT_SAMPLE_RATE,
    OmniDiffusionModelSpecialTokens,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([1, 2, 3], [1, 2, 3]),
        ({"input_ids": [4, 5]}, [4, 5]),
        (SimpleNamespace(input_ids=[6, 7]), [6, 7]),
        (torch.tensor([8, 9]), [8, 9]),
        ([[10, 11]], [10, 11]),
    ],
)
def test_normalize_chat_template_token_ids(value: object, expected: list[int]) -> None:
    assert normalize_chat_template_token_ids(value) == expected


def test_normalize_chat_template_token_ids_rejects_invalid_batches() -> None:
    with pytest.raises(ValueError, match="input_ids"):
        normalize_chat_template_token_ids({"attention_mask": [1]})
    with pytest.raises(ValueError, match="single token sequence"):
        normalize_chat_template_token_ids([[1], [2]])


def test_encode_prompt_applies_official_chat_template() -> None:
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = {"input_ids": [[1, 2, 3]]}

    result = OmniDiffusionMultiModalProcessor._encode_prompt_to_token_ids(
        "describe this image",
        tokenizer,
    )

    assert result == [1, 2, 3]
    tokenizer.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": "describe this image"}],
        add_generation_prompt=True,
        tokenize=True,
    )


def test_encode_rendered_or_tokenized_prompt() -> None:
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [4, 5]
    rendered = "<|im_start|>user\nhello<|im_end|>"

    assert OmniDiffusionMultiModalProcessor._encode_prompt_to_token_ids(rendered, tokenizer) == [4, 5]
    assert OmniDiffusionMultiModalProcessor._encode_prompt_to_token_ids([6, 7], tokenizer) == [6, 7]


def test_ensure_official_chat_template() -> None:
    tokenizer = MagicMock()
    OmniDiffusionMultiModalProcessor._ensure_official_chat_template(tokenizer)
    assert tokenizer.chat_template == OMNI_DIFFUSION_CHAT_TEMPLATE


@pytest.mark.parametrize(
    ("prompt", "counts", "expected"),
    [
        ("hello", {}, "hello"),
        ("hello", {"audio": 1}, "hello\n<|audio|>"),
        ("hello\n<|audio|>", {"audio": 1}, "hello\n<|audio|>"),
        ([1, 2], {"audio": 1}, [1, 2]),
    ],
)
def test_ensure_audio_placeholder(
    prompt: str | list[int],
    counts: dict[str, int],
    expected: str | list[int],
) -> None:
    assert OmniDiffusionMultiModalProcessor._ensure_audio_placeholder(prompt, counts) == expected


def test_ensure_audio_placeholder_stays_in_user_turn() -> None:
    prompt = "question<|im_end|>\n<|im_start|>assistant\n"
    result = OmniDiffusionMultiModalProcessor._ensure_audio_placeholder(prompt, {"audio": 1})
    assert isinstance(result, str)
    assert result.index(OmniDiffusionModelSpecialTokens.AUD_TAG.value) < result.index("<|im_start|>assistant")


@pytest.mark.parametrize(
    ("tokenizer", "expected"),
    [
        (SimpleNamespace(bos_token_id=1, eos_token_id=2, pad_token_id=3), [1]),
        (SimpleNamespace(bos_token_id=None, eos_token_id=2, pad_token_id=3), [2]),
        (SimpleNamespace(bos_token_id=None, eos_token_id=None, pad_token_id=3), [3]),
        (None, [0]),
    ],
)
def test_ensure_non_empty_prompt_ids(tokenizer: object, expected: list[int]) -> None:
    assert OmniDiffusionMultiModalProcessor._ensure_non_empty_prompt_ids([], tokenizer) == expected
    assert OmniDiffusionMultiModalProcessor._ensure_non_empty_prompt_ids([9], tokenizer) == [9]


@pytest.mark.parametrize(
    ("haystack", "needle", "start", "expected"),
    [
        ([1, 2, 3], [1, 2], 0, 0),
        ([1, 2, 1, 2], [1, 2], 1, 2),
        ([1, 2], [3], 0, None),
        ([1, 2], [], 0, None),
    ],
)
def test_find_subsequence(
    haystack: list[int],
    needle: list[int],
    start: int,
    expected: int | None,
) -> None:
    assert OmniDiffusionMultiModalProcessor._find_subsequence(haystack, needle, start) == expected


@pytest.mark.parametrize(
    ("image", "expected_shape"),
    [
        (torch.rand(3, 4, 5), (3, 4, 5)),
        (torch.rand(6, 5, 3), (3, 6, 5)),
        (torch.rand(1, 4, 5), (3, 4, 5)),
        (torch.rand(4, 4, 5), (3, 4, 5)),
        (np.zeros((4, 5, 3), dtype=np.uint8), (3, 4, 5)),
        (PILImage.new("RGB", (5, 4)), (3, 4, 5)),
    ],
)
def test_image_to_chw_float_tensor(image: object, expected_shape: tuple[int, ...]) -> None:
    result = OmniDiffusionMultiModalProcessor._image_to_chw_float_tensor(image)
    assert result.shape == expected_shape
    assert result.dtype == torch.float32
    assert result.is_contiguous()


def test_image_to_chw_float_tensor_normalizes_uint8() -> None:
    result = OmniDiffusionMultiModalProcessor._image_to_chw_float_tensor(torch.full((3, 2, 2), 255, dtype=torch.uint8))
    torch.testing.assert_close(result, torch.ones_like(result))


def test_image_conversion_rejects_invalid_input() -> None:
    with pytest.raises(TypeError, match="Unsupported image"):
        OmniDiffusionMultiModalProcessor._image_to_chw_float_tensor("image")
    with pytest.raises(ValueError, match="3D image"):
        OmniDiffusionMultiModalProcessor._image_to_chw_float_tensor(torch.zeros(1))


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        (PILImage.new("RGB", (5, 4)), [4, 5]),
        (torch.zeros(3, 4, 5), [4, 5]),
        (torch.zeros(6, 5, 3), [6, 5]),
        (np.zeros((4, 5, 3), dtype=np.uint8), [4, 5]),
        (np.zeros((4, 5), dtype=np.uint8), [4, 5]),
    ],
)
def test_image_size_hw_tensor(image: object, expected: list[int]) -> None:
    assert OmniDiffusionMultiModalProcessor._image_size_hw_tensor(image).tolist() == expected


@pytest.mark.parametrize(
    "audio",
    [torch.zeros(16), torch.zeros(2, 16), np.zeros(16, dtype=np.float32)],
)
def test_audio_to_float_tensor(audio: object) -> None:
    result = OmniDiffusionMultiModalProcessor._audio_to_float_tensor(audio)
    assert result.dtype == torch.float32
    assert result.is_contiguous()


def test_audio_tuple_preserves_sample_rate_metadata() -> None:
    audio = (torch.zeros(16), 44100)
    assert OmniDiffusionMultiModalProcessor._audio_to_float_tensor(audio).shape == (16,)
    assert OmniDiffusionMultiModalProcessor._audio_sample_rate_tensor(audio).item() == 44100
    assert (
        OmniDiffusionMultiModalProcessor._audio_sample_rate_tensor(torch.zeros(16)).item()
        == OMNI_DIFFUSION_INPUT_SAMPLE_RATE
    )


def test_audio_to_float_tensor_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        OmniDiffusionMultiModalProcessor._audio_to_float_tensor(torch.zeros(1, 2, 3))


def test_dummy_inputs_include_requested_modalities() -> None:
    builder = OmniDiffusionDummyInputsBuilder(MagicMock())
    text = builder.get_dummy_text({"image": 1, "audio": 1})
    data = builder.get_dummy_mm_data(
        seq_len=16,
        mm_counts={"image": 1, "audio": 1},
    )

    assert "<|image|>" in text
    assert "<|audio|>" in text
    assert len(data["image"]) == 1
    assert len(data["audio"]) == 1
