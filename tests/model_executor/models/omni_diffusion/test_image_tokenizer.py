# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OmniDiffusionImageTokenizer encode/decode/preprocess logic."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_omni.model_executor.models.omni_diffusion.image_tokenizer import (
    _MAGVIT_IMAGE_TOKEN_COUNT,
    OmniDiffusionImageTokenizer,
)
from vllm_omni.model_executor.models.omni_diffusion.utils import (
    OmniDiffusionModelSpecialTokens,
    OmniDiffusionTokenizerBaseData,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _StubMagvit:
    """Minimal stub for MAGVITv2 so tests can exercise encode/decode paths."""

    def __init__(self, codebook_size: int = 8192) -> None:
        self.codebook_size = codebook_size
        self.dtype = torch.float32

    def get_code(self, images: torch.Tensor) -> torch.Tensor:
        # Return a deterministic sequence of codebook IDs per image.
        batch = images.shape[0]
        codes = (torch.arange(_MAGVIT_IMAGE_TOKEN_COUNT) % self.codebook_size).unsqueeze(0)
        return codes.repeat(batch, 1)

    def decode_code(self, image_tokens: torch.Tensor) -> torch.Tensor:
        # Return a normalized image tensor for each batch item.
        batch = image_tokens.shape[0]
        return torch.rand(batch, 3, 512, 512) * 2 - 1  # [-1, 1]

    def to(self, device: torch.device) -> "_StubMagvit":
        return self

    def eval(self) -> "_StubMagvit":
        return self

    def requires_grad_(self, flag: bool) -> "_StubMagvit":
        return self


def _make_tokenizer_base_data() -> OmniDiffusionTokenizerBaseData:
    """Build a TokenizerBaseData with deterministic token IDs for testing."""
    tok = MagicMock()

    def _encode(tokens, add_special_tokens):
        ids = []
        for t in tokens:
            for special in OmniDiffusionModelSpecialTokens:
                if special.value == t:
                    ids.append([100 + list(OmniDiffusionModelSpecialTokens).index(special)])
                    break
            else:
                ids.append([999])
        return SimpleNamespace(input_ids=ids)

    tok.side_effect = _encode
    return OmniDiffusionTokenizerBaseData(tok)


def _make_stub_tokenizer() -> MagicMock:
    """Make a tokenizer stub that returns deterministic token IDs for any input."""
    tok = MagicMock()
    dynamic_token_ids: dict[str, int] = {}

    def _encode(tokens, add_special_tokens):
        del add_special_tokens
        ids = []
        for t in tokens:
            for special in OmniDiffusionModelSpecialTokens:
                if special.value == t:
                    ids.append([100 + list(OmniDiffusionModelSpecialTokens).index(special)])
                    break
            else:
                token_id = dynamic_token_ids.setdefault(t, len(dynamic_token_ids) + 500)
                ids.append([token_id])
        return SimpleNamespace(input_ids=ids)

    tok.side_effect = _encode
    return tok


# ---------------------------------------------------------------------------
# Image tokenizer: construction and preprocess
# ---------------------------------------------------------------------------


class TestImageTokenizerConstruction:
    def test_init_loads_model_and_sets_device(self) -> None:
        with patch.object(OmniDiffusionImageTokenizer, "_load_model", return_value=_StubMagvit()):
            tok = OmniDiffusionImageTokenizer(model_path="/fake/magvit", device=torch.device("cpu"))
            assert tok.device == torch.device("cpu")
            assert tok.tokenizer is not None

    def test_preprocess_accepts_chw_tensor(self) -> None:
        with patch.object(OmniDiffusionImageTokenizer, "_load_model", return_value=_StubMagvit()):
            tok = OmniDiffusionImageTokenizer(model_path="/fake/magvit", device=torch.device("cpu"))
            image = torch.rand(3, 128, 256)
            result = tok._preprocess(image)
            assert result.ndim == 4
            assert result.shape[0] == 1
            assert result.shape[1] == 3

    def test_preprocess_accepts_bchw_tensor(self) -> None:
        with patch.object(OmniDiffusionImageTokenizer, "_load_model", return_value=_StubMagvit()):
            tok = OmniDiffusionImageTokenizer(model_path="/fake/magvit", device=torch.device("cpu"))
            image = torch.rand(2, 3, 128, 256)
            result = tok._preprocess(image)
            assert result.ndim == 4
            assert result.shape[0] == 2

    def test_preprocess_rejects_wrong_channel_count(self) -> None:
        with patch.object(OmniDiffusionImageTokenizer, "_load_model", return_value=_StubMagvit()):
            tok = OmniDiffusionImageTokenizer(model_path="/fake/magvit", device=torch.device("cpu"))
            with pytest.raises(ValueError, match="RGB"):
                tok._preprocess(torch.rand(1, 128, 256))

    def test_preprocess_rejects_wrong_ndim(self) -> None:
        with patch.object(OmniDiffusionImageTokenizer, "_load_model", return_value=_StubMagvit()):
            tok = OmniDiffusionImageTokenizer(model_path="/fake/magvit", device=torch.device("cpu"))
            with pytest.raises(ValueError, match="CHW or BCHW"):
                tok._preprocess(torch.rand(3, 3, 128, 256, 3))


# ---------------------------------------------------------------------------
# Image tokenizer: encode
# ---------------------------------------------------------------------------


class TestImageTokenizerEncode:
    def test_encode_returns_codebook_ids(self) -> None:
        with patch.object(OmniDiffusionImageTokenizer, "_load_model", return_value=_StubMagvit()):
            tok = OmniDiffusionImageTokenizer(model_path="/fake/magvit", device=torch.device("cpu"))
            # 3x512x512 RGB image
            image = torch.rand(3, 512, 512)
            codes = tok.encode(image)
            assert codes.ndim == 2  # [B, num_tokens]
            assert codes.shape[0] == 1
            assert codes.shape[1] == _MAGVIT_IMAGE_TOKEN_COUNT

    def test_encode_batch_returns_per_image_codes(self) -> None:
        with patch.object(OmniDiffusionImageTokenizer, "_load_model", return_value=_StubMagvit()):
            tok = OmniDiffusionImageTokenizer(model_path="/fake/magvit", device=torch.device("cpu"))
            images = torch.rand(3, 3, 512, 512)
            codes = tok.encode(images)
            assert codes.shape[0] == 3
            assert codes.shape[1] == _MAGVIT_IMAGE_TOKEN_COUNT


# ---------------------------------------------------------------------------
# Image tokenizer: decode
# ---------------------------------------------------------------------------


class TestImageTokenizerDecode:
    def test_decode_returns_normalized_image(self) -> None:
        with patch.object(OmniDiffusionImageTokenizer, "_load_model", return_value=_StubMagvit()):
            tok = OmniDiffusionImageTokenizer(model_path="/fake/magvit", device=torch.device("cpu"))
            tokens = torch.randint(0, 8192, (_MAGVIT_IMAGE_TOKEN_COUNT,))
            image = tok.decode(tokens)
            assert image.ndim == 4  # [B, C, H, W]
            assert image.shape[0] == 1
            assert image.shape[1] == 3
            # Values must be in [0, 1] range.
            assert image.min() >= 0.0
            assert image.max() <= 1.0

    def test_decode_accepts_batched_tokens(self) -> None:
        with patch.object(OmniDiffusionImageTokenizer, "_load_model", return_value=_StubMagvit()):
            tok = OmniDiffusionImageTokenizer(model_path="/fake/magvit", device=torch.device("cpu"))
            tokens = torch.randint(0, 8192, (2, _MAGVIT_IMAGE_TOKEN_COUNT))
            image = tok.decode(tokens)
            assert image.shape[0] == 2

    def test_decode_pads_short_token_sequences(self) -> None:
        with patch.object(OmniDiffusionImageTokenizer, "_load_model", return_value=_StubMagvit()):
            tok = OmniDiffusionImageTokenizer(model_path="/fake/magvit", device=torch.device("cpu"))
            # Fewer tokens than the MAGVIT grid expects.
            tokens = torch.randint(0, 8192, (128,))
            image = tok.decode(tokens)
            assert image.shape[0] == 1
            assert image.shape[1] == 3

    def test_decode_rejects_empty_tokens(self) -> None:
        with patch.object(OmniDiffusionImageTokenizer, "_load_model", return_value=_StubMagvit()):
            tok = OmniDiffusionImageTokenizer(model_path="/fake/magvit", device=torch.device("cpu"))
            with pytest.raises(ValueError, match="empty"):
                tok.decode(torch.zeros(0, dtype=torch.int64))

    def test_decode_clamps_out_of_range_ids(self) -> None:
        with patch.object(OmniDiffusionImageTokenizer, "_load_model", return_value=_StubMagvit()):
            tok = OmniDiffusionImageTokenizer(model_path="/fake/magvit", device=torch.device("cpu"))
            # Token IDs that exceed the valid codebook range.
            tokens = torch.full((_MAGVIT_IMAGE_TOKEN_COUNT,), 99999, dtype=torch.int64)
            image = tok.decode(tokens)
            assert image.min() >= 0.0
            assert image.max() <= 1.0


# ---------------------------------------------------------------------------
# Image tokenizer: prepare_image_token_inputs
# ---------------------------------------------------------------------------


class TestPrepareImageTokenInputs:
    def test_replaces_single_placeholder(self) -> None:
        with patch.object(OmniDiffusionImageTokenizer, "_load_model", return_value=_StubMagvit()):
            tok = OmniDiffusionImageTokenizer(model_path="/fake/magvit", device=torch.device("cpu"))
            base_data = _make_tokenizer_base_data()
            tokenizer = _make_stub_tokenizer()
            img_tag_id = base_data.get_token_id(OmniDiffusionModelSpecialTokens.IMG_TAG)

            input_ids = [10, 20, img_tag_id, 30, 40]
            image_tensor = torch.rand(3, 512, 512)

            result = tok.prepare_image_token_inputs(
                input_ids=input_ids,
                images=image_tensor,
                tokenizer=tokenizer,
                tokenizer_base_data=base_data,
            )
            assert isinstance(result, list)
            assert len(result) > len(input_ids)
            # The placeholder should be replaced — no IMG_TAG in result.
            assert img_tag_id not in result

    def test_replaces_multiple_placeholders(self) -> None:
        with patch.object(OmniDiffusionImageTokenizer, "_load_model", return_value=_StubMagvit()):
            tok = OmniDiffusionImageTokenizer(model_path="/fake/magvit", device=torch.device("cpu"))
            base_data = _make_tokenizer_base_data()
            tokenizer = _make_stub_tokenizer()
            img_tag_id = base_data.get_token_id(OmniDiffusionModelSpecialTokens.IMG_TAG)

            input_ids = [img_tag_id, 20, img_tag_id, 40]
            images = torch.rand(2, 3, 512, 512)

            result = tok.prepare_image_token_inputs(
                input_ids=input_ids,
                images=images,
                tokenizer=tokenizer,
                tokenizer_base_data=base_data,
            )
            assert img_tag_id not in result

    def test_accepts_list_of_image_tensors(self) -> None:
        with patch.object(OmniDiffusionImageTokenizer, "_load_model", return_value=_StubMagvit()):
            tok = OmniDiffusionImageTokenizer(model_path="/fake/magvit", device=torch.device("cpu"))
            base_data = _make_tokenizer_base_data()
            tokenizer = _make_stub_tokenizer()
            img_tag_id = base_data.get_token_id(OmniDiffusionModelSpecialTokens.IMG_TAG)

            input_ids = [img_tag_id, 20]
            images = [torch.rand(3, 512, 512)]

            result = tok.prepare_image_token_inputs(
                input_ids=input_ids,
                images=images,
                tokenizer=tokenizer,
                tokenizer_base_data=base_data,
            )
            assert img_tag_id not in result

    def test_raises_on_count_mismatch(self) -> None:
        with patch.object(OmniDiffusionImageTokenizer, "_load_model", return_value=_StubMagvit()):
            tok = OmniDiffusionImageTokenizer(model_path="/fake/magvit", device=torch.device("cpu"))
            base_data = _make_tokenizer_base_data()
            tokenizer = _make_stub_tokenizer()
            img_tag_id = base_data.get_token_id(OmniDiffusionModelSpecialTokens.IMG_TAG)

            input_ids = [img_tag_id, img_tag_id]  # 2 placeholders
            images = torch.rand(1, 3, 512, 512)  # only 1 image

            with pytest.raises(ValueError, match="placeholder"):
                tok.prepare_image_token_inputs(
                    input_ids=input_ids,
                    images=images,
                    tokenizer=tokenizer,
                    tokenizer_base_data=base_data,
                )

    def test_raises_on_invalid_image_type(self) -> None:
        with patch.object(OmniDiffusionImageTokenizer, "_load_model", return_value=_StubMagvit()):
            tok = OmniDiffusionImageTokenizer(model_path="/fake/magvit", device=torch.device("cpu"))
            base_data = _make_tokenizer_base_data()
            tokenizer = _make_stub_tokenizer()
            img_tag_id = base_data.get_token_id(OmniDiffusionModelSpecialTokens.IMG_TAG)

            with pytest.raises(TypeError, match="tensor or sequence"):
                tok.prepare_image_token_inputs(
                    input_ids=[img_tag_id],
                    images="not_a_tensor",
                    tokenizer=tokenizer,
                    tokenizer_base_data=base_data,
                )
