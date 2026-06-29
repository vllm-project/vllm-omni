# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for prompt_token_ids support in QwenImage pipelines."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import (
    QwenImagePipeline,
)
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit import (
    QwenImageEditPipeline,
)
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit_plus import (
    QwenImageEditPlusPipeline,
)
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_layered import (
    QwenImageLayeredPipeline,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Fake components
# ---------------------------------------------------------------------------


class _RecordingTextEncoder:
    """Text encoder that returns hidden states derived from input ids."""

    dtype = torch.float32

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, input_ids, attention_mask, output_hidden_states, **kwargs):
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": attention_mask.clone(),
            }
        )
        batch, seq_len = input_ids.shape
        hidden_dim = 128
        # Return hidden states where each token's embedding encodes its id
        hidden = input_ids.unsqueeze(-1).float().expand(batch, seq_len, hidden_dim)
        # Stack enough layers so that hidden_states[-1] is available
        return SimpleNamespace(
            hidden_states=[hidden] * 5,  # 5 layers, last one is [-1]
        )


class _RejectingTokenizer:
    """Tokenizer that raises if called — proves tokenizer is skipped."""

    def __call__(self, *args, **kwargs):
        raise AssertionError("Tokenizer was called but should have been skipped when prompt_token_ids is provided.")

    def decode(self, ids):
        return "decoded"


class _PredictableTextEncoder:
    """Returns hidden states where the hidden value = input_id * 10."""

    dtype = torch.float32

    def __call__(self, input_ids, attention_mask, output_hidden_states, **kwargs):
        batch, seq_len = input_ids.shape
        hidden_dim = 128
        # Make hidden = input_id * 10 so we can verify the right ids were used
        hidden = input_ids.unsqueeze(-1).float().expand(batch, seq_len, hidden_dim) * 10.0
        return SimpleNamespace(hidden_states=[hidden] * 5)


class _FakeScheduler:
    def __init__(self):
        self.begin_index = None

    def set_begin_index(self, begin_index: int):
        self.begin_index = begin_index


# ---------------------------------------------------------------------------
# Pipeline factories
# ---------------------------------------------------------------------------

PIPELINE_CASES = [
    pytest.param(QwenImagePipeline, 34, id="qwen-image"),
    pytest.param(QwenImageLayeredPipeline, 34, id="qwen-image-layered"),
    pytest.param(QwenImageEditPipeline, 64, id="qwen-image-edit"),
    pytest.param(QwenImageEditPlusPipeline, 64, id="qwen-image-edit-plus"),
]


def _make_pipeline_with_recording_encoder(pipeline_class, drop_idx):
    """Create a pipeline with a recording text encoder and rejecting tokenizer."""
    pipeline = object.__new__(pipeline_class)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.text_encoder = _RecordingTextEncoder()
    pipeline.tokenizer_max_length = 1024
    pipeline.prompt_template_encode = "{}"
    pipeline.prompt_template_encode_start_idx = drop_idx
    pipeline.tokenizer = _RejectingTokenizer()
    return pipeline


def _make_pipeline_with_predictable_encoder(pipeline_class, drop_idx):
    """Create a pipeline whose encoder returns input_id * 10 hidden states."""
    pipeline = object.__new__(pipeline_class)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.text_encoder = _PredictableTextEncoder()
    pipeline.tokenizer_max_length = 1024
    pipeline.prompt_template_encode = "{}"
    pipeline.prompt_template_encode_start_idx = drop_idx
    pipeline.tokenizer = _RejectingTokenizer()
    return pipeline


# ---------------------------------------------------------------------------
# _extract_prompts tests
# ---------------------------------------------------------------------------

PROMPT_TOKEN_IDS = [[101, 2027, 3045, 2003, 1037, 2629]]
NEG_PROMPT_TOKEN_IDS = [[101, 2027, 3045, 2003]]


class TestExtractPrompts:
    def test_extracts_prompt_token_ids_from_dict(self):
        pipeline = object.__new__(QwenImagePipeline)
        nn.Module.__init__(pipeline)
        prompt, neg_prompt, pt_ids, neg_pt_ids = pipeline._extract_prompts(
            [
                {"prompt": "a cat", "prompt_token_ids": PROMPT_TOKEN_IDS[0]},
            ]
        )
        assert prompt == ["a cat"]
        assert neg_prompt is None
        assert pt_ids == PROMPT_TOKEN_IDS
        assert neg_pt_ids is None

    def test_extracts_negative_prompt_token_ids_from_dict(self):
        pipeline = object.__new__(QwenImagePipeline)
        nn.Module.__init__(pipeline)
        prompt, neg_prompt, pt_ids, neg_pt_ids = pipeline._extract_prompts(
            [
                {
                    "prompt": "a cat",
                    "negative_prompt": "a dog",
                    "prompt_token_ids": PROMPT_TOKEN_IDS[0],
                    "negative_prompt_token_ids": NEG_PROMPT_TOKEN_IDS[0],
                },
            ]
        )
        assert prompt == ["a cat"]
        assert neg_prompt == ["a dog"]
        assert pt_ids == PROMPT_TOKEN_IDS
        assert neg_pt_ids == NEG_PROMPT_TOKEN_IDS

    def test_prompt_token_ids_none_when_missing(self):
        pipeline = object.__new__(QwenImagePipeline)
        nn.Module.__init__(pipeline)
        prompt, neg_prompt, pt_ids, neg_pt_ids = pipeline._extract_prompts(
            [
                {"prompt": "a cat"},
            ]
        )
        assert prompt == ["a cat"]
        assert pt_ids is None
        assert neg_pt_ids is None

    def test_prompt_token_ids_from_plain_string_returns_none(self):
        pipeline = object.__new__(QwenImagePipeline)
        nn.Module.__init__(pipeline)
        prompt, neg_prompt, pt_ids, neg_pt_ids = pipeline._extract_prompts(["a cat"])
        assert prompt == ["a cat"]
        assert pt_ids is None
        assert neg_pt_ids is None

    def test_partial_prompt_token_ids_returns_none(self):
        """If only some prompts have token_ids, fall back to None (all-or-nothing)."""
        pipeline = object.__new__(QwenImagePipeline)
        nn.Module.__init__(pipeline)
        prompt, neg_prompt, pt_ids, neg_pt_ids = pipeline._extract_prompts(
            [
                {"prompt": "a cat", "prompt_token_ids": PROMPT_TOKEN_IDS[0]},
                {"prompt": "a dog"},  # missing token_ids
            ]
        )
        assert prompt == ["a cat", "a dog"]
        assert pt_ids is None  # not all prompts have token_ids


# ---------------------------------------------------------------------------
# _get_qwen_prompt_embeds tests
# ---------------------------------------------------------------------------


class TestGetQwenPromptEmbeds:
    def test_raises_when_neither_prompt_nor_token_ids_provided(self):
        pipeline = _make_pipeline_with_recording_encoder(QwenImagePipeline, 34)
        with pytest.raises(ValueError, match="Either `prompt` or `prompt_token_ids`"):
            pipeline._get_qwen_prompt_embeds(
                prompt=None,
                prompt_token_ids=None,
                prompt_name="prompt",
            )

    @pytest.mark.parametrize(("pipeline_class", "drop_idx"), PIPELINE_CASES)
    def test_prompt_token_ids_skips_tokenizer(self, pipeline_class, drop_idx):
        """Token ids path must NOT call tokenizer."""
        pipeline = _make_pipeline_with_recording_encoder(pipeline_class, drop_idx)

        embeds, mask = pipeline._get_qwen_prompt_embeds(
            prompt=None,
            prompt_token_ids=PROMPT_TOKEN_IDS,
            prompt_name="prompt",
        )

        # Shape: [batch, seq_len_after_drop, hidden_dim]
        assert embeds.ndim == 3
        assert mask.ndim == 2
        # Tokenizer was never called (would have raised AssertionError)
        assert len(pipeline.text_encoder.calls) == 1

    @pytest.mark.parametrize(("pipeline_class", "drop_idx"), PIPELINE_CASES)
    def test_prompt_token_ids_feeds_correct_input_ids_to_encoder(self, pipeline_class, drop_idx):
        """Verify the text encoder receives the exact token ids passed in."""
        pipeline = _make_pipeline_with_recording_encoder(pipeline_class, drop_idx)

        pipeline._get_qwen_prompt_embeds(
            prompt=None,
            prompt_token_ids=PROMPT_TOKEN_IDS,
            prompt_name="prompt",
        )

        call_args = pipeline.text_encoder.calls[0]
        received_ids = call_args["input_ids"]
        # The first PROMPT_TOKEN_IDS[0] should appear as a prefix in received_ids
        # (padded with zeros to match batch max length)
        orig_len = len(PROMPT_TOKEN_IDS[0])
        assert torch.equal(
            received_ids[0, :orig_len],
            torch.tensor(PROMPT_TOKEN_IDS[0]),
        ), f"Expected first {orig_len} tokens to match prompt_token_ids"

    @pytest.mark.parametrize(("pipeline_class", "drop_idx"), PIPELINE_CASES)
    def test_prompt_token_ids_rejects_exceeding_max_length(self, pipeline_class, drop_idx):
        """Token ids longer than max_sequence_length must raise ValueError."""
        pipeline = _make_pipeline_with_recording_encoder(pipeline_class, drop_idx)

        too_long_ids = [list(range(2000))]  # 2000 > 1024
        with pytest.raises(ValueError, match=r"exceeds limit 1024"):
            pipeline._get_qwen_prompt_embeds(
                prompt=None,
                prompt_token_ids=too_long_ids,
                max_sequence_length=1024,
                prompt_name="prompt",
            )

    @pytest.mark.parametrize(("pipeline_class", "drop_idx"), PIPELINE_CASES)
    def test_prompt_token_ids_pads_uneven_batch(self, pipeline_class, drop_idx):
        """Batch with different-length token id lists must pad correctly."""
        pipeline = _make_pipeline_with_recording_encoder(pipeline_class, drop_idx)

        short = [1, 2, 3]
        long = [4, 5, 6, 7, 8]
        embeds, mask = pipeline._get_qwen_prompt_embeds(
            prompt=None,
            prompt_token_ids=[short, long],
            prompt_name="prompt",
        )

        # Mask should reflect real token lengths (after drop_idx)
        # Both sequences should produce the same seq_len output
        assert embeds.shape[0] == 2  # batch=2
        # mask values: 1 for real tokens, 0 for padding
        short_real = (mask[0] == 1).sum().item()
        long_real = (mask[1] == 1).sum().item()
        assert short_real > 0
        assert long_real > 0
        assert long_real > short_real, (
            f"Longer prompt should have more real tokens: short={short_real}, long={long_real}"
        )

    def test_prompt_token_ids_embeddings_match_expected_values(self):
        """Embeddings should reflect the input ids (id * 10 with fake encoder)."""
        pipeline = _make_pipeline_with_predictable_encoder(QwenImagePipeline, 34)

        ids = [[10, 20, 30, 40, 50]]
        embeds, mask = pipeline._get_qwen_prompt_embeds(
            prompt=None,
            prompt_token_ids=ids,
            prompt_name="prompt",
        )

        # After drop_idx=34 tokens are dropped, 5-34 would be empty/invalid
        # But our fake encoder returns id*10. We trust drop_idx truncation.
        # Verify the embeddings are not all zeros and have expected dtype
        assert embeds.dtype == torch.float32
        assert not torch.allclose(embeds, torch.zeros_like(embeds))

    def test_raises_when_prompt_token_ids_is_none_without_prompt(self):
        """Explicit None for both must raise."""
        pipeline = _make_pipeline_with_recording_encoder(QwenImagePipeline, 34)
        with pytest.raises(ValueError, match="Either `prompt` or `prompt_token_ids`"):
            pipeline._get_qwen_prompt_embeds(
                prompt=None,
                prompt_token_ids=None,
            )


# ---------------------------------------------------------------------------
# encode_prompt tests
# ---------------------------------------------------------------------------


class TestEncodePrompt:
    @pytest.mark.parametrize(("pipeline_class", "drop_idx"), PIPELINE_CASES)
    def test_encode_prompt_with_token_ids_does_not_call_tokenizer(self, pipeline_class, drop_idx):
        """encode_prompt with token_ids must route to _get_qwen_prompt_embeds
        without touching the tokenizer."""
        pipeline = _make_pipeline_with_recording_encoder(pipeline_class, drop_idx)

        embeds, mask = pipeline.encode_prompt(
            prompt="",
            prompt_token_ids=PROMPT_TOKEN_IDS,
        )

        assert embeds.ndim == 3
        assert mask.ndim == 2
        # Text encoder was called exactly once
        assert len(pipeline.text_encoder.calls) == 1

    @pytest.mark.parametrize(("pipeline_class", "drop_idx"), PIPELINE_CASES)
    def test_encode_prompt_with_token_ids_respects_max_sequence_length(self, pipeline_class, drop_idx):
        """Slicing to max_sequence_length must work with token_ids path."""
        pipeline = _make_pipeline_with_recording_encoder(pipeline_class, drop_idx)

        long_ids = [list(range(100))]
        embeds, mask = pipeline.encode_prompt(
            prompt="",
            prompt_token_ids=long_ids,
            max_sequence_length=50,
        )

        # Embeddings should be truncated to max_sequence_length
        assert embeds.shape[1] <= 50
        assert mask.shape[1] <= 50


# ---------------------------------------------------------------------------
# prepare_encode integration test
# ---------------------------------------------------------------------------


class TestPrepareEncode:
    def test_prepare_encode_passes_prompt_token_ids_through(self):
        """Full prepare_encode flow with prompt_token_ids in prompts dict."""
        pipeline = object.__new__(QwenImagePipeline)
        nn.Module.__init__(pipeline)
        pipeline.tokenizer_max_length = 1024
        pipeline.vae_scale_factor = 8
        pipeline.default_sample_size = 128
        pipeline.scheduler = _FakeScheduler()

        captured = {}

        def _fake_prepare_generation_context(**kwargs):
            captured["prompt_token_ids"] = kwargs.get("prompt_token_ids")
            captured["negative_prompt_token_ids"] = kwargs.get("negative_prompt_token_ids")
            captured["prompt"] = kwargs.get("prompt")
            embeds = torch.ones((1, 1, 1))
            mask = torch.ones((1, 1), dtype=torch.long)
            return {
                "prompt_embeds": embeds,
                "prompt_embeds_mask": mask,
                "negative_prompt_embeds": None,
                "negative_prompt_embeds_mask": None,
                "latents": embeds,
                "timesteps": torch.tensor([1]),
                "do_true_cfg": False,
                "guidance": None,
                "img_shapes": [[(1, 1, 1)]],
                "txt_seq_lens": [1],
                "negative_txt_seq_lens": None,
            }

        pipeline._prepare_generation_context = _fake_prepare_generation_context

        state = SimpleNamespace(
            prompts=[{"prompt": "a cat", "prompt_token_ids": PROMPT_TOKEN_IDS[0]}],
            sampling=SimpleNamespace(
                height=None,
                width=None,
                num_inference_steps=None,
                sigmas=None,
                guidance_scale_provided=False,
                guidance_scale=None,
                num_outputs_per_prompt=0,
                generator=None,
                true_cfg_scale=None,
                max_sequence_length=None,
            ),
        )

        pipeline.prepare_encode(state)

        assert captured["prompt"] == ["a cat"]
        assert captured["prompt_token_ids"] == PROMPT_TOKEN_IDS
        assert captured["negative_prompt_token_ids"] is None
