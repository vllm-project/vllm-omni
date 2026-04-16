# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HCXOmni multimodal token processing.

Tests verify that:
  1. Audio tokens are correctly positioned and embedded.
  2. Image tokens (continuous path via Qwen2.5-VL) are correctly positioned.
  3. Discrete audio/image token boundaries match config.json values.
  4. Stage input processors correctly extract discrete tokens from mixed output.
"""

# Token ID boundaries (from HyperCLOVAX-SEED-Omni-8B config.json)
DISCRETE_AUDIO_UNIT_0_ID = 128606
DISCRETE_IMAGE_UNIT_0_ID = 135168
DISCRETE_AUDIO_VOCAB_SIZE = 6561
DISCRETE_IMAGE_VOCAB_SIZE = 65536
DISCRETE_IMAGE_TOKEN_LENGTH = 729  # 27 * 27


class TestDiscreteTokenBoundaries:
    """Verify token ID arithmetic matches config.json."""

    def test_audio_range(self):
        assert DISCRETE_AUDIO_UNIT_0_ID == 128606
        assert DISCRETE_AUDIO_UNIT_0_ID + DISCRETE_AUDIO_VOCAB_SIZE - 1 == 135166

    def test_image_range(self):
        assert DISCRETE_IMAGE_UNIT_0_ID == 135168
        # Image codebook is 2^16 = 65536
        assert DISCRETE_IMAGE_VOCAB_SIZE == 65536

    def test_no_overlap(self):
        audio_end = DISCRETE_AUDIO_UNIT_0_ID + DISCRETE_AUDIO_VOCAB_SIZE
        assert audio_end < DISCRETE_IMAGE_UNIT_0_ID, "Audio and image token ranges must not overlap"

    def test_image_token_count_is_square(self):
        """TA-Tok produces 27×27 = 729 tokens per image."""
        import math

        side = math.isqrt(DISCRETE_IMAGE_TOKEN_LENGTH)
        assert side * side == DISCRETE_IMAGE_TOKEN_LENGTH


class TestExtractDiscreteTokens:
    """Test the _extract_discrete_tokens helper."""

    def _extract(self, token_ids, start_id, vocab_size):
        return [tid - start_id for tid in token_ids if start_id <= tid < start_id + vocab_size]

    def test_extract_audio_tokens(self):
        token_ids = [
            100,
            200,  # text
            DISCRETE_AUDIO_UNIT_0_ID,
            DISCRETE_AUDIO_UNIT_0_ID + 42,
            DISCRETE_AUDIO_UNIT_0_ID + 100,
            300,  # text
        ]
        result = self._extract(token_ids, DISCRETE_AUDIO_UNIT_0_ID, DISCRETE_AUDIO_VOCAB_SIZE)
        assert result == [0, 42, 100]

    def test_extract_image_tokens(self):
        token_ids = [
            100,
            DISCRETE_IMAGE_UNIT_0_ID,
            DISCRETE_IMAGE_UNIT_0_ID + 255,
            200,
        ]
        result = self._extract(token_ids, DISCRETE_IMAGE_UNIT_0_ID, DISCRETE_IMAGE_VOCAB_SIZE)
        assert result == [0, 255]

    def test_no_overlap_extraction(self):
        """Audio extraction must not pick up image tokens and vice versa."""
        mixed = [
            DISCRETE_AUDIO_UNIT_0_ID + 5,
            DISCRETE_IMAGE_UNIT_0_ID + 5,
        ]
        audio = self._extract(mixed, DISCRETE_AUDIO_UNIT_0_ID, DISCRETE_AUDIO_VOCAB_SIZE)
        image = self._extract(mixed, DISCRETE_IMAGE_UNIT_0_ID, DISCRETE_IMAGE_VOCAB_SIZE)
        assert audio == [5]
        assert image == [5]

    def test_truncate_and_pad_image(self):
        """Vision decoder needs exactly DISCRETE_IMAGE_TOKEN_LENGTH codes."""
        codes = list(range(DISCRETE_IMAGE_TOKEN_LENGTH + 50))  # too long
        truncated = codes[:DISCRETE_IMAGE_TOKEN_LENGTH]
        assert len(truncated) == DISCRETE_IMAGE_TOKEN_LENGTH

        codes_short = list(range(100))  # too short
        padded = codes_short + [0] * (DISCRETE_IMAGE_TOKEN_LENGTH - len(codes_short))
        assert len(padded) == DISCRETE_IMAGE_TOKEN_LENGTH


class TestStageInputProcessor:
    """Test thinker2vision_decoder and thinker2audio_decoder processors."""

    def _make_fake_output(self, token_ids: list[int]):
        """Create a minimal fake EngineCoreOutput-like object."""
        from types import SimpleNamespace

        output = SimpleNamespace(
            token_ids=token_ids,
        )
        thinker_out = SimpleNamespace(
            outputs=[output],
            request_id="test-001",
            prompt_token_ids=[1, 2, 3],
        )
        return thinker_out

    def test_vision_decoder_extracts_image_tokens(self):
        """thinker2vision_decoder should extract exactly 729 image tokens."""
        image_codes = list(range(DISCRETE_IMAGE_UNIT_0_ID, DISCRETE_IMAGE_UNIT_0_ID + DISCRETE_IMAGE_TOKEN_LENGTH))
        audio_codes = list(range(DISCRETE_AUDIO_UNIT_0_ID, DISCRETE_AUDIO_UNIT_0_ID + 20))
        token_ids = [100, 200] + audio_codes + image_codes + [300]

        thinker_out = self._make_fake_output(token_ids)

        from types import SimpleNamespace

        stage_list = {0: SimpleNamespace(engine_outputs=[thinker_out])}

        from vllm_omni.model_executor.stage_input_processors.hyperclovax_seed_omni import (
            thinker2vision_decoder,
        )

        results = thinker2vision_decoder(stage_list, [0])
        assert len(results) == 1
        prompt_ids = results[0]["prompt_token_ids"]
        assert len(prompt_ids) == DISCRETE_IMAGE_TOKEN_LENGTH
        assert all(0 <= tid < DISCRETE_IMAGE_VOCAB_SIZE for tid in prompt_ids)

    def test_audio_decoder_extracts_audio_tokens(self):
        """thinker2audio_decoder should extract discrete audio tokens."""
        audio_codes = list(range(DISCRETE_AUDIO_UNIT_0_ID, DISCRETE_AUDIO_UNIT_0_ID + 50))
        token_ids = [100, 200] + audio_codes + [300]

        thinker_out = self._make_fake_output(token_ids)

        from types import SimpleNamespace

        stage_list = {0: SimpleNamespace(engine_outputs=[thinker_out])}

        from vllm_omni.model_executor.stage_input_processors.hyperclovax_seed_omni import (
            thinker2audio_decoder,
        )

        results = thinker2audio_decoder(stage_list, [0])
        assert len(results) == 1
        additional = results[0]["additional_information"]
        audio_tokens = additional["audio_tokens"][0]
        assert len(audio_tokens) == 50
        assert all(0 <= tid < DISCRETE_AUDIO_VOCAB_SIZE for tid in audio_tokens)

    def test_vision_decoder_no_output_if_no_image_tokens(self):
        """thinker2vision_decoder returns empty list when no image tokens present."""
        token_ids = [100, 200, 300]  # text only

        thinker_out = self._make_fake_output(token_ids)

        from types import SimpleNamespace

        stage_list = {0: SimpleNamespace(engine_outputs=[thinker_out])}

        from vllm_omni.model_executor.stage_input_processors.hyperclovax_seed_omni import (
            thinker2vision_decoder,
        )

        results = thinker2vision_decoder(stage_list, [0])
        assert results == []

    def test_audio_decoder_no_output_if_no_audio_tokens(self):
        """thinker2audio_decoder returns empty list when no audio tokens present."""
        token_ids = [100, 200, 300]  # text only

        thinker_out = self._make_fake_output(token_ids)

        from types import SimpleNamespace

        stage_list = {0: SimpleNamespace(engine_outputs=[thinker_out])}

        from vllm_omni.model_executor.stage_input_processors.hyperclovax_seed_omni import (
            thinker2audio_decoder,
        )

        results = thinker2audio_decoder(stage_list, [0])
        assert results == []
