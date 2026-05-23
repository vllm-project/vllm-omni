# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for FunCineForge utility functions."""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestConcatTextWithPromptIds:
    """Tests for concat_text_with_prompt_ids utility."""

    def test_basic_sequence_layout(self):
        """Test the expected sequence: [SOS, clue, text, type, timespk, TOS]."""
        from vllm_omni.model_executor.models.funcineforge.utils import (
            concat_text_with_prompt_ids,
        )

        text = torch.tensor([[10, 11, 12]], dtype=torch.long)
        clue = torch.tensor([[20, 21]], dtype=torch.long)

        result, length = concat_text_with_prompt_ids(
            text,
            3,
            clue,
            2,
            sos=6561,
            turn_of_speech=6563,
            type_id=1502,
            startofclue_token=151646,
            endofclue_token=151647,
        )

        ids = result[0].tolist()
        # SOS at start
        assert ids[0] == 6561
        # startofclue + clue + endofclue
        assert ids[1] == 151646
        assert ids[2:4] == [20, 21]
        assert ids[4] == 151647
        # text tokens
        assert ids[5:8] == [10, 11, 12]
        # type_id
        assert ids[8] == 1502
        # turn_of_speech at end
        assert ids[-1] == 6563
        assert length == len(ids)

    def test_no_prompt(self):
        """Test without clue (lm_use_prompt=False)."""
        from vllm_omni.model_executor.models.funcineforge.utils import (
            concat_text_with_prompt_ids,
        )

        text = torch.tensor([[10, 11]], dtype=torch.long)
        clue = torch.tensor([[20, 21]], dtype=torch.long)

        result, length = concat_text_with_prompt_ids(
            text,
            2,
            clue,
            2,
            lm_use_prompt=False,
        )

        ids = result[0].tolist()
        # Should be [SOS, text, type_id, TOS] — no clue tokens
        assert ids[0] == 6561  # SOS
        assert ids[1:3] == [10, 11]  # text
        assert ids[-1] == 6563  # turn_of_speech

    def test_empty_clue(self):
        """Test with zero-length clue."""
        from vllm_omni.model_executor.models.funcineforge.utils import (
            concat_text_with_prompt_ids,
        )

        text = torch.tensor([[10, 11]], dtype=torch.long)
        clue = torch.tensor([[]], dtype=torch.long).reshape(1, 0)

        result, length = concat_text_with_prompt_ids(text, 2, clue, 0)
        ids = result[0].tolist()
        # No startofclue/endofclue wrapper when clue is empty
        assert 151646 not in ids
        assert 151647 not in ids

    def test_with_timespk_ids(self):
        """Test with timespeaker tag IDs included."""
        from vllm_omni.model_executor.models.funcineforge.utils import (
            concat_text_with_prompt_ids,
        )

        text = torch.tensor([[10]], dtype=torch.long)
        clue = torch.tensor([[]], dtype=torch.long).reshape(1, 0)

        result, length = concat_text_with_prompt_ids(
            text,
            1,
            clue,
            0,
            timespk_ids=[1504, 1508, 1511],  # male, adult, speaker_0
        )

        ids = result[0].tolist()
        # timespk IDs should appear between type_id and turn_of_speech
        assert 1504 in ids
        assert 1508 in ids
        assert 1511 in ids
        # turn_of_speech still at end
        assert ids[-1] == 6563

    def test_output_shape(self):
        """Test output tensor is 2D (batch=1, seq_len)."""
        from vllm_omni.model_executor.models.funcineforge.utils import (
            concat_text_with_prompt_ids,
        )

        text = torch.tensor([[10, 11, 12]], dtype=torch.long)
        clue = torch.tensor([[20]], dtype=torch.long)

        result, length = concat_text_with_prompt_ids(text, 3, clue, 1)
        assert result.ndim == 2
        assert result.shape[0] == 1
        assert result.shape[1] == length


class TestLoadFaceEmbedding:
    """Tests for load_face_embedding utility."""

    def test_zero_padding(self, tmp_path):
        """Test that unmatched frames are zero-padded."""
        import numpy as np

        from vllm_omni.model_executor.models.funcineforge.utils import (
            load_face_embedding,
        )

        emb = np.random.randn(512).astype(np.float32)
        face_path = str(tmp_path / "face.npz")
        np.savez(face_path, embeddings=np.array([emb]), faceI=np.array([2]))

        result = load_face_embedding(face_path, speech_len=10, face_size=512)
        assert result.shape == (1, 10, 512)
        # Frame 0-1 should be zeros
        assert result[0, 0].abs().sum() == 0
        assert result[0, 1].abs().sum() == 0
        # Frame 2 should have the embedding
        assert result[0, 2].abs().sum() > 0
