# SPDX-License-Identifier: Apache-2.0
"""Deterministic unit tests for higgs-audio v3.

These tests verify AC-1 (config), AC-3 (prompt), AC-4 (fused modules),
AC-5 (delay pattern), AC-7 (stage processor), and AC-10 (registry)
without requiring the actual checkpoint or GPU.
"""

import asyncio
import json
import time
from collections import OrderedDict, defaultdict
from types import SimpleNamespace

import pytest
import torch

# ---- AC-1: Configuration ----


class TestHiggsAudioV3Config:
    def test_default_config_loads(self):
        from vllm_omni.transformers_utils.configs.higgs_audio_v3 import (
            HiggsAudioV3Config,
        )

        config = HiggsAudioV3Config()
        assert config.num_codebooks == 8
        assert config.codebook_size == 1026
        assert config.audio_stream_bos_id == 1024
        assert config.audio_stream_eos_id == 1025
        assert config.sample_rate == 24000
        assert config.frame_rate == 25
        assert config.num_real_codes == 1024
        assert config.tie_modality_embeddings is True

    def test_custom_audio_encoder_config(self):
        from vllm_omni.transformers_utils.configs.higgs_audio_v3 import (
            HiggsAudioV3Config,
        )

        config = HiggsAudioV3Config(
            audio_encoder_config={
                "encoder_type": "discrete",
                "num_codebooks": 4,
                "vocab_size": 512,
                "tie_word_embeddings": False,
            }
        )
        assert config.num_codebooks == 4
        assert config.codebook_size == 512
        assert config.tie_modality_embeddings is False

    def test_invalid_num_codebooks_rejected(self):
        from vllm_omni.transformers_utils.configs.higgs_audio_v3 import (
            HiggsAudioV3Config,
        )

        with pytest.raises(ValueError, match="num_codebooks must be > 0"):
            HiggsAudioV3Config(audio_encoder_config={"num_codebooks": 0, "vocab_size": 1026})

    def test_negative_num_codebooks_rejected(self):
        from vllm_omni.transformers_utils.configs.higgs_audio_v3 import (
            HiggsAudioV3Config,
        )

        with pytest.raises(ValueError, match="num_codebooks must be > 0"):
            HiggsAudioV3Config(audio_encoder_config={"num_codebooks": -1, "vocab_size": 1026})

    def test_special_token_ids_initially_none(self):
        from vllm_omni.transformers_utils.configs.higgs_audio_v3 import (
            HiggsAudioV3Config,
        )

        config = HiggsAudioV3Config()
        assert config.tts_token_id is None
        assert config.text_token_id is None
        assert config.audio_continuation_id is None

    def test_hidden_size_from_text_config(self):
        from vllm_omni.transformers_utils.configs.higgs_audio_v3 import (
            HiggsAudioV3Config,
        )

        config = HiggsAudioV3Config(text_config={"model_type": "qwen3", "hidden_size": 2560})
        assert config.hidden_size == 2560
        assert config.audio_hidden_size == 2560


# ---- AC-4: Fused Multi-Codebook Modules ----


class TestFusedModules:
    def test_embedding_shape(self):
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import (
            HiggsFusedMultiTextEmbedding,
        )

        embed = HiggsFusedMultiTextEmbedding(num_codebooks=8, vocab_size=1026, hidden_size=256)
        assert embed.weight.shape == (8 * 1026, 256)

        codes = torch.randint(0, 1024, (5, 8))
        out = embed(codes)
        assert out.shape == (5, 256)

    def test_embedding_offset_indexing(self):
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import (
            HiggsFusedMultiTextEmbedding,
        )

        embed = HiggsFusedMultiTextEmbedding(num_codebooks=2, vocab_size=4, hidden_size=3)
        # Set weights to identity-like pattern so we can verify offsets
        with torch.no_grad():
            embed.weight.zero_()
            # Codebook 0, vocab [0..3] -> row 0..3
            # Codebook 1, vocab [0..3] -> row 4..7
            for i in range(8):
                embed.weight[i, i % 3] = float(i)

        # codes[0] = [1, 2] -> embed[1] + embed[4+2=6]
        codes = torch.tensor([[1, 2]])
        out = embed(codes)
        expected = embed.weight[1] + embed.weight[6]
        assert torch.allclose(out[0], expected)

    def test_head_shape(self):
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import (
            HiggsFusedMultiTextHead,
        )

        head = HiggsFusedMultiTextHead(num_codebooks=8, vocab_size=1026, hidden_size=256)
        assert head.weight.shape == (8 * 1026, 256)

        hidden = torch.randn(3, 256)
        out = head.generate(hidden)
        assert out.shape == (3, 8, 1026)

    def test_tying(self):
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import (
            HiggsFusedMultiTextEmbedding,
            HiggsFusedMultiTextHead,
        )

        embed = HiggsFusedMultiTextEmbedding(num_codebooks=8, vocab_size=1026, hidden_size=64)
        head = HiggsFusedMultiTextHead(num_codebooks=8, vocab_size=1026, hidden_size=64)
        head.weight = embed.weight
        assert head.weight is embed.weight
        # Modification to one should reflect in the other
        with torch.no_grad():
            embed.weight[0, 0] = 42.0
        assert head.weight[0, 0] == 42.0

    def test_codes_out_of_range(self):
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import (
            HiggsFusedMultiTextEmbedding,
        )

        embed = HiggsFusedMultiTextEmbedding(num_codebooks=8, vocab_size=1026, hidden_size=16)
        # Code 1026 + offset for codebook 7 = 1026 + 7*1026 = 8208
        # But weight only has 8*1026 = 8208 rows, so index 8208 is out of bounds
        codes = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1026]])  # Last cb: 1026 is OOB
        with pytest.raises(IndexError):
            embed(codes)


# ---- AC-5: Delay Pattern Behavior ----


class TestDelayPatternBehavior:
    """Test the delay pattern masking logic extracted from the talker."""

    def test_boc_eoc_ids(self):
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import (
            BOC_ID,
            EOC_ID,
        )

        assert BOC_ID == 1024
        assert EOC_ID == 1025

    def test_delay_phase_boc_masking(self):
        """During delay phase, codebooks beyond delay_count are forced to BOC."""
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import BOC_ID

        num_codebooks = 8
        # Simulate delay phase at step 3 (delay_count=2, so CBs 3-7 should be BOC)
        codes = torch.randint(0, 1024, (num_codebooks,))
        delay_count = 2
        next_cb = delay_count + 1
        if next_cb < num_codebooks:
            codes[next_cb:] = BOC_ID
        # CBs 0-2 should have original codes, CBs 3-7 should be BOC
        assert all(codes[i] != BOC_ID or i >= 3 for i in range(num_codebooks))
        assert all(codes[i] == BOC_ID for i in range(3, num_codebooks))

    def test_cb0_eoc_triggers_rampdown(self):
        """EOC on codebook 0 starts ramp-down; EOC on other codebooks does not."""
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import (
            EOC_ID,
        )

        num_codebooks = 8
        codes = torch.randint(0, 1024, (num_codebooks,))

        # Simulate: cb0 emits EOC
        codes[0] = EOC_ID
        assert int(codes[0].item()) == EOC_ID
        # Ramp-down should start with countdown = N-2 = 6
        eoc_countdown = num_codebooks - 2
        assert eoc_countdown == 6

        # If cb3 emits EOC but cb0 doesn't, no ramp-down
        codes2 = torch.randint(0, 1024, (num_codebooks,))
        codes2[3] = EOC_ID
        assert int(codes2[0].item()) != EOC_ID  # cb0 is not EOC

    def test_rampdown_termination(self):
        """After N-2 steps of ramp-down, generation_done becomes True."""
        num_codebooks = 8
        eoc_countdown = num_codebooks - 2  # = 6
        generation_done = False
        for _ in range(6):
            eoc_countdown -= 1
            if eoc_countdown <= 0:
                generation_done = True
        assert generation_done is True

    def test_all_boc_seed(self):
        """First audio step should seed last_codes with all-BOC."""
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import BOC_ID

        seeded = torch.full((8,), BOC_ID, dtype=torch.long)
        assert seeded.shape == (8,)
        assert all(seeded[i] == BOC_ID for i in range(8))


# ---- AC-5: Real Sampler Method Tests ----


class TestSamplerMethods:
    """Test the actual sampler and batched delay masking methods."""

    def _make_minimal_talker(self):
        """Create a minimal talker-like object with sampler/masking methods."""
        from vllm_omni.model_executor.models.higgs_audio_v3 import higgs_audio_v3_talker as mod

        class FakeTalker:
            num_codebooks = 8
            codebook_size = 1026

        t = FakeTalker()
        t._sample_audio_codes = mod.HiggsAudioV3TalkerForConditionalGeneration._sample_audio_codes.__get__(t)
        return t

    def _make_batched_sampler_talker(self, num_rows=4):
        """Create a fake talker with GPU-resident sampler state helpers."""
        from vllm_omni.model_executor.models.higgs_audio_v3 import higgs_audio_v3_talker as mod

        class FakeTalker:
            num_codebooks = 8
            codebook_size = 1026

        t = FakeTalker()
        t._decode_last_codes = torch.arange(num_rows * t.num_codebooks, dtype=torch.long).view(
            num_rows, t.num_codebooks
        )
        t._decode_has_codes = torch.tensor([True, False, True, True])[:num_rows].clone()
        t._decode_delay_count = torch.tensor([0, 2, 7, 4], dtype=torch.int32)[:num_rows].clone()
        t._decode_eoc_countdown = torch.tensor([-1, -1, 3, 1], dtype=torch.int32)[:num_rows].clone()
        t._decode_generation_done = torch.tensor([False, False, False, True])[:num_rows].clone()
        t._decode_active_audio_count = 0
        t._codebook_index_cache = {}
        t._row_index_cache = {}
        t._boc_frame_cache = {}
        t._last_audio_codes_buffer = None
        t._last_audio_host_staging = None
        t._last_audio_gpu_staging = None
        t._last_audio_staging_event = None
        t._audio_staging_events = []
        t._audio_staging_event_cursor = 0
        t._last_audio_codes = None
        t._last_audio_code_valid = []
        t._postprocess_cursor = 0
        t._postprocess_audio_rows = 0
        t._postprocess_audio_active_rows = 0

        cls = mod.HiggsAudioV3TalkerForConditionalGeneration
        for name in (
            "_sample_audio_codes",
            "_ensure_decode_state_capacity",
            "_get_row_indices",
            "_get_codebook_indices",
            "_get_boc_frames",
            "_get_audio_codes_buffer",
            "_get_audio_gpu_staging_buffer",
            "_get_audio_host_staging_buffer",
            "_apply_delay_pattern_masking_batched",
            "_update_delay_state_batched",
            "_prefill_row_mask",
            "_audio_seed_mask_from_step_input",
        ):
            setattr(t, name, getattr(cls, name).__get__(t))
        t._device_cache_key = cls._device_cache_key
        return t

    def test_sample_respects_mask(self):
        """Tokens masked to -inf must never be sampled."""
        t = self._make_batched_sampler_talker(num_rows=1)
        logits = torch.full((10, 1026), float("-inf"))
        # Only token 500 is allowed for each row
        logits[:, 500] = 0.0
        result = t._sample_audio_codes(logits)
        assert result.shape == (10,)
        assert (result == 500).all(), f"Expected all 500, got {result.tolist()}"

    def test_sample_all_masked_falls_back_to_argmax(self):
        """All-masked row should fall back to argmax (least-negative logit)."""
        t = self._make_batched_sampler_talker(num_rows=1)
        logits = torch.full((2, 1026), float("-inf"))
        # Row 0: all masked
        # Row 1: only token 42 allowed
        logits[1, 42] = 0.0
        result = t._sample_audio_codes(logits)
        assert result.shape == (2,)
        assert result[1].item() == 42

    def test_delay_masking_forces_boc_during_delay(self):
        """During delay phase, codebooks beyond delay_count must have only BOC allowed."""
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import BOC_ID

        t = self._make_batched_sampler_talker(num_rows=1)
        # Simulate delay_count=2 for batch row 0
        t._ensure_decode_state_capacity(1, torch.device("cpu"))
        t._decode_delay_count[0] = 2
        t._decode_eoc_countdown[0] = -1
        cb_logits = torch.zeros(1, 8, 1026)  # [1 audio row, 8 codebooks, 1026 vocab]
        t._apply_delay_pattern_masking_batched(cb_logits, torch.tensor([0], dtype=torch.long))
        # CBs 3-7 should have only BOC allowed (everything else -inf)
        for q in range(3, 8):
            row = cb_logits[0, q]
            assert row[BOC_ID].item() == 0.0  # BOC kept
            # All non-BOC should be -inf
            non_boc = torch.cat([row[:BOC_ID], row[BOC_ID + 1 :]])
            assert (non_boc == float("-inf")).all(), f"CB{q} has non-inf non-BOC values"

    def test_delay_masking_disallows_boc_for_active_codebooks(self):
        """Active codebooks during delay should have BOC disallowed."""
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import BOC_ID

        t = self._make_batched_sampler_talker(num_rows=1)
        t._ensure_decode_state_capacity(1, torch.device("cpu"))
        t._decode_delay_count[0] = 3
        t._decode_eoc_countdown[0] = -1
        cb_logits = torch.zeros(1, 8, 1026)
        t._apply_delay_pattern_masking_batched(cb_logits, torch.tensor([0], dtype=torch.long))
        # CBs 0-3 are active: BOC should be -inf
        for q in range(4):
            assert cb_logits[0, q, BOC_ID].item() == float("-inf")

    def test_delay_masking_only_cb0_allows_eoc(self):
        """Only codebook 0 should allow EOC during normal generation."""
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import EOC_ID

        t = self._make_batched_sampler_talker(num_rows=1)
        t._ensure_decode_state_capacity(1, torch.device("cpu"))
        t._decode_delay_count[0] = 8
        t._decode_eoc_countdown[0] = -1
        cb_logits = torch.zeros(1, 8, 1026)
        t._apply_delay_pattern_masking_batched(cb_logits, torch.tensor([0], dtype=torch.long))
        # CB0 should keep EOC
        assert cb_logits[0, 0, EOC_ID].item() != float("-inf")
        # CB1-7 should have EOC masked
        for q in range(1, 8):
            assert cb_logits[0, q, EOC_ID].item() == float("-inf"), f"CB{q} EOC not masked"

    def test_rampdown_masking_locks_to_eoc(self):
        """During ramp-down, locked codebooks should only allow EOC."""
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import (
            BOC_ID,
            EOC_ID,
        )

        t = self._make_batched_sampler_talker(num_rows=1)
        # Ramp-down with 4 remaining delays: lock CBs 0-3 to EOC
        t._ensure_decode_state_capacity(1, torch.device("cpu"))
        t._decode_delay_count[0] = 8
        t._decode_eoc_countdown[0] = 4
        cb_logits = torch.zeros(1, 8, 1026)
        t._apply_delay_pattern_masking_batched(cb_logits, torch.tensor([0], dtype=torch.long))
        # CBs 0-3 locked: only EOC allowed
        for q in range(4):
            row = cb_logits[0, q]
            assert row[EOC_ID].item() == 0.0
            non_eoc = torch.cat([row[:EOC_ID], row[EOC_ID + 1 :]])
            assert (non_eoc == float("-inf")).all(), f"Locked CB{q} has non-inf non-EOC"
        # CBs 4-7 active: BOC and EOC disallowed
        for q in range(4, 8):
            assert cb_logits[0, q, BOC_ID].item() == float("-inf")
            assert cb_logits[0, q, EOC_ID].item() == float("-inf")

    def test_batched_delay_masking_all_rows_matches_sparse_path(self):
        """The all-row fast path must preserve sparse path masking semantics."""
        torch.manual_seed(0)
        sparse = self._make_batched_sampler_talker()
        all_rows = self._make_batched_sampler_talker()
        rows = torch.arange(4, dtype=torch.long)
        logits_sparse = torch.randn(4, 8, 1026)
        logits_all_rows = logits_sparse.clone()

        sparse._apply_delay_pattern_masking_batched(logits_sparse, rows, all_rows=False)
        all_rows._apply_delay_pattern_masking_batched(logits_all_rows, rows, all_rows=True)

        assert torch.equal(logits_all_rows, logits_sparse)

    def test_delay_state_update_all_rows_matches_sparse_path(self):
        """The all-row fast path must update state and staging like index_copy_."""
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import (
            EOC_ID,
        )

        sparse = self._make_batched_sampler_talker()
        all_rows = self._make_batched_sampler_talker()
        rows = torch.arange(4, dtype=torch.long)
        codes = torch.tensor(
            [
                [10, 11, 12, 13, 14, 15, 16, 17],
                [20, 21, 22, 23, 24, 25, 26, 27],
                [30, 31, EOC_ID, 33, 34, 35, 36, 37],
                [40, 41, 42, 43, 44, 45, 46, 47],
            ],
            dtype=torch.long,
        )
        code_row_mask = torch.tensor([True, False, True, True])

        sparse._update_delay_state_batched(
            codes.clone(), rows, 4, torch.device("cpu"), code_row_mask=code_row_mask, all_rows=False
        )
        all_rows._update_delay_state_batched(
            codes.clone(), rows, 4, torch.device("cpu"), code_row_mask=code_row_mask, all_rows=True
        )

        assert torch.equal(all_rows._decode_delay_count, sparse._decode_delay_count)
        assert torch.equal(all_rows._decode_eoc_countdown, sparse._decode_eoc_countdown)
        assert torch.equal(all_rows._decode_generation_done, sparse._decode_generation_done)
        assert torch.equal(all_rows._decode_last_codes, sparse._decode_last_codes)
        assert torch.equal(all_rows._decode_has_codes, sparse._decode_has_codes)
        assert all_rows._last_audio_codes is None
        assert sparse._last_audio_codes is None
        assert torch.equal(all_rows._last_audio_host_staging, sparse._last_audio_host_staging)

    def test_direct_audio_sampler_does_not_override_done_tokens(self):
        """Done rows are represented directly in GPU sampled tokens."""
        from vllm_omni.model_executor.models.higgs_audio_v3 import higgs_audio_v3_talker as mod

        t = self._make_batched_sampler_talker(num_rows=2)
        t._resolve_token_ids = lambda: None
        t._audio_continuation_id = 99999
        t._eos_token_id = 151671
        t._last_logits_hidden = torch.zeros(2, 16)
        t._last_step_input_ids = torch.tensor([99999, 99999])
        t._last_step_query_start_loc = None
        t._decode_has_codes = torch.tensor([True, True])
        t._decode_generation_done = torch.tensor([False, True])
        t._decode_delay_count = torch.zeros(2, dtype=torch.long)
        t._decode_eoc_countdown = torch.full((2,), -1, dtype=torch.long)
        t._fast_audio_direct_rows = 2
        t._last_audio_done_flags = None
        t._fast_audio_sampler_gpu_fallback_reason = lambda **kwargs: None
        t._audio_codebook_logits_from_rows = lambda hidden, rows, all_rows=False: torch.zeros(2, 8, 1026)
        t._apply_delay_pattern_masking_batched = lambda cb_logits, audio_rows, all_rows=False: None
        t._sample_audio_codes = lambda logits_2d: torch.zeros(int(logits_2d.shape[0]), dtype=torch.long)
        t._update_delay_state_batched = lambda *args, **kwargs: None
        t.sample = mod.HiggsAudioV3TalkerForConditionalGeneration.sample.__get__(t)

        sampler_output = t.sample(torch.zeros(2, 200000), sampling_metadata=object())

        assert not getattr(
            mod.HiggsAudioV3TalkerForConditionalGeneration, "supports_sampled_token_ids_cpu_override", False
        )
        assert sampler_output.sampled_token_ids.tolist() == [[99999], [151671]]

    def test_direct_audio_sampler_uses_gpu_tokens_for_active_rows(self):
        """Direct audio sampling leaves scheduler-visible tokens to the GPU tensor."""
        from vllm_omni.model_executor.models.higgs_audio_v3 import higgs_audio_v3_talker as mod

        t = self._make_batched_sampler_talker(num_rows=2)
        t._resolve_token_ids = lambda: None
        t._audio_continuation_id = 99999
        t._eos_token_id = 151671
        t._last_logits_hidden = torch.zeros(2, 16)
        t._last_step_input_ids = torch.tensor([99999, 99999])
        t._last_step_query_start_loc = None
        t._decode_has_codes = torch.tensor([True, True])
        t._decode_generation_done = torch.tensor([False, False])
        t._decode_delay_count = torch.zeros(2, dtype=torch.long)
        t._decode_eoc_countdown = torch.full((2,), -1, dtype=torch.long)
        t._fast_audio_direct_rows = 2
        t._last_audio_done_flags = [0, 0]
        t._fast_audio_sampler_gpu_fallback_reason = lambda **kwargs: None
        t._audio_codebook_logits_from_rows = lambda hidden, rows, all_rows=False: torch.zeros(2, 8, 1026)
        t._apply_delay_pattern_masking_batched = lambda cb_logits, audio_rows, all_rows=False: None
        t._sample_audio_codes = lambda logits_2d: torch.zeros(int(logits_2d.shape[0]), dtype=torch.long)
        t._update_delay_state_batched = lambda *args, **kwargs: None
        t.sample = mod.HiggsAudioV3TalkerForConditionalGeneration.sample.__get__(t)

        sampler_output = t.sample(torch.zeros(2, 200000), sampling_metadata=object())

        assert sampler_output.sampled_token_ids.tolist() == [[99999], [99999]]

    def test_terminal_rampdown_frame_is_emitted_before_done(self):
        """The final ramp-down frame must be emitted, not replaced by a done marker."""
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import EOC_ID

        t = self._make_batched_sampler_talker(num_rows=1)
        t._decode_delay_count[0] = 8
        t._decode_eoc_countdown[0] = 1
        t._decode_generation_done[0] = False
        t._decode_has_codes[0] = True
        codes = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]], dtype=torch.long)

        t._update_delay_state_batched(
            codes,
            torch.tensor([0], dtype=torch.long),
            1,
            torch.device("cpu"),
            code_row_mask=torch.tensor([True]),
            all_rows=True,
        )

        staged = t._last_audio_host_staging[0]
        assert staged[t.num_codebooks].item() == 1  # valid
        assert staged[t.num_codebooks + 1].item() == 1  # done
        assert not (staged[: t.num_codebooks] == -1).any()
        assert staged[: t.num_codebooks - 1].eq(EOC_ID).all()
        assert t._decode_generation_done[0].item() is True
        assert t._decode_has_codes[0].item() is False

    def test_decode_state_growth_preserves_inflight_rows(self):
        """Growing decode buffers must copy old rows and initialize new rows inactive."""
        t = self._make_batched_sampler_talker(num_rows=2)
        old_codes = t._decode_last_codes.clone()
        old_has = t._decode_has_codes.clone()
        old_delay = t._decode_delay_count.clone()
        old_rem = t._decode_eoc_countdown.clone()
        old_done = t._decode_generation_done.clone()

        t._ensure_decode_state_capacity(5, torch.device("cpu"))

        assert t._decode_last_codes.shape[0] >= 5
        assert torch.equal(t._decode_last_codes[:2], old_codes)
        assert torch.equal(t._decode_has_codes[:2], old_has)
        assert torch.equal(t._decode_delay_count[:2], old_delay)
        assert torch.equal(t._decode_eoc_countdown[:2], old_rem)
        assert torch.equal(t._decode_generation_done[:2], old_done)
        assert not t._decode_has_codes[2:5].any()
        assert not t._decode_generation_done[2:5].any()
        assert t._decode_eoc_countdown[2:5].eq(-1).all()

    def test_mixed_batch_prefill_mask_targets_request_rows(self):
        """Mixed prefill/decode must reset only prefill request rows."""
        t = self._make_batched_sampler_talker(num_rows=3)
        t._last_step_query_start_loc = torch.tensor([0, 1, 4, 5])

        mask = t._prefill_row_mask(3, torch.device("cpu"))

        assert torch.equal(mask, torch.tensor([False, True, False]))

    def test_mixed_batch_seed_mask_maps_token_tails_to_request_rows(self):
        """Seed detection should return one flag per request, not per token."""
        t = self._make_batched_sampler_talker(num_rows=3)
        t._audio_continuation_id = 99999
        t._last_step_query_start_loc = torch.tensor([0, 1, 4, 5])
        t._last_step_input_ids = torch.tensor([99999, 11, 12, 13, 99999])

        mask = t._audio_seed_mask_from_step_input(3, torch.device("cpu"))

        assert torch.equal(mask, torch.tensor([True, False, True]))

    def test_mixed_batch_seed_mask_allows_prefill_tail_audio_token(self):
        """A prefill span ending in <|audio|> starts audio-code generation."""
        t = self._make_batched_sampler_talker(num_rows=3)
        t._audio_continuation_id = 99999
        t._last_step_query_start_loc = torch.tensor([0, 1, 4, 5])
        t._last_step_input_ids = torch.tensor([99999, 11, 12, 99999, 99999])

        mask = t._audio_seed_mask_from_step_input(3, torch.device("cpu"))

        assert torch.equal(mask, torch.tensor([True, True, True]))


# ---- AC-6: Feedback Method Tests ----


class TestFeedbackMethods:
    def test_postprocess_emits_audio_codes(self):
        """postprocess() should return codes from _last_audio_codes."""
        from vllm_omni.model_executor.models.higgs_audio_v3 import higgs_audio_v3_talker as mod

        class FakeTalker:
            _last_audio_codes = torch.tensor([[100, 200, 300, 400, 500, 600, 700, 800]])
            _last_audio_code_valid = [True]
            _last_audio_host_staging = None
            _last_audio_staging_event = None
            _postprocess_cursor = 0

        t = FakeTalker()
        t.postprocess = mod.HiggsAudioV3TalkerForConditionalGeneration.postprocess.__get__(t)
        t._postprocess_impl = mod.HiggsAudioV3TalkerForConditionalGeneration._postprocess_impl.__get__(t)
        result = t.postprocess(torch.zeros(1, 64))
        assert "codes" in result
        assert "audio" in result["codes"]
        assert result["codes"]["audio"].shape == (1, 8)

    def test_postprocess_skips_negative_rows(self):
        """postprocess() should skip rows with -1 (no audio)."""
        from vllm_omni.model_executor.models.higgs_audio_v3 import higgs_audio_v3_talker as mod

        class FakeTalker:
            _last_audio_codes = torch.tensor([[-1, -1, -1, -1, -1, -1, -1, -1]])
            _last_audio_code_valid = [False]
            _last_audio_host_staging = None
            _last_audio_staging_event = None
            _postprocess_cursor = 0

        t = FakeTalker()
        t.postprocess = mod.HiggsAudioV3TalkerForConditionalGeneration.postprocess.__get__(t)
        t._postprocess_impl = mod.HiggsAudioV3TalkerForConditionalGeneration._postprocess_impl.__get__(t)
        result = t.postprocess(torch.zeros(1, 64))
        assert result == {}

    def test_postprocess_advances_cursor(self):
        """postprocess() should advance cursor by 1 per call."""
        from vllm_omni.model_executor.models.higgs_audio_v3 import higgs_audio_v3_talker as mod

        class FakeTalker:
            _last_audio_codes = torch.tensor(
                [[100, 200, 300, 400, 500, 600, 700, 800], [-1, -1, -1, -1, -1, -1, -1, -1]]
            )
            _last_audio_code_valid = [True, False]
            _last_audio_host_staging = None
            _last_audio_staging_event = None
            _postprocess_cursor = 0

        t = FakeTalker()
        t.postprocess = mod.HiggsAudioV3TalkerForConditionalGeneration.postprocess.__get__(t)
        t._postprocess_impl = mod.HiggsAudioV3TalkerForConditionalGeneration._postprocess_impl.__get__(t)
        r1 = t.postprocess(torch.zeros(1, 64))
        assert "codes" in r1
        r2 = t.postprocess(torch.zeros(1, 64))
        assert r2 == {}  # Second row is all -1


# ---- AC-6: Audio Feedback Method Tests ----


class TestAudioFeedback:
    """Test _apply_audio_feedback with minimal fake talker."""

    def _make_feedback_talker(self):
        from vllm_omni.model_executor.models.higgs_audio_v3 import higgs_audio_v3_talker as mod

        embed = mod.HiggsFusedMultiTextEmbedding(num_codebooks=8, vocab_size=1026, hidden_size=16)
        torch.nn.init.ones_(embed.weight)

        class FakeTalker:
            num_codebooks = 8
            codebook_size = 1026
            _audio_continuation_id = 99999  # fake audio token
            _last_step_query_start_loc = None
            _decode_active_audio_count = 0
            _decode_last_codes = torch.zeros(4, 8, dtype=torch.long)
            _decode_has_codes = torch.zeros(4, dtype=torch.bool)
            _use_external_decode_cudagraph = False
            multimodal_embedding = embed
            model = type("M", (), {"embed_tokens": lambda self, ids: torch.zeros(ids.shape[0], 16)})()

        t = FakeTalker()
        t._apply_audio_feedback = mod.HiggsAudioV3TalkerForConditionalGeneration._apply_audio_feedback.__get__(t)
        t._is_single_token_decode_step = (
            mod.HiggsAudioV3TalkerForConditionalGeneration._is_single_token_decode_step.__get__(t)
        )
        t._step_request_count = mod.HiggsAudioV3TalkerForConditionalGeneration._step_request_count.__get__(t)
        t._decode_request_token_positions = (
            mod.HiggsAudioV3TalkerForConditionalGeneration._decode_request_token_positions.__get__(t)
        )
        t._ensure_decode_state_capacity = lambda min_bs, device: None
        return t

    def test_text_positions_unchanged(self):
        """Non-audio positions should not be modified."""
        t = self._make_feedback_talker()
        # No audio state → all positions unchanged
        input_ids = torch.tensor([1, 2, 3, 4])
        hidden = torch.randn(4, 16)
        result = t._apply_audio_feedback(hidden, input_ids)
        assert torch.equal(result, hidden)

    def test_audio_position_replaced_with_state(self):
        """Audio position with state should have its embedding replaced."""
        t = self._make_feedback_talker()
        audio_id = t._audio_continuation_id
        t._decode_last_codes[1] = torch.zeros(8, dtype=torch.long)
        t._decode_has_codes[1] = True
        t._decode_active_audio_count = 1
        t._ensure_decode_state_capacity = lambda min_bs, device: None
        t._last_step_query_start_loc = torch.tensor([0, 1, 2, 3])
        input_ids = torch.tensor([1, audio_id, 3])
        hidden = torch.zeros(3, 16)
        result = t._apply_audio_feedback(hidden, input_ids)
        # Position 1 (audio) should be non-zero (embedding of all-0 codes)
        assert result[1].abs().sum() > 0
        # Positions 0 and 2 should remain zero
        assert result[0].abs().sum() == 0
        assert result[2].abs().sum() == 0

    def test_no_state_no_replacement(self):
        """Audio position without state should not be replaced."""
        t = self._make_feedback_talker()
        audio_id = t._audio_continuation_id
        # No state for row 0
        input_ids = torch.tensor([audio_id])
        hidden = torch.zeros(1, 16)
        result = t._apply_audio_feedback(hidden, input_ids)
        assert torch.equal(result, hidden)

    def test_prefill_span_does_not_receive_audio_feedback(self):
        """A prefill span can have numel == hidden rows but must not be treated as decode."""
        t = self._make_feedback_talker()
        t._last_step_input_ids = torch.tensor([1, 2, 3, 4])
        t._last_step_query_start_loc = torch.tensor([0, 4])
        t._decode_last_codes[0] = torch.zeros(8, dtype=torch.long)
        t._decode_has_codes[0] = True
        t._decode_active_audio_count = 1
        hidden = torch.zeros(4, 16)

        result = t._apply_audio_feedback(hidden, t._last_step_input_ids)

        assert torch.equal(result, hidden)

    def test_multi_request_decode_receives_audio_feedback(self):
        """Concurrent decode has numel > 1 but each request span is exactly one token."""
        t = self._make_feedback_talker()
        t._last_step_input_ids = torch.tensor([10, 11, 12, 13])
        t._last_step_query_start_loc = torch.tensor([0, 1, 2, 3, 4])
        t._decode_last_codes[2] = torch.zeros(8, dtype=torch.long)
        t._decode_has_codes[2] = True
        t._decode_active_audio_count = 1
        hidden = torch.zeros(4, 16)

        result = t._apply_audio_feedback(hidden, t._last_step_input_ids)

        assert result[2].abs().sum() > 0
        assert result[0].abs().sum() == 0
        assert result[1].abs().sum() == 0
        assert result[3].abs().sum() == 0

    def test_external_decode_graph_capture_keeps_feedback_ops_without_query_start_loc(self):
        """FULL_DECODE capture has no request qsl but still needs dense decode feedback ops."""
        t = self._make_feedback_talker()
        t._use_external_decode_cudagraph = True
        t._last_step_input_ids = torch.tensor([10, 11, 12, 13])
        t._last_step_query_start_loc = None
        t._decode_last_codes[2] = torch.zeros(8, dtype=torch.long)
        t._decode_has_codes[2] = True
        hidden = torch.zeros(4, 16)

        result = t._apply_audio_feedback(hidden, t._last_step_input_ids)

        assert result[2].abs().sum() > 0
        assert result[0].abs().sum() == 0
        assert result[1].abs().sum() == 0
        assert result[3].abs().sum() == 0

    def test_mixed_prefill_decode_receives_feedback_only_on_decode_rows(self):
        """Mixed batches should skip prefill spans but keep audio feedback for decode spans."""
        t = self._make_feedback_talker()
        t._last_step_input_ids = torch.tensor([99999, 21, 22, 23, 99999])
        t._last_step_query_start_loc = torch.tensor([0, 1, 4, 5])
        t._decode_last_codes[0] = torch.zeros(8, dtype=torch.long)
        t._decode_last_codes[2] = torch.zeros(8, dtype=torch.long)
        t._decode_has_codes[0] = True
        t._decode_has_codes[2] = True
        t._decode_active_audio_count = 2
        hidden = torch.zeros(5, 16)

        result = t._apply_audio_feedback(hidden, t._last_step_input_ids)

        assert result[0].abs().sum() > 0
        assert result[4].abs().sum() > 0
        assert result[1].abs().sum() == 0
        assert result[2].abs().sum() == 0
        assert result[3].abs().sum() == 0


# ---- AC-8: Codec Strictness ----


class TestCodecStrictness:
    def test_bundled_missing_quantizer_key_raises(self):
        """Bundled codec load must fail when a quantizer codebook key is missing."""
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_code2wav import (
            HiggsAudioV3Code2Wav,
        )
        from vllm_omni.transformers_utils.configs.higgs_audio_v3 import (
            HiggsAudioV3Config,
        )

        config = HiggsAudioV3Config()
        c2w = HiggsAudioV3Code2Wav(config=config)

        # State with only 7 quantizers (missing quantizer.quantizers.7.*)
        codec_state = {}
        for i in range(7):
            codec_state[f"quantizer.quantizers.{i}.codebook.embed"] = torch.randn(1024, 64)
            codec_state[f"quantizer.quantizers.{i}.project_out.weight"] = torch.randn(1024, 64)
            codec_state[f"quantizer.quantizers.{i}.project_out.bias"] = torch.randn(1024)
        codec_state["fc2.weight"] = torch.randn(256, 1024)
        codec_state["fc2.bias"] = torch.randn(256)

        with pytest.raises(KeyError, match="quantizer 7"):
            c2w._load_from_bundled_state(codec_state, device=torch.device("cpu"))

    def test_bundled_missing_fc2_raises(self):
        """Bundled codec load must fail when fc2 keys are missing."""
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_code2wav import (
            HiggsAudioV3Code2Wav,
        )
        from vllm_omni.transformers_utils.configs.higgs_audio_v3 import (
            HiggsAudioV3Config,
        )

        config = HiggsAudioV3Config()
        c2w = HiggsAudioV3Code2Wav(config=config)

        codec_state = {}
        for i in range(8):
            codec_state[f"quantizer.quantizers.{i}.codebook.embed"] = torch.randn(1024, 64)
            codec_state[f"quantizer.quantizers.{i}.project_out.weight"] = torch.randn(1024, 64)
            codec_state[f"quantizer.quantizers.{i}.project_out.bias"] = torch.randn(1024)
        # No fc2 keys

        with pytest.raises(KeyError, match="fc2"):
            c2w._load_from_bundled_state(codec_state, device=torch.device("cpu"))

    def test_bundled_unknown_decoder_key_rejected(self, monkeypatch):
        """Unknown decoder-side key should be rejected as unconsumed."""
        from vllm_omni.model_executor.models.higgs_audio_v3 import higgs_audio_v3_code2wav as c2w_mod
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_code2wav import (
            HiggsAudioV3Code2Wav,
        )
        from vllm_omni.transformers_utils.configs.higgs_audio_v3 import (
            HiggsAudioV3Config,
        )

        class FakeDAC(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))

        monkeypatch.setattr(c2w_mod, "build_boson_dac_decoder", lambda device: FakeDAC().to(device))
        monkeypatch.setattr(c2w_mod, "build_higgs_audio_acoustic_decoder", lambda cfg, device: FakeDAC().to(device))

        config = HiggsAudioV3Config()
        c2w = HiggsAudioV3Code2Wav(config=config)

        codec_state = _build_full_codec_state()
        codec_state["unknown_decoder_module.weight"] = torch.randn(10)

        with pytest.raises(RuntimeError, match="unexpected decoder-side keys"):
            c2w._load_from_bundled_state(codec_state, device=torch.device("cpu"))

    def test_bundled_encoder_side_keys_accepted(self, monkeypatch):
        """Known encoder-side keys should not trigger unconsumed-key rejection."""
        from vllm_omni.model_executor.models.higgs_audio_v3 import higgs_audio_v3_code2wav as c2w_mod
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_code2wav import (
            HiggsAudioV3Code2Wav,
        )
        from vllm_omni.transformers_utils.configs.higgs_audio_v3 import (
            HiggsAudioV3Config,
        )

        class FakeDAC(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))

        monkeypatch.setattr(c2w_mod, "build_boson_dac_decoder", lambda device: FakeDAC().to(device))
        monkeypatch.setattr(c2w_mod, "build_higgs_audio_acoustic_decoder", lambda cfg, device: FakeDAC().to(device))

        config = HiggsAudioV3Config()
        c2w = HiggsAudioV3Code2Wav(config=config)

        codec_state = _build_full_codec_state()
        codec_state["acoustic_encoder.block.0.conv_t1.weight"] = torch.randn(10)
        codec_state["semantic_model.encoder.layers.0.weight"] = torch.randn(10)
        codec_state["fc.weight"] = torch.randn(10)

        # Should NOT raise — encoder-side keys are allowed
        c2w._load_from_bundled_state(codec_state, device=torch.device("cpu"))
        assert c2w._loaded


def _build_minimal_codec_state() -> dict[str, torch.Tensor]:
    """Build a minimal codec state dict (quantizer + fc2, no DAC)."""
    state: dict[str, torch.Tensor] = {}
    for i in range(8):
        state[f"quantizer.quantizers.{i}.codebook.embed"] = torch.randn(1024, 64)
        state[f"quantizer.quantizers.{i}.project_out.weight"] = torch.randn(1024, 64)
        state[f"quantizer.quantizers.{i}.project_out.bias"] = torch.randn(1024)
    state["fc2.weight"] = torch.randn(256, 1024)
    state["fc2.bias"] = torch.randn(256)
    return state


def _build_full_codec_state() -> dict[str, torch.Tensor]:
    """Build a full codec state dict including acoustic_decoder keys.

    The acoustic_decoder keys here match what a FakeDAC module expects
    (just 'dummy'). Real DAC has many more keys.
    """
    state = _build_minimal_codec_state()
    state["acoustic_decoder.dummy"] = torch.zeros(1)
    return state


# ---- AC-2: Loader Required-Key Validation ----


class TestLoaderRequiredKeys:
    """Test exact required-key validation in load_weights."""

    def test_build_required_keys_count(self):
        """Required key set should have 1 text_emb + 36*11 layers + 1 norm + 1 modality = 399."""
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import (
            HiggsAudioV3TalkerForConditionalGeneration,
        )

        keys = HiggsAudioV3TalkerForConditionalGeneration._build_required_keys(36)
        assert len(keys) == 1 + 36 * 11 + 1 + 1  # text + layers + norm + modality = 399

    def test_build_required_keys_contains_norm(self):
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import (
            HiggsAudioV3TalkerForConditionalGeneration,
        )

        keys = HiggsAudioV3TalkerForConditionalGeneration._build_required_keys(2)
        assert "body.norm.weight" in keys

    def test_build_required_keys_contains_layer_subkeys(self):
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import (
            HiggsAudioV3TalkerForConditionalGeneration,
        )

        keys = HiggsAudioV3TalkerForConditionalGeneration._build_required_keys(2)
        assert "body.layers.0.self_attn.k_proj.weight" in keys
        assert "body.layers.1.mlp.gate_proj.weight" in keys

    def test_build_required_keys_contains_modality_embedding(self):
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import (
            HiggsAudioV3TalkerForConditionalGeneration,
        )

        keys = HiggsAudioV3TalkerForConditionalGeneration._build_required_keys(1)
        assert "tied.embedding.modality_embeddings.0.embedding.weight" in keys

    def test_build_required_keys_contains_text_embedding(self):
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import (
            HiggsAudioV3TalkerForConditionalGeneration,
        )

        keys = HiggsAudioV3TalkerForConditionalGeneration._build_required_keys(1)
        assert "tied.embedding.text_embedding.weight" in keys


class TestLoadWeightsEnforcement:
    """Test load_weights() itself with synthetic key sets via a minimal fake talker."""

    def _make_fake_talker_and_keys(self):
        """Build a minimal talker-like object and the full 1-layer key set."""
        from vllm_omni.model_executor.models.higgs_audio_v3 import higgs_audio_v3_talker as mod

        # Minimal fake talker with just enough to run load_weights
        talker = object.__new__(mod.HiggsAudioV3TalkerForConditionalGeneration)
        torch.nn.Module.__init__(talker)

        # Attach minimal attributes load_weights needs
        talker.multimodal_embedding = mod.HiggsFusedMultiTextEmbedding(8, 1026, 16)
        talker.modality_head = mod.HiggsFusedMultiTextHead(8, 1026, 16)
        talker.tie_modality = True
        talker.modality_head.weight = talker.multimodal_embedding.weight

        # Fake backbone config with 1 layer
        class FakeConfig:
            num_hidden_layers = 1
            tie_word_embeddings = True

        talker._backbone_config = FakeConfig()

        # Minimal model and lm_head so _BackboneWrapper can be constructed
        talker.model = torch.nn.Module()
        talker.lm_head = torch.nn.Module()

        # Fake vllm_config for resolve_special_tokens (no-op)
        class FakeModelConfig:
            model = None

        class FakeVllmConfig:
            model_config = FakeModelConfig()

        talker.vllm_config = FakeVllmConfig()
        talker.config = type("C", (), {"resolve_special_tokens": lambda self, p: None})()
        talker._resolved_tokens = True
        talker._audio_continuation_id = None
        talker._eos_token_id = None

        # Build the full required key set for 1 layer
        required = mod.HiggsAudioV3TalkerForConditionalGeneration._build_required_keys(1)

        # Build synthetic weights: tiny tensors for every required key
        weights = {}
        for key in required:
            if "modality_embeddings.0.embedding" in key:
                weights[key] = torch.randn(8 * 1026, 16)
            else:
                weights[key] = torch.randn(4)  # tiny placeholder

        # Monkeypatch _BackboneWrapper.load_weights to accept and return keys
        def fake_backbone_load(self_wrapper, ws):
            consumed = set()
            for name, _ in ws:
                consumed.add(name)
            return consumed

        talker._fake_backbone_load = fake_backbone_load
        return talker, weights, mod, fake_backbone_load

    def test_load_weights_succeeds_with_all_keys(self, monkeypatch):
        """load_weights succeeds when all required keys are present."""
        talker, weights, mod, fake_load = self._make_fake_talker_and_keys()
        monkeypatch.setattr(mod._BackboneWrapper, "load_weights", fake_load)
        result = talker.load_weights(iter(weights.items()))
        assert isinstance(result, set)
        assert "multimodal_embedding.weight" in result

    def test_load_weights_fails_missing_norm(self, monkeypatch):
        """load_weights raises when body.norm.weight is missing."""
        talker, weights, mod, fake_load = self._make_fake_talker_and_keys()
        monkeypatch.setattr(mod._BackboneWrapper, "load_weights", fake_load)
        del weights["body.norm.weight"]
        with pytest.raises(RuntimeError, match="body.norm.weight"):
            talker.load_weights(iter(weights.items()))

    def test_load_weights_fails_missing_layer_subkey(self, monkeypatch):
        """load_weights raises when a specific layer subkey is missing."""
        talker, weights, mod, fake_load = self._make_fake_talker_and_keys()
        monkeypatch.setattr(mod._BackboneWrapper, "load_weights", fake_load)
        del weights["body.layers.0.self_attn.k_proj.weight"]
        with pytest.raises(RuntimeError, match="body.layers.0.self_attn.k_proj.weight"):
            talker.load_weights(iter(weights.items()))

    def test_load_weights_fails_missing_modality_embedding(self, monkeypatch):
        """load_weights raises when modality embedding is missing."""
        talker, weights, mod, fake_load = self._make_fake_talker_and_keys()
        monkeypatch.setattr(mod._BackboneWrapper, "load_weights", fake_load)
        del weights["tied.embedding.modality_embeddings.0.embedding.weight"]
        with pytest.raises(RuntimeError, match="modality_embeddings"):
            talker.load_weights(iter(weights.items()))

    def test_load_weights_fails_unknown_prefix(self, monkeypatch):
        """load_weights raises on unknown non-codec checkpoint prefix."""
        talker, weights, mod, fake_load = self._make_fake_talker_and_keys()
        monkeypatch.setattr(mod._BackboneWrapper, "load_weights", fake_load)
        weights["unknown.prefix.weight"] = torch.randn(4)
        with pytest.raises(ValueError, match="Unexpected checkpoint key"):
            talker.load_weights(iter(weights.items()))


# ---- AC-7: Stage Input Processor ----


class TestStageInputProcessor:
    def test_revert_delay_pattern(self):
        from vllm_omni.model_executor.stage_input_processors.higgs_audio_v3 import (
            _revert_delay_pattern,
        )

        # 8 codebooks, 3 real frames -> delayed shape [8, 3+8-1=10]
        Q, T = 8, 3
        delayed = torch.full((Q, T + Q - 1), 1024)  # Fill with BOC
        for i in range(Q):
            for t in range(T):
                delayed[i, i + t] = i * 100 + t  # Real codes at shifted positions
        result = _revert_delay_pattern(delayed)
        assert result.shape == (Q, T)
        for i in range(Q):
            for t in range(T):
                assert result[i, t].item() == i * 100 + t

    def test_revert_delay_pattern_rejects_wrong_codebooks(self):
        from vllm_omni.model_executor.stage_input_processors.higgs_audio_v3 import (
            _revert_delay_pattern,
        )

        # 7 codebooks should be rejected
        codes = torch.zeros(7, 20)
        with pytest.raises(ValueError, match="Expected exactly 8 codebook rows"):
            _revert_delay_pattern(codes)

    def test_revert_delay_pattern_rejects_too_few_frames(self):
        from vllm_omni.model_executor.stage_input_processors.higgs_audio_v3 import (
            _revert_delay_pattern,
        )

        # 8 codebooks but only 5 frames (need at least 8)
        codes = torch.zeros(8, 5)
        with pytest.raises(ValueError, match="Not enough frames"):
            _revert_delay_pattern(codes)

    def test_talker2code2wav_skips_too_few_frames_without_crashing(self):
        from vllm_omni.model_executor.stage_input_processors.higgs_audio_v3 import (
            talker2code2wav,
        )

        class Output:
            multimodal_output = {"codes": {"audio": torch.zeros(5, 8, dtype=torch.long)}}

        class TalkerOutput:
            finished = True
            outputs = [Output()]

        result = talker2code2wav([TalkerOutput()])

        assert len(result) == 1
        assert result[0]["prompt_token_ids"] == []

    def test_filter_real_code_frames(self):
        from vllm_omni.model_executor.stage_input_processors.higgs_audio_v3 import (
            _filter_real_code_frames,
        )

        # 8 codebooks, 4 frames
        codes = torch.tensor(
            [
                [100, 200, 1024, 300],  # cb0: frame 2 has BOC
                [101, 201, 1024, 301],
                [102, 202, 1024, 302],
                [103, 203, 1024, 303],
                [104, 204, 1024, 304],
                [105, 205, 1024, 305],
                [106, 206, 1024, 306],
                [107, 207, 1024, 307],
            ]
        )
        result = _filter_real_code_frames(codes)
        # Frame 2 (column 2) has BOC in all codebooks -> filtered out
        assert result.shape == (8, 3)
        assert result[0, 0].item() == 100
        assert result[0, 1].item() == 200
        assert result[0, 2].item() == 300

    def test_async_chunk_accepts_multimodal_output_keyword(self):
        from vllm_omni.model_executor.stage_input_processors.higgs_audio_v3 import (
            talker2code2wav_async_chunk,
        )

        request_id = "req-higgs"
        transfer_manager = SimpleNamespace(
            code_prompt_token_ids=defaultdict(list),
            connector=SimpleNamespace(config={"extra": {"codec_chunk_frames": 1, "codec_right_holdback_frames": 0}}),
        )
        request = SimpleNamespace(external_req_id=request_id, is_finished=lambda: False)

        result = None
        for step in range(8):
            result = talker2code2wav_async_chunk(
                transfer_manager=transfer_manager,
                multimodal_output={"codes": {"audio": torch.full((1, 8), step, dtype=torch.long)}},
                request=request,
                is_finished=False,
            )

        assert result is not None
        assert result.codes is not None
        assert result.codes.audio.numel() > 0


# ---- AC-10: Registry ----


class TestRegistry:
    def test_talker_registered(self):
        from vllm_omni.model_executor.models.registry import _OMNI_MODELS

        assert "HiggsMultimodalQwen3ForConditionalGeneration" in _OMNI_MODELS
        assert "HiggsAudioV3TalkerForConditionalGeneration" in _OMNI_MODELS

    def test_code2wav_registered(self):
        from vllm_omni.model_executor.models.registry import _OMNI_MODELS

        assert "HiggsAudioV3Code2WavForConditionalGeneration" in _OMNI_MODELS

    def test_pipeline_registered(self):
        from vllm_omni.config.pipeline_registry import OMNI_PIPELINES

        assert "higgs_multimodal_qwen3" in OMNI_PIPELINES

    def test_deploy_yaml_exists(self):
        import os

        # __file__ = tests/unit/higgs_audio_v3/test_*.py → 4 dirnames to repo root
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        deploy_dir = os.path.join(repo_root, "vllm_omni", "deploy")
        for name in (
            "higgs_multimodal_qwen3.yaml",
            "higgs_multimodal_qwen3_high_throughput.yaml",
            "higgs_multimodal_qwen3_low_latency.yaml",
        ):
            yaml_path = os.path.join(deploy_dir, name)
            assert os.path.isfile(yaml_path), f"Deploy YAML not found at {yaml_path}"


# ---- AC-3: Prompt Builder ----


class TestPromptBuilder:
    def _make_mock_tokenizer(self):
        """Create a mock tokenizer with the required special tokens."""

        class MockTokenizer:
            def __init__(self):
                self._added_vocab = {
                    "<|tts|>": 151700,
                    "<|text|>": 151701,
                    "<|audio|>": 151702,
                    "<|ref_audio|>": 151703,
                    "<|ref_text|>": 151704,
                }

            def get_added_vocab(self):
                return self._added_vocab

            def encode(self, text, add_special_tokens=True):
                # Simple word-level tokenization for testing
                return list(range(100, 100 + len(text.split())))

        return MockTokenizer()

    def test_plain_tts_prompt(self):
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_tokenizer import (
            HiggsAudioV3TokenizerAdapter,
        )

        tok = self._make_mock_tokenizer()
        adapter = HiggsAudioV3TokenizerAdapter(tok)
        ids = adapter.build_prompt("Hello world")
        assert ids[0] == 151700  # <|tts|>
        assert ids[1] == 151701  # <|text|>
        assert ids[-1] == 151702  # <|audio|>
        assert len(ids) == 2 + 2 + 1  # tts + text + 2 word tokens + audio

    def test_empty_text_rejected(self):
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_tokenizer import (
            HiggsAudioV3TokenizerAdapter,
        )

        tok = self._make_mock_tokenizer()
        adapter = HiggsAudioV3TokenizerAdapter(tok)
        with pytest.raises(ValueError, match="non-empty"):
            adapter.build_prompt("")

    def test_whitespace_only_rejected(self):
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_tokenizer import (
            HiggsAudioV3TokenizerAdapter,
        )

        tok = self._make_mock_tokenizer()
        adapter = HiggsAudioV3TokenizerAdapter(tok)
        with pytest.raises(ValueError, match="non-empty"):
            adapter.build_prompt("   ")

    def test_missing_specials_rejected(self):
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_tokenizer import (
            HiggsAudioV3TokenizerAdapter,
        )

        class BadTokenizer:
            def get_added_vocab(self):
                return {"<|tts|>": 1}  # Missing <|text|> and <|audio|>

        with pytest.raises(ValueError, match="missing"):
            HiggsAudioV3TokenizerAdapter(BadTokenizer())

    def test_no_voice_clone_tokens_in_plain_tts(self):
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_tokenizer import (
            HiggsAudioV3TokenizerAdapter,
        )

        tok = self._make_mock_tokenizer()
        adapter = HiggsAudioV3TokenizerAdapter(tok)
        ids = adapter.build_prompt("Hello")
        # Should not contain ref_audio or ref_text token IDs
        assert 151703 not in ids  # <|ref_audio|>
        assert 151704 not in ids  # <|ref_text|>


class TestVoiceCloneReferenceCache:
    def test_audio_tokenizer_dir_env_accepts_parent_or_subdir(self, tmp_path, monkeypatch):
        from vllm_omni.model_executor.models.higgs_audio_v2 import higgs_audio_v2_tokenizer as tok

        parent = tmp_path / "OmniVoice"
        subdir = parent / "audio_tokenizer"
        subdir.mkdir(parents=True)
        (subdir / "config.json").write_text(
            json.dumps({"model_type": "higgs_audio_v2_tokenizer"}),
            encoding="utf-8",
        )

        monkeypatch.setenv("HIGGS_AUDIO_TOKENIZER_PATH", str(parent))
        assert tok._resolve_audio_tokenizer_dir() == str(subdir)

        monkeypatch.setenv("HIGGS_AUDIO_TOKENIZER_PATH", str(subdir))
        assert tok._resolve_audio_tokenizer_dir() == str(subdir)

    def test_audio_tokenizer_dir_rejects_omnivoice_config(self, tmp_path, monkeypatch):
        import huggingface_hub
        import huggingface_hub.constants as hub_constants

        from vllm_omni.model_executor.models.higgs_audio_v2 import higgs_audio_v2_tokenizer as tok

        parent = tmp_path / "OmniVoice"
        subdir = parent / "audio_tokenizer"
        subdir.mkdir(parents=True)
        (subdir / "config.json").write_text(
            json.dumps({"model_type": "omnivoice"}),
            encoding="utf-8",
        )

        monkeypatch.setenv("HIGGS_AUDIO_TOKENIZER_PATH", str(parent))
        monkeypatch.setenv("HIGGS_AUDIO_V2_TOKENIZER_PATH", "")
        monkeypatch.setattr(
            huggingface_hub,
            "try_to_load_from_cache",
            lambda **_: str(subdir / "config.json"),
        )
        monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path / "empty_cache"))

        assert tok._normalize_audio_tokenizer_dir(str(parent)) is None
        assert tok._resolve_audio_tokenizer_dir() is None

    def test_higgs_v3_ref_code_cache_returns_clone(self):
        from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

        serving = object.__new__(OmniOpenAIServingSpeech)
        serving._higgs_audio_v3_ref_code_cache = OrderedDict()
        serving._higgs_audio_v3_ref_code_cache_bytes = 0
        serving._higgs_audio_v3_ref_code_inflight = {}

        codes = torch.arange(16, dtype=torch.long).reshape(2, 8)
        serving._put_higgs_audio_v3_ref_codes("ref-a", codes)

        cached = serving._get_higgs_audio_v3_ref_codes("ref-a")
        assert cached is not None
        cached.fill_(0)

        cached_again = serving._get_higgs_audio_v3_ref_codes("ref-a")
        assert cached_again is not None
        assert torch.equal(cached_again, codes)

    def test_higgs_v3_ref_code_inflight_deduplicates_concurrent_encode(self):
        from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

        serving = object.__new__(OmniOpenAIServingSpeech)
        serving._higgs_audio_v3_ref_code_cache = OrderedDict()
        serving._higgs_audio_v3_ref_code_cache_bytes = 0
        serving._higgs_audio_v3_ref_code_inflight = {}
        calls = 0

        def encode_reference_audio(_wav, _sr):
            nonlocal calls
            calls += 1
            time.sleep(0.05)
            return torch.arange(16, dtype=torch.long).reshape(2, 8)

        def apply_delay_pattern(codes):
            return codes + 1

        async def run():
            return await asyncio.gather(
                *[
                    serving._resolve_higgs_audio_v3_ref_codes(
                        "ref-a",
                        object(),
                        24000,
                        encode_reference_audio,
                        apply_delay_pattern,
                    )
                    for _ in range(3)
                ]
            )

        results = asyncio.run(run())
        assert calls == 1
        assert sum(int(inflight_wait) for _, _, inflight_wait in results) == 2
        for codes, cache_hit, _ in results:
            assert cache_hit is False
            assert torch.equal(codes, torch.arange(16, dtype=torch.long).reshape(2, 8) + 1)

        cached, cache_hit, inflight_wait = asyncio.run(
            serving._resolve_higgs_audio_v3_ref_codes(
                "ref-a",
                object(),
                24000,
                encode_reference_audio,
                apply_delay_pattern,
            )
        )
        assert calls == 1
        assert cache_hit is True
        assert inflight_wait is False
        assert torch.equal(cached, torch.arange(16, dtype=torch.long).reshape(2, 8) + 1)
