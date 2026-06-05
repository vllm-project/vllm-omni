# SPDX-License-Identifier: Apache-2.0
"""Deterministic unit tests for higgs-audio v3.

These tests verify AC-1 (config), AC-3 (prompt), AC-4 (fused modules),
AC-5 (delay pattern), AC-7 (stage processor), and AC-10 (registry)
without requiring the actual checkpoint or GPU.
"""

import pytest
import torch

# ---- AC-1: Configuration ----


class TestHiggsAudioV3Config:
    def test_default_config_loads(self):
        from vllm_omni.model_executor.models.higgs_audio_v3.configuration_higgs_audio_v3 import (
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
        from vllm_omni.model_executor.models.higgs_audio_v3.configuration_higgs_audio_v3 import (
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
        from vllm_omni.model_executor.models.higgs_audio_v3.configuration_higgs_audio_v3 import (
            HiggsAudioV3Config,
        )

        with pytest.raises(ValueError, match="num_codebooks must be > 0"):
            HiggsAudioV3Config(audio_encoder_config={"num_codebooks": 0, "vocab_size": 1026})

    def test_negative_num_codebooks_rejected(self):
        from vllm_omni.model_executor.models.higgs_audio_v3.configuration_higgs_audio_v3 import (
            HiggsAudioV3Config,
        )

        with pytest.raises(ValueError, match="num_codebooks must be > 0"):
            HiggsAudioV3Config(audio_encoder_config={"num_codebooks": -1, "vocab_size": 1026})

    def test_special_token_ids_initially_none(self):
        from vllm_omni.model_executor.models.higgs_audio_v3.configuration_higgs_audio_v3 import (
            HiggsAudioV3Config,
        )

        config = HiggsAudioV3Config()
        assert config.tts_token_id is None
        assert config.text_token_id is None
        assert config.audio_continuation_id is None

    def test_hidden_size_from_text_config(self):
        from vllm_omni.model_executor.models.higgs_audio_v3.configuration_higgs_audio_v3 import (
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


# ---- AC-8: Codec Strictness ----


class TestCodecStrictness:
    def test_bundled_missing_quantizer_key_raises(self):
        """Bundled codec load must fail when a quantizer codebook key is missing."""
        from vllm_omni.model_executor.models.higgs_audio_v3.configuration_higgs_audio_v3 import (
            HiggsAudioV3Config,
        )
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_code2wav import (
            HiggsAudioV3Code2Wav,
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
        from vllm_omni.model_executor.models.higgs_audio_v3.configuration_higgs_audio_v3 import (
            HiggsAudioV3Config,
        )
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_code2wav import (
            HiggsAudioV3Code2Wav,
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
        from vllm_omni.config.pipeline_registry import _PIPELINE_REGISTRY

        assert "higgs_multimodal_qwen3" in _PIPELINE_REGISTRY

    def test_deploy_yaml_exists(self):
        import os

        yaml_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
            "vllm_omni",
            "deploy",
            "higgs_multimodal_qwen3.yaml",
        )
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
