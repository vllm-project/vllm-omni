"""Unit tests for Qwen3 Omni stage input processors.

Tests the thinker->talker and talker->code2wav transition functions,
with special focus on PD (Prefill-Decode) disaggregation embedding merge
logic that is critical for correct audio generation.

All tests run on CPU without requiring a GPU or model weights.
"""

import warnings
from collections import defaultdict
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Suppress noisy DeprecationWarnings from optional Swig bindings.
warnings.filterwarnings(
    "ignore",
    message=r"builtin type SwigPy.*has no __module__ attribute",
    category=DeprecationWarning,
)

# ---------------------------------------------------------------------------
# Constants mirroring the production code
# ---------------------------------------------------------------------------
_EMBED_LAYER_KEY = "0"
_HIDDEN_LAYER_KEY = "24"

# Token IDs used in _compute_talker_prompt_ids_length
_IM_START_TOKEN_ID = 151644
_SYSTEM_TOKEN_ID = 8948
_USER_TOKEN_ID = 872
_ASSISTANT_TOKEN_ID = 77091


# ---------------------------------------------------------------------------
# Fixture: force CPU device for thinker2talker / talker2code2wav
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _force_cpu_platform(monkeypatch):
    """Ensure current_platform.device_type == 'cpu' for all tests.

    thinker2talker() uses ``torch.device(current_platform.device_type)``
    internally, which would fail on machines without CUDA.
    """
    try:
        import vllm.platforms as plat_mod

        monkeypatch.setattr(plat_mod.current_platform, "device_type", "cpu")
    except (ImportError, AttributeError):
        pass  # vllm not installed with platform support — tests still work


# ---------------------------------------------------------------------------
# Fake stage / output helpers
# ---------------------------------------------------------------------------


class _FakeCompletionOutput:
    """Minimal stand-in for vLLM CompletionOutput."""

    def __init__(
        self,
        token_ids: list[int],
        multimodal_output: dict[str, Any] | None = None,
    ):
        self.token_ids = token_ids
        self.multimodal_output = multimodal_output or {}


class _FakeRequestOutput:
    """Minimal stand-in for vLLM RequestOutput."""

    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        outputs: list[_FakeCompletionOutput],
    ):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.outputs = outputs


class _FakeStage:
    """Lightweight stage stub for testing stage input processors."""

    def __init__(
        self,
        stage_id: int = 0,
        is_prefill_only: bool = False,
        is_decode_only: bool = False,
        engine_outputs: list | None = None,
    ):
        self.stage_id = stage_id
        self.is_prefill_only = is_prefill_only
        self.is_decode_only = is_decode_only
        self.engine_outputs = engine_outputs


def _make_multimodal_output(
    embed: torch.Tensor,
    hidden: torch.Tensor,
    *,
    tts_bos: torch.Tensor | None = None,
    tts_eos: torch.Tensor | None = None,
    tts_pad: torch.Tensor | None = None,
    codec_codes: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Build a multimodal_output dict like the thinker/talker produces."""
    mm: dict[str, Any] = {
        _EMBED_LAYER_KEY: embed,
        _HIDDEN_LAYER_KEY: hidden,
    }
    if tts_bos is not None:
        mm["tts_bos_embed"] = tts_bos
    if tts_eos is not None:
        mm["tts_eos_embed"] = tts_eos
    if tts_pad is not None:
        mm["tts_pad_embed"] = tts_pad
    if codec_codes is not None:
        mm["code_predictor_codes"] = codec_codes
    return mm


def _rand(rows: int, dim: int = 16) -> torch.Tensor:
    return torch.randn(rows, dim)


# ---------------------------------------------------------------------------
# Helpers to build realistic token sequences
# ---------------------------------------------------------------------------


def _build_chat_token_ids(
    user_len: int = 10,
    assistant_generated_len: int = 5,
) -> tuple[list[int], list[int]]:
    """Build (prompt_token_ids, output_token_ids) with realistic structure.

    Structure: <|im_start|>user <tokens...> <|im_start|>assistant
    """
    # User turn: <|im_start|> user <user_tokens...>
    user_turn = [_IM_START_TOKEN_ID, _USER_TOKEN_ID] + list(range(100, 100 + user_len))
    # Assistant turn prefix: <|im_start|> assistant
    assistant_prefix = [_IM_START_TOKEN_ID, _ASSISTANT_TOKEN_ID]
    prompt_token_ids = user_turn + assistant_prefix

    # Generated tokens
    output_token_ids = list(range(200, 200 + assistant_generated_len))

    return prompt_token_ids, output_token_ids


# ===================================================================
# Tests: _merge_pd_embeddings
# ===================================================================


class TestMergePDEmbeddings:
    """Tests for _merge_pd_embeddings() -- the core PD embedding merge."""

    def _import(self):
        from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
            _merge_pd_embeddings,
        )

        return _merge_pd_embeddings

    def test_basic_merge_no_overlap(self):
        """When prefill_len + decode_len == expected_total, no overlap."""
        merge = self._import()
        prefill_emb = _rand(10)
        prefill_hid = _rand(10)
        decode_emb = _rand(5)
        decode_hid = _rand(5)

        prefill_mm = {
            _EMBED_LAYER_KEY: prefill_emb,
            _HIDDEN_LAYER_KEY: prefill_hid,
        }

        merged_emb, merged_hid = merge(
            decode_emb,
            decode_hid,
            prefill_mm,
            device=torch.device("cpu"),
            expected_total=15,
        )

        assert merged_emb.shape[0] == 15
        assert merged_hid.shape[0] == 15
        # First 10 rows should come from prefill
        assert torch.allclose(merged_emb[:10], prefill_emb)
        # Last 5 rows should come from decode
        assert torch.allclose(merged_emb[10:], decode_emb)

    def test_merge_with_overlap(self):
        """When prefill_len + decode_len > expected_total, skip overlap."""
        merge = self._import()
        prefill_emb = _rand(10)
        prefill_hid = _rand(10)
        decode_emb = _rand(8)
        decode_hid = _rand(8)

        prefill_mm = {
            _EMBED_LAYER_KEY: prefill_emb,
            _HIDDEN_LAYER_KEY: prefill_hid,
        }

        # 10 + 8 = 18, expected = 15, so overlap = 3
        merged_emb, merged_hid = merge(
            decode_emb,
            decode_hid,
            prefill_mm,
            device=torch.device("cpu"),
            expected_total=15,
        )

        assert merged_emb.shape[0] == 15
        assert merged_hid.shape[0] == 15
        # First 10 come from prefill
        assert torch.allclose(merged_emb[:10], prefill_emb)
        # Last 5 come from decode[3:] (skipping 3 overlap tokens)
        assert torch.allclose(merged_emb[10:], decode_emb[3:])

    def test_merge_without_expected_total(self):
        """Without expected_total, should concatenate fully (no overlap skip)."""
        merge = self._import()
        prefill_emb = _rand(10)
        prefill_hid = _rand(10)
        decode_emb = _rand(5)
        decode_hid = _rand(5)

        prefill_mm = {
            _EMBED_LAYER_KEY: prefill_emb,
            _HIDDEN_LAYER_KEY: prefill_hid,
        }

        merged_emb, merged_hid = merge(
            decode_emb,
            decode_hid,
            prefill_mm,
            device=torch.device("cpu"),
            expected_total=None,
        )

        assert merged_emb.shape[0] == 15
        assert merged_hid.shape[0] == 15

    def test_merge_preserves_hidden_consistency(self):
        """Embedding and hidden state merges should have same length."""
        merge = self._import()
        dim_emb, dim_hid = 16, 32
        prefill_emb = torch.randn(10, dim_emb)
        prefill_hid = torch.randn(10, dim_hid)
        decode_emb = torch.randn(5, dim_emb)
        decode_hid = torch.randn(5, dim_hid)

        prefill_mm = {
            _EMBED_LAYER_KEY: prefill_emb,
            _HIDDEN_LAYER_KEY: prefill_hid,
        }

        merged_emb, merged_hid = merge(
            decode_emb,
            decode_hid,
            prefill_mm,
            device=torch.device("cpu"),
            expected_total=15,
        )

        assert merged_emb.shape == (15, dim_emb)
        assert merged_hid.shape == (15, dim_hid)

    def test_empty_prefill_returns_decode_only(self):
        """If prefill embeddings are empty, return decode unchanged."""
        merge = self._import()
        decode_emb = _rand(5)
        decode_hid = _rand(5)

        prefill_mm = {
            _EMBED_LAYER_KEY: torch.empty(0, 16),
            _HIDDEN_LAYER_KEY: torch.empty(0, 16),
        }

        merged_emb, merged_hid = merge(
            decode_emb,
            decode_hid,
            prefill_mm,
            device=torch.device("cpu"),
            expected_total=5,
        )

        assert torch.equal(merged_emb, decode_emb)
        assert torch.equal(merged_hid, decode_hid)

    def test_empty_decode_returns_decode_only(self):
        """If decode embeddings are empty, return decode unchanged."""
        merge = self._import()
        decode_emb = torch.empty(0, 16)
        decode_hid = torch.empty(0, 16)

        prefill_mm = {
            _EMBED_LAYER_KEY: _rand(10),
            _HIDDEN_LAYER_KEY: _rand(10),
        }

        merged_emb, merged_hid = merge(
            decode_emb,
            decode_hid,
            prefill_mm,
            device=torch.device("cpu"),
            expected_total=10,
        )

        # When decode is empty, function returns decode unchanged
        assert torch.equal(merged_emb, decode_emb)
        assert torch.equal(merged_hid, decode_hid)

    def test_missing_key_returns_decode_unchanged(self):
        """If prefill_mm is missing required keys, return decode as-is."""
        merge = self._import()
        decode_emb = _rand(5)
        decode_hid = _rand(5)

        # Missing _EMBED_LAYER_KEY
        prefill_mm = {_HIDDEN_LAYER_KEY: _rand(10)}

        merged_emb, merged_hid = merge(
            decode_emb,
            decode_hid,
            prefill_mm,
            device=torch.device("cpu"),
            expected_total=15,
        )

        assert torch.equal(merged_emb, decode_emb)
        assert torch.equal(merged_hid, decode_hid)

    def test_overlap_equals_decode_len_gives_prefill_only(self):
        """If computed overlap >= decode_len, all decode tokens are skipped."""
        merge = self._import()
        prefill_emb = _rand(10)
        prefill_hid = _rand(10)
        decode_emb = _rand(3)
        decode_hid = _rand(3)

        prefill_mm = {
            _EMBED_LAYER_KEY: prefill_emb,
            _HIDDEN_LAYER_KEY: prefill_hid,
        }

        # 10 + 3 = 13, expected = 10 -> overlap = 3 -> decode[3:] is empty
        merged_emb, merged_hid = merge(
            decode_emb,
            decode_hid,
            prefill_mm,
            device=torch.device("cpu"),
            expected_total=10,
        )

        # Result should be just the prefill embeddings
        assert merged_emb.shape[0] == 10
        assert torch.allclose(merged_emb, prefill_emb)

    def test_expected_total_smaller_than_both_uses_no_overlap(self):
        """When raw_total <= expected_total, overlap=0 (simple concat)."""
        merge = self._import()
        prefill_emb = _rand(5)
        prefill_hid = _rand(5)
        decode_emb = _rand(3)
        decode_hid = _rand(3)

        prefill_mm = {
            _EMBED_LAYER_KEY: prefill_emb,
            _HIDDEN_LAYER_KEY: prefill_hid,
        }

        # 5 + 3 = 8 <= 20 -> overlap = 0
        merged_emb, merged_hid = merge(
            decode_emb,
            decode_hid,
            prefill_mm,
            device=torch.device("cpu"),
            expected_total=20,
        )

        assert merged_emb.shape[0] == 8
        assert torch.allclose(merged_emb[:5], prefill_emb)
        assert torch.allclose(merged_emb[5:], decode_emb)


# ===================================================================
# Tests: _get_prefill_stage
# ===================================================================


class TestGetPrefillStage:
    """Tests for _get_prefill_stage() -- prefill stage detection for PD mode."""

    def _import(self):
        from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
            _get_prefill_stage,
        )

        return _get_prefill_stage

    def test_returns_prefill_stage_when_pd_active(self):
        """Should return the prefill stage when source is decode-only."""
        get_prefill = self._import()

        prefill = _FakeStage(
            stage_id=0,
            is_prefill_only=True,
            engine_outputs=[MagicMock()],
        )
        decode = _FakeStage(stage_id=1, is_decode_only=True)
        stage_list = [prefill, decode]

        result = get_prefill(stage_list, source_stage_id=1)
        assert result is prefill

    def test_returns_none_when_source_is_not_decode_only(self):
        """Non-PD pipeline: source stage is not decode-only."""
        get_prefill = self._import()

        thinker = _FakeStage(stage_id=0)
        talker = _FakeStage(stage_id=1)
        stage_list = [thinker, talker]

        result = get_prefill(stage_list, source_stage_id=1)
        assert result is None

    def test_returns_none_when_source_id_is_zero(self):
        """First stage cannot have a preceding prefill."""
        get_prefill = self._import()

        stage_list = [_FakeStage(stage_id=0)]
        result = get_prefill(stage_list, source_stage_id=0)
        assert result is None

    def test_returns_none_when_prefill_has_no_outputs(self):
        """Prefill stage exists but has no outputs yet."""
        get_prefill = self._import()

        prefill = _FakeStage(
            stage_id=0,
            is_prefill_only=True,
            engine_outputs=None,
        )
        decode = _FakeStage(stage_id=1, is_decode_only=True)
        stage_list = [prefill, decode]

        result = get_prefill(stage_list, source_stage_id=1)
        assert result is None

    def test_returns_none_when_prev_not_prefill_only(self):
        """Previous stage is not marked as prefill-only."""
        get_prefill = self._import()

        normal = _FakeStage(
            stage_id=0,
            is_prefill_only=False,
            engine_outputs=[MagicMock()],
        )
        decode = _FakeStage(stage_id=1, is_decode_only=True)
        stage_list = [normal, decode]

        result = get_prefill(stage_list, source_stage_id=1)
        assert result is None


# ===================================================================
# Tests: thinker2talker (non-PD mode)
# ===================================================================


class TestThinker2TalkerNonPD:
    """Tests for thinker2talker() in standard (non-PD) mode."""

    def _import(self):
        from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
            thinker2talker,
        )

        return thinker2talker

    def _make_thinker_output(
        self,
        prompt_len: int = 14,
        output_len: int = 5,
        dim: int = 16,
    ) -> _FakeRequestOutput:
        """Create a fake thinker output with correct embeddings."""
        prompt_ids, output_ids = _build_chat_token_ids(
            user_len=prompt_len - 4,  # subtract im_start + user + im_start + assistant
            assistant_generated_len=output_len,
        )
        total_len = len(prompt_ids) + len(output_ids)

        mm_output = _make_multimodal_output(
            embed=_rand(total_len, dim),
            hidden=_rand(total_len, dim),
            tts_bos=_rand(1, dim),
            tts_eos=_rand(1, dim),
            tts_pad=_rand(1, dim),
        )

        return _FakeRequestOutput(
            request_id="req-0",
            prompt_token_ids=prompt_ids,
            outputs=[
                _FakeCompletionOutput(
                    token_ids=output_ids,
                    multimodal_output=mm_output,
                )
            ],
        )

    def test_produces_talker_input(self):
        """Basic thinker2talker should produce a valid OmniTokensPrompt."""
        thinker2talker = self._import()
        thinker_out = self._make_thinker_output()

        thinker = _FakeStage(stage_id=0, engine_outputs=[thinker_out])
        stage_list = [thinker]

        results = thinker2talker(stage_list, engine_input_source=[0])

        assert len(results) == 1
        result = results[0]
        assert "prompt_token_ids" in result
        assert "additional_information" in result

        info = result["additional_information"]
        assert "thinker_prefill_embeddings" in info
        assert "thinker_hidden_states" in info
        assert "thinker_sequences" in info
        assert "thinker_input_ids" in info
        assert "tts_bos_embed" in info
        assert "tts_eos_embed" in info
        assert "tts_pad_embed" in info

    def test_prompt_token_ids_has_correct_length(self):
        """prompt_token_ids length should match _compute_talker_prompt_ids_length."""
        thinker2talker = self._import()
        thinker_out = self._make_thinker_output(prompt_len=14, output_len=5)

        thinker = _FakeStage(stage_id=0, engine_outputs=[thinker_out])
        results = thinker2talker([thinker], engine_input_source=[0])

        # prompt_token_ids is [0]*prompt_len, should be all zeros
        prompt_tids = results[0]["prompt_token_ids"]
        assert all(t == 0 for t in prompt_tids)
        assert len(prompt_tids) > 0

    def test_thinker_sequences_is_full_concat(self):
        """thinker_sequences should be prompt + output token ids."""
        thinker2talker = self._import()
        thinker_out = self._make_thinker_output(prompt_len=14, output_len=5)

        thinker = _FakeStage(stage_id=0, engine_outputs=[thinker_out])
        results = thinker2talker([thinker], engine_input_source=[0])
        info = results[0]["additional_information"]

        expected_seq = thinker_out.prompt_token_ids + thinker_out.outputs[0].token_ids
        assert info["thinker_sequences"] == expected_seq

    def test_thinker_input_ids_is_prompt_only(self):
        """thinker_input_ids should be only the prompt token ids."""
        thinker2talker = self._import()
        thinker_out = self._make_thinker_output()

        thinker = _FakeStage(stage_id=0, engine_outputs=[thinker_out])
        results = thinker2talker([thinker], engine_input_source=[0])
        info = results[0]["additional_information"]

        assert info["thinker_input_ids"] == thinker_out.prompt_token_ids

    def test_embeddings_shape_matches_total_sequence(self):
        """Embeddings dim-0 should equal len(prompt) + len(output)."""
        thinker2talker = self._import()
        prompt_len, output_len, dim = 14, 5, 16
        thinker_out = self._make_thinker_output(prompt_len, output_len, dim)

        thinker = _FakeStage(stage_id=0, engine_outputs=[thinker_out])
        results = thinker2talker([thinker], engine_input_source=[0])
        info = results[0]["additional_information"]

        total = len(thinker_out.prompt_token_ids) + len(thinker_out.outputs[0].token_ids)
        assert info["thinker_prefill_embeddings"].shape[0] == total
        assert info["thinker_hidden_states"].shape[0] == total

    def test_invalid_stage_raises(self):
        """Empty engine_input_source should raise."""
        thinker2talker = self._import()
        with pytest.raises(ValueError, match="cannot be empty"):
            thinker2talker([], engine_input_source=[])

    def test_no_outputs_raises(self):
        """Stage with no outputs should raise."""
        thinker2talker = self._import()
        thinker = _FakeStage(stage_id=0, engine_outputs=None)

        with pytest.raises(RuntimeError, match="no outputs"):
            thinker2talker([thinker], engine_input_source=[0])

    def test_multiple_outputs(self):
        """Multiple thinker outputs should produce multiple talker inputs."""
        thinker2talker = self._import()
        out1 = self._make_thinker_output(prompt_len=14, output_len=3)
        out2 = self._make_thinker_output(prompt_len=14, output_len=7)

        thinker = _FakeStage(stage_id=0, engine_outputs=[out1, out2])
        results = thinker2talker([thinker], engine_input_source=[0])
        assert len(results) == 2


# ===================================================================
# Tests: thinker2talker (PD mode)
# ===================================================================


class TestThinker2TalkerPDMode:
    """Tests for thinker2talker() when PD disaggregation is active.

    In PD mode:
    - Stage 0 = prefill (is_prefill_only=True), has prompt embeddings
    - Stage 1 = decode (is_decode_only=True), has generated embeddings
    - thinker2talker should merge both to form the full sequence
    """

    def _import(self):
        from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
            thinker2talker,
        )

        return thinker2talker

    def _make_pd_stages(
        self,
        prompt_len: int = 14,
        output_len: int = 5,
        dim: int = 16,
        overlap: int = 0,
        prefill_has_tts: bool = True,
        decode_has_tts: bool = True,
    ):
        """Build prefill + decode stages for PD testing.

        Returns (stage_list, expected_total_embeddings).
        """
        prompt_ids, output_ids = _build_chat_token_ids(
            user_len=prompt_len - 4,
            assistant_generated_len=output_len,
        )
        total_len = len(prompt_ids) + len(output_ids)

        # Prefill: embeddings for prompt tokens
        prefill_emb_len = len(prompt_ids)
        prefill_mm = _make_multimodal_output(
            embed=_rand(prefill_emb_len, dim),
            hidden=_rand(prefill_emb_len, dim),
            tts_bos=_rand(1, dim) if prefill_has_tts else None,
            tts_eos=_rand(1, dim) if prefill_has_tts else None,
            tts_pad=_rand(1, dim) if prefill_has_tts else None,
        )

        prefill_out = _FakeRequestOutput(
            request_id="req-0",
            prompt_token_ids=prompt_ids,
            outputs=[
                _FakeCompletionOutput(
                    token_ids=[output_ids[0]],  # prefill produces 1 token
                    multimodal_output=prefill_mm,
                )
            ],
        )

        # Decode: embeddings for generated tokens (+ possible overlap)
        decode_emb_len = len(output_ids) + overlap
        decode_mm = _make_multimodal_output(
            embed=_rand(decode_emb_len, dim),
            hidden=_rand(decode_emb_len, dim),
            tts_bos=_rand(1, dim) if decode_has_tts else None,
            tts_eos=_rand(1, dim) if decode_has_tts else None,
            tts_pad=_rand(1, dim) if decode_has_tts else None,
        )

        decode_out = _FakeRequestOutput(
            request_id="req-0",
            prompt_token_ids=prompt_ids,
            outputs=[
                _FakeCompletionOutput(
                    token_ids=output_ids,
                    multimodal_output=decode_mm,
                )
            ],
        )

        prefill_stage = _FakeStage(
            stage_id=0,
            is_prefill_only=True,
            engine_outputs=[prefill_out],
        )
        decode_stage = _FakeStage(
            stage_id=1,
            is_decode_only=True,
            engine_outputs=[decode_out],
        )

        return [prefill_stage, decode_stage], total_len

    def test_pd_merge_basic(self):
        """PD mode should merge prefill + decode embeddings."""
        thinker2talker = self._import()
        stage_list, total_len = self._make_pd_stages(
            prompt_len=14,
            output_len=5,
            overlap=0,
        )

        results = thinker2talker(stage_list, engine_input_source=[1])

        assert len(results) == 1
        info = results[0]["additional_information"]
        emb = info["thinker_prefill_embeddings"]
        hid = info["thinker_hidden_states"]
        # Merged length should equal prompt + output
        assert emb.shape[0] == total_len
        assert hid.shape[0] == total_len

    def test_pd_merge_with_overlap(self):
        """PD mode with overlapping tokens should handle deduplication."""
        thinker2talker = self._import()
        stage_list, total_len = self._make_pd_stages(
            prompt_len=14,
            output_len=5,
            overlap=2,
        )

        results = thinker2talker(stage_list, engine_input_source=[1])
        info = results[0]["additional_information"]
        emb = info["thinker_prefill_embeddings"]

        # Should still be total_len despite overlap
        assert emb.shape[0] == total_len

    def test_pd_tts_fallback_to_prefill(self):
        """When decode lacks TTS embeds, should fall back to prefill's."""
        thinker2talker = self._import()
        stage_list, _ = self._make_pd_stages(
            prompt_len=14,
            output_len=5,
            prefill_has_tts=True,
            decode_has_tts=False,
        )

        results = thinker2talker(stage_list, engine_input_source=[1])
        info = results[0]["additional_information"]

        # TTS embeds should be present (from prefill fallback)
        assert info["tts_bos_embed"] is not None
        assert info["tts_eos_embed"] is not None
        assert info["tts_pad_embed"] is not None

    def test_pd_tts_from_decode_when_available(self):
        """When decode has TTS embeds, should use them (not prefill's)."""
        thinker2talker = self._import()
        stage_list, _ = self._make_pd_stages(
            prompt_len=14,
            output_len=5,
            prefill_has_tts=True,
            decode_has_tts=True,
        )

        # Get the decode stage's TTS embed for comparison
        decode_tts_bos = stage_list[1].engine_outputs[0].outputs[0].multimodal_output["tts_bos_embed"]

        results = thinker2talker(stage_list, engine_input_source=[1])
        info = results[0]["additional_information"]

        # _tts() does: val.detach().to(device=cpu, dtype=torch.float)
        # decode_tts_bos is already float32 on CPU from _rand(), so values match
        assert torch.equal(info["tts_bos_embed"], decode_tts_bos.detach().float())

    def test_pd_no_tts_anywhere_gives_none(self):
        """When neither decode nor prefill has TTS embeds, result is None."""
        thinker2talker = self._import()
        stage_list, _ = self._make_pd_stages(
            prompt_len=14,
            output_len=5,
            prefill_has_tts=False,
            decode_has_tts=False,
        )

        results = thinker2talker(stage_list, engine_input_source=[1])
        info = results[0]["additional_information"]

        assert info["tts_bos_embed"] is None
        assert info["tts_eos_embed"] is None
        assert info["tts_pad_embed"] is None

    def test_pd_sequences_are_full(self):
        """In PD mode, thinker_sequences should still be full prompt + output."""
        thinker2talker = self._import()
        stage_list, _ = self._make_pd_stages(prompt_len=14, output_len=5)

        results = thinker2talker(stage_list, engine_input_source=[1])
        info = results[0]["additional_information"]

        decode_out = stage_list[1].engine_outputs[0]
        expected_seq = decode_out.prompt_token_ids + decode_out.outputs[0].token_ids
        assert info["thinker_sequences"] == expected_seq

    def test_pd_prefill_merge_error_is_graceful(self):
        """If prefill embeddings are corrupted, merge fails gracefully
        and decode embeddings are used as-is (logged as warning)."""
        thinker2talker = self._import()
        stage_list, _ = self._make_pd_stages(prompt_len=14, output_len=5)

        # Corrupt prefill multimodal_output to trigger exception in merge
        stage_list[0].engine_outputs[0].outputs[0].multimodal_output = "not-a-dict"

        # Should not raise; falls back to decode-only embeddings
        results = thinker2talker(stage_list, engine_input_source=[1])
        assert len(results) == 1
        info = results[0]["additional_information"]
        assert info["thinker_prefill_embeddings"] is not None


# ===================================================================
# Tests: talker2code2wav
# ===================================================================


class TestTalker2Code2Wav:
    """Tests for talker2code2wav() -- the talker -> code2wav transition."""

    def _import(self):
        from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
            talker2code2wav,
        )

        return talker2code2wav

    def _make_talker_output(
        self,
        seq_len: int = 20,
        num_quantizers: int = 8,
    ) -> _FakeRequestOutput:
        """Create a fake talker output with codec codes.

        The talker produces token_ids of length seq_len+1 (including a
        start/padding token), and codec codes of shape
        [num_quantizers, seq_len+1].  talker2code2wav slices the last
        seq_len columns via ``codes[:, -seq_len:]``.
        """
        codec_codes = torch.randint(0, 1024, (num_quantizers, seq_len + 1))
        mm_output = {"code_predictor_codes": codec_codes}

        # token_ids length = seq_len + 1 (code uses len(token_ids) - 1)
        token_ids = list(range(seq_len + 1))

        return _FakeRequestOutput(
            request_id="req-0",
            prompt_token_ids=[0] * 10,  # dummy prompt
            outputs=[
                _FakeCompletionOutput(
                    token_ids=token_ids,
                    multimodal_output=mm_output,
                )
            ],
        )

    def test_produces_code2wav_input(self):
        """Should produce a valid OmniTokensPrompt with flattened codec codes."""
        talker2code2wav = self._import()
        talker_out = self._make_talker_output(seq_len=20, num_quantizers=8)

        talker = _FakeStage(stage_id=1, engine_outputs=[talker_out])
        stage_list = [_FakeStage(stage_id=0), talker]

        results = talker2code2wav(stage_list, engine_input_source=[1])

        assert len(results) == 1
        result = results[0]
        assert "prompt_token_ids" in result
        # Flattened: seq_len * num_quantizers
        assert len(result["prompt_token_ids"]) == 20 * 8

    def test_flattened_code_values_match_source(self):
        """Flattened codes should match transposed + reshaped original."""
        talker2code2wav = self._import()
        seq_len = 10
        num_q = 8
        talker_out = self._make_talker_output(seq_len=seq_len, num_quantizers=num_q)

        talker = _FakeStage(stage_id=1, engine_outputs=[talker_out])
        stage_list = [_FakeStage(stage_id=0), talker]

        results = talker2code2wav(stage_list, engine_input_source=[1])
        result_codes = results[0]["prompt_token_ids"]

        # Manually compute expected: codes[:, -seq_len:].transpose(0,1).reshape(-1)
        original_codes = talker_out.outputs[0].multimodal_output["code_predictor_codes"]
        expected = original_codes[:, -seq_len:].to(torch.long).transpose(0, 1).reshape(-1).tolist()
        assert result_codes == expected

    def test_codes_are_all_ints(self):
        """All flattened codes should be Python ints (for serialization)."""
        talker2code2wav = self._import()
        talker_out = self._make_talker_output(seq_len=15, num_quantizers=8)

        talker = _FakeStage(stage_id=1, engine_outputs=[talker_out])
        stage_list = [_FakeStage(stage_id=0), talker]

        results = talker2code2wav(stage_list, engine_input_source=[1])
        assert all(isinstance(c, int) for c in results[0]["prompt_token_ids"])

    def test_multiple_talker_outputs(self):
        """Should handle multiple talker outputs (batch)."""
        talker2code2wav = self._import()
        out1 = self._make_talker_output(seq_len=10, num_quantizers=8)
        out2 = self._make_talker_output(seq_len=15, num_quantizers=8)

        talker = _FakeStage(stage_id=1, engine_outputs=[out1, out2])
        stage_list = [_FakeStage(stage_id=0), talker]

        results = talker2code2wav(stage_list, engine_input_source=[1])
        assert len(results) == 2
        assert len(results[0]["prompt_token_ids"]) == 10 * 8
        assert len(results[1]["prompt_token_ids"]) == 15 * 8

    def test_16_quantizer_layers(self):
        """Should work with 16-layer RVQ (Qwen3-Omni-MoE)."""
        talker2code2wav = self._import()
        talker_out = self._make_talker_output(seq_len=20, num_quantizers=16)

        talker = _FakeStage(stage_id=1, engine_outputs=[talker_out])
        stage_list = [_FakeStage(stage_id=0), talker]

        results = talker2code2wav(stage_list, engine_input_source=[1])
        assert len(results[0]["prompt_token_ids"]) == 20 * 16

    def test_no_outputs_raises(self):
        """Should raise when talker has no outputs."""
        talker2code2wav = self._import()
        talker = _FakeStage(stage_id=1, engine_outputs=None)
        stage_list = [_FakeStage(stage_id=0), talker]

        with pytest.raises(RuntimeError, match="no outputs"):
            talker2code2wav(stage_list, engine_input_source=[1])


# ===================================================================
# Tests: _compute_talker_prompt_ids_length
# ===================================================================


class TestComputeTalkerPromptIdsLength:
    """Tests for _compute_talker_prompt_ids_length() -- determines
    how many prompt tokens the talker needs for masking.
    """

    def _import(self):
        from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
            _compute_talker_prompt_ids_length,
        )

        return _compute_talker_prompt_ids_length

    def test_user_turn_only(self):
        """Single user turn: user_len + assistant(9)."""
        compute = self._import()
        prompt_ids, output_ids = _build_chat_token_ids(
            user_len=10,
            assistant_generated_len=5,
        )
        full_seq = prompt_ids + output_ids

        info = {
            "thinker_sequences": full_seq,
            "thinker_input_ids": prompt_ids,
        }

        result = compute(info, device="cpu")
        # User turn: im_start(1) + user(1) + 10 tokens = 12
        # Assistant turn: 9  (hard-coded constant in source)
        assert result == 12 + 9

    def test_with_system_turn(self):
        """System turn should be skipped in counting."""
        compute = self._import()

        # System turn + user turn + assistant
        system_turn = [_IM_START_TOKEN_ID, _SYSTEM_TOKEN_ID] + [999] * 5
        user_turn = [_IM_START_TOKEN_ID, _USER_TOKEN_ID] + list(range(100, 110))
        assistant_prefix = [_IM_START_TOKEN_ID, _ASSISTANT_TOKEN_ID]
        prompt_ids = system_turn + user_turn + assistant_prefix
        output_ids = [200, 201, 202]
        full_seq = prompt_ids + output_ids

        info = {
            "thinker_sequences": full_seq,
            "thinker_input_ids": prompt_ids,
        }

        result = compute(info, device="cpu")
        # System turn: skipped
        # User turn: im_start(1) + user(1) + 10 tokens = 12
        # Assistant: 9
        assert result == 12 + 9

    def test_returns_positive_for_empty_generation(self):
        """Even with zero generated tokens, result should be positive."""
        compute = self._import()
        prompt_ids, _ = _build_chat_token_ids(user_len=10, assistant_generated_len=0)
        full_seq = list(prompt_ids)  # no output tokens

        info = {
            "thinker_sequences": full_seq,
            "thinker_input_ids": prompt_ids,
        }

        result = compute(info, device="cpu")
        assert result > 0


# ===================================================================
# Tests: _ensure_list
# ===================================================================


class TestEnsureList:
    """Tests for _ensure_list() -- converts various list-like types."""

    def _import(self):
        from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
            _ensure_list,
        )

        return _ensure_list

    def test_regular_list(self):
        ensure = self._import()
        assert ensure([1, 2, 3]) == [1, 2, 3]

    def test_constant_list_with_x(self):
        """ConstantList objects have a _x attribute."""
        ensure = self._import()

        class FakeConstantList:
            def __init__(self, data):
                self._x = data

        result = ensure(FakeConstantList([4, 5, 6]))
        assert result == [4, 5, 6]

    def test_non_list_passthrough(self):
        """Non-list, non-ConstantList values pass through unchanged."""
        ensure = self._import()
        assert ensure("hello") == "hello"
        assert ensure(42) == 42

    def test_tuple_passthrough(self):
        """Tuples don't have _x, aren't lists -> pass through."""
        ensure = self._import()
        result = ensure((1, 2))
        assert result == (1, 2)


# ===================================================================
# Tests: _validate_stage_inputs
# ===================================================================


class TestValidateStageInputs:
    """Tests for _validate_stage_inputs()."""

    def _import(self):
        from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
            _validate_stage_inputs,
        )

        return _validate_stage_inputs

    def test_returns_outputs_on_valid(self):
        validate = self._import()
        outputs = [MagicMock()]
        stage = _FakeStage(stage_id=0, engine_outputs=outputs)

        result = validate([stage], engine_input_source=[0])
        assert result is outputs

    def test_empty_source_raises(self):
        validate = self._import()
        with pytest.raises(ValueError, match="cannot be empty"):
            validate([], engine_input_source=[])

    def test_invalid_stage_id_raises(self):
        validate = self._import()
        with pytest.raises(IndexError, match="Invalid stage_id"):
            validate([_FakeStage(stage_id=0)], engine_input_source=[5])

    def test_no_outputs_raises(self):
        validate = self._import()
        stage = _FakeStage(stage_id=0, engine_outputs=None)
        with pytest.raises(RuntimeError, match="no outputs"):
            validate([stage], engine_input_source=[0])


# ===================================================================
# Tests: talker2code2wav_async_chunk
# ===================================================================


class TestTalker2Code2WavAsyncChunk:
    """Tests for talker2code2wav_async_chunk() -- chunked codec code processing."""

    def _import(self):
        from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
            talker2code2wav_async_chunk,
        )

        return talker2code2wav_async_chunk

    def _make_transfer_manager(self):
        mgr = MagicMock()
        mgr.code_prompt_token_ids = defaultdict(list)
        return mgr

    def _make_request(self, request_id="req-0", is_finished=False):
        req = MagicMock()
        req.external_req_id = request_id
        req.is_finished = MagicMock(return_value=is_finished)
        return req

    def test_returns_none_when_no_codec_codes_key(self):
        """Should return None when pooling output lacks code_predictor_codes."""
        chunk_fn = self._import()
        mgr = self._make_transfer_manager()
        req = self._make_request()

        result = chunk_fn(mgr, {}, req)
        assert result is None

    def test_returns_none_for_none_codes(self):
        """Should return None when code_predictor_codes is None."""
        chunk_fn = self._import()
        mgr = self._make_transfer_manager()
        req = self._make_request()

        result = chunk_fn(mgr, {"code_predictor_codes": None}, req)
        assert result is None

    def test_returns_none_for_empty_tensor(self):
        """Should return None for empty tensor."""
        chunk_fn = self._import()
        mgr = self._make_transfer_manager()
        req = self._make_request()

        result = chunk_fn(mgr, {"code_predictor_codes": torch.empty(0)}, req)
        assert result is None

    def test_returns_none_for_all_zero_tensor(self):
        """Should return None when all codes are zero."""
        chunk_fn = self._import()
        mgr = self._make_transfer_manager()
        req = self._make_request()

        codes = torch.zeros(8, 10, dtype=torch.long)
        result = chunk_fn(mgr, {"code_predictor_codes": codes}, req)
        assert result is None

    def test_returns_info_when_finished(self):
        """Should return info dict when request is finished (any chunk count)."""
        chunk_fn = self._import()
        mgr = self._make_transfer_manager()
        req = self._make_request(is_finished=True)

        codes = torch.randint(1, 100, (8, 5))
        result = chunk_fn(mgr, {"code_predictor_codes": codes}, req, is_finished=True)

        assert result is not None
        assert "code_predictor_codes" in result
        assert "finished" in result
        assert result["finished"].item() is True

    def test_buffers_until_chunk_boundary(self):
        """Should buffer codec codes and only emit at chunk_size=25 boundary."""
        chunk_fn = self._import()
        mgr = self._make_transfer_manager()
        req = self._make_request(request_id="req-0", is_finished=False)

        codes = torch.randint(1, 100, (8, 5))

        # 24 calls: not at chunk boundary (24 % 25 != 0), not finished -> None
        for _ in range(24):
            result = chunk_fn(mgr, {"code_predictor_codes": codes}, req)
            assert result is None

        # 25th call hits boundary (25 % 25 == 0) -> output emitted
        result = chunk_fn(mgr, {"code_predictor_codes": codes}, req)
        assert result is not None
        assert "code_predictor_codes" in result


# ===================================================================
# Tests: thinker2talker_async_chunk
# ===================================================================


class TestThinker2TalkerAsyncChunk:
    """Tests for thinker2talker_async_chunk() -- pooling-based transition."""

    def _import(self):
        from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
            thinker2talker_async_chunk,
        )

        return thinker2talker_async_chunk

    def _make_transfer_manager(self):
        mgr = MagicMock()
        mgr.put_req_chunk = defaultdict(int)
        mgr.request_payload = {}
        return mgr

    def _make_request(self, request_id="req-0", is_finished=True):
        req = MagicMock()
        req.external_req_id = request_id
        req.all_token_ids = [1, 2, 3, 4, 5]
        req.prompt_token_ids = [1, 2, 3]
        req.output_token_ids = [4, 5]
        req.is_finished = MagicMock(return_value=is_finished)
        return req

    def test_first_chunk_finished_returns_info(self):
        """First chunk with finished request should return full info."""
        chunk_fn = self._import()
        mgr = self._make_transfer_manager()
        req = self._make_request(is_finished=True)

        pooling_output = {
            _EMBED_LAYER_KEY: _rand(5),
            _HIDDEN_LAYER_KEY: _rand(5),
            "tts_bos_embed": _rand(1),
            "tts_eos_embed": _rand(1),
            "tts_pad_embed": _rand(1),
        }

        result = chunk_fn(mgr, pooling_output, req, is_finished=True)

        assert result is not None
        assert "thinker_prefill_embeddings" in result
        assert "thinker_hidden_states" in result
        assert "thinker_sequences" in result
        assert "thinker_input_ids" in result
        assert "tts_bos_embed" in result
        assert result["finished"].item() is True

    def test_first_chunk_not_finished_stores_payload(self):
        """First chunk with unfinished request should store payload, return None."""
        chunk_fn = self._import()
        mgr = self._make_transfer_manager()
        req = self._make_request(is_finished=False)

        pooling_output = {
            _EMBED_LAYER_KEY: _rand(3),
            _HIDDEN_LAYER_KEY: _rand(3),
            "tts_bos_embed": _rand(1),
            "tts_eos_embed": _rand(1),
            "tts_pad_embed": _rand(1),
        }

        result = chunk_fn(mgr, pooling_output, req)

        assert result is None
        assert "req-0" in mgr.request_payload

    def test_second_chunk_merges_with_stored(self):
        """Second call with stored payload should concatenate embeddings."""
        chunk_fn = self._import()
        mgr = self._make_transfer_manager()
        req1 = self._make_request(is_finished=False)

        # First call: store partial
        pooling_output_1 = {
            _EMBED_LAYER_KEY: _rand(3),
            _HIDDEN_LAYER_KEY: _rand(3),
            "tts_bos_embed": _rand(1),
            "tts_eos_embed": _rand(1),
            "tts_pad_embed": _rand(1),
        }
        chunk_fn(mgr, pooling_output_1, req1)

        # Second call: finished request, should merge with stored
        req2 = self._make_request(is_finished=True)
        pooling_output_2 = {
            _EMBED_LAYER_KEY: _rand(2),
            _HIDDEN_LAYER_KEY: _rand(2),
            "tts_bos_embed": _rand(1),
            "tts_eos_embed": _rand(1),
            "tts_pad_embed": _rand(1),
        }
        result = chunk_fn(mgr, pooling_output_2, req2)

        assert result is not None
        # Embeddings should be merged: 3 + 2 = 5
        assert result["thinker_prefill_embeddings"].shape[0] == 5
        assert result["thinker_hidden_states"].shape[0] == 5

    def test_sequences_are_all_token_ids(self):
        """thinker_sequences should be all_token_ids (prompt + decode)."""
        chunk_fn = self._import()
        mgr = self._make_transfer_manager()
        req = self._make_request(is_finished=True)
        req.all_token_ids = [10, 20, 30, 40, 50]

        pooling_output = {
            _EMBED_LAYER_KEY: _rand(5),
            _HIDDEN_LAYER_KEY: _rand(5),
            "tts_bos_embed": _rand(1),
            "tts_eos_embed": _rand(1),
            "tts_pad_embed": _rand(1),
        }

        result = chunk_fn(mgr, pooling_output, req, is_finished=True)
        assert result["thinker_sequences"] == [10, 20, 30, 40, 50]


# ===================================================================
# Tests: Full PD audio pipeline integration (unit-level)
# ===================================================================


class TestPDAudioPipelineIntegration:
    """Integration-style tests verifying the full PD audio data flow:
    prefill -> decode -> thinker2talker -> talker2code2wav

    These test the complete chain at the unit level (no GPU, no model).
    """

    def test_full_pd_audio_chain(self):
        """Simulate a full PD audio pipeline and verify data flows correctly
        through all transition functions."""
        from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
            talker2code2wav,
            thinker2talker,
        )

        dim = 16
        prompt_ids, output_ids = _build_chat_token_ids(
            user_len=10,
            assistant_generated_len=5,
        )
        total_len = len(prompt_ids) + len(output_ids)

        # --- Stage 0: Prefill (produces prompt embeddings) ---
        prefill_mm = _make_multimodal_output(
            embed=_rand(len(prompt_ids), dim),
            hidden=_rand(len(prompt_ids), dim),
            tts_bos=_rand(1, dim),
            tts_eos=_rand(1, dim),
            tts_pad=_rand(1, dim),
        )
        prefill_out = _FakeRequestOutput(
            request_id="req-0",
            prompt_token_ids=prompt_ids,
            outputs=[
                _FakeCompletionOutput(
                    token_ids=[output_ids[0]],
                    multimodal_output=prefill_mm,
                )
            ],
        )

        # --- Stage 1: Decode (produces generated embeddings) ---
        decode_mm = _make_multimodal_output(
            embed=_rand(len(output_ids), dim),
            hidden=_rand(len(output_ids), dim),
            tts_bos=_rand(1, dim),
            tts_eos=_rand(1, dim),
            tts_pad=_rand(1, dim),
        )
        decode_out = _FakeRequestOutput(
            request_id="req-0",
            prompt_token_ids=prompt_ids,
            outputs=[
                _FakeCompletionOutput(
                    token_ids=output_ids,
                    multimodal_output=decode_mm,
                )
            ],
        )

        prefill_stage = _FakeStage(
            stage_id=0,
            is_prefill_only=True,
            engine_outputs=[prefill_out],
        )
        decode_stage = _FakeStage(
            stage_id=1,
            is_decode_only=True,
            engine_outputs=[decode_out],
        )

        # --- thinker2talker: merge prefill + decode -> talker input ---
        stage_list_t2t = [prefill_stage, decode_stage]
        talker_inputs = thinker2talker(stage_list_t2t, engine_input_source=[1])

        assert len(talker_inputs) == 1
        info = talker_inputs[0]["additional_information"]
        assert info["thinker_prefill_embeddings"].shape[0] == total_len
        assert info["thinker_hidden_states"].shape[0] == total_len
        assert info["tts_bos_embed"] is not None

        # --- Stage 2: Talker (produces codec codes) ---
        num_q = 8
        talker_seq_len = 30
        codec_codes = torch.randint(1, 1024, (num_q, talker_seq_len + 1))
        talker_mm = {"code_predictor_codes": codec_codes}
        talker_out = _FakeRequestOutput(
            request_id="req-0",
            prompt_token_ids=[0] * 10,
            outputs=[
                _FakeCompletionOutput(
                    token_ids=list(range(talker_seq_len + 1)),
                    multimodal_output=talker_mm,
                )
            ],
        )

        talker_stage = _FakeStage(stage_id=2, engine_outputs=[talker_out])

        # --- talker2code2wav: codec codes -> code2wav input ---
        stage_list_t2c = [prefill_stage, decode_stage, talker_stage]
        code2wav_inputs = talker2code2wav(stage_list_t2c, engine_input_source=[2])

        assert len(code2wav_inputs) == 1
        flattened_codes = code2wav_inputs[0]["prompt_token_ids"]
        assert len(flattened_codes) == talker_seq_len * num_q
        # All codes should be valid integers
        assert all(isinstance(c, int) for c in flattened_codes)

    def test_pd_audio_preserves_prompt_context(self):
        """Verify that in PD mode, the merged embeddings preserve the full
        prompt context that the talker needs for coherent audio generation.

        Uses distinguishable embeddings (positive for prefill, negative for
        decode) to verify the merge places them correctly.
        """
        from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
            thinker2talker,
        )

        dim = 16
        prompt_ids, output_ids = _build_chat_token_ids(
            user_len=20,
            assistant_generated_len=10,
        )

        # Create distinguishable prefill embeddings (all positive)
        prefill_emb = torch.abs(_rand(len(prompt_ids), dim)) + 1.0
        prefill_hid = torch.abs(_rand(len(prompt_ids), dim)) + 1.0

        # Create distinguishable decode embeddings (all negative)
        decode_emb = -torch.abs(_rand(len(output_ids), dim)) - 1.0
        decode_hid = -torch.abs(_rand(len(output_ids), dim)) - 1.0

        prefill_mm = _make_multimodal_output(
            embed=prefill_emb,
            hidden=prefill_hid,
            tts_bos=_rand(1, dim),
            tts_eos=_rand(1, dim),
            tts_pad=_rand(1, dim),
        )
        decode_mm = _make_multimodal_output(
            embed=decode_emb,
            hidden=decode_hid,
            tts_bos=_rand(1, dim),
            tts_eos=_rand(1, dim),
            tts_pad=_rand(1, dim),
        )

        prefill_out = _FakeRequestOutput(
            request_id="req-0",
            prompt_token_ids=prompt_ids,
            outputs=[
                _FakeCompletionOutput(
                    token_ids=[output_ids[0]],
                    multimodal_output=prefill_mm,
                )
            ],
        )
        decode_out = _FakeRequestOutput(
            request_id="req-0",
            prompt_token_ids=prompt_ids,
            outputs=[
                _FakeCompletionOutput(
                    token_ids=output_ids,
                    multimodal_output=decode_mm,
                )
            ],
        )

        prefill_stage = _FakeStage(
            stage_id=0,
            is_prefill_only=True,
            engine_outputs=[prefill_out],
        )
        decode_stage = _FakeStage(
            stage_id=1,
            is_decode_only=True,
            engine_outputs=[decode_out],
        )

        results = thinker2talker([prefill_stage, decode_stage], engine_input_source=[1])
        merged_emb = results[0]["additional_information"]["thinker_prefill_embeddings"]

        # First part (prompt) should be from prefill (positive values)
        prompt_part = merged_emb[: len(prompt_ids)]
        assert (prompt_part > 0).all(), "Prompt embeddings should come from prefill"

        # Second part (generated) should be from decode (negative values)
        decode_part = merged_emb[len(prompt_ids) :]
        assert (decode_part < 0).all(), "Generated embeddings should come from decode"

    def test_non_pd_audio_chain(self):
        """Verify the non-PD path also works end-to-end for comparison."""
        from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
            talker2code2wav,
            thinker2talker,
        )

        dim = 16
        prompt_ids, output_ids = _build_chat_token_ids(user_len=10, assistant_generated_len=5)
        total_len = len(prompt_ids) + len(output_ids)

        # Single thinker stage (no PD split)
        thinker_mm = _make_multimodal_output(
            embed=_rand(total_len, dim),
            hidden=_rand(total_len, dim),
            tts_bos=_rand(1, dim),
            tts_eos=_rand(1, dim),
            tts_pad=_rand(1, dim),
        )
        thinker_out = _FakeRequestOutput(
            request_id="req-0",
            prompt_token_ids=prompt_ids,
            outputs=[
                _FakeCompletionOutput(
                    token_ids=output_ids,
                    multimodal_output=thinker_mm,
                )
            ],
        )
        thinker_stage = _FakeStage(stage_id=0, engine_outputs=[thinker_out])

        # thinker2talker (non-PD: source_stage_id=0, no prefill stage)
        talker_inputs = thinker2talker([thinker_stage], engine_input_source=[0])
        assert len(talker_inputs) == 1
        info = talker_inputs[0]["additional_information"]
        assert info["thinker_prefill_embeddings"].shape[0] == total_len

        # talker2code2wav
        num_q = 8
        talker_seq = 25
        codec_codes = torch.randint(1, 1024, (num_q, talker_seq + 1))
        talker_out2 = _FakeRequestOutput(
            request_id="req-0",
            prompt_token_ids=[0] * 10,
            outputs=[
                _FakeCompletionOutput(
                    token_ids=list(range(talker_seq + 1)),
                    multimodal_output={"code_predictor_codes": codec_codes},
                )
            ],
        )
        talker_stage = _FakeStage(stage_id=1, engine_outputs=[talker_out2])

        c2w_inputs = talker2code2wav([thinker_stage, talker_stage], engine_input_source=[1])
        assert len(c2w_inputs) == 1
        assert len(c2w_inputs[0]["prompt_token_ids"]) == talker_seq * num_q
