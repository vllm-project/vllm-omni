# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for :mod:`vllm_omni.core.transfer_types`.

Covers the reviewer-requested scenarios from PR #9014:

* Nested payload / tensor preservation (from_dict → as_dict round-trip).
* Chunk phase transitions across prefill / decode / final.
* Legacy processor path — omitting optional kwargs still produces a valid
  payload that downstream consumers can read via ``as_dict``.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.core.transfer_types import ChunkMeta, ChunkPhase, TransferPayload

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestChunkMeta:
    """Chunk-phase inference and factory helpers."""

    def test_prefill_is_inferred_from_zero_index(self):
        meta = ChunkMeta.from_chunk_index(0, sequence_id=3, is_final=False)
        assert meta.phase is ChunkPhase.PREFILL
        assert meta.sequence_id == 3
        assert meta.chunk_index == 0
        assert meta.is_prefill()
        assert not meta.is_decode()

    def test_decode_is_inferred_from_positive_index(self):
        meta = ChunkMeta.from_chunk_index(1, sequence_id=3, is_final=False)
        assert meta.phase is ChunkPhase.DECODE
        assert meta.chunk_index == 1
        assert meta.is_decode()
        assert not meta.is_prefill()

    def test_explicit_prefill_factory(self):
        meta = ChunkMeta.prefill(sequence_id=7, is_final=True, total_chunks=4)
        assert meta.phase is ChunkPhase.PREFILL
        assert meta.chunk_index == 0
        assert meta.sequence_id == 7
        assert meta.is_final
        assert meta.total_chunks == 4
        assert meta.is_terminal()

    def test_explicit_decode_factory(self):
        meta = ChunkMeta.decode(5, sequence_id=2, is_final=True)
        assert meta.phase is ChunkPhase.DECODE
        assert meta.chunk_index == 5
        assert meta.is_final
        assert meta.is_terminal()

    def test_decode_rejects_zero_index(self):
        import pytest

        with pytest.raises(AssertionError, match="decode ChunkMeta requires chunk_index > 0"):
            ChunkMeta.decode(0)

    def test_preempt_factory(self):
        meta = ChunkMeta.preempt(sequence_id=1)
        assert meta.phase is ChunkPhase.PREEMPT
        assert meta.chunk_index == 0
        assert meta.is_prefill()  # PREEMPT counts as prefill-ish for routing

    def test_final_factory(self):
        meta = ChunkMeta.final(sequence_id=0, chunk_index=4)
        assert meta.phase is ChunkPhase.FINAL
        assert meta.is_final
        assert meta.is_terminal()
        assert meta.total_chunks == 5

    def test_prefill_to_decode_transition(self):
        """prefill → decode chunk counter must be monotonic."""
        prefill = ChunkMeta.prefill(sequence_id=0)
        decodes = [ChunkMeta.from_chunk_index(i, sequence_id=0) for i in range(1, 4)]
        assert prefill.phase is ChunkPhase.PREFILL
        assert all(d.phase is ChunkPhase.DECODE for d in decodes)
        assert [d.chunk_index for d in decodes] == [1, 2, 3]

    def test_final_chunk_terminates_sequence(self):
        final = ChunkMeta.final(sequence_id=0, chunk_index=3)
        assert final.is_terminal()
        assert final.is_final


class TestTransferPayload:
    """TransferPayload construction, nested preservation, device policy."""

    def test_round_trip_preserves_top_level_keys(self):
        h = torch.randn(2, 8)
        e = torch.randn(2, 8)
        raw = {"hidden_states": h, "embed": e, "language": "en"}
        payload = TransferPayload.from_dict(raw)
        out = payload.as_dict()
        assert torch.equal(out["hidden_states"], h)
        assert torch.equal(out["embed"], e)
        assert out["language"] == "en"

    def test_nested_tensors_are_flattened_on_construction(self):
        """from_dict flattens arbitrarily nested tensors into dotted keys."""
        layers = [torch.randn(2, 8) for _ in range(4)]
        raw = {"hidden_states": {"layers": {i: layers[i] for i in range(4)}}}
        payload = TransferPayload.from_dict(raw)
        for i in range(4):
            assert f"hidden_states.layers.{i}" in payload.tensors
            assert torch.equal(payload.tensors[f"hidden_states.layers.{i}"], layers[i])

    def test_nested_tensors_are_rebuilt_on_serialisation(self):
        """as_dict rebuilds the dotted-keyed structure for legacy consumers."""
        h = torch.randn(2, 8)
        e = torch.randn(2, 8)
        raw = {"embed": {"prefill": e}, "hidden_states": h, "meta": {"lang": "en"}}
        payload = TransferPayload.from_dict(raw)
        out = payload.as_dict()
        # Legacy consumer reads payload["embed"]["prefill"]
        assert torch.equal(out["embed"]["prefill"], e)
        assert torch.equal(out["hidden_states"], h)
        assert out["meta"]["lang"] == "en"

    def test_from_dict_preserves_non_tensor_nested_metadata(self):
        raw = {"config": {"num_layers": 12, "hidden_size": 1024}}
        payload = TransferPayload.from_dict(raw)
        out = payload.as_dict()
        assert out["config"]["num_layers"] == 12
        assert out["config"]["hidden_size"] == 1024

    def test_enforce_device_policy_cpu_moves_gpu_tensors(self):
        t_cpu = torch.randn(2, 8, device="cpu")
        payload = TransferPayload(tensors={"t": t_cpu}, device_policy="cpu")
        result = payload.enforce_device_policy()
        # Returns self for chaining
        assert result is payload
        for v in result.tensors.values():
            assert v.device.type == "cpu"

    def test_legacy_processor_path_no_optional_kwargs(self):
        """Calling from_dict without chunk / source_stage / target_stage / request_id
        must still produce a valid payload — legacy code paths should not break."""
        h = torch.randn(2, 8)
        payload = TransferPayload.from_dict({"hidden_states": h})
        assert payload.chunk is None
        assert payload.source_stage is None
        assert payload.target_stage is None
        assert payload.request_id is None
        assert payload.device_policy == "cpu"  # default
        # and round-trips cleanly
        out = payload.as_dict()
        assert torch.equal(out["hidden_states"], h)

    def test_get_tensor_default_fallback(self):
        h = torch.randn(2, 8)
        payload = TransferPayload.from_dict({"hidden_states": h})
        assert payload.get_tensor("hidden_states") is not None
        assert payload.get_tensor("missing") is None
        sentinel = torch.zeros(1)
        assert payload.get_tensor("missing", sentinel) is sentinel

    def test_set_tensor_replaces_and_drops_none(self):
        payload = TransferPayload()
        payload.set_tensor("a", torch.zeros(1))
        assert "a" in payload.tensors
        payload.set_tensor("a", None)
        assert "a" not in payload.tensors
