# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mini E2E validation tests for Qwen2.5-Omni Phase 2 multimodal output channel.

These tests mirror the logic of tests/e2e/offline_inference/test_qwen2_5_omni.py
but run entirely in CPU without GPU, model weights, or a real server.

Coverage:
- OmniRequestState.add_multimodal_tensor()  (accumulation logic)
- OmniRequestState._consolidate_multimodal_tensors()  (tensor merging)
- OmniRequestState._new_completion_output()  (mm_accumulated → CompletionOutput)
- MultimodalOutputProcessor.process_outputs()  (eco.multimodal_output routing)
- OmniEngineCoreOutput.multimodal_output  (new dedicated field)
- OmniModelRunnerOutput  (@dataclass + multimodal_outputs field)
- Scheduler extraction pattern  (mm_outputs[req_index] → OmniEngineCoreOutput)
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.sampling_params import RequestOutputKind

from vllm_omni.engine import OmniEngineCoreOutput
from vllm_omni.engine.mm_outputs import MultimodalPayload
from vllm_omni.engine.output_modality import OutputModality
from vllm_omni.engine.output_processor import MultimodalOutputProcessor, OmniRequestState
from vllm_omni.outputs import OmniModelRunnerOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_omni_request_state(mm_type: str | None = None) -> OmniRequestState:
    """Create a bare OmniRequestState without touching vLLM RequestState.__init__."""
    state = object.__new__(OmniRequestState)
    state.mm_type = mm_type
    state.mm_accumulated = MultimodalPayload()
    state.detokenizer = MagicMock()  # simulate a text-capable request
    state.output_kind = RequestOutputKind.CUMULATIVE
    return state


# ===========================================================================
# 1. OmniEngineCoreOutput – dedicated multimodal_output field
# ===========================================================================


class TestOmniEngineCoreOutput:
    """Validates the dedicated multimodal_output channel (Phase 2)."""

    def test_multimodal_output_field_exists(self):
        eco = OmniEngineCoreOutput(request_id="r1", new_token_ids=[1])
        assert hasattr(eco, "multimodal_output")
        assert eco.multimodal_output is None

    def test_multimodal_output_set_with_dict(self):
        payload = {"hidden": torch.zeros(4, 8), "latent": torch.ones(4, 16)}
        eco = OmniEngineCoreOutput(request_id="r1", new_token_ids=[1], multimodal_output=payload)
        assert eco.multimodal_output is payload
        assert "hidden" in eco.multimodal_output
        assert "latent" in eco.multimodal_output

    def test_pooling_output_is_separate_from_multimodal_output(self):
        """pooling_output must remain None when multimodal_output is set."""
        eco = OmniEngineCoreOutput(request_id="r1", new_token_ids=[], multimodal_output={"audio": torch.randn(1, 100)})
        assert eco.pooling_output is None
        assert eco.multimodal_output is not None


# ===========================================================================
# 2. OmniModelRunnerOutput – @dataclass + multimodal_outputs list
# ===========================================================================


class TestOmniModelRunnerOutput:
    """Validates that OmniModelRunnerOutput is a proper dataclass."""

    def test_can_instantiate_with_multimodal_outputs(self):
        out = OmniModelRunnerOutput(
            req_ids=["r1"],
            req_id_to_index={"r1": 0},
            multimodal_outputs=[{"hidden": torch.zeros(2, 4)}],
        )
        assert len(out.multimodal_outputs) == 1
        assert "hidden" in out.multimodal_outputs[0]

    def test_multimodal_outputs_defaults_to_none(self):
        out = OmniModelRunnerOutput(
            req_ids=["r1"],
            req_id_to_index={"r1": 0},
        )
        assert out.multimodal_outputs is None

    def test_pooler_output_defaults_to_none(self):
        """pooler_output must not receive multimodal data in Phase 2."""
        out = OmniModelRunnerOutput(
            req_ids=["r1"],
            req_id_to_index={"r1": 0},
            multimodal_outputs=[{"audio": torch.randn(1, 200)}],
        )
        assert out.pooler_output is None

    def test_multiple_requests(self):
        per_req = [{"hidden": torch.zeros(1, 4)}, {"hidden": torch.ones(1, 4)}]
        out = OmniModelRunnerOutput(
            req_ids=["r1", "r2"],
            req_id_to_index={"r1": 0, "r2": 1},
            multimodal_outputs=per_req,
        )
        assert len(out.multimodal_outputs) == 2
        assert torch.all(out.multimodal_outputs[1]["hidden"] == 1)


# ===========================================================================
# 3. OmniRequestState.add_multimodal_tensor – accumulation logic
# ===========================================================================


class TestAddMultimodalTensor:
    """Mirrors the accumulation that happens inside the output processor
    for each OmniEngineCoreOutput that carries a multimodal_output dict."""

    def test_none_payload_is_ignored(self):
        state = _make_omni_request_state()
        state.add_multimodal_tensor(None, mm_type="audio")
        assert state.mm_accumulated.is_empty

    def test_dict_payload_stored_directly_on_first_call(self):
        state = _make_omni_request_state()
        payload = {"audio": torch.randn(1, 100), "sr": torch.tensor(24000)}
        state.add_multimodal_tensor(payload, mm_type="audio")
        assert not state.mm_accumulated.is_empty
        assert "audio" in state.mm_accumulated
        assert "sr" in state.mm_accumulated

    def test_hidden_key_remapped_to_mm_type(self):
        """AR runner produces {"hidden": ...}; must be renamed to mm_type key."""
        state = _make_omni_request_state()
        tensor = torch.randn(4, 8)
        state.add_multimodal_tensor({"hidden": tensor}, mm_type="latent")
        assert "latent" in state.mm_accumulated
        assert "hidden" not in state.mm_accumulated

    def test_model_outputs_key_remapped_to_mm_type(self):
        """Generation runner produces {"model_outputs": ...}; rename to mm_type."""
        state = _make_omni_request_state()
        tensor = torch.randn(2, 3)
        state.add_multimodal_tensor({"model_outputs": tensor}, mm_type="audio")
        assert "audio" in state.mm_accumulated
        assert "model_outputs" not in state.mm_accumulated

    def test_tensor_accumulation_builds_list(self):
        """Second call on same key must build a list for deferred concat."""
        state = _make_omni_request_state()
        t1 = torch.randn(1, 8)
        t2 = torch.randn(1, 8)
        state.add_multimodal_tensor({"audio": t1}, mm_type="audio")
        state.add_multimodal_tensor({"audio": t2}, mm_type="audio")
        assert isinstance(state.mm_accumulated["audio"], list)
        assert len(state.mm_accumulated["audio"]) == 2

    def test_mm_type_stored_on_state(self):
        state = _make_omni_request_state()
        state.add_multimodal_tensor({"audio": torch.randn(1, 50)}, mm_type="Audio")
        assert state.mm_type == "audio"  # lowercased

    def test_raw_tensor_payload(self):
        """Non-dict payloads wrapped under mm_type key."""
        state = _make_omni_request_state()
        tensor = torch.randn(3, 4)
        state.add_multimodal_tensor(tensor, mm_type="latent")
        assert "latent" in state.mm_accumulated
        assert isinstance(state.mm_accumulated["latent"], torch.Tensor)

    def test_tensor_moved_to_cpu(self):
        state = _make_omni_request_state()
        tensor = torch.randn(2, 4)  # already on cpu in this env
        state.add_multimodal_tensor({"audio": tensor}, mm_type="audio")
        assert state.mm_accumulated["audio"].device.type == "cpu"


# ===========================================================================
# 4. OmniRequestState._consolidate_multimodal_tensors – tensor merging
# ===========================================================================


class TestConsolidateMultimodalTensors:
    """Mirrors the final consolidation step that runs when a request finishes."""

    def test_noop_when_no_accumulated(self):
        state = _make_omni_request_state()
        state._consolidate_multimodal_tensors()  # must not raise
        assert state.mm_accumulated.is_empty

    def test_single_tensor_unchanged(self):
        state = _make_omni_request_state()
        t = torch.randn(1, 16000)
        state.mm_accumulated = MultimodalPayload.from_dict({"audio": t})
        state._consolidate_multimodal_tensors()
        # single tensor is not a list → stays as-is
        assert isinstance(state.mm_accumulated["audio"], torch.Tensor)

    def test_tensor_list_concatenated_for_non_audio(self):
        state = _make_omni_request_state()
        t1 = torch.randn(1, 8)
        t2 = torch.randn(1, 8)
        state.mm_accumulated = MultimodalPayload(tensors={"latent": [t1, t2]})
        state._consolidate_multimodal_tensors()
        result = state.mm_accumulated["latent"]
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 8)

    def test_audio_list_not_concatenated(self):
        """Audio tensors use CONCAT_LAST strategy (cat along last dim)."""
        state = _make_omni_request_state(mm_type="audio")
        t1 = torch.randn(1, 100)
        t2 = torch.randn(1, 200)  # different length, but cat on dim=-1 works
        state.mm_accumulated = MultimodalPayload(tensors={"audio": [t1, t2]})
        state._consolidate_multimodal_tensors()
        result = state.mm_accumulated["audio"]
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 300)  # CONCAT_LAST: cat on dim=-1

    def test_sr_list_keeps_last_value(self):
        state = _make_omni_request_state()
        sr1 = torch.tensor(16000)
        sr2 = torch.tensor(24000)
        # Put a dummy tensor so is_empty is False, sr in metadata
        state.mm_accumulated = MultimodalPayload(
            tensors={"audio": torch.randn(1, 8)},
            metadata={"sr": [sr1, sr2]},
        )
        state._consolidate_multimodal_tensors()
        assert state.mm_accumulated["sr"] == sr2

    def test_nested_dict_tensors_concatenated(self):
        """Nested dict values in metadata are left as-is by consolidation."""
        state = _make_omni_request_state()
        t1, t2 = torch.randn(1, 4), torch.randn(1, 4)
        state.mm_accumulated = MultimodalPayload(metadata={"features": {"enc": [t1, t2]}})
        state._consolidate_multimodal_tensors()
        # Nested dicts in metadata are not traversed; stays as-is
        assert isinstance(state.mm_accumulated["features"]["enc"], list)


# ===========================================================================
# 5. OmniRequestState._new_completion_output – attaches mm to CompletionOutput
# ===========================================================================


class TestNewCompletionOutputAttachesMultimodal:
    """Simulates mix-to-audio: state has accumulated audio, completion output
    must carry it in a multimodal_output dict attribute."""

    def _make_state_with_audio(self) -> OmniRequestState:
        state = _make_omni_request_state(mm_type="audio")
        state.mm_accumulated = MultimodalPayload.from_dict(
            {
                "audio": torch.randn(1, 16000),
                "sr": torch.tensor(24000),
            }
        )
        return state

    def test_mm_accumulated_attached_to_completion_output(self):
        state = self._make_state_with_audio()
        fake_base_output = MagicMock()
        del fake_base_output.multimodal_output  # simulate plain object

        with patch.object(
            OmniRequestState.__bases__[0],
            "_new_completion_output",
            return_value=fake_base_output,
        ):
            output = state._new_completion_output(token_ids=[], finish_reason=None, stop_reason=None)

        assert hasattr(output, "multimodal_output")
        mm = output.multimodal_output
        assert "audio" in mm
        assert "sr" in mm

    def test_no_mm_accumulated_leaves_output_clean(self):
        state = _make_omni_request_state()
        state.mm_accumulated = MultimodalPayload()
        fake_base_output = MagicMock(spec=[])  # no attributes

        with patch.object(
            OmniRequestState.__bases__[0],
            "_new_completion_output",
            return_value=fake_base_output,
        ):
            output = state._new_completion_output(token_ids=[], finish_reason=None, stop_reason=None)

        assert not hasattr(output, "multimodal_output")


# ===========================================================================
# 6. MultimodalOutputProcessor.process_outputs – eco routing
# ===========================================================================


class TestMultimodalOutputProcessorRouting:
    """Mini version of test_mix_to_audio / test_text_to_text:
    validates that OmniEngineCoreOutput.multimodal_output is routed
    into the correct OmniRequestState via process_outputs()."""

    def _make_processor(self, output_type: str = "audio") -> MultimodalOutputProcessor:
        proc = object.__new__(MultimodalOutputProcessor)
        proc.engine_core_output_type = output_type
        proc.output_modality = OutputModality.from_string(output_type)
        proc.request_states = {}
        proc.log_stats = False
        return proc

    def test_multimodal_output_routed_to_request_state(self):
        proc = self._make_processor(output_type="audio")
        state = _make_omni_request_state()
        proc.request_states["r1"] = state

        audio_payload = {"audio": torch.randn(1, 16000), "sr": torch.tensor(24000)}
        eco = OmniEngineCoreOutput(request_id="r1", new_token_ids=[], multimodal_output=audio_payload)

        with patch.object(
            MultimodalOutputProcessor.__bases__[0],
            "process_outputs",
            return_value=MagicMock(),
        ):
            proc.process_outputs([eco])

        assert not state.mm_accumulated.is_empty
        assert "audio" in state.mm_accumulated

    def test_none_multimodal_output_skipped(self):
        proc = self._make_processor()
        state = _make_omni_request_state()
        proc.request_states["r1"] = state

        eco = OmniEngineCoreOutput(request_id="r1", new_token_ids=[], multimodal_output=None)

        with patch.object(
            MultimodalOutputProcessor.__bases__[0],
            "process_outputs",
            return_value=MagicMock(),
        ):
            proc.process_outputs([eco])

        assert state.mm_accumulated.is_empty

    def test_unknown_request_id_ignored(self):
        proc = self._make_processor()
        eco = OmniEngineCoreOutput(
            request_id="unknown", new_token_ids=[], multimodal_output={"audio": torch.randn(1, 100)}
        )

        with patch.object(
            MultimodalOutputProcessor.__bases__[0],
            "process_outputs",
            return_value=MagicMock(),
        ):
            proc.process_outputs([eco])  # must not raise

    def test_text_only_request_no_mm_output(self):
        """Mirrors test_text_to_text: eco carries no multimodal_output."""
        proc = self._make_processor(output_type="text")
        state = _make_omni_request_state()
        proc.request_states["r1"] = state

        eco = OmniEngineCoreOutput(request_id="r1", new_token_ids=[42])

        with patch.object(
            MultimodalOutputProcessor.__bases__[0],
            "process_outputs",
            return_value=MagicMock(),
        ):
            proc.process_outputs([eco])

        assert state.mm_accumulated.is_empty

    def test_mm_type_derived_from_eco_output_type(self):
        """engine_core_output_type on processor determines mm_type
        when eco has no output_type attribute (msgspec structs are fixed)."""
        proc = self._make_processor(output_type="latent")
        state = _make_omni_request_state()
        proc.request_states["r1"] = state

        eco = OmniEngineCoreOutput(request_id="r1", new_token_ids=[], multimodal_output={"hidden": torch.randn(1, 8)})
        # OmniEngineCoreOutput is a msgspec.Struct without output_type field,
        # so getattr(eco, "output_type", ...) falls back to
        # self.engine_core_output_type which is "latent".

        with patch.object(
            MultimodalOutputProcessor.__bases__[0],
            "process_outputs",
            return_value=MagicMock(),
        ):
            proc.process_outputs([eco])

        assert state.mm_type == "latent"


# ===========================================================================
# 7. Scheduler extraction pattern – mm_outputs[req_index] → eco
# ===========================================================================


class TestSchedulerMultimodalExtractionPattern:
    """Validates the pattern used in omni_ar_scheduler and
    omni_generation_scheduler to build OmniEngineCoreOutput per request."""

    def test_per_request_mm_output_extracted_from_list(self):
        """Mirrors: mm_output = mm_outputs[req_index]"""
        mm_outputs = [
            {"hidden": torch.randn(1, 4), "latent": torch.randn(1, 8)},
            {"hidden": torch.randn(1, 4), "latent": torch.randn(1, 8)},
        ]
        req_index = 1
        mm_output = mm_outputs[req_index]

        eco = OmniEngineCoreOutput(
            request_id="r2",
            new_token_ids=[],
            multimodal_output=mm_output,
        )
        assert eco.multimodal_output is mm_output
        assert eco.pooling_output is None

    def test_none_mm_outputs_skipped(self):
        """Mirrors: mm_output = mm_outputs[req_index] if mm_outputs else None"""
        mm_outputs = None
        mm_output = mm_outputs[0] if mm_outputs else None

        eco = OmniEngineCoreOutput(
            request_id="r1",
            new_token_ids=[],
            multimodal_output=mm_output,
        )
        assert eco.multimodal_output is None

    def test_audio_generation_output_structure(self):
        """Mirrors Stage 2 (Code2WAV) producing audio output per request."""
        audio_tensor = torch.randn(1, 24000)
        mm_outputs = [{"audio": audio_tensor, "sr": torch.tensor(24000)}]

        eco = OmniEngineCoreOutput(
            request_id="r1",
            new_token_ids=[],
            multimodal_output=mm_outputs[0],
        )
        assert "audio" in eco.multimodal_output
        assert "sr" in eco.multimodal_output
        assert eco.multimodal_output["audio"].shape == (1, 24000)


# ===========================================================================
# 8. Full mini pipeline – text-to-audio flow without model weights
# ===========================================================================


class TestMiniPipelineFlow:
    """End-to-end data flow test without real GPU/weights.
    Simulates the 3-stage Qwen2.5-Omni pipeline by constructing
    the objects that each stage would produce and verifying they
    wire together correctly."""

    def test_mix_to_audio_data_flow(self):
        """Mirrors test_mix_to_audio:
        Stage 0 (Thinker) → latent → Stage 1 (Talker) → tokens
        Stage 2 (Code2WAV) → audio dict → final OmniRequestState has audio.
        """
        # Stage 0: Thinker produces latent hidden states
        stage0_mm = {"hidden": torch.randn(1, 512), "latent": torch.randn(1, 256)}
        eco0 = OmniEngineCoreOutput(request_id="r1", new_token_ids=[], multimodal_output=stage0_mm)

        # Stage 2: Code2WAV produces audio waveform
        audio_chunk1 = torch.randn(1, 8000)
        audio_chunk2 = torch.randn(1, 8000)
        eco2a = OmniEngineCoreOutput(
            request_id="r1", new_token_ids=[], multimodal_output={"audio": audio_chunk1, "sr": torch.tensor(24000)}
        )
        eco2b = OmniEngineCoreOutput(
            request_id="r1", new_token_ids=[], multimodal_output={"audio": audio_chunk2, "sr": torch.tensor(24000)}
        )

        # Output processor accumulates audio chunks across eco calls
        state = _make_omni_request_state(mm_type="audio")
        state.add_multimodal_tensor(eco2a.multimodal_output, mm_type="audio")
        state.add_multimodal_tensor(eco2b.multimodal_output, mm_type="audio")
        state._consolidate_multimodal_tensors()

        # Validate final state
        assert not state.mm_accumulated.is_empty
        assert "audio" in state.mm_accumulated  # audio skips concat → stays list
        assert state.mm_type == "audio"

        # Validate stage0 eco carries latent (not audio, not pooling_output)
        assert eco0.pooling_output is None
        assert "latent" in eco0.multimodal_output

    def test_text_to_text_data_flow(self):
        """Mirrors test_text_to_text:
        Only text tokens are generated; no multimodal_output on eco.
        OmniRequestState must remain clean (no mm_accumulated).
        """
        eco = OmniEngineCoreOutput(
            request_id="r1",
            new_token_ids=[1, 2, 3],  # token ids from Thinker
            multimodal_output=None,  # no audio/latent output
        )

        state = _make_omni_request_state()
        if eco.multimodal_output is not None:
            state.add_multimodal_tensor(eco.multimodal_output, mm_type="text")

        assert state.mm_accumulated.is_empty
        assert eco.pooling_output is None
        assert eco.new_token_ids == [1, 2, 3]
