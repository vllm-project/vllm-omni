# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import re
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm_omni.worker import sparse_audio
from vllm_omni.worker.output import payload_build
from vllm_omni.worker.sampling_utils import clamp_prompt_ids_to_penalty_padding

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_sparse_mm_req_ids_requires_sparse_audio_marker():
    assert sparse_audio.resolve_sparse_mm_req_ids({"meta": {"req_id": ["r1"]}}) is None
    assert sparse_audio.resolve_sparse_mm_req_ids({"meta.req_id": ["r1"]}) is None

    assert sparse_audio.resolve_sparse_mm_req_ids({"meta": {"req_id": ["r1"], "sparse_audio": ["1"]}}) == ["r1"]
    assert sparse_audio.resolve_sparse_mm_req_ids({"meta.req_id": ["r1"], "meta.sparse_audio": ["1"]}) == ["r1"]
    assert sparse_audio.resolve_sparse_mm_req_ids({"meta": {"req_id": ["r1"]}, "meta.sparse_audio": ["1"]}) == ["r1"]
    assert sparse_audio.resolve_sparse_mm_req_ids({"meta": {"sparse_audio": ["1"]}, "meta.req_id": ["r1"]}) == ["r1"]


# --- builder gap-fill: contract corners not covered by the tests above ---


class TestPromptIdsClampPaddingConvention:
    """Pin `sampling_utils.clamp_prompt_ids_to_penalty_padding` (the clamp
    `sample_tokens` applies before penalties).

    The audit (RFC #5450 C2) originally flagged the clamp as an off-by-one
    (OOB index). It is not: upstream penalties allocate `vocab_size + 1` bins
    and drop the last column, so `vocab_size` is the designed padding value
    (vllm/model_executor/layers/utils.py::get_token_bin_counts_and_mask).
    These tests call the runner's own helper, so a future "fix" to the clamp
    fails here.
    """

    LOGITS_VOCAB = 8
    BATCH_VOCAB = 10  # wider input_batch vocab; its pad value is BATCH_VOCAB

    def _padded_prompt(self):
        # Two real tokens + two padding slots padded with the batch vocab size.
        return torch.tensor([[1, 2, self.BATCH_VOCAB, self.BATCH_VOCAB]])

    def test_clamp_to_logits_vocab_keeps_padding_inert(self):
        from vllm.model_executor.layers.utils import get_token_bin_counts_and_mask

        # The runner's own helper, not a re-implementation.
        clamped = clamp_prompt_ids_to_penalty_padding(self._padded_prompt(), self.LOGITS_VOCAB)

        _, mask = get_token_bin_counts_and_mask(clamped, self.LOGITS_VOCAB, 1)

        expected = torch.zeros(1, self.LOGITS_VOCAB, dtype=torch.bool)
        expected[0, 1] = True
        expected[0, 2] = True
        # Pad ids landed in the dropped (vocab_size-th) bin: no penalty effect.
        assert torch.equal(mask, expected)

    def test_clamp_one_lower_would_corrupt_last_token_penalties(self):
        from vllm.model_executor.layers.utils import get_token_bin_counts_and_mask

        # The "fix" the audit first proposed - shown here to be the actual bug:
        # padding would count as occurrences of the last real vocab token.
        clamped = self._padded_prompt().clamp(max=self.LOGITS_VOCAB - 1)

        _, mask = get_token_bin_counts_and_mask(clamped, self.LOGITS_VOCAB, 1)

        assert bool(mask[0, self.LOGITS_VOCAB - 1])

    def test_unclamped_padding_overflows_penalty_bins(self):
        from vllm.model_executor.layers.utils import get_token_bin_counts_and_mask

        # Why the clamp exists: the batch-level pad value (BATCH_VOCAB) is out
        # of range even for the vocab_size+1 penalty bins of a narrower head.
        with pytest.raises(RuntimeError):
            get_token_bin_counts_and_mask(self._padded_prompt(), self.LOGITS_VOCAB, 1)


class TestSparseAudioMarkerRobustness:
    """RFC #5450 C3: `sparse_audio.is_sparse_audio_marker` must not crash on tensor/array
    markers, and the combined prefix-cache payload builder must never ship one
    request's data as another's."""

    def test_marker_list_and_str_forms(self):
        marker = sparse_audio.is_sparse_audio_marker
        assert marker(["1"]) is True
        assert marker(["0"]) is False
        assert marker("true") is True
        assert marker("off") is False

    def test_marker_enforced_canonical_forms_no_error(self):
        # The designed carrier (list-of-str; bare str accepted) and marker
        # absence (None) are the ONLY silent forms.
        marker = sparse_audio.is_sparse_audio_marker
        with patch.object(sparse_audio.logger, "error") as err:
            assert marker(["1"]) is True
            assert marker(["0"]) is False
            assert marker(["0", "1"]) is True
            assert marker("true") is True
            assert marker("off") is False
            assert marker(None) is False
        err.assert_not_called()

    def test_marker_illegal_carriers_raise_one_error_per_type(self):
        # Enforcement, not normalization: any carrier outside the design
        # contract is a producer bug - it raises InvalidSparseDeclarationError
        # (caught inside the module by resolve_sparse_mm_routing, which
        # fails closed)
        # and logs at error level once per offending type. Interpreting
        # deviants would need a second truth predicate, and predicate
        # divergence is exactly how the np.array(["0"])-routes-as-sparse bug
        # happened.
        marker = sparse_audio.is_sparse_audio_marker
        for value in (torch.tensor(1), torch.tensor([1, 1]), np.array(["1"]), 1, 1.0, True, ["1", 1]):
            with (
                patch.object(sparse_audio, "_logged_offenders", set()),
                patch.object(sparse_audio.logger, "error") as err,
            ):
                with pytest.raises(sparse_audio.InvalidSparseDeclarationError):
                    marker(value)
                assert err.call_count == 1, f"expected one error for {value!r}"
                with pytest.raises(sparse_audio.InvalidSparseDeclarationError):
                    marker(value)
                assert err.call_count == 1  # deduped per type, not per step

    def test_marker_device_tensor_rejected_without_read(self):
        # Deviant carriers are never read element-wise, so a device tensor
        # can never introduce a D2H sync here. Meta tensors hold no data -
        # any read (`.item()`/`bool()`) raises a RuntimeError, so seeing OUR
        # exception type (not RuntimeError) proves the data is never touched.
        marker = sparse_audio.is_sparse_audio_marker
        with patch.object(sparse_audio, "_logged_offenders", set()):
            with pytest.raises(sparse_audio.InvalidSparseDeclarationError):
                marker(torch.ones((), device="meta"))
            with pytest.raises(sparse_audio.InvalidSparseDeclarationError):
                marker(torch.ones(3, device="meta"))

    def test_invalid_marker_fails_closed_at_routing(self):
        # A present-but-invalid sparse declaration most likely belongs to a
        # producer that INTENDED sparse output; falling back to dense would
        # batch-index-assign its payloads to the WRONG requests. Routing
        # must fail closed: no payload routed for the step (empty sparse
        # set), never dense misrouting.
        with patch.object(sparse_audio, "_logged_offenders", set()):
            downstream, index, is_sparse = sparse_audio.resolve_sparse_mm_routing(
                engine_output_type="audio",
                req_ids_output_copy=["r0", "r1"],
                downstream_req_ids=["r0", "r1"],
                multimodal_outputs={
                    "meta": {"req_id": ["r0"], "sparse_audio": torch.tensor([1, 1])},
                    "model_outputs": [torch.zeros(1)],
                },
            )
        assert downstream == []
        assert index == {}
        assert is_sparse is True

    def test_fail_closed_routing_error_remains_observable(self):
        def route_invalid_marker():
            return sparse_audio.resolve_sparse_mm_routing(
                engine_output_type="audio",
                req_ids_output_copy=["r0"],
                downstream_req_ids=["r0"],
                multimodal_outputs={
                    "meta": {"req_id": ["r0"], "sparse_audio": torch.tensor([1, 1])},
                    "model_outputs": [torch.zeros(1)],
                },
            )

        with (
            patch.object(sparse_audio, "_logged_offenders", set()),
            patch.object(sparse_audio, "_fail_closed_last_log_at", None),
            patch.object(sparse_audio, "_fail_closed_suppressed", 0),
            patch.object(sparse_audio.time, "monotonic", side_effect=[100.0, 120.0, 161.0]),
            patch.object(sparse_audio.logger, "error") as err,
        ):
            for _ in range(3):
                assert route_invalid_marker() == ([], {}, True)

        routing_errors = [call for call in err.call_args_list if "no multimodal payload" in call.args[0]]
        assert len(routing_errors) == 2
        assert [call.args[1] for call in routing_errors] == [0, 1]

    def test_fail_closed_logging_is_thread_safe(self):
        workers = 8
        ready = Barrier(workers)

        def route_invalid_marker():
            return sparse_audio.resolve_sparse_mm_routing(
                engine_output_type="audio",
                req_ids_output_copy=["r0"],
                downstream_req_ids=["r0"],
                multimodal_outputs={
                    "meta": {"req_id": ["r0"], "sparse_audio": torch.tensor([1, 1])},
                },
            )

        def route_concurrently(_):
            ready.wait(timeout=10)
            return route_invalid_marker()

        with (
            patch.object(sparse_audio, "_logged_offenders", set()),
            patch.object(sparse_audio, "_fail_closed_last_log_at", None),
            patch.object(sparse_audio, "_fail_closed_suppressed", 0),
            patch.object(sparse_audio.time, "monotonic", return_value=100.0) as clock,
            patch.object(sparse_audio.logger, "error") as err,
        ):
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(route_concurrently, range(workers)))
            assert results == [([], {}, True)] * workers

            clock.return_value = 161.0
            assert route_invalid_marker() == ([], {}, True)

        routing_errors = [call for call in err.call_args_list if "no multimodal payload" in call.args[0]]
        assert len(routing_errors) == 2
        assert [call.args[1] for call in routing_errors] == [0, workers - 1]
        classifier_errors = [call for call in err.call_args_list if "Illegal sparse-audio marker" in call.args[0]]
        assert len(classifier_errors) == 1

    def test_duplicate_sparse_metadata_must_be_consistent(self):
        equal = {
            "meta": {"req_id": ["r1"], "sparse_audio": ["1"]},
            "meta.req_id": ["r1"],
            "meta.sparse_audio": ["true"],
        }
        assert sparse_audio.resolve_sparse_mm_req_ids(equal) == ["r1"]

        invalid_declarations = (
            {
                "meta": {"req_id": ["r1"], "sparse_audio": ["1"]},
                "meta.req_id": ["r2"],
            },
            {
                "meta": {"req_id": ["r1"], "sparse_audio": ["1"]},
                "meta.req_id": "r1",
            },
            {
                "meta": {"req_id": ["r1"], "sparse_audio": ["0"]},
                "meta.sparse_audio": ["1"],
            },
            {"meta": {"req_id": ["r1", "r1"], "sparse_audio": ["1"]}},
        )
        with patch.object(sparse_audio, "_logged_offenders", set()):
            for declaration in invalid_declarations:
                with pytest.raises(sparse_audio.InvalidSparseDeclarationError):
                    sparse_audio.resolve_sparse_mm_req_ids(declaration)

    def test_conflicting_sparse_metadata_fails_closed_at_public_boundary(self):
        downstream, index, is_sparse = sparse_audio.resolve_sparse_mm_routing(
            engine_output_type="audio",
            req_ids_output_copy=["r1", "r2"],
            downstream_req_ids=["r1", "r2"],
            multimodal_outputs={
                "meta": {"req_id": ["r1"], "sparse_audio": ["1"]},
                "meta.req_id": ["r2"],
            },
        )
        assert downstream == []
        assert index == {}
        assert is_sparse is True

    def test_flat_marker_with_nested_req_ids_is_not_erased(self):
        downstream, index, is_sparse = sparse_audio.resolve_sparse_mm_routing(
            engine_output_type="audio",
            req_ids_output_copy=["r0", "r1"],
            downstream_req_ids=["r0", "r1"],
            multimodal_outputs={"meta": {"req_id": ["r1"]}, "meta.sparse_audio": ["1"]},
        )
        assert downstream == ["r1"]
        assert index == {"r1": 0}
        assert is_sparse is True

    def test_sparse_marker_with_unusable_req_ids_fails_closed(self):
        # The marker says "sparse" but the alignment list is unusable - the
        # declaration as a whole is out of contract; same fail-closed shape.
        with patch.object(sparse_audio, "_logged_offenders", set()):
            downstream, index, is_sparse = sparse_audio.resolve_sparse_mm_routing(
                engine_output_type="audio",
                req_ids_output_copy=["r0"],
                downstream_req_ids=["r0"],
                multimodal_outputs={"meta": {"req_id": "r0", "sparse_audio": ["1"]}},
            )
        assert downstream == []
        assert is_sparse is True

    def test_nested_marker_without_req_id_fails_closed(self):
        # {"meta": {"sparse_audio": ["1"]}} - marker set, req_id missing.
        # The flattened-encoding fallback must not ERASE the nested marker
        # (which used to silently demote the declaration to dense); this is
        # an unusable-req_id invalid declaration -> fail closed.
        with patch.object(sparse_audio, "_logged_offenders", set()):
            downstream, index, is_sparse = sparse_audio.resolve_sparse_mm_routing(
                engine_output_type="audio",
                req_ids_output_copy=["r0"],
                downstream_req_ids=["r0"],
                multimodal_outputs={"meta": {"sparse_audio": ["1"]}},
            )
        assert downstream == []
        assert is_sparse is True

    def test_invalid_marker_on_non_audio_pipeline_keeps_dense_routing(self):
        # Non-audio pipelines never consult sparse routing; the carrier
        # error is logged by the classifier but routing is unaffected.
        with patch.object(sparse_audio, "_logged_offenders", set()):
            downstream, index, is_sparse = sparse_audio.resolve_sparse_mm_routing(
                engine_output_type="latent",
                req_ids_output_copy=["r0", "r1"],
                downstream_req_ids=["r0", "r1"],
                multimodal_outputs={"meta": {"req_id": ["r0"], "sparse_audio": torch.tensor([1, 1])}},
            )
        assert downstream == ["r0", "r1"]
        assert is_sparse is False

    def test_legal_marker_still_routes_sparse(self):
        downstream, index, is_sparse = sparse_audio.resolve_sparse_mm_routing(
            engine_output_type="audio",
            req_ids_output_copy=["r0", "r1", "r2"],
            downstream_req_ids=["r0", "r1", "r2"],
            multimodal_outputs={"meta": {"req_id": ["r1", "r2"], "sparse_audio": ["1"]}},
        )
        assert downstream == ["r1", "r2"]
        assert index == {"r1": 0, "r2": 1}
        assert is_sparse is True

    def test_combined_payload_unwraps_aligned_list(self):
        combined = {"codes.audio": {"req-1": [torch.zeros(1), torch.ones(1)]}}
        payload = payload_build.build_combined_prefix_cache_mm_payload(combined, rid="req-1", list_idx=1)
        assert torch.equal(payload["codes.audio"], torch.ones(1))

    def test_combined_payload_broadcasts_qwen_request_invariant_singletons(self):
        req_ids = ("r0", "r1")
        tts_embeddings = {
            "embed.tts_bos": torch.full((1, 1, 2), 1.0),
            "embed.tts_eos": torch.full((1, 1, 2), 2.0),
            "embed.tts_pad": torch.full((1, 1, 2), 3.0),
        }
        combined = {key: {rid: [value.clone()] for rid in req_ids} for key, value in tts_embeddings.items()}

        for idx, rid in enumerate(req_ids):
            payload = payload_build.build_omni_mm_payload(
                combined_multimodal_outputs=combined,
                mm_cpu=None,
                rid=rid,
                idx=idx,
                start=idx,
                end=idx + 1,
                audio_sparse_output=False,
                sparse_mm_index={},
                hidden_seq_len=len(req_ids),
                scheduled_seq_len=len(req_ids),
            )
            assert payload.keys() == tts_embeddings.keys()
            for key, expected in tts_embeddings.items():
                assert torch.equal(payload[key], expected)

    def test_combined_payload_drops_sparse_protocol_metadata(self):
        combined = {
            "meta.sparse_audio": {"req-1": ["1"]},
            "meta.req_id": {"req-1": ["req-1"]},
            "sr": {"req-1": 22050},
        }
        payload = payload_build.build_combined_prefix_cache_mm_payload(combined, rid="req-1", list_idx=2)
        assert payload == {"sr": 22050}

    def test_combined_payload_misalignment_drops_key_not_request_zero(self):
        # A short multi-element list used to silently return `v[0]` - request
        # 0's payload shipped under request `rid`. Never substitute another
        # request's data; the key is dropped with an error log (same policy
        # as the non-combined sparse mismatch) rather than raised - v1
        # runners don't catch per-request exceptions, so a raise would turn a
        # per-request protocol break into a whole-stage outage.
        combined = {
            "codes.audio": {"req-3": [torch.zeros(1), torch.ones(1)]},
            "sr": {"req-3": 22050},
        }
        payload = payload_build.build_combined_prefix_cache_mm_payload(combined, rid="req-3", list_idx=2)
        assert "codes.audio" not in payload
        assert payload["sr"] == 22050

    def test_direct_path_sparse_mismatch_drops_key_and_keeps_payload_shape(self):
        # Mirror of the combined-path policy on the non-prefix-cache path:
        # a sparse index beyond the per-request list drops that key (error
        # log) and the rest of the request's payload survives.
        payload = payload_build.build_omni_mm_payload(
            combined_multimodal_outputs=None,
            mm_cpu={"codes.audio": [torch.zeros(1)]},
            rid="r2",
            idx=1,
            start=0,
            end=1,
            audio_sparse_output=True,
            sparse_mm_index={"r2": 1},
            hidden_seq_len=1,
            scheduled_seq_len=1,
        )
        assert payload == {}

    def test_combined_payload_uses_sparse_index_under_sparse_routing(self):
        # Batch [r0, r1, r2], sparse outputs only for [r1, r2]: producers
        # align per-request lists to the SPARSE order, so r1 (batch idx 1,
        # sparse idx 0) must get element 0 and r2 (batch idx 2, sparse idx 1)
        # element 1. Indexing by batch idx shipped r2's payload to r1 and
        # went out of range for r2. The sparse->list_idx resolution lives in
        # `_build_omni_mm_payload` (the combined helper itself is
        # sparse-agnostic), so this exercises the caller.
        sparse_mm_index = {"r1": 0, "r2": 1}
        combined = {
            "codes.audio": {
                "r1": [torch.zeros(1), torch.ones(1)],
                "r2": [torch.zeros(1), torch.ones(1)],
            }
        }
        payloads = {}
        for rid, idx in (("r1", 1), ("r2", 2)):
            payloads[rid] = payload_build.build_omni_mm_payload(
                combined_multimodal_outputs=combined,
                mm_cpu=None,
                rid=rid,
                idx=idx,
                start=0,
                end=1,
                audio_sparse_output=True,
                sparse_mm_index=sparse_mm_index,
                hidden_seq_len=1,
                scheduled_seq_len=1,
            )
        assert torch.equal(payloads["r1"]["codes.audio"], torch.zeros(1))
        assert torch.equal(payloads["r2"]["codes.audio"], torch.ones(1))


# Matches a real ASSIGNMENT of the capability (optionally `self.`-qualified
# and/or annotated), not a bare mention: a comment, a docstring, or a read
# such as `getattr(self.model, "prefer_model_sampler", False)` must not count
# as declaring it. `=(?!=)` keeps `==` comparisons out.
_PREFER_MODEL_SAMPLER_ASSIGNMENT = re.compile(
    r"^[ \t]*(?:self\.)?prefer_model_sampler[ \t]*(?::[^=\n]*)?=(?!=)[ \t]*(?P<rhs>[^\n]*)",
    re.MULTILINE,
)
