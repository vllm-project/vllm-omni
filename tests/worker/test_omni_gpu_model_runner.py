# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_qwen3_thinker_embedding_adapter_materializes_only_committed_ids(dtype) -> None:
    """The model-side A2 adapter maps verifier IDs to Thinker embeddings."""
    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
        Qwen3OmniMoeForConditionalGeneration,
    )

    class _Thinker(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = torch.nn.Module()
            self.language_model.model = torch.nn.Module()
            self.language_model.model.embed_tokens = torch.nn.Embedding(6, 2, dtype=dtype)
            with torch.no_grad():
                self.language_model.model.embed_tokens.weight.copy_(torch.arange(12).reshape(6, 2))

        def embed_input_ids(self, *, input_ids):
            return self.language_model.model.embed_tokens(input_ids)

    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model_stage = "thinker"
    model.thinker = _Thinker()
    model.thinker_config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=2))

    def prepare(samples):
        mm = model.prepare_speculative_multimodal_outputs(
            multimodal_outputs={},
            req_ids=["r"],
            speculative_request_ids={"r"},
            invalid_req_indices=[],
            sampled_token_ids=torch.tensor([samples]),
            input_token_ids=torch.tensor([4]),
            input_positions=torch.tensor([3]),
            query_start_loc=[0, 1],
            prompt_lens=[2],
            vocab_size=6,
            consumed_input_span=False,
        )
        mm = model.finalize_multimodal_outputs_from_cpu_snapshot(mm)
        return model.finalize_pooler_payload({}, multimodal_outputs=mm, idx=0)["hidden_states.layer_0"]

    rows = prepare([4, -1, 1])
    assert torch.equal(rows, torch.tensor([[8.0, 9.0], [2.0, 3.0]]))
    empty = prepare([-1, -1, -1])
    assert empty.shape == (0, 2)
    assert empty.dtype == rows.dtype == dtype
    assert empty.device == rows.device
    assert torch.cat([empty, rows]).dtype == dtype


# No real forward needed for these tests.


def _make_full_payload_accumulation_runner(
    model_arch="Qwen3OmniMoeForConditionalGeneration",
    model_stage="talker",
    async_chunk=False,
    final_output=False,
    custom_process_next_stage_input_func="module.full_payload",
):
    runner = object.__new__(OmniConnectorModelRunnerMixin)
    runner.model_config = SimpleNamespace(
        model_arch=model_arch,
        model_stage=model_stage,
        async_chunk=async_chunk,
        final_output=final_output,
        custom_process_next_stage_input_func=custom_process_next_stage_input_func,
    )
    runner._custom_process_func = object()
    runner._pending_full_payload_send = {}
    runner._stage_id = 1
    runner._omni_connector = object()
    return runner


def test_accumulate_full_payload_output_preserves_aligned_all_zero_qwen3_omni_codec_rows():
    runner = _make_full_payload_accumulation_runner()
    request = SimpleNamespace(output_token_ids=[0, 1])
    codes = torch.zeros((2, 3), dtype=torch.long)

    OmniConnectorModelRunnerMixin.accumulate_full_payload_output(runner, "r1", {"codes.audio": codes}, request)

    stored, _ = OmniConnectorModelRunnerMixin._materialize_full_payload_entry(runner._pending_full_payload_send["r1"])
    assert torch.equal(stored["codes.audio"], codes)


def test_accumulate_full_payload_output_keeps_misaligned_all_zero_qwen3_omni_codec_rows():
    # After removing the sender-side zero filter, the accumulator keeps every
    # codec row including misaligned all-zero rows. The downstream consumer
    # (_extract_qwen3_full_payload_codec_rows) is the authoritative crop and
    # filters by output_token_ids.
    runner = _make_full_payload_accumulation_runner()
    request = SimpleNamespace(output_token_ids=[0, 1])
    codes = torch.zeros((1, 3), dtype=torch.long)

    OmniConnectorModelRunnerMixin.accumulate_full_payload_output(runner, "r1", {"codes.audio": codes}, request)

    stored, _ = OmniConnectorModelRunnerMixin._materialize_full_payload_entry(runner._pending_full_payload_send["r1"])
    assert "codes.audio" in stored
    assert torch.equal(stored["codes.audio"], codes)


def test_accumulate_full_payload_output_preserves_incremental_aligned_all_zero_qwen3_omni_codec_rows():
    runner = _make_full_payload_accumulation_runner()
    request = SimpleNamespace(output_token_ids=[0, 1])
    runner._pending_full_payload_send["r1"] = (
        {"codes.audio": torch.ones((1, 3), dtype=torch.long)},
        request,
    )
    codes = torch.zeros((1, 3), dtype=torch.long)

    OmniConnectorModelRunnerMixin.accumulate_full_payload_output(runner, "r1", {"codes.audio": codes}, request)

    stored, _ = OmniConnectorModelRunnerMixin._materialize_full_payload_entry(runner._pending_full_payload_send["r1"])
    assert stored["codes.audio"].shape == (2, 3)
    assert torch.equal(stored["codes.audio"][1], torch.zeros(3, dtype=torch.long))


def test_accumulate_full_payload_output_keeps_all_zero_qwen3_omni_prefill_placeholder():
    # Prefill placeholder rows (output_token_ids empty) are no longer dropped
    # at the sender. The consumer-side crop trims them off using
    # output_token_ids, so the end-to-end semantics are unchanged.
    runner = _make_full_payload_accumulation_runner()
    request = SimpleNamespace(output_token_ids=[])
    codes = torch.zeros((2, 3), dtype=torch.long)

    OmniConnectorModelRunnerMixin.accumulate_full_payload_output(runner, "r1", {"codes.audio": codes}, request)

    stored, _ = OmniConnectorModelRunnerMixin._materialize_full_payload_entry(runner._pending_full_payload_send["r1"])
    assert "codes.audio" in stored
    assert torch.equal(stored["codes.audio"], codes)


def test_full_payload_output_accumulation_hook_matrix():
    assert _make_full_payload_accumulation_runner(model_stage="thinker")._should_accumulate_full_payload_output()
    assert _make_full_payload_accumulation_runner(model_stage="talker")._should_accumulate_full_payload_output()
    # A stage may publish a user-visible output and still feed another stage
    # (Qwen3-Omni thinker publishes text while sending audio state to talker).
    assert _make_full_payload_accumulation_runner(
        model_stage="thinker", final_output=True
    )._should_accumulate_full_payload_output()
    assert not _make_full_payload_accumulation_runner(
        model_stage="token2audio",
        custom_process_next_stage_input_func=None,
    )._should_accumulate_full_payload_output()
    assert not _make_full_payload_accumulation_runner(
        model_stage="talker", async_chunk=True
    )._should_accumulate_full_payload_output()
    for model_arch in (
        "Qwen3TTSForConditionalGeneration",
        "Qwen2_5OmniForConditionalGeneration",
    ):
        runner = _make_full_payload_accumulation_runner(model_arch=model_arch)
        runner._custom_process_func = None
        assert not runner._should_accumulate_full_payload_output()


# ---------------------------------------------------------------------------
# LMCache HS chunk-boundary buffering tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# LMCache HS restore: one consumer, and a short store is a failure
# ---------------------------------------------------------------------------
