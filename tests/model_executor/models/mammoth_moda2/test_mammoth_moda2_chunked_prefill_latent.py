# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU coverage for MammothModa2 AR→DiT latent reconstruction under chunked prefill (#7116).

Chunked prefill emits hidden-state rows in successive pooler payloads. The
full-payload accumulator must CONCAT those rows so ``ar2diffusion`` still sees
``full_hidden_states.shape[0] == len(full_token_ids)``.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.mammoth_moda2 import ar2diffusion
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _runner() -> OmniConnectorModelRunnerMixin:
    runner = object.__new__(OmniConnectorModelRunnerMixin)
    runner.model_config = SimpleNamespace(
        model_arch="MammothModa2ForConditionalGeneration",
        model_stage="ar",
        async_chunk=False,
        final_output=False,
        custom_process_next_stage_input_func=(
            "vllm_omni.model_executor.stage_input_processors.mammoth_moda2.ar2diffusion"
        ),
    )
    runner._custom_process_func = ar2diffusion
    runner._pending_full_payload_send = {}
    runner._stage_id = 0
    runner._omni_connector = object()
    return runner


def test_should_accumulate_latent_for_mammoth_ar_producer() -> None:
    assert _runner()._should_accumulate_full_payload_output() is True


def test_chunked_prefill_latent_rows_concat_to_full_sequence() -> None:
    runner = _runner()
    request = SimpleNamespace(request_id="mm2-chunked")
    chunks = [
        torch.arange(0, 8, dtype=torch.float32).reshape(4, 2),
        torch.arange(8, 14, dtype=torch.float32).reshape(3, 2),
        torch.arange(14, 20, dtype=torch.float32).reshape(3, 2),
        torch.arange(20, 22, dtype=torch.float32).reshape(1, 2),
    ]
    for chunk in chunks:
        OmniConnectorModelRunnerMixin.accumulate_full_payload_output(runner, "r0", {"latent": chunk}, request)

    materialised, _ = OmniConnectorModelRunnerMixin._materialize_full_payload_entry(
        runner._pending_full_payload_send["r0"]
    )
    latent = materialised["latent"]
    assert latent.shape == (11, 2)
    torch.testing.assert_close(latent, torch.cat(chunks, dim=0))


def test_ar2diffusion_accepts_concatenated_chunked_latent() -> None:
    prompt_ids = list(range(7))
    gen_ids = list(range(7, 10))
    completion = SimpleNamespace(
        cumulative_token_ids=gen_ids + [999],
        multimodal_output={"latent": torch.arange(20, dtype=torch.float32).reshape(10, 2)},
    )
    ar_output = SimpleNamespace(
        request_id="r0",
        prompt_token_ids=prompt_ids,
        outputs=[completion],
    )
    out = ar2diffusion([ar_output], prompt={"mm_processor_kwargs": {"target_h": 1024, "target_w": 1024}})
    info = out["additional_information"]
    assert info["full_hidden_states"].shape[0] == len(info["full_token_ids"]) == 10
    assert info["answer_start_index"] == 7
