# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.moss_tts_local import (
    _is_codes_empty,
    llm2decoder,
    llm2decoder_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_stage(codes, request_id="rid"):
    return SimpleNamespace(
        engine_outputs=[
            SimpleNamespace(
                request_id=request_id,
                outputs=[
                    SimpleNamespace(
                        multimodal_output={
                            "code_predictor_codes": codes,
                        }
                    )
                ],
            )
        ]
    )


def test_all_zero_rvq_codes_are_not_empty():
    assert not _is_codes_empty(torch.zeros((1, 1, 3, 1), dtype=torch.long))
    assert _is_codes_empty(torch.empty(0, dtype=torch.long))
    assert _is_codes_empty(None)


def test_llm2decoder_keeps_all_zero_rvq_codes():
    codes = torch.zeros((2, 1, 3, 1), dtype=torch.long)
    prompts = llm2decoder([_make_stage(codes)], engine_input_source=[0])

    assert len(prompts) == 1
    assert prompts[0]["prompt_token_ids"] == [0, 0, 0, 0, 0, 0]


def test_async_chunk_accumulates_all_zero_rvq_rows():
    transfer_manager = SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        connector=SimpleNamespace(config={"extra": {"codec_chunk_frames": 1}}),
    )
    request = SimpleNamespace(external_req_id="rid")

    payload = llm2decoder_async_chunk(
        transfer_manager,
        {"code_predictor_codes": torch.zeros(3, dtype=torch.long)},
        request,
        is_finished=False,
    )

    assert payload is not None
    assert payload["code_predictor_codes"] == [0, 0, 0]
    assert payload["code_flat_numel"] == 3
    assert payload["finished"] == torch.tensor(False, dtype=torch.bool)
