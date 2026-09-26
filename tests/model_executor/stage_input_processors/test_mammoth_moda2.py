# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regressions for MammothModa2 AR-to-DiT and prefix caching."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.core.prefix_cache import OmniTensorPrefixCache
from vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit import (
    MammothModa2DiTPipeline,
)
from vllm_omni.engine.serialization import (
    deserialize_additional_information,
    serialize_additional_information,
)
from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient
from vllm_omni.model_executor.models.mammoth_moda2.pipeline import MAMMOTH_MODA2_PIPELINE
from vllm_omni.model_executor.stage_input_processors.mammoth_moda2 import ar2diffusion

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _CompletionOutputStub:
    cumulative_token_ids: list[int]
    multimodal_output: dict[str, torch.Tensor]


@dataclass
class _AROutputStub:
    request_id: str
    prompt_token_ids: list[int]
    outputs: list[_CompletionOutputStub]


@dataclass
class _TextConfigStub:
    gen_vocab_start_index: int


@dataclass
class _MammothConfigStub:
    llm_config: _TextConfigStub
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int


def _source_output(*, include_latent: bool = True) -> _AROutputStub:
    multimodal_output = {"latent": torch.arange(32, dtype=torch.float32).reshape(4, 8)} if include_latent else {}
    completion = _CompletionOutputStub(
        cumulative_token_ids=[100, 101, 102],
        multimodal_output=multimodal_output,
    )
    return _AROutputStub(
        request_id="req-7",
        prompt_token_ids=[10, 11],
        outputs=[completion],
    )


def test_ar2diffusion_builds_one_prompt_with_raw_ar_conditions() -> None:
    result = ar2diffusion(
        [_source_output()],
        {"prompt": "a cat", "mm_processor_kwargs": {"target_h": 512, "target_w": 768}},
    )
    assert not isinstance(result, list)
    assert result["prompt"] == ""
    assert result["height"] == 512
    assert result["width"] == 768
    info = result["additional_information"]
    assert info["full_token_ids"] == [10, 11, 100, 101]
    assert info["answer_start_index"] == 2
    torch.testing.assert_close(
        info["full_hidden_states"],
        torch.arange(32, dtype=torch.float32).reshape(4, 8),
    )
    assert info["full_hidden_states"].is_contiguous()


def test_ar2diffusion_uses_prompt_dimension_fallbacks() -> None:
    result = ar2diffusion(
        [_source_output()],
        {"additional_information": {"image_height": [256], "image_width": [384]}},
    )
    assert (result["height"], result["width"]) == (256, 384)


def test_ar2diffusion_preserves_request_level_sampling_fallbacks() -> None:
    result = ar2diffusion(
        [_source_output()],
        {
            "additional_information": {
                "text_guidance_scale": [1.5],
                "num_inference_steps": [3],
                "cfg_range": [0.25, 0.75],
            }
        },
    )

    info = result["additional_information"]
    assert info["text_guidance_scale"] == [1.5]
    assert info["num_inference_steps"] == [3]
    assert info["cfg_range"] == [0.25, 0.75]


def test_ar2diffusion_unwraps_the_orchestrator_prompt_list() -> None:
    result = ar2diffusion(
        [_source_output()],
        [{"mm_processor_kwargs": {"target_h": 640, "target_w": 960}}],
    )
    assert (result["height"], result["width"]) == (640, 960)


def test_ar2diffusion_rejects_multiple_source_requests() -> None:
    with pytest.raises(ValueError, match="exactly one AR output"):
        ar2diffusion([_source_output(), _source_output()], {})


def test_ar2diffusion_reports_missing_latent_with_request_id() -> None:
    with pytest.raises(ValueError, match="req-7"):
        ar2diffusion([_source_output(include_latent=False)], {})


def test_ar2diffusion_rejects_hidden_state_length_mismatch() -> None:
    source = _source_output()
    source.outputs[0].multimodal_output["latent"] = torch.zeros(3, 8)
    with pytest.raises(ValueError, match="Hidden states length mismatch"):
        ar2diffusion([source], {})


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_ar2diffusion_preserves_low_precision_through_engine_core_payload(dtype: torch.dtype) -> None:
    source = _source_output()
    hidden_states = torch.arange(32, dtype=dtype).reshape(4, 8)
    source.outputs[0].multimodal_output["latent"] = hidden_states
    diffusion_input = ar2diffusion([source], {})

    wire_payload = serialize_additional_information(diffusion_input["additional_information"])
    assert wire_payload is not None
    restored = deserialize_additional_information(wire_payload)
    restored_hidden_states = restored["full_hidden_states"]

    assert isinstance(restored_hidden_states, torch.Tensor)
    assert restored_hidden_states.dtype == dtype
    assert restored_hidden_states.is_contiguous()
    assert torch.equal(restored_hidden_states, hidden_states)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_dit_condition_split_preserves_compact_transfer_dtype(dtype: torch.dtype) -> None:
    pipeline = object.__new__(MammothModa2DiTPipeline)
    object.__setattr__(
        pipeline,
        "config",
        _MammothConfigStub(
            llm_config=_TextConfigStub(gen_vocab_start_index=100),
            image_token_id=20,
            video_token_id=21,
            vision_start_token_id=22,
            vision_end_token_id=23,
        ),
    )
    hidden_states = torch.arange(20, dtype=dtype).reshape(5, 4)

    text_cond, image_cond = pipeline._split_ar_conditions(
        full_hidden_states=hidden_states,
        full_token_ids=[7, 20, 8, 101, 102],
        answer_start_index=3,
    )

    assert text_cond.dtype == dtype
    assert image_cond.dtype == dtype
    assert text_cond.is_contiguous()
    assert image_cond.is_contiguous()
    assert torch.equal(text_cond, hidden_states[[0, 2]])
    assert torch.equal(image_cond, hidden_states[[3, 4]])


def test_mammoth_pipeline_uses_standard_completed_ar_forwarding() -> None:
    stage0, stage1 = MAMMOTH_MODA2_PIPELINE.stages

    assert stage0.custom_process_next_stage_input_func is None
    assert stage1.custom_process_input_func.endswith(".ar2diffusion")
    assert stage1.sync_process_input_func is None
    assert stage1.requires_full_payload_input is False


def test_stage_client_forwards_completed_ar_output_to_mammoth_adapter() -> None:
    source = _source_output()
    client = object.__new__(StageEngineCoreClient)
    client.custom_process_input_func = ar2diffusion
    client.requires_multimodal_data = False

    diffusion_input = client.process_engine_inputs(
        [source],
        {"mm_processor_kwargs": {"target_h": 512, "target_w": 768}},
    )

    assert diffusion_input["height"] == 512
    assert diffusion_input["width"] == 768
    info = diffusion_input["additional_information"]
    assert info["full_token_ids"] == [10, 11, 100, 101]
    assert info["answer_start_index"] == 2
    assert torch.equal(info["full_hidden_states"], source.outputs[0].multimodal_output["latent"])


class _InputBatch:
    def __init__(self, block_ids: torch.Tensor, num_computed_tokens: int):
        block_table = SimpleNamespace(cpu=block_ids)
        block_group = SimpleNamespace(block_table=block_table)
        self.block_table = _BlockTable(block_table, block_group)
        self.req_ids = ["request-0"]
        self.req_id_to_index = {"request-0": 0}
        self.num_computed_tokens_cpu = torch.tensor([num_computed_tokens])


class _BlockTable:
    def __init__(self, block_table, block_group):
        self.block_tables = [block_table]
        self._block_group = block_group

    def __getitem__(self, index):
        assert index == 0
        return self._block_group


def _prefix_cache_ar_output(
    prompt_token_ids: list[int],
    generated_token_ids: list[int],
    hidden_states: torch.Tensor,
):
    completion = SimpleNamespace(
        cumulative_token_ids=generated_token_ids,
        multimodal_output={"latent": hidden_states},
    )
    return SimpleNamespace(
        request_id="request-0",
        prompt_token_ids=prompt_token_ids,
        outputs=[completion],
    )


def test_prefix_cache_miss_hit_preserves_ar_to_dit_alignment():
    block_size = 4
    hidden_size = 3
    cached_tokens = 8
    cache = OmniTensorPrefixCache(
        num_blocks=8,
        block_size=block_size,
        hidden_size=hidden_size,
        hs_dtype=torch.float32,
    )

    cached_hidden = torch.arange(cached_tokens * hidden_size, dtype=torch.float32).reshape(cached_tokens, hidden_size)
    cached_slots = torch.arange(2 * block_size, 4 * block_size)
    cache.update_omni_tensor_prefix_cache(
        hidden_states=cached_hidden,
        multimodal_outputs=None,
        num_tokens_unpadded=cached_tokens,
        slot_mapping=cached_slots,
    )

    new_hidden = torch.arange(12, dtype=torch.float32).reshape(4, hidden_size) + 100
    miss_hidden = torch.cat([cached_hidden, new_hidden], dim=0)

    input_batch = _InputBatch(
        block_ids=torch.tensor([[2, 3]], dtype=torch.long),
        num_computed_tokens=cached_tokens,
    )
    cache.add_prefix_cached_new_req_id("request-0")
    hit_hidden = cache.get_merged_hidden_states(
        query_start_loc=torch.tensor([0]),
        input_batch=input_batch,
        hidden_states=new_hidden,
        num_scheduled_tokens={"request-0": len(new_hidden)},
    )["request-0"]

    prompt_token_ids = list(range(10))
    # ar2diffusion intentionally drops the final generated token because no
    # hidden state is produced for it.
    generated_token_ids = [20, 21, 22]
    prompt = {"additional_information": {"image_height": [256], "image_width": [256]}}
    miss = ar2diffusion(
        [
            _prefix_cache_ar_output(
                prompt_token_ids,
                generated_token_ids,
                miss_hidden,
            )
        ],
        prompt,
    )
    hit = ar2diffusion(
        [
            _prefix_cache_ar_output(
                prompt_token_ids,
                generated_token_ids,
                hit_hidden,
            )
        ],
        prompt,
    )

    miss_info = miss["additional_information"]
    hit_info = hit["additional_information"]
    assert hit_info["full_token_ids"] == prompt_token_ids + generated_token_ids[:-1]
    assert hit_info["answer_start_index"] == len(prompt_token_ids)
    assert torch.equal(hit_info["full_hidden_states"], miss_info["full_hidden_states"])
    assert hit_info["full_hidden_states"].shape[0] == len(hit_info["full_token_ids"])
