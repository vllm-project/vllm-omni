# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resolve a codec sub-config's vocabulary before constructing sampler state."""

import pytest
import torch
from transformers import PretrainedConfig
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.worker import gpu_input_batch
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_omni.config.model import OmniModelArchConfigConvertor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "values,expected",
    [
        ({}, 0),
        ({"num_audio_tokens": 6562}, 6562),
        ({"vocab_size": 64, "num_audio_tokens": 6562}, 64),
        ({"vocab_size": 0, "num_audio_tokens": 6562}, 0),
    ],
)
def test_audio_vocab_fallback_preserves_explicit_vocab(values, expected):
    text_config = PretrainedConfig(**values)
    converter = OmniModelArchConfigConvertor(PretrainedConfig(), text_config, stage_config_name="tts_config")
    assert converter.get_vocab_size() == expected


def test_remote_codec_vocab_retains_top_k_in_real_v1_metadata(monkeypatch):
    # The released remote TTS config omits vocab_size. Exercise the real V1
    # admission/metadata path: testing SamplingParams alone misses the bug.
    codec_config = PretrainedConfig(num_audio_tokens=6562)
    converter = OmniModelArchConfigConvertor(PretrainedConfig(), codec_config, stage_config_name="tts_config")
    monkeypatch.setattr(gpu_input_batch, "PIN_MEMORY", False)
    batch = InputBatch(
        max_num_reqs=2,
        max_model_len=8,
        max_num_batched_tokens=8,
        device=torch.device("cpu"),
        vocab_size=converter.get_vocab_size(),
        block_sizes=[1],
        kernel_block_sizes=[1],
        max_num_blocks_per_req=[8],
        logitsprocs=LogitsProcessors(),
    )
    requests = {}
    for request_id, top_k in [("filtered", 25), ("unfiltered", -1)]:
        requests[request_id] = CachedRequestState(
            req_id=request_id,
            prompt_token_ids=[0, 0],
            mm_features=[],
            sampling_params=SamplingParams(temperature=0.8, top_k=top_k, top_p=0.85),
            generator=None,
            block_ids=([],),
            num_computed_tokens=0,
            output_token_ids=[],
        )
        batch.add_request(requests[request_id])
    batch.refresh_metadata()
    assert batch.sampling_metadata.top_k is not None
    assert batch.sampling_metadata.top_k.tolist() == [25, 6562]
    # Removal and re-admission must retain the request's filter, as they do
    # when a resumable duplex request receives another conditioning chunk.
    batch.remove_request("filtered")
    batch.condense()
    batch.refresh_metadata()
    assert batch.sampling_metadata.top_k is None
    batch.add_request(requests["filtered"])
    batch.refresh_metadata()
    metadata_after_readmission = batch.sampling_metadata
    assert metadata_after_readmission.top_k is not None
    assert metadata_after_readmission.top_k.tolist() == [6562, 25]
