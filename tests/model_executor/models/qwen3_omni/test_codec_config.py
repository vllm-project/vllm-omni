# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-Omni applies the codec ramp and Code2Wav graph groups on CUDA only."""

import pytest
import torch

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.stage_config import load_deploy_config
from vllm_omni.model_executor.models.qwen3_omni import codec_config
from vllm_omni.model_executor.models.qwen3_omni.codec_config import qwen3_omni_codec_extra
from vllm_omni.model_executor.stage_input_processors.qwen3_omni import talker2code2wav_async_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_EXTRA = load_deploy_config(get_deploy_config_path("qwen3_omni_moe.yaml")).connectors["connector_of_shared_memory"][
    "extra"
]


class _Platform:
    def __init__(self, cuda: bool) -> None:
        self._cuda = cuda

    def is_cuda(self) -> bool:
        return self._cuda


@pytest.mark.parametrize("cuda", [True, False])
def test_default_deploy_codec_extras_follow_platform(monkeypatch, cuda) -> None:
    monkeypatch.setattr(codec_config, "current_omni_platform", _Platform(cuda))
    extra = qwen3_omni_codec_extra(_EXTRA)
    assert ("codec_chunk_ramp" in extra) is cuda
    assert ("codec_graph_batch_sizes" in extra) is cuda
    assert extra["initial_codec_chunk_frames"] == 4
    assert extra["talker_first_audio"] is True


class _Connector:
    config = {"extra": _EXTRA}


class _TransferManager:
    def __init__(self) -> None:
        self.connector = _Connector()
        self.code_prompt_token_ids: dict[str, list[torch.Tensor]] = {"r": []}
        self.put_req_chunk = {"r": 0}
        self.ramp_chunk_count: dict[str, int] = {}


class _Request:
    external_req_id = "r"
    sampling_params = None


@pytest.mark.parametrize(("cuda", "first_chunk_frames"), [(True, 1), (False, 4)])
def test_talker_first_chunk_follows_platform(monkeypatch, cuda, first_chunk_frames) -> None:
    monkeypatch.setattr(codec_config, "current_omni_platform", _Platform(cuda))
    manager = _TransferManager()
    emitted = None
    frames = 0
    while emitted is None and frames < 25:
        frames += 1
        emitted = talker2code2wav_async_chunk(
            manager,
            {"codes": {"audio": torch.ones((1, 16), dtype=torch.long)}},
            _Request(),
            is_finished=False,
        )
    assert frames == first_chunk_frames
