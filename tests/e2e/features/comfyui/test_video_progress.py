# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import AsyncMock

import pytest
from comfyui_vllm_omni.utils import api_client

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


async def test_progress_reaches_100_only_after_decode(monkeypatch):
    reports = []
    statuses = [
        {"id": "video", "status": "queued"},
        {"status": "in_progress", "progress": 25},
        {"status": "in_progress", "progress": 10},
        {"status": "in_progress", "progress": "invalid"},
        {"status": "completed", "progress": 100},
        {},
    ]
    monkeypatch.setattr(api_client, "url_json", AsyncMock(side_effect=statuses))

    async def download(*args):
        assert 100 not in reports
        return b"video"

    def decode(data):
        assert 100 not in reports
        return data

    monkeypatch.setattr(api_client, "url_bytes", download)
    monkeypatch.setattr(api_client, "bytes_to_video", decode)
    client = api_client.VLLMOmniClient("http://localhost/v1", poll_interval=0)
    kwargs = dict(
        model="MiniMaxAI/MiniMax-H3",
        prompt="test",
        width=896,
        height=512,
        num_frames=90,
        fps=24,
        on_progress=reports.append,
    )
    assert await client.generate_video(**kwargs) == b"video"
    assert reports == [0, 25, 25, 25, 99, 100]


async def test_generate_node_updates_progress(monkeypatch):
    from types import SimpleNamespace

    import comfy.utils
    from comfyui_vllm_omni.nodes import VLLMOmniGenerateVideo

    reports = []
    totals = []

    def make_bar(total):
        totals.append(total)
        return SimpleNamespace(update_absolute=reports.append)

    async def generate(self, **kwargs):
        kwargs["on_progress"](40)
        kwargs["on_progress"](100)
        return "video"

    monkeypatch.setattr(comfy.utils, "ProgressBar", make_bar)
    monkeypatch.setattr(api_client.VLLMOmniClient, "generate_video", generate)
    output = await VLLMOmniGenerateVideo().generate(
        url="http://localhost/v1",
        model="MiniMaxAI/MiniMax-H3",
        prompt="test",
        width=896,
        height=512,
        fps=24,
        duration=2,
    )
    assert totals == [100]
    assert reports == [0, 40, 100]
    assert output == ("video",)
