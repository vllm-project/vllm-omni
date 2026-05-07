import pytest

from benchmarks.diffusion.backends import RequestFuncInput, async_request_chat_completions


class _MockResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return str(self._payload)


class _MockSession:
    def __init__(self, payload: dict):
        self._payload = payload

    def post(self, *args, **kwargs):
        return _MockResponse(self._payload)


@pytest.mark.core_model
@pytest.mark.benchmark
@pytest.mark.cpu
@pytest.mark.asyncio
async def test_chat_completions_metrics_fallback_to_top_level():
    payload = {
        "choices": [
            {
                "message": {
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,abc"},
                        }
                    ]
                }
            }
        ],
        "metrics": {
            "stage_durations": {"diffusion": 1.25},
            "peak_memory_mb": 4096.0,
        },
    }

    output = await async_request_chat_completions(
        RequestFuncInput(
            prompt="draw a cat",
            api_url="http://test.local/v1/chat/completions",
            model="ByteDance-Seed/BAGEL-7B-MoT",
        ),
        session=_MockSession(payload),
    )

    assert output.success is True
    assert output.stage_durations == {"diffusion": 1.25}
    assert output.peak_memory_mb == 4096.0


@pytest.mark.core_model
@pytest.mark.benchmark
@pytest.mark.cpu
@pytest.mark.asyncio
async def test_chat_completions_metrics_message_level_takes_precedence():
    payload = {
        "choices": [
            {
                "message": {
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,abc"},
                            "stage_durations": {"message_stage": 0.7},
                            "peak_memory_mb": 1234.0,
                        }
                    ]
                }
            }
        ],
        "metrics": {
            "stage_durations": {"top_level_stage": 9.9},
            "peak_memory_mb": 9999.0,
        },
    }

    output = await async_request_chat_completions(
        RequestFuncInput(
            prompt="draw a dog",
            api_url="http://test.local/v1/chat/completions",
            model="ByteDance-Seed/BAGEL-7B-MoT",
        ),
        session=_MockSession(payload),
    )

    assert output.success is True
    assert output.stage_durations == {"message_stage": 0.7}
    assert output.peak_memory_mb == 1234.0
