import json
import sys
from dataclasses import dataclass

import pytest
from pytest_mock import MockerFixture

from vllm_omni.entrypoints.cli.main import main as omni_cli_main


class MockResponse:
    def __init__(self, status: int, chunks: list[bytes]):
        self.status = status
        self.reason = "OK" if status == 200 else "Error"
        self._chunks = chunks
        self.content = self

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


@dataclass
class _PostCall:
    url: str


class MockClientSession:
    def __init__(self, response: MockResponse):
        self._response = response
        self.post_calls: list[_PostCall] = []

    def post(self, url: str, *args, **kwargs):
        self.post_calls.append(_PostCall(url=url))
        return self._response

    async def close(self):
        return None


@pytest.mark.core_model
@pytest.mark.benchmark
@pytest.mark.cpu
def test_bench_serve_cli_mocks_http_request(mocker: MockerFixture, tmp_path):
    num_prompts = 5
    result_filename = "bench-result.json"
    result_path = tmp_path / result_filename

    chunks = [
        b'data: {"choices":[{"delta":{"content":"hi"}}],"modality":"text"}\n\n',
        b'data: {"choices":[{"delta":{"content":" there"}}],"modality":"text","metrics":{"num_tokens_out":4,"num_tokens_in":5}}\n\n',
        b"data: [DONE]\n\n",
    ]
    mock_response = MockResponse(200, chunks)
    mock_session = MockClientSession(mock_response)

    # Patch the aiohttp session used by vllm-omni's benchmark implementation.
    # We keep benchmark logic intact but intercept outbound HTTP calls.
    mocker.patch("vllm_omni.benchmarks.patch.patch.aiohttp.ClientSession", return_value=mock_session)
    mocker.patch("vllm_omni.benchmarks.patch.patch.aiohttp.TCPConnector", return_value=mocker.Mock())

    argv = [
        "vllm",
        "bench",
        "serve",
        "--omni",
        "--model",
        "Qwen/Qwen2.5-Omni-7B",
        "--port",
        "18000",
        "--dataset-name",
        "random",
        "--random-input-len",
        "32",
        "--random-output-len",
        "4",
        "--num-prompts",
        str(num_prompts),
        "--endpoint",
        "/v1/chat/completions",
        "--backend",
        "openai-chat-omni",
        "--disable-tqdm",
        "--num-warmups",
        "0",
        "--ready-check-timeout-sec",
        "0",
        "--save-result",
        "--result-dir",
        str(tmp_path),
        "--result-filename",
        result_filename,
    ]
    mocker.patch.object(sys, "argv", argv)

    omni_cli_main()

    assert result_path.exists()
    result = json.loads(result_path.read_text(encoding="utf-8"))

    sent_requests = len(mock_session.post_calls)
    assert result["completed"] == sent_requests == num_prompts
    assert any(call.url.endswith("/v1/chat/completions") for call in mock_session.post_calls)
