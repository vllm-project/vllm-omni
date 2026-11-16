# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for multimodal OpenAI example clients."""

import base64
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import soundfile

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass(frozen=True)
class _Args:
    model: str = "Qwen/Qwen2.5-Omni-7B"
    query_type: str = "text"
    video_path: None = None
    image_path: None = None
    audio_path: None = None
    prompt: str = "Summarize the input."
    modalities: str = "audio,text"
    stream: bool = False
    seed: int | None = None
    temperature: float | None = None
    num_concurrent_requests: int = 1
    prompts: None = None
    speaker: None = None
    speakers: None = None
    output_audio_path: str | None = None


def _load_client_module(client_dir: str = "") -> ModuleType:
    repo_root = Path(__file__).resolve().parents[3]
    client_path = (
        repo_root
        / "examples"
        / "online_serving"
        / client_dir
        / "openai_chat_completion_client_for_multimodal_generation.py"
    )
    spec = importlib.util.spec_from_file_location(f"multimodal_generation_client_{client_dir}", client_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capture_request(mocker, args: _Args) -> dict:
    module = _load_client_module()
    completion = mocker.Mock(choices=[])
    create = mocker.Mock(return_value=completion)
    client = mocker.Mock()
    client.chat.completions.create = create

    module.run_multimodal_generation(args, client)

    create.assert_called_once()
    return create.call_args.kwargs


def test_multimodal_generation_omits_sampling_defaults(mocker) -> None:
    request = _capture_request(mocker, _Args())

    assert "seed" not in request
    assert "temperature" not in request


def test_multimodal_generation_forwards_explicit_sampling(mocker) -> None:
    request = _capture_request(mocker, _Args(seed=42, temperature=0))

    assert request["seed"] == 42
    assert request["temperature"] == 0


@pytest.mark.parametrize("client_dir", ["minicpmo", "mimo_audio"])
def test_streaming_clients_save_one_wav_per_request_and_choice(client_dir, tmp_path, monkeypatch, mocker) -> None:
    from vllm.entrypoints.generate.base.protocol import DeltaMessage

    from vllm_omni.entrypoints.openai.audio_utils_mixin import create_wav_header
    from vllm_omni.entrypoints.openai.protocol.chat_completion import (
        OmniChatCompletionResponseStreamChoice,
        OmniChatCompletionStreamResponse,
    )

    client = mocker.Mock()
    mocker.patch("openai.OpenAI", return_value=client)
    # MiniCPM imports the shared client by its script name; restore both import
    # paths and the module alias after this test.
    shared = _load_client_module()
    monkeypatch.syspath_prepend(str(Path(shared.__file__).parent))
    monkeypatch.setitem(sys.modules, "openai_chat_completion_client_for_multimodal_generation", shared)
    module = _load_client_module(client_dir)
    monkeypatch.chdir(tmp_path)

    streams = []
    expected_files = {}
    for request_index in range(2):
        request_id = f"chatcmpl-request-{request_index}"

        def event(index, content, modality="audio"):
            return OmniChatCompletionStreamResponse(
                id=request_id,
                created=0,
                model="test-model",
                modality=modality,
                choices=[OmniChatCompletionResponseStreamChoice(index=index, delta=DeltaMessage(content=content))],
            )

        chunks = [event(0, ""), event(1, "hello", "text")]
        waveforms = {}
        for index in (1, 0):
            samples = np.array([1000, -2000, 3000, -4000, 5000], dtype="<i2") + request_index * 200 + index * 100
            waveform = np.column_stack((samples, -samples)) if index else samples[:, None]
            waveforms[index] = waveform
            sample_rate = 22050 + request_index * 1950
            header = create_wav_header(sample_rate, waveform.shape[1])
            expected_files[header + waveform.astype("<i2").tobytes()] = (request_id, sample_rate, waveform)
            chunks.append(event(index, base64.b64encode(header + waveform[:2].astype("<i2").tobytes()).decode()))
        # Reverse choice order for the next chunks and interleave text/empty audio.
        chunks.extend([event(1, ""), event(0, "more text", "text")])
        for index in (0, 1):
            chunks.append(event(index, base64.b64encode(waveforms[index][2:].astype("<i2").tobytes()).decode()))
        streams.append(iter(chunks))

    client.chat.completions.create.side_effect = streams
    args = _Args(stream=True, num_concurrent_requests=2, output_audio_path=str(tmp_path))
    if client_dir == "mimo_audio":
        module.run_multimodal_generation(args)
    else:
        module.run_multimodal_generation(args, client)

    assert client.chat.completions.create.call_count == 2
    files = list(tmp_path.rglob("*.wav"))
    assert len(files) == len(expected_files) == 4
    for path in files:
        request_id, sample_rate, waveform = expected_files.pop(path.read_bytes())
        assert request_id in str(path)
        decoded, rate = soundfile.read(path, dtype="int16", always_2d=True)
        assert rate == sample_rate
        np.testing.assert_array_equal(decoded, waveform)
    assert not expected_files
