# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CosyVoice3 online serving example clients."""

import argparse
import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

EXAMPLE_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "examples"
    / "online_serving"
    / "text_to_speech"
    / "cosyvoice3"
)


def _load_example_module(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EXAMPLE_DIR / filename)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_speech_client_builds_non_streaming_payload(tmp_path: Path):
    client = _load_example_module("cosyvoice3_speech_client", "speech_client.py")
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"RIFFfake")

    args = argparse.Namespace(
        model="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        text="Hello CosyVoice3.",
        ref_audio=str(ref_audio),
        ref_text="You are a helpful assistant.<|endofprompt|>reference text",
        response_format="wav",
        stream=False,
        seed=123,
        max_new_tokens=64,
        extra_params='{"custom_flag": true}',
    )

    payload = client.build_payload(args)

    assert payload["model"] == args.model
    assert payload["input"] == args.text
    assert payload["ref_audio"].startswith("data:audio/wav;base64,")
    assert payload["ref_text"] == args.ref_text
    assert payload["response_format"] == "wav"
    assert "stream" not in payload
    assert payload["seed"] == 123
    assert payload["max_new_tokens"] == 64
    assert payload["extra_params"] == {"custom_flag": True}


def test_speech_client_builds_streaming_pcm_payload():
    client = _load_example_module("cosyvoice3_speech_client_stream", "speech_client.py")
    args = argparse.Namespace(
        model="cosyvoice3",
        text="Stream this.",
        ref_audio="https://example.com/ref.wav",
        ref_text="reference text",
        response_format="wav",
        stream=True,
        seed=None,
        max_new_tokens=None,
        extra_params=None,
    )

    payload = client.build_payload(args)

    assert payload["ref_audio"] == "https://example.com/ref.wav"
    assert payload["response_format"] == "pcm"
    assert payload["stream"] is True


def test_streaming_client_imports_without_websockets_and_builds_config():
    client = _load_example_module("cosyvoice3_streaming_speech_client", "streaming_speech_client.py")
    args = argparse.Namespace(
        model="cosyvoice3",
        response_format="pcm",
        ref_audio="data:audio/wav;base64,AAAA",
        ref_text="reference text",
        split_granularity="clause",
        stream_audio=True,
        max_new_tokens=128,
    )

    config = client.build_session_config(args)

    assert config == {
        "model": "cosyvoice3",
        "response_format": "pcm",
        "ref_audio": "data:audio/wav;base64,AAAA",
        "ref_text": "reference text",
        "split_granularity": "clause",
        "stream_audio": True,
        "max_new_tokens": 128,
    }


def test_cosyvoice3_deploy_uses_pr_aligned_stage0_sampling_defaults():
    deploy_path = Path(__file__).parent.parent.parent.parent / "vllm_omni" / "deploy" / "cosyvoice3.yaml"
    text = deploy_path.read_text(encoding="utf-8")
    match = re.search(r"stage_id:\s*0.*?default_sampling_params:\n(?P<body>.*?)(?:\n\s{4}\S|\n\n)", text, re.S)
    assert match is not None
    sampling_body = match.group("body")

    assert re.search(r"temperature:\s*1\.0\b", sampling_body)
    assert re.search(r"top_p:\s*0\.8\b", sampling_body)
    assert re.search(r"top_k:\s*25\b", sampling_body)
    assert re.search(r"repetition_penalty:\s*2\.0\b", sampling_body)
    assert re.search(r"stop_token_ids:\s*\n\s*-\s*6562\b", sampling_body)
    assert re.search(r"detokenize:\s*false\b", sampling_body)
