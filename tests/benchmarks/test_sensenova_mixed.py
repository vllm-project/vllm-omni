# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validate mixed SenseNova benchmark responses before recording latency."""

import base64
import io
import sys

import pytest
from PIL import Image

from benchmarks.diffusion import sensenova_mixed

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


def _response(content):
    return {"choices": [{"message": {"content": content}}]}


def _image_content(size=(32, 16)):
    buffer = io.BytesIO()
    Image.new("RGB", size).save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    return [{"type": "image_url", "image_url": {"url": url}}]


@pytest.mark.parametrize("case", ["t2i:32x16", "t2i-think:32x16"])
def test_image_response_requires_decodable_image_with_requested_size(case):
    sensenova_mixed._validate_response(case, _response(_image_content()))
    sensenova_mixed._validate_response(case, _response([{"type": "text", "text": "Done."}, *_image_content()]))

    with pytest.raises(RuntimeError, match="Wrong image dimensions"):
        sensenova_mixed._validate_response(case, _response(_image_content((16, 16))))
    with pytest.raises(RuntimeError, match="Undecodable image"):
        sensenova_mixed._validate_response(
            case, _response([{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}])
        )
    with pytest.raises(RuntimeError, match="No image returned"):
        sensenova_mixed._validate_response(case, _response("Image generation completed but no images were produced."))
    with pytest.raises(RuntimeError, match="No image returned"):
        sensenova_mixed._validate_response(
            case, _response([{"type": "text", "text": "Image generation completed but no images were produced."}])
        )


@pytest.mark.parametrize("case", ["t2t", "i2t"])
def test_text_response_requires_nonempty_text(case):
    sensenova_mixed._validate_response(case, _response("A red cube."))
    with pytest.raises(RuntimeError, match="No text returned"):
        sensenova_mixed._validate_response(case, _response("  "))
    with pytest.raises(RuntimeError, match="No text returned"):
        sensenova_mixed._validate_response(case, _response(_image_content()))


def test_invalid_image_does_not_enter_benchmark_report(monkeypatch, tmp_path):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return _response("Image generation completed but no images were produced.")

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def post(self, *_args, **_kwargs):
            return Response()

    report = tmp_path / "result.json"
    monkeypatch.setattr(sensenova_mixed.requests, "Session", Session)
    monkeypatch.setattr(
        sys,
        "argv",
        ["sensenova_mixed.py", "--rounds", "1", "--case", "t2i:32x16", "--output", str(report)],
    )

    with pytest.raises(RuntimeError, match="No image returned"):
        sensenova_mixed.main()

    assert not report.exists()


def test_selected_model_is_sent_for_every_case(monkeypatch, tmp_path):
    payloads = []

    class Response:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            pass

        def json(self):
            return _response(self.content)

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def post(self, *_args, json, **_kwargs):
            payloads.append(json)
            content = _image_content() if json["modalities"] == ["image"] else "A red cube."
            return Response(content)

    report = tmp_path / "result.json"
    monkeypatch.setattr(sensenova_mixed.requests, "Session", Session)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sensenova_mixed.py",
            "--model",
            "selected/model",
            "--rounds",
            "1",
            "--case",
            "t2t",
            "--case",
            "i2t",
            "--case",
            "t2i:32x16",
            "--case",
            "t2i-think:32x16",
            "--output",
            str(report),
        ],
    )

    sensenova_mixed.main()

    assert len(payloads) == 4
    assert all(payload["model"] == "selected/model" for payload in payloads)
    assert '"model": "selected/model"' in report.read_text()
