# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contracts for the SheetSage2 transcription server."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tools import sheetsage2_serve as serve

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SCORE = "X:1\nM:4/4\nK:C\nC4|\n"


@pytest.fixture
def model(mocker):
    model = mocker.Mock()
    model.transcribe.return_value = {"abc": SCORE, "midi": b"MThd\x00\x00\x00\x06"}
    return model


@pytest.fixture
def client(model):
    return TestClient(serve.create_app(model, "m-a-p/SheetSage2", "bf16", max_upload_bytes=64))


def upload(client, data=b"recording", **form):
    return client.post("/v1/audio/transcriptions", files={"file": ("song.M4A", data)}, data=form)


@pytest.mark.parametrize("melody_only", [True, False])
def test_transcription_returns_score_and_forwards_options(client, model, melody_only):
    response = upload(client, model="m-a-p/SheetSage2", melody_only=str(melody_only).lower(), max_seconds="45")

    assert response.status_code == 200
    assert response.json() == {"text": SCORE}
    (audio,), options = model.transcribe.call_args
    assert Path(audio).suffix == ".m4a"  # FFmpeg keeps the container hint.
    assert options["melody_only"] is melody_only
    assert options["max_seconds"] == 45
    assert options["dtype"] == "bf16"
    assert not Path(audio).exists()  # Uploads and annotations are temporary.


def test_text_format_returns_the_bare_score(client):
    response = upload(client, response_format="text")
    assert response.status_code == 200
    assert response.text == SCORE


@pytest.mark.parametrize(
    "form,status",
    [
        ({"model": "m-a-p/YuE2-3B"}, 404),
        ({"response_format": "srt"}, 400),
        ({"max_seconds": "0"}, 400),
        ({"max_seconds": "nan"}, 400),
    ],
)
def test_invalid_requests_fail_before_transcription(client, model, form, status):
    response = upload(client, **form)
    assert response.status_code == status
    assert response.json()["error"]["code"] == status
    model.transcribe.assert_not_called()


@pytest.mark.parametrize("data,status", [(b"", 400), (b"x" * 65, 413)])
def test_empty_and_oversized_uploads_are_rejected(client, model, data, status):
    assert upload(client, data=data).status_code == status
    model.transcribe.assert_not_called()


def test_undecodable_recording_is_a_client_error(client, model):
    model.transcribe.side_effect = ValueError("Cannot decode audio")
    response = upload(client)
    assert response.status_code == 400
    assert "Cannot decode audio" in response.json()["error"]["message"]


def test_failed_score_export_is_a_server_error(client, model):
    model.transcribe.return_value = {"abc": "", "abc_error": "no beats", "midi": b"MThd"}
    response = upload(client)
    assert response.status_code == 500
    assert "no beats" in response.json()["error"]["message"]


def test_server_requires_reviewed_remote_code():
    args = serve.create_parser().parse_args([])
    with pytest.raises(ValueError, match="--trust-remote-code"):
        serve.validate_args(args)
