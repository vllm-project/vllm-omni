# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contracts for the SheetSage2 transcription server."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tools import sheetsage2_serve as serve

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SCORE = "X:1\nM:4/4\nK:C\nC4|\n"
LIMIT = 4096


@pytest.fixture
def model(mocker):
    model = mocker.Mock()
    model.transcribe.return_value = {"abc": SCORE, "midi": b"MThd\x00\x00\x00\x06"}
    return model


@pytest.fixture
def client(model):
    return TestClient(serve.create_app(model, "m-a-p/SheetSage2", "bf16", max_upload_bytes=LIMIT))


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


def assert_error(response, status, message=""):
    assert response.status_code == status
    error = response.json()["error"]
    assert error["code"] == status
    assert error["type"] == ("invalid_request_error" if status < 500 else "server_error")
    assert message in error["message"]


@pytest.mark.parametrize(
    "form,status,message",
    [
        ({"model": "m-a-p/YuE2-3B"}, 404, "does not exist"),
        ({"response_format": "srt"}, 400, "response_format"),
        ({"max_seconds": "0"}, 400, "max_seconds"),
        ({"max_seconds": "nan"}, 400, "max_seconds"),
        # FastAPI rejects these before the endpoint runs.
        ({"max_seconds": "abc"}, 400, "max_seconds: "),
        ({"melody_only": "invalid"}, 400, "melody_only: "),
    ],
)
def test_invalid_requests_fail_before_transcription(client, model, form, status, message):
    assert_error(upload(client, **form), status, message)
    model.transcribe.assert_not_called()


def test_missing_file_is_a_client_error(client, model):
    response = client.post("/v1/audio/transcriptions", data={"melody_only": "true"})
    assert_error(response, 400, "file: Field required")
    model.transcribe.assert_not_called()


def test_empty_upload_is_rejected(client, model):
    assert_error(upload(client, data=b""), 400, "file is empty")
    model.transcribe.assert_not_called()


def test_oversized_upload_is_rejected_while_received(client, model):
    assert_error(upload(client, data=b"x" * LIMIT), 413, "Request body exceeds")
    model.transcribe.assert_not_called()


def test_oversized_chunked_upload_is_rejected_while_received(client, model):
    def body():  # No Content-Length: the server counts what arrives.
        for _ in range(8):
            yield b"x" * (LIMIT // 2)

    response = client.post(
        "/v1/audio/transcriptions",
        content=body(),
        headers={"content-type": "multipart/form-data; boundary=b"},
    )
    assert_error(response, 413, "Request body exceeds")
    model.transcribe.assert_not_called()


def test_undecodable_recording_is_a_client_error(client, model, caplog):
    model.transcribe.side_effect = ValueError("Cannot decode audio /tmp/sheetsage2-x/input.m4a")
    response = upload(client)
    assert_error(response, 400, "cannot decode this recording")
    assert "/tmp" not in response.text  # Upstream details stay in the server log.
    assert "Cannot decode audio" in caplog.text


def test_failed_score_export_is_a_server_error(client, model, caplog):
    model.transcribe.return_value = {"abc": "", "abc_error": "no beats", "midi": b"MThd"}
    response = upload(client)
    assert_error(response, 500, "see the server log")
    assert "no beats" not in response.text
    assert "no beats" in caplog.text


def test_server_requires_reviewed_remote_code():
    args = serve.create_parser().parse_args([])
    with pytest.raises(ValueError, match="--trust-remote-code"):
        serve.validate_args(args)
