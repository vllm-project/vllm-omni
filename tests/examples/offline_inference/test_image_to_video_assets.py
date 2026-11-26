# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import socket
from unittest.mock import Mock, patch

import pytest
import requests

from tests.examples.offline_inference import test_image_to_video as example

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _dns_connection_error(error_number: int) -> requests.ConnectionError:
    """Get the real requests/urllib3 wrapper chain without accessing a network."""
    dns_error = socket.gaierror(error_number, "name resolution failed")
    with requests.Session() as session, patch.object(socket, "getaddrinfo", side_effect=dns_error):
        session.trust_env = False
        with pytest.raises(requests.ConnectionError) as raised:
            session.get(example._IMAGE_URL, timeout=1)
    return raised.value


@pytest.fixture
def image_download(monkeypatch, tmp_path):
    # Ensure the test cannot use a developer's preseeded asset.
    monkeypatch.setattr(example, "I2V_SCRIPT", tmp_path / "source" / "image_to_video.py")
    get = Mock()
    sleep = Mock()
    monkeypatch.setattr(example.requests, "get", get)
    monkeypatch.setattr(example.time, "sleep", sleep)
    return get, sleep


def test_asset_download_recovers_from_temporary_dns_error(tmp_path, image_download):
    get, sleep = image_download
    response = Mock()
    response.content = b"exact image bytes"
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    get.side_effect = [_dns_connection_error(socket.EAI_AGAIN), _dns_connection_error(socket.EAI_AGAIN), response]

    example._ensure_test_image(tmp_path)

    assert (tmp_path / example._IMAGE_NAME).read_bytes() == response.content
    assert get.call_count == 3
    assert [call.args for call in sleep.call_args_list] == [(1,), (2,)]
    response.raise_for_status.assert_called_once_with()
    response.__exit__.assert_called_once()


@pytest.mark.parametrize(
    ("error", "expected_attempts"),
    [
        (_dns_connection_error(socket.EAI_AGAIN), 3),
        (_dns_connection_error(socket.EAI_NONAME), 1),
        (requests.ConnectionError("connection refused"), 1),
        (requests.ReadTimeout("read timed out"), 1),
    ],
    ids=["persistent-temporary-dns", "permanent-dns", "other-connection-error", "read-timeout"],
)
def test_asset_download_preserves_connection_failures(tmp_path, image_download, error, expected_attempts):
    get, sleep = image_download
    get.side_effect = error

    with pytest.raises(type(error)) as raised:
        example._ensure_test_image(tmp_path)

    assert raised.value is error
    assert not (tmp_path / example._IMAGE_NAME).exists()
    assert get.call_count == expected_attempts
    assert sleep.call_count == expected_attempts - 1


def test_asset_download_does_not_retry_http_error(tmp_path, image_download):
    get, sleep = image_download
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    error = requests.HTTPError("404 Client Error")
    response.raise_for_status.side_effect = error
    get.return_value = response

    with pytest.raises(requests.HTTPError) as raised:
        example._ensure_test_image(tmp_path)

    assert raised.value is error
    assert not (tmp_path / example._IMAGE_NAME).exists()
    get.assert_called_once_with(example._IMAGE_URL, timeout=60)
    sleep.assert_not_called()
    response.__exit__.assert_called_once()


@pytest.mark.parametrize("existing_location", ["destination", "source"])
def test_existing_asset_needs_no_download(tmp_path, image_download, existing_location):
    get, sleep = image_download
    existing_dir = tmp_path if existing_location == "destination" else example.I2V_SCRIPT.parent
    existing_dir.mkdir(parents=True, exist_ok=True)
    (existing_dir / example._IMAGE_NAME).write_bytes(b"preseeded image")

    example._ensure_test_image(tmp_path)

    assert (tmp_path / example._IMAGE_NAME).read_bytes() == b"preseeded image"
    get.assert_not_called()
    sleep.assert_not_called()


def test_cli_examples_share_download_and_get_separate_copies(tmp_path, image_download):
    get, sleep = image_download
    response = Mock()
    response.content = b"shared example image"
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    get.return_value = response
    runner = Mock(output_root=tmp_path / "runs")
    runner.run.return_value = Mock(assets=[])
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()
    snippets = [snippet for snippet in example.README_SNIPPETS if not snippet.skip[0]][:2]
    assert len(snippets) == 2

    for snippet in snippets:
        example.test_image_to_video(snippet, runner, shared_dir)
        destination = runner.output_root / example.EXAMPLE_OUTPUT_SUBFOLDER / snippet.test_id / example._IMAGE_NAME
        assert destination.read_bytes() == response.content
        assert destination != shared_dir / example._IMAGE_NAME

    get.assert_called_once_with(example._IMAGE_URL, timeout=60)
    assert runner.run.call_count == 2
    sleep.assert_not_called()
