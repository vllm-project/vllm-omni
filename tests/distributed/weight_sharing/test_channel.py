# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU framing and Linux endpoint ownership tests; no CUDA initialization."""

from __future__ import annotations

import importlib
import os
import socket
import struct
import sys
import threading
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import contextmanager

import msgspec
import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def channel(weight_sharing):
    return importlib.import_module(f"{weight_sharing.__name__}.channel")


@pytest.fixture
def messages(weight_sharing):
    return importlib.import_module(f"{weight_sharing.__name__}.contracts")


@pytest.fixture
def sockets():
    left, right = socket.socketpair()
    yield left, right
    left.close()
    right.close()


def _frame(data: bytes) -> bytes:
    return struct.pack("!I", len(data)) + data


@contextmanager
def _background(operation, timeout=3):
    """Capture background results and exceptions while keeping shutdown bounded."""
    result: Future[object] = Future()
    started = threading.Event()

    def run():
        started.set()
        try:
            result.set_result(operation())
        except BaseException as error:
            result.set_exception(error)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert started.wait(1)
        yield result
    finally:
        thread.join(timeout)
        assert not thread.is_alive()


def test_roundtrip_and_message_boundaries(channel, messages, sockets):
    limits = channel.Limits(timeout=2)
    sender = channel.FramedConnection(sockets[0], limits)
    receiver = channel.FramedConnection(sockets[1], limits)
    large = messages.Message("error", "session", 1, error="x" * 20_000)
    following = messages.Message("accepted", "session", 1)

    def send():
        sender.send(large)
        sender.send(following)

    with _background(send) as sent:
        try:
            assert receiver.recv() == large
            assert receiver.recv() == following
        finally:
            receiver.close()
            sender.close()
        sent.result(timeout=3)


@pytest.mark.parametrize("length", [0, 513, 2**32 - 1])
def test_frame_length_rejected_before_reading_payload(channel, sockets, length):
    receiver = channel.FramedConnection(sockets[1], channel.Limits(max_bundle_bytes=512))
    sockets[0].sendall(struct.pack("!I", length))
    with pytest.raises(channel.ChannelError, match="size limit"):
        receiver.recv()


def test_fragmented_header_and_body(channel, messages, sockets):
    message = messages.Message("accepted", "session", 1)
    wire = _frame(msgspec.msgpack.encode(message))
    chunks = [wire[:1], wire[1:3], wire[3:4], wire[4:9], wire[9:]]

    def send():
        for chunk in chunks:
            sockets[0].sendall(chunk)
            time.sleep(0.01)

    with _background(send) as sent:
        assert channel.FramedConnection(sockets[1], channel.Limits(timeout=2)).recv() == message
        sent.result(timeout=3)


@pytest.mark.parametrize(
    "payload",
    [
        b"\xc1",
        b"\x81\xa2op",
        b"\x80\x80",
        msgspec.msgpack.encode({"op": "execute_callable", "session_id": "s", "generation": 1}),
        msgspec.msgpack.encode({"op": "hello", "session_id": "s", "generation": 1, "unexpected": "field"}),
        msgspec.msgpack.encode({"op": "hello", "session_id": "s", "generation": "wrong type"}),
        msgspec.msgpack.encode(["hello", "s", 1]),
    ],
)
def test_invalid_messagepack_or_schema_is_rejected(channel, sockets, payload):
    sockets[0].sendall(_frame(payload))
    with pytest.raises(channel.ChannelError, match="valid metadata"):
        channel.FramedConnection(sockets[1]).recv()


def test_send_rejects_arbitrary_objects(channel, sockets):
    with pytest.raises(channel.ChannelError, match="declared protocol"):
        channel.FramedConnection(sockets[0]).send({"op": "hello"})


def test_oversized_bundle_rejected_before_send(channel, messages, sockets):
    sender = channel.FramedConnection(sockets[0], channel.Limits(max_bundle_bytes=64))
    with pytest.raises(channel.ChannelError, match="size limit"):
        sender.send(messages.Message("error", "s", 1, error="x" * 200))
    sockets[1].setblocking(False)
    with pytest.raises(BlockingIOError):
        sockets[1].recv(1)


@pytest.mark.parametrize("wait_for_message", [False, True])
def test_partial_reads_share_one_deadline(channel, messages, sockets, wait_for_message):
    receiver = channel.FramedConnection(sockets[1], channel.Limits(timeout=0.15))
    wire = _frame(msgspec.msgpack.encode(messages.Message("accepted", "s", 1)))
    stopped = threading.Event()

    def drip():
        try:
            for byte in wire:
                if stopped.wait(0.02):
                    break
                sockets[0].sendall(bytes([byte]))
        except OSError:
            pass  # The test closes the socket when the total deadline expires.

    with _background(drip, timeout=2) as sent:
        before = time.monotonic()
        try:
            with pytest.raises(channel.ChannelError):
                receiver.recv(wait_for_message=wait_for_message)
            assert time.monotonic() - before < 1.0
        finally:
            stopped.set()
            receiver.close()
        sent.result(timeout=2)


def test_idle_session_can_receive_after_operation_timeout(channel, messages, sockets):
    receiver = channel.FramedConnection(sockets[1], channel.Limits(timeout=0.1))
    sender = channel.FramedConnection(sockets[0])
    message = messages.Message("accepted", "session", 1)

    with _background(lambda: receiver.recv(wait_for_message=True), timeout=1) as received:
        try:
            with pytest.raises(FutureTimeoutError):
                received.result(timeout=0.3)
            sender.send(message)
            assert received.result(timeout=1) == message
        finally:
            receiver.close()
            sender.close()


@pytest.mark.parametrize("wait_for_message", [False, True])
def test_close_interrupts_pending_receive_and_is_idempotent(channel, sockets, wait_for_message):
    receiver = channel.FramedConnection(sockets[1], channel.Limits(timeout=5))

    with _background(lambda: receiver.recv(wait_for_message=wait_for_message), timeout=1) as received:
        receiver.close()
        receiver.close()
        with pytest.raises(channel.ChannelError):
            received.result(timeout=1)


@pytest.mark.parametrize("wait_for_message", [False, True])
def test_peer_close_between_messages_is_eof(channel, sockets, wait_for_message):
    sockets[0].close()
    with pytest.raises(EOFError, match="between messages"):
        channel.FramedConnection(sockets[1]).recv(wait_for_message=wait_for_message)


@pytest.mark.parametrize("partial", [b"\x00\x00", struct.pack("!I", 4), struct.pack("!I", 4) + b"\x81"])
@pytest.mark.parametrize("wait_for_message", [False, True])
def test_peer_close_with_partial_frame_is_protocol_error(channel, sockets, partial, wait_for_message):
    sockets[0].sendall(partial)
    sockets[0].close()
    with pytest.raises(channel.ChannelError, match="during an operation"):
        channel.FramedConnection(sockets[1]).recv(wait_for_message=wait_for_message)


@pytest.mark.parametrize(
    "options",
    [
        {"timeout": 0},
        {"timeout": float("inf")},
        {"max_bundle_bytes": 0},
        {"max_bundle_bytes": 2**32},
        {"max_entries": 0},
    ],
)
def test_invalid_limits_rejected(channel, options):
    with pytest.raises(ValueError):
        channel.Limits(**options)


linux_only = pytest.mark.skipif(sys.platform != "linux", reason="Linux Unix peer credentials required")


@linux_only
def test_private_unix_endpoint_and_joined_shutdown(channel, messages, tmp_path):
    tmp_path.chmod(0o700)
    endpoint = str(tmp_path / "weights.sock")
    handled = threading.Event()

    def handler(connection):
        request = connection.recv()
        connection.send(messages.Message("accepted", request.session_id, request.generation))
        handled.set()

    server = channel.SessionServer(endpoint, channel.Limits(timeout=2), handler)
    server.start()
    try:
        with channel.connect(endpoint) as connection:
            connection.send(messages.Message("hello", "s", 1))
            assert connection.recv().op == "accepted"
        assert handled.wait(2)
        assert os.stat(endpoint).st_mode & 0o777 == 0o600
    finally:
        server.stop()
        server.stop()
    assert not os.path.exists(endpoint)
    assert server.last_error is None


@linux_only
def test_existing_endpoint_is_never_removed(channel, tmp_path):
    tmp_path.chmod(0o700)
    endpoint = tmp_path / "weights.sock"
    endpoint.write_bytes(b"belongs to another session")
    server = channel.SessionServer(str(endpoint), channel.Limits(timeout=1), lambda connection: None)
    with pytest.raises(channel.ChannelError, match="start"):
        server.start()
    server.stop()
    assert endpoint.read_bytes() == b"belongs to another session"
    assert server.last_error is not None


@linux_only
def test_endpoint_cleanup_checks_inode(channel, tmp_path):
    tmp_path.chmod(0o700)
    endpoint = tmp_path / "weights.sock"
    server = channel.SessionServer(str(endpoint), channel.Limits(timeout=1), lambda connection: None)
    server.start()
    endpoint.unlink()
    endpoint.write_bytes(b"replacement")
    server.stop()
    assert endpoint.read_bytes() == b"replacement"


@linux_only
def test_shared_directory_is_rejected(channel, tmp_path):
    tmp_path.chmod(0o755)
    server = channel.SessionServer(str(tmp_path / "weights.sock"), channel.Limits(timeout=1), lambda connection: None)
    with pytest.raises(channel.ChannelError, match="start"):
        server.start()
    assert not (tmp_path / "weights.sock").exists()


@linux_only
def test_stop_wakes_server_handler_waiting_for_message(channel, tmp_path):
    tmp_path.chmod(0o700)
    started = threading.Event()

    def handler(connection):
        started.set()
        connection.recv(wait_for_message=True)

    endpoint = str(tmp_path / "weights.sock")
    server = channel.SessionServer(endpoint, channel.Limits(timeout=2), handler)
    server.start()
    with channel.connect(endpoint):
        assert started.wait(1)
        server.stop()
    assert not os.path.exists(endpoint)
