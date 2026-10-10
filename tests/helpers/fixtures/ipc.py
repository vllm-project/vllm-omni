# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native executor serialization for small trusted test payloads."""

import pytest
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue


@pytest.fixture
def executor_roundtrip():
    writer = MessageQueue(n_reader=1, n_local_reader=1, max_chunk_bytes=65536, max_chunks=2)
    reader = None
    try:
        reader = MessageQueue.create_from_handle(writer.export_handle(), rank=0)
        writer.local_socket.rcvtimeo = reader.local_socket.rcvtimeo = 5000
        writer.wait_until_ready()
        reader.wait_until_ready()

        def roundtrip(value):
            writer.enqueue(value, timeout=5)
            return reader.dequeue(timeout=5)

        yield roundtrip
    finally:
        for queue in (reader, writer):
            if queue is None:
                continue
            queue.shutdown()
            queue.local_socket.context.destroy(linger=0)
            # ShmRingBuffer owns close/unlink, including creator-only unlink.
            del queue.buffer
