# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Save received MP4 packets without changing pixels or presentation timestamps."""

import io
import logging
from typing import Any, cast

import av

logger = logging.getLogger(__name__)


def finalize_streaming_video_bytes(video_bytes: bytes, *, input_format: str, fps: float = 25.0) -> bytes:
    """Remux received tracks; fps is a label, never a retiming instruction."""
    if not video_bytes:
        return video_bytes
    if input_format.lower() != "m4s":
        raise ValueError(f"Unsupported streaming video format: {input_format}")
    try:
        with cast(Any, av.open(io.BytesIO(video_bytes), format="mp4")) as container:
            buffer = io.BytesIO()
            streams = [stream for stream in container.streams if stream.type in ("video", "audio")]
            with cast(Any, av.open(buffer, mode="w", format="mp4")) as output:
                targets = {stream.index: output.add_stream_from_template(stream) for stream in streams}
                for packet in container.demux(*streams):
                    if packet.dts is None:
                        continue
                    packet.stream = targets[packet.stream.index]
                    output.mux(packet)
            return buffer.getvalue()
    except Exception as error:
        # Retain the input on saving errors; independent validators still decode it.
        logger.warning("Could not remux streamed video; retaining received bytes: %s", error)
        return video_bytes
