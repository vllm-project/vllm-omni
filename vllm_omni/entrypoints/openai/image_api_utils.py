# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
Shared media utilities for OpenAI-compatible image APIs.

PUT HERE:
  - Pure media helpers with no FastAPI Request / app-state / job semantics:
    size parsing, image encode/base64, layered-layer validation.

DO NOT PUT HERE:
  - ``/v1/images*`` request/engine/job helpers — those go in ``images.helpers``.

LONGEVITY:
  - This root utils file is a **temporary shared home**.
  - TODO(#5227, P1.2): tidy up / absorb into the images family (e.g. ``images/media.py``
    or equivalent) in the P1.2 image modality PR; do not treat this file as the
    long-term owner.
  - ``images.helpers`` is the longer home for endpoint-specific logic through
    P0.2/P0.3 until P1.2 further splits it.

See ``openai/README.md`` and ``images/README.md`` (utils vs helpers, no overlap).
"""

import io
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import PIL.Image
import pybase64 as base64

SUPPORTED_LAYERED_RESOLUTIONS = (640, 1024)
SUPPORTED_LAYERED_LAYERS_RANGE = range(2, 11)


def parse_size(size_str: str) -> tuple[int, int]:
    """Parse size string to width and height tuple.

    Args:
        size_str: Size in format "WIDTHxHEIGHT" (e.g., "1024x1024")

    Returns:
        Tuple of (width, height)

    Raises:
        ValueError: If size format is invalid
    """
    if not size_str or not isinstance(size_str, str):
        raise ValueError(
            f"Size must be a non-empty string in format 'WIDTHxHEIGHT' (e.g., '1024x1024'), got: {size_str}"
        )

    parts = size_str.split("x")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid size format: '{size_str}'. Expected format: 'WIDTHxHEIGHT' (e.g., '1024x1024'). "
            f"Did you mean to use 'x' as separator?"
        )

    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid size format: '{size_str}'. Width and height must be integers.")

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid size: {width}x{height}. Width and height must be positive integers.")

    return width, height


def encode_image_base64(image: PIL.Image.Image) -> str:
    """Encode PIL Image to base64 PNG string.

    Args:
        image: PIL Image object

    Returns:
        Base64-encoded PNG image as string
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def prepare_image_for_output_format(image: PIL.Image.Image, format: str) -> PIL.Image.Image:
    fmt = format.lower()
    if fmt not in {"jpg", "jpeg"}:
        return image

    if image.mode == "RGB":
        return image

    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        alpha_image = image.convert("RGBA")
        flattened = PIL.Image.new("RGB", alpha_image.size, (255, 255, 255))
        flattened.paste(alpha_image, mask=alpha_image.getchannel("A"))
        return flattened

    return image.convert("RGB")


def encode_image_base64_with_compression(
    image: PIL.Image.Image, format: str = "png", output_compression: int = 100
) -> str:
    """Encode a PIL image to a base64 image string.

    Args:
        image: PIL Image object.
        format: Output image format, such as "png", "jpeg", or "webp".
        output_compression: Compression level (0-100), where 100 keeps the
            best quality for lossy formats and uses the lowest PNG compression.

    Returns:
        Base64-encoded image string.
    """
    buffer = io.BytesIO()
    image = prepare_image_for_output_format(image, format)
    pil_format = "jpeg" if format == "jpg" else format
    save_kwargs = {}
    if format in ("jpg", "jpeg", "webp"):
        save_kwargs["quality"] = output_compression
    elif format == "png":
        save_kwargs["compress_level"] = max(0, min(9, 9 - output_compression // 11))

    if save_kwargs.get("compress_level") == 0 and image.mode in _UNCOMPRESSED_PNG_MODES:
        # Level 0 stores the image uncompressed; PIL still runs per-row
        # adaptive filtering, which cannot shrink stored data.
        return base64.b64encode(encode_png_uncompressed(image)).decode("utf-8")
    image.save(buffer, format=pil_format, **save_kwargs)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


# PIL mode -> (PNG color type, channels) for 8-bit images.
_UNCOMPRESSED_PNG_MODES = {"L": (0, 1), "RGB": (2, 3), "RGBA": (6, 4)}
_STORED_BLOCK = 65535
_IDAT_CHUNK = 8 << 20
_PNG_THREADS = 8
_PNG_POOL: ThreadPoolExecutor | None = None


def _png_pool() -> ThreadPoolExecutor:
    global _PNG_POOL
    if _PNG_POOL is None:
        _PNG_POOL = ThreadPoolExecutor(max_workers=_PNG_THREADS, thread_name_prefix="png-encode")
    return _PNG_POOL


def _parallel_ranges(fn, total: int) -> None:
    """Run ``fn(start, stop)`` over ``total`` items split across the pool (numpy
    copies and zlib checksums release the GIL)."""
    if total <= 0:
        return
    step = -(-total // _PNG_THREADS)
    list(_png_pool().map(lambda index: fn(index * step, min(total, (index + 1) * step)), range(-(-total // step))))


def encode_png_uncompressed(image: PIL.Image.Image) -> memoryview:
    """Encode an 8-bit L/RGB/RGBA image as an uncompressed (zlib level 0) PNG.

    Every scanline uses filter type None and the zlib stream holds stored
    deflate blocks, so this decodes to exactly the pixels of ``image``. The
    file is assembled in one buffer with the copies and checksums spread over
    a thread pool; it is several times faster than PIL's level-0 encoder.
    """
    color_type, channels = _UNCOMPRESSED_PNG_MODES[image.mode]
    width, height = image.size
    pixels = np.frombuffer(image.tobytes(), dtype=np.uint8).reshape(height, width * channels)
    raw = np.empty((height, 1 + width * channels), dtype=np.uint8)

    def fill_rows(start: int, stop: int) -> None:
        raw[start:stop, 0] = 0
        raw[start:stop, 1:] = pixels[start:stop]

    _parallel_ranges(fill_rows, height)
    raw = raw.reshape(-1)
    size = raw.size
    blocks = max(1, -(-size // _STORED_BLOCK))
    lengths = np.full(blocks, _STORED_BLOCK, dtype=np.int64)
    lengths[-1] = size - _STORED_BLOCK * (blocks - 1)
    headers = np.zeros((blocks, 5), dtype=np.uint8)
    headers[-1, 0] = 1  # BFINAL
    headers[:, 1], headers[:, 2] = lengths & 0xFF, lengths >> 8
    headers[:, 3], headers[:, 4] = ~lengths & 0xFF, (~lengths >> 8) & 0xFF
    stream = np.empty(2 + size + 5 * blocks + 4, dtype=np.uint8)
    stream[:2] = (0x78, 0x01)
    body = stream[2:-4]
    full = blocks - 1
    if full:
        grid = body[: full * (_STORED_BLOCK + 5)].reshape(full, _STORED_BLOCK + 5)

        def fill_blocks(start: int, stop: int) -> None:
            grid[start:stop, :5] = headers[start:stop]
            grid[start:stop, 5:] = raw[start * _STORED_BLOCK : stop * _STORED_BLOCK].reshape(
                stop - start, _STORED_BLOCK
            )

        _parallel_ranges(fill_blocks, full)
    tail = body[full * (_STORED_BLOCK + 5) :]
    tail[:5] = headers[-1]
    tail[5:] = raw[full * _STORED_BLOCK :]
    stream[-4:] = np.frombuffer(struct.pack(">I", zlib.adler32(raw)), dtype=np.uint8)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    head = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr)
    end = _png_chunk(b"IEND", b"")
    chunks = -(-stream.size // _IDAT_CHUNK)
    output = np.empty(len(head) + stream.size + 12 * chunks + len(end), dtype=np.uint8)
    output[: len(head)] = np.frombuffer(head, dtype=np.uint8)
    output[output.size - len(end) :] = np.frombuffer(end, dtype=np.uint8)
    idat_crc = zlib.crc32(b"IDAT")

    def write_chunk(index: int) -> None:
        start, stop = index * _IDAT_CHUNK, min(stream.size, (index + 1) * _IDAT_CHUNK)
        offset = len(head) + index * (_IDAT_CHUNK + 12)
        payload = stream[start:stop]
        crc = zlib.crc32(memoryview(payload), idat_crc) & 0xFFFFFFFF
        output[offset : offset + 8] = np.frombuffer(struct.pack(">I", stop - start) + b"IDAT", dtype=np.uint8)
        output[offset + 8 : offset + 8 + payload.size] = payload
        output[offset + 8 + payload.size : offset + 12 + payload.size] = np.frombuffer(
            struct.pack(">I", crc), dtype=np.uint8
        )

    list(_png_pool().map(write_chunk, range(chunks)))
    return output.data.cast("B")


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(data, zlib.crc32(tag)) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def validate_layered_layers(layers: int | None) -> int | None:
    """Validate the Qwen-Image-Layered ``layers`` parameter."""
    if layers is None:
        return None
    if layers not in SUPPORTED_LAYERED_LAYERS_RANGE:
        raise ValueError(
            f"Invalid layers value {layers}. layers must be between "
            f"{SUPPORTED_LAYERED_LAYERS_RANGE.start} and "
            f"{SUPPORTED_LAYERED_LAYERS_RANGE.stop - 1} inclusive."
        )
    return layers
