# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""L1 tests for the image response encoder's compression contract."""

import base64
import io
import multiprocessing
import threading

import pytest
from PIL import Image

from vllm_omni.entrypoints.openai.image_api_utils import (
    encode_image_base64_with_compression,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(scope="module")
def image():
    # Photographic-content proxy: noise-plus-gradient content compresses very
    # differently across PNG compress levels, which is what the mapping under
    # test controls.
    import numpy as np

    rng = np.random.default_rng(0)
    grad = np.linspace(0, 255, 256, dtype=np.float32)
    arr = np.clip(grad[None, :, None] + rng.integers(-24, 25, (256, 256, 3)), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_png_output_compression_maps_to_compress_level(image):
    """output_compression 100 -> compress_level 0, 1 -> compress_level 9."""

    def encode(compression):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", compress_level=max(0, min(9, 9 - compression // 11)))
        return len(base64.b64encode(buffer.getvalue()))

    fast = encode_image_base64_with_compression(image, format="png", output_compression=100)
    small = encode_image_base64_with_compression(image, format="png", output_compression=1)

    # Level-0 PNG barely compresses; the mapped level-9 encode is smaller.
    assert len(small) < len(fast)
    # The mapping must produce the same sizes as the equivalent compress_level.
    # Level 0 is stored uncompressed (by a faster encoder than PIL's, with the
    # same pixels), so its size is the raw scanline bytes plus framing.
    raw = image.height * (1 + image.width * 3)
    assert raw <= len(base64.b64decode(fast)) <= raw * 1.01
    with io.BytesIO(base64.b64decode(fast)) as buffer:
        assert Image.open(buffer).tobytes() == image.tobytes()
    assert len(small) == encode(1)


def test_jpeg_output_compression_maps_to_quality(image):
    high = base64.b64decode(encode_image_base64_with_compression(image, format="jpeg", output_compression=100))
    low = base64.b64decode(encode_image_base64_with_compression(image, format="jpeg", output_compression=10))

    assert len(high) > len(low)


@pytest.mark.parametrize(
    ("mode", "shape"),
    [("RGB", (256, 256, 3)), ("RGBA", (40, 1500, 4)), ("L", (300, 200)), ("RGB", (1500, 3000, 3)), ("RGB", (1, 1, 3))],
)
def test_uncompressed_png_decodes_to_the_same_pixels(mode, shape):
    import numpy as np

    from vllm_omni.entrypoints.openai.image_api_utils import encode_png_uncompressed

    array = np.random.default_rng(1).integers(0, 256, shape, dtype=np.uint8)
    image = Image.fromarray(array, mode=mode)
    data = bytes(encode_png_uncompressed(image))
    decoded = Image.open(io.BytesIO(data))
    decoded.load()  # validates every chunk CRC and the zlib checksum
    assert decoded.mode == mode and decoded.size == image.size
    assert np.array_equal(np.asarray(decoded), array)
    encoded = encode_image_base64_with_compression(image, format="png", output_compression=100)
    assert base64.b64decode(encoded) == data


def test_compressed_png_still_uses_pil(image):
    encoded = encode_image_base64_with_compression(image, format="png", output_compression=50)
    with io.BytesIO(base64.b64decode(encoded)) as buffer:
        assert Image.open(buffer).tobytes() == image.tobytes()


def _encode_uncompressed_png() -> None:
    from vllm_omni.entrypoints.openai.image_api_utils import encode_png_uncompressed

    image = Image.new("RGB", (64, 64), color=(12, 34, 56))
    with Image.open(io.BytesIO(encode_png_uncompressed(image))) as decoded:
        assert decoded.size == image.size and decoded.mode == image.mode
        assert decoded.tobytes() == image.tobytes()


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="requires fork")
def test_uncompressed_png_encodes_in_a_forked_child():
    from vllm_omni.entrypoints.openai import image_api_utils

    # Start every pool worker in this process, then encode in a forked child.
    workers = image_api_utils._PNG_THREADS
    barrier = threading.Barrier(workers)
    pool = image_api_utils._png_pool()
    list(pool.map(lambda _: barrier.wait(timeout=10), range(workers)))

    child = multiprocessing.get_context("fork").Process(target=_encode_uncompressed_png)
    child.start()
    try:
        child.join(timeout=30)
    finally:
        if child.is_alive():
            child.kill()
            child.join()
        exitcode = child.exitcode
        child.close()
    assert exitcode == 0
    assert image_api_utils._png_pool() is pool
    _encode_uncompressed_png()
