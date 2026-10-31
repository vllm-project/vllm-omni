# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import base64
import io
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PIL import Image, ImageFile

from tests.helpers import media

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("existing_cache", [False, True], ids=["cold-cache", "replace-cache"])
def test_image_cache_readers_never_observe_incomplete_encoding(monkeypatch, tmp_path, existing_cache):
    previous = media.generate_synthetic_image(64, 64, cache_dir=tmp_path, seed=41) if existing_cache else None
    encoding = threading.Event()
    resume = threading.Event()
    original_save = ImageFile._save

    def paused_encoding(*args, **kwargs):
        if threading.current_thread().name.startswith("image-cache-writer"):
            encoding.set()
            assert resume.wait(20), "Reader did not finish during the paused encoding"
        return original_save(*args, **kwargs)

    monkeypatch.setattr(ImageFile, "_save", paused_encoding)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="image-cache-writer") as pool:
        writer = pool.submit(media.generate_synthetic_image, 64, 64, cache_dir=tmp_path, seed=42, force_regenerate=True)
        try:
            assert encoding.wait(20), "Writer did not enter the JPEG encoder"
            read = media.generate_synthetic_image(64, 64, cache_dir=tmp_path, seed=42)
            raw = base64.b64decode(read["base64"])
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                assert image.size == (64, 64)
            if previous is not None:
                assert read["base64"] == previous["base64"]
        finally:
            resume.set()
        written = writer.result(timeout=20)
    with Image.open(written["file_path"]) as image:
        image.load()
        assert image.size == (64, 64)
    assert not list(tmp_path.glob(".*.tmp"))


def test_cached_image_pixels_and_bytes_share_one_published_snapshot(monkeypatch, tmp_path):
    previous = media.generate_synthetic_image(64, 64, cache_dir=tmp_path, seed=41)
    original_open = Image.open
    replacement = tmp_path / "replacement.jpg"
    Image.new("RGB", (64, 64), (250, 10, 10)).save(replacement, format="JPEG")

    def replace_after_open(*args, **kwargs):
        image = original_open(*args, **kwargs)
        replacement.replace(previous["file_path"])
        return image

    monkeypatch.setattr(Image, "open", replace_after_open)
    cached = media.generate_synthetic_image(64, 64, cache_dir=tmp_path, seed=42)

    assert cached["base64"] == previous["base64"]
    with original_open(io.BytesIO(base64.b64decode(cached["base64"]))) as image:
        image.load()
        assert np.array_equal(cached["np_array"], np.array(image))
