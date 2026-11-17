# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import base64
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.helpers import media

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("existing_cache", [False, True])
def test_readers_do_not_see_an_incomplete_jpeg(tmp_path, monkeypatch, existing_cache):
    previous = media.generate_synthetic_image(32, 24, cache_dir=tmp_path) if existing_cache else None
    encoding = threading.Event()
    release = threading.Event()
    save = Image.Image.save

    def pause_writer(image, fp, *args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            # Reproduce Pillow's final-path truncation before encoding starts.
            # Encoding into memory must leave the published cache untouched.
            if isinstance(fp, (str, Path)):
                with open(fp, "w+b") as stream:
                    encoding.set()
                    assert release.wait(10)
                    return save(image, stream, *args, **kwargs)
            encoding.set()
            assert release.wait(10)
        return save(image, fp, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", pause_writer)
    with ThreadPoolExecutor(max_workers=1) as pool:
        writer = pool.submit(media.generate_synthetic_image, 32, 24, cache_dir=tmp_path, force_regenerate=True)
        try:
            assert encoding.wait(10)
            reader = media.generate_synthetic_image(32, 24, cache_dir=tmp_path)
            with Image.open(io.BytesIO(base64.b64decode(reader["base64"]))) as image:
                image.load()
                assert image.size == (32, 24)
            if previous is not None:
                assert reader["base64"] == previous["base64"]
        finally:
            release.set()
        published = writer.result(timeout=10)

    path = Path(published["file_path"])
    assert path.read_bytes() == base64.b64decode(published["base64"])
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("failure_at", ["encode", "publish"])
def test_failed_image_regeneration_preserves_cache(tmp_path, monkeypatch, failure_at):
    previous = media.generate_synthetic_image(32, 24, cache_dir=tmp_path)
    path = Path(previous["file_path"])

    def fail(*args, **kwargs):
        raise OSError("image publication failed")

    if failure_at == "encode":
        monkeypatch.setattr(Image.Image, "save", fail)
    else:
        monkeypatch.setattr(media.os, "replace", fail)
    with pytest.raises(OSError, match="image publication failed"):
        media.generate_synthetic_image(32, 24, cache_dir=tmp_path, force_regenerate=True)

    assert path.read_bytes() == base64.b64decode(previous["base64"])
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("broken_bytes", [b"", b"not a JPEG", b"\xff\xd8\xff\xe0"])
def test_incomplete_image_cache_is_regenerated(tmp_path, broken_bytes):
    path = tmp_path / "synth_image_w32_h24.jpg"
    path.write_bytes(broken_bytes)

    result = media.generate_synthetic_image(32, 24, cache_dir=tmp_path)

    assert path.read_bytes() == base64.b64decode(result["base64"])
    with Image.open(path) as image:
        image.load()
        assert image.size == (32, 24)
    assert list(tmp_path.iterdir()) == [path]


def test_cached_pixels_and_payload_use_the_same_generation(tmp_path, monkeypatch):
    media.generate_synthetic_image(32, 24, cache_dir=tmp_path)
    cache = tmp_path / "synth_image_w32_h24.jpg"
    replacement = tmp_path / "replacement.jpg"
    Image.new("RGB", (32, 24), "black").save(replacement)
    open_image = Image.open

    def replace_after_decode(fp, *args, **kwargs):
        image = open_image(fp, *args, **kwargs)
        image.load()
        replacement.replace(cache)
        return image

    with monkeypatch.context() as patch:
        patch.setattr(Image, "open", replace_after_decode)
        result = media.generate_synthetic_image(32, 24, cache_dir=tmp_path)

    with Image.open(io.BytesIO(base64.b64decode(result["base64"]))) as image:
        np.testing.assert_array_equal(result["np_array"], np.array(image))
