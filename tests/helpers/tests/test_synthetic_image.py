# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import base64
from pathlib import Path

import pytest
from PIL import Image

from tests.helpers import media

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_image_cache_is_published_atomically(mocker, tmp_path):
    cache_path = tmp_path / "synth_image_w64_h32.jpg"
    replace = media.os.replace

    def observe_replace(source, destination):
        assert Path(destination) == cache_path
        assert Path(source).parent == cache_path.parent
        assert not cache_path.exists()
        with Image.open(source) as image:
            image.load()
            assert image.size == (64, 32)
        replace(source, destination)

    publish = mocker.patch.object(media.os, "replace", side_effect=observe_replace)
    result = media.generate_synthetic_image(64, 32, cache_dir=tmp_path, seed=43)

    publish.assert_called_once()
    assert base64.b64decode(result["base64"]) == cache_path.read_bytes()
    assert list(tmp_path.iterdir()) == [cache_path]


def test_failed_image_cache_publication_preserves_old_file(mocker, tmp_path):
    original = media.generate_synthetic_image(64, 32, cache_dir=tmp_path, seed=42)
    cache_path = Path(original["file_path"])
    old_bytes = cache_path.read_bytes()
    mocker.patch.object(media.os, "replace", side_effect=OSError("publication failed"))

    with pytest.raises(OSError, match="publication failed"):
        media.generate_synthetic_image(64, 32, force_regenerate=True, cache_dir=tmp_path, seed=43)

    assert cache_path.read_bytes() == old_bytes
    assert list(tmp_path.iterdir()) == [cache_path]
