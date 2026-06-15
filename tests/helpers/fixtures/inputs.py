from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
import requests
from PIL import Image

QWEN_BEAR_IMAGE_URL = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/omni-assets/qwen-bear.png"


def load_qwen_bear_image(cache_dir: Path) -> Image.Image:
    """Load the shared Qwen bear image, downloading it into cache_dir if needed."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_path = cache_dir / "qwen_bear.png"
    if image_path.exists():
        with Image.open(image_path) as image:
            return image.convert("RGB")

    response = requests.get(QWEN_BEAR_IMAGE_URL, timeout=60)
    response.raise_for_status()
    with Image.open(BytesIO(response.content)) as image:
        rgb_image = image.convert("RGB")
    rgb_image.save(image_path)
    return rgb_image


@pytest.fixture(scope="session")
def shared_artifact_root() -> Path:
    return Path(__file__).resolve().parents[2] / "artifacts"


@pytest.fixture(scope="session")
def qwen_bear_image(shared_artifact_root: Path) -> Image.Image:
    """Shared Qwen bear image for image-edit tests."""
    image = load_qwen_bear_image(shared_artifact_root)
    yield image
    image.close()
