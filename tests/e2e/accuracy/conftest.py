from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
import requests
from PIL import Image


def pytest_addoption(parser):
    group = parser.getgroup("accuracy-e2e")
    group.addoption(
        "--wan22-i2v-image-source",
        action="store",
        default=None,
        help="Image source for Wan2.2 I2V accuracy tests. Can be local path or remote URL",
    )
    group.addoption(
        "--wan22-i2v-online-timeout-seconds",
        action="store",
        type=int,
        default=1200,
        help="Online serving timeout in seconds for Wan2.2 I2V accuracy tests.",
    )
    group.addoption(
        "--hunyuanvideo15-i2v-image-source",
        action="store",
        default=None,
        help="Image source for HunyuanVideo-1.5 I2V accuracy tests. Can be local path or remote URL",
    )
    group.addoption(
        "--hunyuanvideo15-online-timeout-seconds",
        action="store",
        type=int,
        default=3600,
        help="Online serving timeout in seconds for HunyuanVideo-1.5 accuracy tests.",
    )


@pytest.fixture(scope="session")
def wan22_i2v_image_source(request: pytest.FixtureRequest) -> str | None:
    value = request.config.getoption("wan22_i2v_image_source")
    return str(value) if value else None


@pytest.fixture(scope="session")
def wan22_i2v_online_timeout_seconds(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("wan22_i2v_online_timeout_seconds"))


@pytest.fixture(scope="session")
def hunyuanvideo15_i2v_image_source(request: pytest.FixtureRequest) -> str | None:
    value = request.config.getoption("hunyuanvideo15_i2v_image_source")
    return str(value) if value else None


@pytest.fixture(scope="session")
def hunyuanvideo15_online_timeout_seconds(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("hunyuanvideo15_online_timeout_seconds"))


@pytest.fixture(scope="session")
def accuracy_artifact_root() -> Path:
    root = Path(__file__).resolve().parent / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(scope="session")
def qwen_bear_image(accuracy_artifact_root: Path):
    """Download the Qwen bear image from the URL and save it to the accuracy artifact root."""
    QWEN_BEAR_IMAGE_URL = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/omni-assets/qwen-bear.png"
    image_path = accuracy_artifact_root / "qwen_bear.png"
    if image_path.exists():
        image = Image.open(image_path).convert("RGB")
        yield image
        image.close()
        return
    response = requests.get(QWEN_BEAR_IMAGE_URL, timeout=60)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image.save(image_path)
    yield image
    image.close()


@pytest.fixture(scope="session")
def rabbit_image(accuracy_artifact_root: Path):
    """Download the rabbit image from the URL and save it to the accuracy artifact root."""
    RABBIT_IMAGE_URL = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/omni-assets/rabbit.png"
    image_path = accuracy_artifact_root / "rabbit.png"
    if image_path.exists():
        image = Image.open(image_path).convert("RGB")
        yield image
        image.close()
        return
    response = requests.get(RABBIT_IMAGE_URL, timeout=60)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image.save(image_path)
    yield image
    image.close()
