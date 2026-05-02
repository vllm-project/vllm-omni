# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit_plus import (
    get_qwen_image_edit_plus_pre_process_func,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_qwen_image_edit_plus_rejects_too_many_input_images(tmp_path: Path):
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    # Keep the mock config intentionally minimal: this test only needs the
    # fields touched during pre-process initialization.
    (vae_dir / "config.json").write_text(json.dumps({"z_dim": 16}))

    pre_process = get_qwen_image_edit_plus_pre_process_func(SimpleNamespace(model=str(tmp_path)))
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    request = SimpleNamespace(
        prompts=[
            {
                "prompt": "combine",
                "multi_modal_data": {"image": [image, image, image, image, image]},
            }
        ],
        sampling_params=SimpleNamespace(height=None, width=None),
    )

    with pytest.raises(ValueError, match=r"At most 4 images are supported by this model"):
        pre_process(request)
