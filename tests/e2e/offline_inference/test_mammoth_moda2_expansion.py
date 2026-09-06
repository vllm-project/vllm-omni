# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
End-to-end test for MammothModa2 text-to-image generation.

Verifies that the AR->DiT pipeline produces a postprocessed PIL image. Pixel
values are compared with a golden reference when one is explicitly supplied.

Model Hub repo id: ``bytedance-research/MammothModa2-Preview``.
Deploy config: ``get_deploy_config_path("mammoth_moda2.yaml")`` -> ``vllm_omni/deploy/mammoth_moda2.yaml``

Golden pixel file: ``tests/e2e/offline_inference/fixtures/mammoth_moda2_t2i_golden.json``
  Regenerate with: ``UPDATE_GOLDEN=1 pytest tests/e2e/offline_inference/test_mammoth_moda2_expansion.py``
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from vllm.sampling_params import SamplingParams

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = pytest.mark.advanced_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_IMAGE_TOKEN_ID = 151655  # "<|image_pad|>"
_VIDEO_TOKEN_ID = 151656  # "<|video_pad|>"
_VISION_START_TOKEN_ID = 151652  # "<|vision_start|>"
_VISION_END_TOKEN_ID = 151653  # "<|vision_end|>"
_AR_PATCH_SIZE = 16

MODEL_PATH = "bytedance-research/MammothModa2-Preview"
T2I_DEPLOY_CONFIG = get_deploy_config_path("mammoth_moda2.yaml")

_OMNI_RUNNER_PARAM = (MODEL_PATH, T2I_DEPLOY_CONFIG)

# Golden pixel reference file.  Set UPDATE_GOLDEN=1 to regenerate.
_GOLDEN_T2I_PATH = Path(__file__).parent / "fixtures" / "mammoth_moda2_t2i_golden.json"
# Fixed sampling coordinates: (channel, row_fraction, col_fraction)
# Covers corners, centre, and mid-edges across all 3 channels.
_PIXEL_SAMPLE_COORDS = [
    (0, 0.0, 0.0),
    (0, 0.5, 0.5),
    (0, 1.0, 1.0),
    (0, 0.25, 0.75),
    (1, 0.0, 1.0),
    (1, 0.5, 0.0),
    (1, 0.75, 0.25),
    (1, 1.0, 0.5),
    (2, 0.0, 0.5),
    (2, 0.5, 1.0),
    (2, 0.75, 0.75),
    (2, 1.0, 0.0),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_t2i_gen_config(repo_id: str) -> dict:
    weights_dir = Path(snapshot_download(repo_id))
    cfg_path = weights_dir / "t2i_generation_config.json"
    if not cfg_path.exists():
        pytest.skip(f"t2i_generation_config.json not found at {cfg_path}")
    with cfg_path.open() as f:
        return json.load(f)


def _format_t2i_prompt(user_prompt: str, ar_width: int, ar_height: int) -> str:
    return (
        "<|im_start|>system\nYou are a helpful image generator.<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"<|image start|>{ar_width}*{ar_height}<|image token|>"
    )


def _sample_pixels(image: Image.Image) -> list[float]:
    """Sample normalized RGB values after the shared VAE postprocess boundary."""
    width, height = image.size
    values = []
    for c, rh, rw in _PIXEL_SAMPLE_COORDS:
        ri = min(int(rh * (height - 1)), height - 1)
        ci = min(int(rw * (width - 1)), width - 1)
        values.append(round(image.getpixel((ci, ri))[c] / 255.0, 6))
    return values


def _iter_image_tensors(outputs: list[object]):
    """Yield images from shared diffusion ``OmniRequestOutput`` objects."""
    for out in outputs:
        ro_list = out if isinstance(out, list) else [out]
        for ro in ro_list:
            images = getattr(ro, "images", None)
            if isinstance(images, list):
                yield from images


@pytest.mark.cpu
def test_diffusion_output_exposes_images_at_top_level():
    image = torch.zeros((3, 16, 16))
    output = OmniRequestOutput.from_diffusion(request_id="diffusion-test", images=[image])

    assert output.outputs == []
    assert list(_iter_image_tensors([output])) == [image]


@pytest.mark.cpu
def test_golden_sampling_uses_postprocessed_rgb_values():
    image = Image.new("RGB", (16, 16), (0, 127, 255))
    assert _sample_pixels(image) == [0.0] * 4 + [round(127 / 255, 6)] * 4 + [1.0] * 4


@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True)
@hardware_test(res={"cuda": "H100"})
def test_mammothmoda2_t2i_e2e(omni_runner: OmniRunner):
    """
    End-to-end text-to-image generation with MammothModa2 (AR -> DiT).

    Verifies:
      - Omni pipeline initialises with the two-stage YAML config.
      - Shared postprocessing returns a PIL RGB image with the correct size.
      - A fixed set of pixel values matches a golden reference
        (regenerate with ``UPDATE_GOLDEN=1``).
    """
    gen_cfg = _load_t2i_gen_config(MODEL_PATH)
    eol_token_id = int(gen_cfg["eol_token_id"])
    visual_start = int(gen_cfg["visual_token_start_id"])
    visual_end = int(gen_cfg["visual_token_end_id"])

    height, width = 256, 256  # small for CI speed
    ar_height, ar_width = height // _AR_PATCH_SIZE, width // _AR_PATCH_SIZE
    expected_grid_tokens = ar_height * (ar_width + 1)

    prompt_text = "A cat sitting on a laptop keyboard"
    formatted_prompt = _format_t2i_prompt(prompt_text, ar_width, ar_height)

    omni = omni_runner.omni
    ar_sampling = SamplingParams(
        temperature=0.0,
        top_k=1,
        max_tokens=max(1, expected_grid_tokens + 1),
        detokenize=False,
    )
    dit_sampling = OmniDiffusionSamplingParams(
        height=height,
        width=width,
        seed=42,
        guidance_scale=1.0,
        num_inference_steps=2,
        extra_args={"cfg_range": [0.0, 1.0]},
    )

    outputs = list(
        omni.generate(
            [
                {
                    "prompt": formatted_prompt,
                    "additional_information": {
                        "omni_task": ["t2i"],
                        "ar_width": [ar_width],
                        "ar_height": [ar_height],
                        "eol_token_id": [eol_token_id],
                        "visual_token_start_id": [visual_start],
                        "visual_token_end_id": [visual_end],
                        "image_height": [height],
                        "image_width": [width],
                        "visual_ids": [
                            _IMAGE_TOKEN_ID,
                            _VIDEO_TOKEN_ID,
                            _VISION_START_TOKEN_ID,
                            _VISION_END_TOKEN_ID,
                        ],
                    },
                }
            ],
            [ar_sampling, dit_sampling],
        )
    )

    assert len(outputs) > 0, "Pipeline produced no outputs"

    found_image = False
    for image in _iter_image_tensors(outputs):
        assert isinstance(image, Image.Image), f"Expected postprocessed PIL image, got {type(image)}"
        assert image.mode == "RGB" and image.size == (width, height)

        sampled = _sample_pixels(image)

        if os.environ.get("UPDATE_GOLDEN"):
            _GOLDEN_T2I_PATH.parent.mkdir(parents=True, exist_ok=True)
            _GOLDEN_T2I_PATH.write_text(json.dumps({"pixels": sampled}, indent=2))
            print(f"\nGolden file written to {_GOLDEN_T2I_PATH}")
        elif _GOLDEN_T2I_PATH.exists():
            golden = json.loads(_GOLDEN_T2I_PATH.read_text())["pixels"]
            for i, (got, exp) in enumerate(zip(sampled, golden)):
                assert abs(got - exp) < 1e-4, f"Pixel {i} mismatch: got {got}, expected {exp}"

        found_image = True

    assert found_image, "No postprocessed image found in pipeline output"
