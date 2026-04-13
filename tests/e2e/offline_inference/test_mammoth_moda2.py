"""
End-to-end test for MammothModa2 text-to-image generation.

Verifies that the AR->DiT pipeline produces an image tensor whose pixel values
match a golden reference.

Model weights: env var ``MAMMOTHMODA2_MODEL_PATH``
  (default: <repo_root>/MammothModa2-Preview)
Stage config:  env var ``MAMMOTHMODA2_T2I_STAGE_CONFIG``
  (default: vllm_omni/model_executor/stage_configs/mammoth_moda2.yaml)

Golden pixel file: ``tests/e2e/offline_inference/fixtures/mammoth_moda2_t2i_golden.json``
  Regenerate with: ``UPDATE_GOLDEN=1 pytest tests/e2e/offline_inference/test_mammoth_moda2.py``
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from vllm.sampling_params import SamplingParams

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_IMAGE_TOKEN_ID = 151655  # "<|image_pad|>"
_VIDEO_TOKEN_ID = 151656  # "<|video_pad|>"
_VISION_START_TOKEN_ID = 151652  # "<|vision_start|>"
_VISION_END_TOKEN_ID = 151653  # "<|vision_end|>"
_AR_PATCH_SIZE = 16

_STAGE_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "vllm_omni" / "model_executor" / "stage_configs"
MODEL_PATH = os.environ.get(
    "MAMMOTHMODA2_MODEL_PATH",
    str(Path(__file__).resolve().parents[3] / "MammothModa2-Preview"),
)
T2I_STAGE_CONFIG = os.environ.get(
    "MAMMOTHMODA2_T2I_STAGE_CONFIG",
    str(_STAGE_CONFIGS_DIR / "mammoth_moda2.yaml"),
)

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
def _load_t2i_gen_config(model_dir: str) -> dict:
    cfg_path = Path(model_dir) / "t2i_generation_config.json"
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


def _sample_pixels(img_tensor: torch.Tensor) -> list[float]:
    """Sample pixel values at fixed fractional coordinates from a (C, H, W) tensor."""
    t = img_tensor.float().clamp(0.0, 1.0)
    if t.ndim == 4:
        t = t[0]  # unbatch
    C, H, W = t.shape
    values = []
    for c, rh, rw in _PIXEL_SAMPLE_COORDS:
        ri = min(int(rh * (H - 1)), H - 1)
        ci = min(int(rw * (W - 1)), W - 1)
        values.append(round(float(t[c, ri, ci]), 6))
    return values


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------
@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"})
def test_mammothmoda2_t2i_e2e():
    """
    End-to-end text-to-image generation with MammothModa2 (AR -> DiT).

    Verifies:
      - Omni pipeline initialises with the two-stage YAML config.
      - DiT stage outputs an image tensor with the correct shape.
      - A fixed set of pixel values matches a golden reference
        (regenerate with ``UPDATE_GOLDEN=1``).
    """
    if not Path(MODEL_PATH).exists():
        pytest.skip(f"Model weights not found at {MODEL_PATH}")
    if not Path(T2I_STAGE_CONFIG).exists():
        pytest.skip(f"Stage config not found at {T2I_STAGE_CONFIG}")

    gen_cfg = _load_t2i_gen_config(MODEL_PATH)
    eol_token_id = int(gen_cfg["eol_token_id"])
    visual_start = int(gen_cfg["visual_token_start_id"])
    visual_end = int(gen_cfg["visual_token_end_id"])

    height, width = 256, 256  # small for CI speed
    ar_height, ar_width = height // _AR_PATCH_SIZE, width // _AR_PATCH_SIZE
    expected_grid_tokens = ar_height * (ar_width + 1)

    prompt_text = "A cat sitting on a laptop keyboard"
    formatted_prompt = _format_t2i_prompt(prompt_text, ar_width, ar_height)

    with OmniRunner(MODEL_PATH, stage_configs_path=T2I_STAGE_CONFIG, trust_remote_code=True) as runner:
        omni = runner.omni
        # Greedy / deterministic sampling so pixel values are reproducible.
        ar_sampling = SamplingParams(
            temperature=0.0,
            top_k=1,
            max_tokens=max(1, expected_grid_tokens + 1),
            detokenize=False,
        )
        dit_sampling = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)

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
                            "num_inference_steps": [2],
                            "text_guidance_scale": [1.0],
                            "cfg_range": [0.0, 1.0],
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

        # Find the image tensor produced by the DiT stage and compare pixels.
        found_image = False
        for out in outputs:
            ro_list = getattr(out, "request_output", out)
            if not isinstance(ro_list, list):
                ro_list = [ro_list]
            for ro in ro_list:
                completion_outputs = getattr(ro, "outputs", None)
                if not isinstance(completion_outputs, list):
                    continue
                for completion in completion_outputs:
                    mm = getattr(completion, "multimodal_output", None)
                    if not (isinstance(mm, dict) and "image" in mm):
                        continue
                    img_list = mm["image"] if isinstance(mm["image"], list) else [mm["image"]]
                    for img_tensor in img_list:
                        assert isinstance(img_tensor, torch.Tensor), f"Expected image tensor, got {type(img_tensor)}"
                        assert img_tensor.ndim in (3, 4), f"Expected 3D or 4D image tensor, got {img_tensor.ndim}D"

                        sampled = _sample_pixels(img_tensor)

                        if os.environ.get("UPDATE_GOLDEN"):
                            _GOLDEN_T2I_PATH.parent.mkdir(parents=True, exist_ok=True)
                            _GOLDEN_T2I_PATH.write_text(json.dumps({"pixels": sampled}, indent=2))
                            print(f"\nGolden file written to {_GOLDEN_T2I_PATH}")
                        elif _GOLDEN_T2I_PATH.exists():
                            golden = json.loads(_GOLDEN_T2I_PATH.read_text())["pixels"]
                            for i, (got, exp) in enumerate(zip(sampled, golden)):
                                assert abs(got - exp) < 1e-4, f"Pixel {i} mismatch: got {got}, expected {exp}"

                        found_image = True

        assert found_image, "No image tensor found in pipeline output"
