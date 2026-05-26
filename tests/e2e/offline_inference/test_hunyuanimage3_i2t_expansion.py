# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke test for HunyuanImage-3.0 Image-to-Text (I2T) pipeline."""

from collections.abc import Generator
from pathlib import Path

import pytest
import torch
from PIL import Image

from tests.helpers.runtime import OmniRunner
from vllm_omni import Omni
from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import build_prompt

MODEL_NAME = "tencent/HunyuanImage-3.0-Instruct"
REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE_CONFIG_PATH = REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "hunyuan_image3_i2t.yaml"

# First 20 generated token IDs from the HF greedy reference on this input.
# vllm-omni AR output matches this prefix bitwise; the two implementations
# diverge past this point.
EXPECTED_PREFIX_TOKEN_IDS: list[int] = [
    791,
    2217,
    374,
    264,
    6573,
    11,
    14113,
    6307,
    1933,
    449,
    912,
    27339,
    11,
    6302,
    11,
    477,
    3649,
    3118,
    13,
    1102,
]
# Decoded form, kept only for human-readable assertion messages.
EXPECTED_PREFIX_TEXT = "The image is a solid, uniform green color with no variations, objects, or details present. It"

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]


@pytest.fixture(scope="module")
def omni() -> Generator[Omni, None, None]:
    with OmniRunner(
        MODEL_NAME,
        stage_configs_path=str(STAGE_CONFIG_PATH),
    ) as runner:
        yield runner.omni


@pytest.mark.skipif(torch.accelerator.device_count() < 4, reason="Need at least 4 CUDA GPUs.")
def test_i2t_generates_text(omni: Omni) -> None:
    """Verify I2T output's first 20 token IDs match the HF greedy baseline."""
    # Solid-color image keeps the input self-contained and reproducible.
    input_image = Image.new("RGB", (256, 256), color=(128, 200, 100))

    prompt = build_prompt("Describe the content of the picture.", task="i2t")
    prompt_dict = {
        "prompt": prompt,
        "modalities": ["text"],
        "multi_modal_data": {"image": input_image},
    }

    outputs = omni.generate(prompts=[prompt_dict])
    assert outputs, "No outputs returned from Omni.generate()"

    request_output = outputs[0].request_output
    assert request_output.outputs, "No completion outputs"

    completion = request_output.outputs[0]
    finish_reason = getattr(completion, "finish_reason", None)
    assert finish_reason is not None, "AR generation did not finish (finish_reason is None)"
    assert str(finish_reason) != "abort", f"AR generation aborted: finish_reason={finish_reason!r}"

    token_ids = list(getattr(completion, "token_ids", []) or [])
    n = len(EXPECTED_PREFIX_TOKEN_IDS)
    assert len(token_ids) >= n, (
        f"AR output shorter than {n} tokens (got {len(token_ids)}): token_ids={token_ids!r} text={completion.text!r}"
    )
    assert token_ids[:n] == EXPECTED_PREFIX_TOKEN_IDS, (
        f"AR prefix drift vs HF reference\n"
        f"  expected ids : {EXPECTED_PREFIX_TOKEN_IDS!r}\n"
        f"  actual ids   : {token_ids[:n]!r}\n"
        f"  expected text: {EXPECTED_PREFIX_TEXT!r}\n"
        f"  actual text  : {completion.text!r}"
    )
