# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke test for HunyuanImage-3.0 Image-to-Text (I2T) pipeline."""

from collections.abc import Generator

import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer

from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni import Omni
from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import build_prompt_tokens, resolve_stop_token_ids

MODEL_NAME = "tencent/HunyuanImage-3.0-Instruct"
AR_DEPLOY_CONFIG_PATH = get_deploy_config_path("hunyuan_image3_ar.yaml")

# First 20 generated token IDs from the HF greedy reference on this input.
# Feed pre-tokenized prompt IDs so vLLM-Omni uses the same segmented chat
# template tokenization as HF apply_chat_template; whole-string tokenization
# can merge BPE tokens across template boundaries and drift at token 8.
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
        deploy_config=AR_DEPLOY_CONFIG_PATH,
    ) as runner:
        yield runner.omni


@pytest.mark.skipif(torch.accelerator.device_count() < 4, reason="Need at least 4 CUDA GPUs.")
def test_i2t_generates_text(omni: Omni) -> None:
    """Verify I2T output's first 20 token IDs match the HF greedy baseline."""
    # Solid-color image keeps the input self-contained and reproducible.
    input_image = Image.new("RGB", (256, 256), color=(128, 200, 100))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    prompt_tokens = build_prompt_tokens("Describe the content of the picture.", tokenizer, task="i2t")
    stop_token_ids = resolve_stop_token_ids(task="i2t", bot_task=None, tokenizer=tokenizer)
    prompt_dict = {
        "prompt_token_ids": prompt_tokens.token_ids,
        "modalities": ["text"],
        "multi_modal_data": {"image": input_image},
    }

    sampling_params_list = list(omni.default_sampling_params_list)
    assert len(sampling_params_list) == 1, f"Expected one I2T stage, got {len(sampling_params_list)}"
    sampling_params_list[0].stop_token_ids = stop_token_ids

    outputs = omni.generate(prompts=[prompt_dict], sampling_params_list=sampling_params_list)
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
