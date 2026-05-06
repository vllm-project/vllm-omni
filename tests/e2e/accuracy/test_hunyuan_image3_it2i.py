# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IT2I accuracy regression for HunyuanImage-3.0-Instruct (PR #3107).

The diffusion image produced by the vllm-omni IT2I pipeline must match the
official HunyuanImage-3.0-Instruct HF reference (loaded with
``trust_remote_code=True`` and driven through its bundled ``generate_image``
helper) at PSNR >= 40 dB on the same ``(condition_image, prompt, seed)``
tuple. We follow the same harness shape as
``tests/e2e/accuracy/test_qwen_image_edit.py``:

* vllm-omni side: drive the offline IT2I path (``examples/offline_inference/
  hunyuan_image3/end2end.py`` style invocation through ``Omni``) so we
  exercise the exact two-stage AR -> DiT pipeline this PR ships.
* Reference side: load the same checkpoint through HF transformers
  ``AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`` and
  call ``model.generate_image(...)`` -- the canonical entry point declared
  in the model's ``hunyuan_image_3_pipeline.py``.
* Compare via the shared :func:`assert_similarity` helper, which runs
  torchmetrics PSNR/SSIM and fails loudly with the actual numbers.

PSNR >= 40 dB is borrowed from the threshold suggested in PR #2949's review
discussion -- it gives enough slack for the small numeric drift introduced
by TP=2 NCCL all-reduce vs single-GPU diffusers while still catching
behavioural regressions (wrong system prompt, wrong condition image
preprocessing, missing `<think>` trigger, etc.).

The test is gated on ``--full-model`` and on having >=8 CUDA devices: the
HunyuanImage-3.0-Instruct checkpoint cannot be split across fewer than 4
GPUs at TP=2 for both stages without OOM, and the reference path needs
its own slice for ``device_map="auto"``.
"""

from __future__ import annotations

import gc
from pathlib import Path

import pytest
import torch
from PIL import Image

from tests.e2e.accuracy.helpers import assert_similarity, model_output_dir
from tests.helpers.env import run_post_test_cleanup, run_pre_test_cleanup
from tests.helpers.runtime import OmniRunner

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]


MODEL_NAME = "tencent/HunyuanImage-3.0-Instruct"
SEED = 42
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 5.0
HEIGHT = 1024
WIDTH = 1024

PSNR_THRESHOLD = 40.0
# Loose SSIM threshold: PSNR is the primary gate -- SSIM here just guards
# against gross structural divergence (e.g. a wrong condition image was
# passed through and the reference and ours look like different scenes).
SSIM_THRESHOLD = 0.92

PROMPT = "Add a cute orange cat sitting in the foreground."

REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE_CONFIG_PATH = REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "hunyuan_image3_it2i.yaml"


@pytest.fixture(scope="module")
def condition_image() -> Image.Image:
    """Synthetic 1024x1024 RGB scene -- coloured rectangles on a soft blue
    background. Deterministic per-seed so vllm-omni and the HF reference
    see byte-identical input pixels."""
    import random

    from PIL import ImageDraw

    random.seed(SEED)
    image = Image.new("RGB", (WIDTH, HEIGHT), (180, 200, 220))
    draw = ImageDraw.Draw(image)
    for _ in range(20):
        x = random.randint(0, 900)
        y = random.randint(0, 900)
        w = random.randint(40, 200)
        h = random.randint(40, 200)
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        draw.rectangle([x, y, x + w, y + h], fill=color, outline=(0, 0, 0), width=4)
    return image


def _run_vllm_omni_it2i(
    *,
    condition_image: Image.Image,
    output_path: Path,
) -> Image.Image:
    from transformers import AutoTokenizer

    from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
        _TASK_PRESETS,
        build_prompt_tokens,
    )
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType
    from vllm_omni.platforms import current_omni_platform

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    task = "it2i_think"
    token_ids = build_prompt_tokens(PROMPT, tok, task=task)
    preset_sys_type, _, _ = _TASK_PRESETS[task]

    prompt_dict: dict = {
        "prompt_token_ids": token_ids,
        "prompt": PROMPT,
        "use_system_prompt": preset_sys_type,
        "modalities": ["image"],
        "multi_modal_data": {"image": condition_image},
        "height": condition_image.height,
        "width": condition_image.width,
    }

    with OmniRunner(
        MODEL_NAME,
        stage_configs_path=str(STAGE_CONFIG_PATH),
        mode="text-to-image",
    ) as runner:
        omni = runner.omni
        params_list = list(omni.default_sampling_params_list)

        for sp in params_list:
            if isinstance(sp, OmniDiffusionSamplingParams):
                sp.num_inference_steps = NUM_INFERENCE_STEPS
                sp.guidance_scale = GUIDANCE_SCALE
                sp.seed = SEED
                generator_device = current_omni_platform.device_type or "cuda"
                sp.generator = torch.Generator(device=generator_device).manual_seed(SEED)

        prompts: list[OmniPromptType] = [prompt_dict]
        outputs = list(omni.generate(prompts=prompts, sampling_params_list=params_list))

    if not outputs:
        raise AssertionError("Omni IT2I produced no outputs")
    first = outputs[0]
    images = getattr(first, "images", None)
    if not images:
        ro = getattr(first, "request_output", None)
        images = getattr(ro, "images", None) if ro is not None else None
    if not images:
        raise AssertionError("Omni IT2I output had no `images` field")

    image = images[0].convert("RGB")
    image.save(output_path)
    return image


def _run_hf_reference_it2i(
    *,
    condition_image: Image.Image,
    output_path: Path,
) -> Image.Image:
    """Run the official HF custom-code pipeline as the reference.

    HunyuanImage-3.0-Instruct's repo ships an ``auto_map`` that wires
    ``AutoModelForCausalLM`` to its custom ``HunyuanImage3ForConditionalGeneration``
    class. The class exposes ``generate_image`` as the canonical entry
    point for both T2I and IT2I -- IT2I is signalled by passing
    ``image=`` (the condition image).
    """
    run_pre_test_cleanup(enable_force=True)
    model = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
        if hasattr(model, "load_tokenizer"):
            model.load_tokenizer(tokenizer)

        torch.manual_seed(SEED)
        result = model.generate_image(  # pyright: ignore[reportAttributeAccessIssue]
            prompt=PROMPT,
            image=condition_image,
            seed=SEED,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            image_size=f"{HEIGHT}x{WIDTH}",
            bot_task="think",
        )

        # ``generate_image`` returns either a PIL image or a list/dict
        # depending on revision; normalise.
        if isinstance(result, Image.Image):
            image = result
        elif isinstance(result, (list, tuple)) and result and isinstance(result[0], Image.Image):
            image = result[0]
        elif isinstance(result, dict) and "images" in result:
            image = result["images"][0]
        else:
            raise AssertionError(f"Unexpected return type from HunyuanImage3 generate_image: {type(result)}")

        image = image.convert("RGB")
        image.save(output_path)
        return image
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        run_post_test_cleanup(enable_force=True)


@pytest.mark.skipif(
    torch.cuda.device_count() < 8,
    reason="HunyuanImage-3.0-Instruct IT2I needs 4 GPUs for AR + 4 for DiT (vllm-omni) "
    "and shares the box with the HF reference -- requires 8+ CUDA devices.",
)
def test_hunyuan_image3_it2i_matches_hf_reference_psnr_40(
    accuracy_artifact_root: Path,
    condition_image: Image.Image,
) -> None:
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_NAME)

    omni_path = output_dir / "vllm_omni_it2i.png"
    omni_image = _run_vllm_omni_it2i(
        condition_image=condition_image,
        output_path=omni_path,
    )

    ref_path = output_dir / "hf_reference_it2i.png"
    reference_image = _run_hf_reference_it2i(
        condition_image=condition_image,
        output_path=ref_path,
    )

    assert_similarity(
        model_name=f"{MODEL_NAME} IT2I",
        vllm_image=omni_image,
        diffusers_image=reference_image,
        ssim_threshold=SSIM_THRESHOLD,
        psnr_threshold=PSNR_THRESHOLD,
        width=WIDTH,
        height=HEIGHT,
    )
