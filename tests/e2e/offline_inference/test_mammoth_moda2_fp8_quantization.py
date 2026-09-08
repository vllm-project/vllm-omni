# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU end-to-end FP8 A/B gates for the MammothModa2 AR stage.

These tests load the real ``MammothModa2-Dev`` checkpoint and compare BF16 vs
FP8 generation, scoped to the AR stage:

* ``test_ar_generation_smoke`` — each format loads and produces tokens.
* ``test_bf16_vs_fp8_generation_consistency`` — AR-only understanding: the
  base head / understanding expert under FP8 (token agreement + logprob
  similarity + MAE).
* ``test_bf16_vs_fp8_t2i_image_consistency`` — t2i (AR→DiT): drives the AR
  stage to emit visual tokens, which exercises the generation experts
  (``gen_mlp``) and the extra vocabulary/head (``gen_embed_tokens`` /
  ``gen_head``) under FP8, and compares the decoded image.

The CPU-only structural unit tests live in
``tests/model_executor/models/mammoth_moda2/test_mammoth_moda2_quantization.py``.
"""

from __future__ import annotations

import os

import pytest
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Local Dev checkpoint; override via env for CI / hub ids.
MODEL_PATH = os.environ.get("MAMMOTH_MODA2_MODEL", "/root/autodl-fs/MammothModa2-Dev")

# Quantization cases for the A/B gate. ``None`` is the BF16 baseline (the
# ``quantization`` stage key is removed). FP8 is vLLM's runtime W8A8
# (weight + activation) quantization, computed directly from the BF16
# checkpoint — no pre-quantized weights required.
QUANTIZATION_CASES = [None, "fp8"]

# Hardware gate: any CUDA GPU (H100/B200 datacenter or RTX PRO 6000
# workstation). ``@hardware_test`` pins specific SKUs, so use a plain skip
# so local workstation cards can run the A/B gate too.
_CUDA_ONLY = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")

# Minimum token-level agreement between BF16 and a quantized run, required for
# the A/B test to pass. TODO: tune after measuring real greedy-decoding drift.
MIN_TOKEN_AGREEMENT = 0.9

# Minimum cosine similarity between BF16 and quantized top-1 logprob sequences.
# FP8 W8A8 typically lands ~0.94-0.99 on greedy logprobs (measured 0.9447 on
# RTX PRO 6000); keep a margin below that to catch real breakage, not normal
# quantization drift.
MIN_LOGPROB_COSINE = 0.90

_PROMPT = "Explain multimodal generation in three sentences."

pytestmark = [pytest.mark.slow]


# ---------------------------------------------------------------------------
# AR-only understanding A/B
# ---------------------------------------------------------------------------

def _stage_config(quantization: str | None) -> str:
    """Return a patched AR deploy config for the requested quantization.

    ``None`` deletes the ``quantization`` key (BF16 baseline); otherwise the
    key is set to ``quantization``.
    """
    from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

    base = get_deploy_config_path("mammoth_moda2_ar.yaml")
    if quantization is None:
        return modify_stage_config(base, deletes={"stages": {0: ["quantization"]}})
    return modify_stage_config(base, updates={"stages": {0: {"quantization": quantization}}})


def _generate(model: str, quantization: str | None) -> tuple[list[int], list[float]]:
    """Run the AR-only pipeline greedily, returning (token_ids, top1_logprobs)."""
    from vllm.sampling_params import SamplingParams

    from tests.helpers.runtime import OmniRunner
    from vllm_omni.model_extras import build_x_to_text_prompt, get_x_to_text_model_family

    family = get_x_to_text_model_family(model)
    prompt_dict, _stop_ids = build_x_to_text_prompt(
        model_family=family,
        model=model,
        prompt=_PROMPT,
        has_image=False,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=32, detokenize=False, logprobs=1)

    with OmniRunner(model, seed=42, deploy_config=_stage_config(quantization)) as runner:
        outputs = list(runner.omni.generate([prompt_dict], [sampling_params]))

    for out in outputs:
        for completion in getattr(out, "outputs", None) or []:
            token_ids = list(getattr(completion, "token_ids", None) or [])
            return token_ids, _top1_logprobs(completion)
    return [], []


def _top1_logprobs(completion) -> list[float]:
    """Extract the top-1 logprob at each generated step (empty if unavailable)."""
    logprobs = getattr(completion, "logprobs", None)
    if not logprobs:
        return []
    result: list[float] = []
    for step in logprobs:
        if not step:
            result.append(float("nan"))
            continue
        best = max(step.values(), key=lambda lp: lp.logprob)
        result.append(float(best.logprob))
    return result


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length float sequences."""
    a_t = torch.tensor(a, dtype=torch.float32)
    b_t = torch.tensor(b, dtype=torch.float32)
    return float(torch.nn.functional.cosine_similarity(a_t, b_t, dim=0))


def _mean_abs_diff(a: list[float], b: list[float]) -> float:
    """Mean absolute difference between two equal-length float sequences."""
    a_t = torch.tensor(a, dtype=torch.float32)
    b_t = torch.tensor(b, dtype=torch.float32)
    return float((a_t - b_t).abs().mean())


@_CUDA_ONLY
@pytest.mark.omni
@pytest.mark.parametrize("quantization", QUANTIZATION_CASES, ids=["bf16", "fp8"])
def test_ar_generation_smoke(quantization: str | None):
    """Each supported format loads and produces a non-empty greedy sequence."""
    token_ids, _ = _generate(MODEL_PATH, quantization)
    assert token_ids, f"no tokens generated for quantization={quantization!r}"


@_CUDA_ONLY
@pytest.mark.omni
def test_bf16_vs_fp8_generation_consistency():
    """A/B: BF16 vs FP8 — report token agreement + logprob similarity + MAE."""
    bf16_ids, bf16_lp = _generate(MODEL_PATH, None)
    fp8_ids, fp8_lp = _generate(MODEL_PATH, "fp8")

    assert bf16_ids, "BF16 baseline produced no tokens"
    assert fp8_ids, "FP8 run produced no tokens"

    common = min(len(bf16_ids), len(fp8_ids))
    token_agree = sum(a == b for a, b in zip(bf16_ids[:common], fp8_ids[:common])) / common

    lp_common = min(len(bf16_lp), len(fp8_lp))
    logprob_cos = _cosine_sim(bf16_lp[:lp_common], fp8_lp[:lp_common]) if lp_common > 0 else float("nan")
    logprob_mae = _mean_abs_diff(bf16_lp[:lp_common], fp8_lp[:lp_common]) if lp_common > 0 else float("nan")

    print(
        f"[FP8 A/B] token_agreement={token_agree:.4f} "
        f"logprob_cosine={logprob_cos:.4f} "
        f"logprob_mae={logprob_mae:.4f}"
    )

    assert token_agree >= MIN_TOKEN_AGREEMENT, (
        f"BF16/FP8 greedy sequences diverge too much: agreement={token_agree:.3f} < "
        f"{MIN_TOKEN_AGREEMENT}. bf16={bf16_ids} fp8={fp8_ids}"
    )
    if logprob_cos == logprob_cos:  # not NaN
        assert logprob_cos >= MIN_LOGPROB_COSINE, (
            f"BF16/FP8 logprob sequences diverge too much: cosine={logprob_cos:.4f} < "
            f"{MIN_LOGPROB_COSINE}"
        )


# ---------------------------------------------------------------------------
# t2i A/B — end-to-end trigger for generation experts + extra head
# ---------------------------------------------------------------------------
# The AR-only understanding test above never activates ``gen_mlp`` /
# ``gen_embed_tokens`` / ``gen_head``, because those only fire for image tokens
# (``input_ids >= gen_vocab_start_index == 152064``). The t2i task drives the AR
# stage to emit visual tokens, exercising the generation experts and the extra
# vocabulary/head under FP8 end to end.

_AR_PATCH_SIZE = 16

# Dev (Qwen3-VL) and Preview (Qwen2.5-VL) share the same vision token ids.
_IMAGE_TOKEN_ID = 151655
_VIDEO_TOKEN_ID = 151656
_VISION_START_TOKEN_ID = 151652
_VISION_END_TOKEN_ID = 151653


def _load_t2i_gen_config(model: str) -> dict:
    """Load ``t2i_generation_config.json`` from a local dir or hub id."""
    import json
    from pathlib import Path

    local = Path(model) / "t2i_generation_config.json"
    if local.exists():
        return json.loads(local.read_text(encoding="utf-8"))

    from huggingface_hub import snapshot_download

    weights_dir = Path(snapshot_download(model))
    cfg_path = weights_dir / "t2i_generation_config.json"
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _format_t2i_prompt(user_prompt: str, ar_width: int, ar_height: int) -> str:
    return (
        "<|im_start|>system\nYou are a helpful image generator.<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"<|image start|>{ar_width}*{ar_height}<|image token|>"
    )


def _t2i_stage_config(quantization: str | None) -> str:
    """Patch the AR→DiT deploy config: FP8 on the AR stage only (stage 0)."""
    from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

    base = get_deploy_config_path("mammoth_moda2.yaml")
    if quantization is None:
        return base
    return modify_stage_config(base, updates={"stages": {0: {"quantization": quantization}}})


def _generate_t2i_image(model: str, quantization: str | None) -> torch.Tensor:
    """Run t2i (AR→DiT) and return the decoded image tensor."""
    from vllm.sampling_params import SamplingParams

    from tests.helpers.runtime import OmniRunner

    gen_cfg = _load_t2i_gen_config(model)
    eol_token_id = int(gen_cfg["eol_token_id"])
    visual_start = int(gen_cfg["visual_token_start_id"])
    visual_end = int(gen_cfg["visual_token_end_id"])

    height, width = 256, 256  # small for CI speed
    ar_height, ar_width = height // _AR_PATCH_SIZE, width // _AR_PATCH_SIZE
    expected_grid_tokens = ar_height * (ar_width + 1)

    formatted_prompt = _format_t2i_prompt("A cat sitting on a laptop keyboard", ar_width, ar_height)

    ar_sampling = SamplingParams(
        temperature=0.0,
        top_k=1,
        max_tokens=max(1, expected_grid_tokens + 1),
        detokenize=False,
    )
    dit_sampling = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)

    with OmniRunner(model, seed=42, deploy_config=_t2i_stage_config(quantization)) as runner:
        outputs = list(
            runner.omni.generate(
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

    return _extract_image_tensor(outputs)


def _extract_image_tensor(outputs) -> torch.Tensor:
    """Extract the decoded image as a ``(C, H, W)`` float tensor in ``[0, 1]``.

    Reuses the official ``extract_images_from_outputs`` helper, which knows all
    the payload shapes (``OmniRequestOutput.images`` plus the ``"image"`` /
    ``"images"`` / ``"model_outputs"`` multimodal keys).
    """
    import numpy as np

    from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs

    images = extract_images_from_outputs(outputs)
    if not images:
        debug = [
            f"{type(out).__name__}(images={getattr(out, 'images', None)!r}, "
            f"mm={getattr(out, 'multimodal_output', None)!r})"
            for out in outputs
        ]
        raise AssertionError(f"no image tensor found in pipeline output; outputs={debug}")

    arr = np.asarray(images[0], dtype=np.float32) / 255.0  # (H, W, C)
    return torch.from_numpy(arr).permute(2, 0, 1)  # (C, H, W)


def _image_rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f, b_f = a.float(), b.float()
    return float((b_f - a_f).norm() / a_f.norm())


def _image_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0))


def _image_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """Full pixel-level comparison of two ``(C, H, W)`` images in ``[0, 1]``."""
    import math

    a_f, b_f = a.float(), b.float()
    diff = (b_f - a_f).abs()
    mse = float((b_f - a_f).pow(2).mean())

    return {
        "psnr_db": float(10 * math.log10(1.0 / mse)) if mse > 0 else float("inf"),
        "mae": float(diff.mean()),
        "max_abs": float(diff.max()),
        "cosine": _image_cosine(a_f, b_f),
        "rel_l2": _image_rel_l2(a_f, b_f),
        # Fraction of pixels whose difference is below one uint8 step.
        "pixel_match": float((diff < 1.0 / 255.0).float().mean()),
    }


@_CUDA_ONLY
@pytest.mark.diffusion
def test_bf16_vs_fp8_t2i_image_consistency():
    """A/B: BF16 vs FP8 t2i — decoded image must match (gen experts + head OK)."""
    bf16_img = _generate_t2i_image(MODEL_PATH, None)
    fp8_img = _generate_t2i_image(MODEL_PATH, "fp8")

    assert bf16_img.shape == fp8_img.shape, (bf16_img.shape, fp8_img.shape)

    metrics = _image_metrics(bf16_img, fp8_img)

    print("\n[FP8 t2i A/B] BF16 vs FP8 decoded image:")
    for name in ("psnr_db", "mae", "max_abs", "cosine", "rel_l2", "pixel_match"):
        print(f"  {name:12s} = {metrics[name]:.6f}")

    # Same seed + greedy AR + deterministic DiT: a correctly quantized
    # gen_mlp/gen_head must reproduce the same visual tokens, hence the image.
    assert metrics["cosine"] >= 0.99, (
        f"BF16/FP8 t2i images diverge too much: cosine={metrics['cosine']:.6f} < 0.99"
    )
    assert metrics["rel_l2"] < 0.05, (
        f"BF16/FP8 t2i images diverge too much: rel_l2={metrics['rel_l2']:.6f} >= 0.05"
    )
