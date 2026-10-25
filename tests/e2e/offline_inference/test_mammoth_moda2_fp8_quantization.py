# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU end-to-end FP8 A/B gates for the MammothModa2 AR stage.

These tests load the real ``MammothModa2-Preview`` and ``MammothModa2-Dev``
checkpoints and compare BF16 vs FP8 generation, scoped to the AR stage:

* ``test_ar_generation_smoke`` — each format loads and produces tokens.
* ``test_bf16_vs_fp8_generation_consistency`` — AR-only understanding: the
  base head / understanding expert under FP8 (reported token agreement and
  MAE; shared-prefix logprob similarity gate).
* ``test_bf16_vs_fp8_t2i_image_consistency`` — t2i (AR→DiT): drives the AR
  stage to emit visual tokens, which exercises the generation experts
  (``gen_mlp``) and the extra vocabulary/head (``gen_embed_tokens`` /
  ``gen_head``) under FP8, and compares the decoded image.
"""

from __future__ import annotations

import math
import os
from typing import Any

import pytest
import torch

from tests.helpers.mark import hardware_test

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Override each checkpoint independently for local runs / CI. The legacy
# MAMMOTH_MODA2_MODEL override remains a fallback for Dev only.
MODEL_CASES = [
    pytest.param(
        os.environ.get("MAMMOTH_MODA2_PREVIEW_MODEL", "bytedance-research/MammothModa2-Preview"),
        id="preview",
    ),
    pytest.param(
        os.environ.get(
            "MAMMOTH_MODA2_DEV_MODEL",
            os.environ.get("MAMMOTH_MODA2_MODEL", "bytedance-research/MammothModa2-Dev"),
        ),
        id="dev",
    ),
]

# Quantization cases for the A/B gate. ``None`` is the BF16 baseline (the
# ``quantization`` stage key is removed). FP8 is vLLM's runtime W8A8
# (weight + activation) quantization, computed directly from the BF16
# checkpoint — no pre-quantized weights required.
QUANTIZATION_CASES = [None, "fp8"]

# Hardware gate: any CUDA GPU (H100/B200 datacenter or RTX PRO 6000
# workstation). ``@hardware_test`` supplies the CI SKU/cards marks (H100,
# cards_1) used by the nightly ``-m`` selector; the plain skip below keeps
# non-CUDA hosts from erroring out locally, and does not restrict the marks.
_CUDA_ONLY = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")

# Token agreement is reported only: one greedy-token change can shift the
# continuation and make positional agreement misleading. Require at least five
# identical initial tokens so scoring extends beyond the first four tokens
# that echo the prompt in the Preview run.
MIN_SHARED_PREFIX = 5

# Minimum cosine similarity between BF16 and quantized top-1 logprobs on the
# shared greedy prefix. Steps from divergent continuations are not scored.
MIN_LOGPROB_COSINE = 0.90

_PROMPT = "Explain multimodal generation in three sentences."
_SEED = 42

# L4 nightly suite selected by the dedicated H100 step in
# ``.buildkite/cuda/test-nightly.yml``. Not ``slow``: that marker routes tests
# into the weekly ``-m "slow and ..."`` sweeps, which would run this full
# checkpoint A/B gate a second time.
pytestmark = [pytest.mark.full_model]


@pytest.fixture(scope="module", params=MODEL_CASES)
def model_path(request) -> str:
    """Run every GPU gate independently for Preview and Dev."""
    return request.param


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
    # OmniRunner stores its seed but does not forward it to the engine.
    updates: dict[str, Any] = {"stages": {0: {"seed": _SEED}}}
    if quantization is None:
        return modify_stage_config(base, updates=updates, deletes={"stages": {0: ["quantization"]}})
    updates["stages"][0]["quantization"] = quantization
    return modify_stage_config(base, updates=updates)


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
    sampling_params = SamplingParams(temperature=0.0, max_tokens=32, seed=_SEED, detokenize=True, logprobs=1)

    with OmniRunner(model, seed=_SEED, deploy_config=_stage_config(quantization)) as runner:
        outputs = list(runner.omni.generate([prompt_dict], [sampling_params]))

    for out in outputs:
        for completion in getattr(out, "outputs", None) or []:
            token_ids = list(getattr(completion, "token_ids", None) or [])
            logprobs = _top1_logprobs(completion)
            print(f"[FP8 AR output] model={model} quantization={quantization!r} seed={_SEED}")
            print(f"  text={getattr(completion, 'text', None)!r}")
            print(f"  token_ids={token_ids}")
            print(f"  top1_logprobs={logprobs}")
            return token_ids, logprobs
    return [], []


def _top1_logprobs(completion) -> list[float]:
    """Extract finite top-1 logprobs, rejecting missing or invalid steps."""
    logprobs = getattr(completion, "logprobs", None)
    if not logprobs:
        return []
    result: list[float] = []
    for index, step in enumerate(logprobs):
        assert step, f"missing logprobs at generated step {index}"
        assert all(math.isfinite(float(lp.logprob)) for lp in step.values()), (
            f"non-finite logprobs at generated step {index}"
        )
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
@hardware_test(res={"cuda": "H100"})
@pytest.mark.parametrize("quantization", QUANTIZATION_CASES, ids=["bf16", "fp8"])
def test_ar_generation_smoke(model_path: str, quantization: str | None):
    """Each supported format loads and produces a non-empty greedy sequence."""
    token_ids, _ = _generate(model_path, quantization)
    assert token_ids, f"no tokens generated for quantization={quantization!r}"


@_CUDA_ONLY
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"})
def test_bf16_vs_fp8_generation_consistency(model_path: str):
    """Gate shared-prefix logprob cosine; report token agreement and MAE."""
    bf16_ids, bf16_lp = _generate(model_path, None)
    fp8_ids, fp8_lp = _generate(model_path, "fp8")

    assert bf16_ids, "BF16 baseline produced no tokens"
    assert fp8_ids, "FP8 run produced no tokens"

    assert len(bf16_ids) == len(fp8_ids), (
        f"BF16/FP8 generated different sequence lengths: bf16={len(bf16_ids)} "
        f"fp8={len(fp8_ids)} (bf16={bf16_ids} fp8={fp8_ids})"
    )
    token_agree = sum(a == b for a, b in zip(bf16_ids, fp8_ids)) / len(bf16_ids)
    first_mismatch = next((i for i, (a, b) in enumerate(zip(bf16_ids, fp8_ids)) if a != b), None)
    shared_prefix = len(bf16_ids) if first_mismatch is None else first_mismatch
    print(f"[FP8 AR comparison] model={model_path} first_mismatch_index={first_mismatch}")

    # Validate every generated step, including the unscored continuation.
    # _top1_logprobs also rejects empty steps and non-finite values.
    assert len(bf16_lp) == len(bf16_ids), f"BF16 top-1 logprobs missing: got {len(bf16_lp)} for {len(bf16_ids)} tokens"
    assert len(fp8_lp) == len(fp8_ids), f"FP8 top-1 logprobs missing: got {len(fp8_lp)} for {len(fp8_ids)} tokens"

    assert shared_prefix >= MIN_SHARED_PREFIX, (
        f"BF16/FP8 share only {shared_prefix}/{len(bf16_ids)} initial token(s); "
        f"need at least {MIN_SHARED_PREFIX} steps to score logprobs. "
        f"bf16={bf16_ids} fp8={fp8_ids}"
    )

    # At and after the first token mismatch the predictions concern different
    # tokens, then different histories. Score only the identical initial path.
    logprob_cos = _cosine_sim(bf16_lp[:shared_prefix], fp8_lp[:shared_prefix])
    logprob_mae = _mean_abs_diff(bf16_lp[:shared_prefix], fp8_lp[:shared_prefix])
    assert math.isfinite(logprob_cos), f"non-finite logprob cosine: {logprob_cos}"

    print(
        f"[FP8 A/B] token_agreement={token_agree:.4f} (reported only) "
        f"agree_prefix={shared_prefix}/{len(bf16_ids)} "
        f"logprob_cosine={logprob_cos:.4f} logprob_mae={logprob_mae:.4f}"
    )
    assert logprob_cos >= MIN_LOGPROB_COSINE, (
        f"BF16/FP8 shared-prefix logprobs diverge too much: cosine={logprob_cos:.4f} < {MIN_LOGPROB_COSINE}"
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
_T2I_IMAGE_SIZE = (256, 256)  # PIL (width, height)
_T2I_IMAGE_COUNT = 1

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
    # Only the AR engine accepts a deployment seed. DiT must receive its seed
    # through request sampling parameters (see dit_sampling below).
    updates: dict[str, Any] = {"stages": {0: {"seed": _SEED}}}
    if quantization is None:
        return modify_stage_config(base, updates=updates, deletes={"stages": {0: ["quantization"]}})
    updates["stages"][0]["quantization"] = quantization
    return modify_stage_config(base, updates=updates)


def _generate_t2i_image(model: str, quantization: str | None) -> torch.Tensor:
    """Run t2i (AR→DiT) and return the decoded image tensor."""
    from vllm.sampling_params import SamplingParams

    from tests.helpers.runtime import OmniRunner

    gen_cfg = _load_t2i_gen_config(model)
    eol_token_id = int(gen_cfg["eol_token_id"])
    visual_start = int(gen_cfg["visual_token_start_id"])
    visual_end = int(gen_cfg["visual_token_end_id"])

    width, height = _T2I_IMAGE_SIZE  # small for CI speed
    ar_height, ar_width = height // _AR_PATCH_SIZE, width // _AR_PATCH_SIZE
    expected_grid_tokens = ar_height * (ar_width + 1)

    formatted_prompt = _format_t2i_prompt("A cat sitting on a laptop keyboard", ar_width, ar_height)

    ar_sampling = SamplingParams(
        temperature=0.0,
        top_k=1,
        max_tokens=max(1, expected_grid_tokens + 1),
        seed=_SEED,
        detokenize=False,
    )
    # Diffusion requests without an explicit seed receive a random seed, even
    # with greedy AR decoding and a fixed engine seed. Pin the initial noise.
    dit_sampling = SamplingParams(temperature=0.0, max_tokens=1, seed=_SEED, detokenize=False)

    with OmniRunner(model, seed=_SEED, deploy_config=_t2i_stage_config(quantization)) as runner:
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

    return _extract_image_tensor(outputs, expected_size=(width, height))


def _assert_finite_image_payload(payload, context: str) -> None:
    """Reject non-finite raw image tensors before PIL conversion can mask them."""
    if isinstance(payload, torch.Tensor):
        assert bool(torch.isfinite(payload).all()), f"non-finite values (NaN/Inf) in {context}"
    elif isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            _assert_finite_image_payload(item, f"{context}[{index}]")


def _check_raw_image_tensors(outputs) -> None:
    """Inspect the image-bearing fields handled by the official image helper."""
    from collections.abc import Mapping

    def check_multimodal(value, context: str) -> None:
        if hasattr(value, "to_dict") and callable(value.to_dict):
            value = value.to_dict()
        if isinstance(value, Mapping):
            for key in ("image", "images", "model_outputs"):
                if key in value:
                    _assert_finite_image_payload(value[key], f"{context}.{key}")

    for index, output in enumerate(outputs):
        # OmniRequestOutput.images is the first source used by the helper.
        _assert_finite_image_payload(getattr(output, "images", None), f"outputs[{index}].images")
        check_multimodal(
            getattr(output, "multimodal_output", None),
            f"outputs[{index}].multimodal_output",
        )
        for completion_index, completion in enumerate(getattr(output, "outputs", None) or []):
            check_multimodal(
                getattr(completion, "multimodal_output", None),
                f"outputs[{index}].outputs[{completion_index}].multimodal_output",
            )


def _extract_image_tensor(outputs, expected_size: tuple[int, int] = _T2I_IMAGE_SIZE) -> torch.Tensor:
    """Validate one RGB image of requested size and return CHW floats in [0, 1].

    Validate raw tensor finiteness *before* the official helper converts image
    tensors to uint8 PIL images (which could otherwise mask NaN/Inf values).
    """
    import numpy as np

    from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs

    _check_raw_image_tensors(outputs)
    images = extract_images_from_outputs(outputs)
    assert len(images) == _T2I_IMAGE_COUNT, f"expected {_T2I_IMAGE_COUNT} generated image(s), got {len(images)}"

    image = images[0]
    assert image.size == expected_size, f"expected image size {expected_size}, got {image.size}"
    assert image.mode == "RGB", f"expected RGB image, got {image.mode}"

    arr = np.asarray(image, dtype=np.float32) / 255.0  # (H, W, C)
    assert np.isfinite(arr).all(), "decoded image contains NaN/Inf"
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
@hardware_test(res={"cuda": "H100"})
def test_bf16_vs_fp8_t2i_image_consistency(model_path: str):
    """A/B: BF16 vs FP8 t2i — decoded image must match (gen experts + head OK)."""
    bf16_img = _generate_t2i_image(model_path, None)
    fp8_img = _generate_t2i_image(model_path, "fp8")

    expected_shape = (3, _T2I_IMAGE_SIZE[1], _T2I_IMAGE_SIZE[0])
    for name, image in (("BF16", bf16_img), ("FP8", fp8_img)):
        assert image.shape == expected_shape, f"{name} produced shape {tuple(image.shape)}, expected {expected_shape}"
        assert bool(torch.isfinite(image).all()), f"{name} image contains NaN/Inf"

    metrics = _image_metrics(bf16_img, fp8_img)

    print("\n[FP8 t2i A/B] BF16 vs FP8 decoded image:")
    for name in ("psnr_db", "mae", "max_abs", "cosine", "rel_l2", "pixel_match"):
        print(f"  {name:12s} = {metrics[name]:.6f}")

    # Same seed + greedy AR + deterministic DiT: a correctly quantized
    # gen_mlp/gen_head must reproduce the same visual tokens, hence the image.
    assert metrics["cosine"] >= 0.99, f"BF16/FP8 t2i images diverge too much: cosine={metrics['cosine']:.6f} < 0.99"
    assert metrics["rel_l2"] < 0.05, f"BF16/FP8 t2i images diverge too much: rel_l2={metrics['rel_l2']:.6f} >= 0.05"
