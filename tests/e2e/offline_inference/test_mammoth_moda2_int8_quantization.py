# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU end-to-end INT8 W8A8 A/B gates for the MammothModa2 AR stage.

Unlike the FP8 gate (``test_mammoth_moda2_fp8_quantization.py``), which applies
vLLM's *runtime* W8A8 quantization to the BF16 ``MammothModa2-Dev`` checkpoint,
the INT8 W8A8 path here loads a *serialized* ``compressed-tensors`` checkpoint
(``MammothModa2-Dev-W8A8``, produced by llm-compressor with SmoothQuant +
GPTQModifier, scheme ``W8A8``). The A/B gate therefore compares two checkpoints:

* ``bf16`` — ``MammothModa2-Dev`` (no quantization).
* ``int8`` — ``MammothModa2-Dev-W8A8`` (``quantization: compressed-tensors``).

Coverage:

* ``test_ar_generation_smoke`` — each checkpoint loads and produces tokens.
* ``test_bf16_vs_int8_generation_consistency`` — AR-only understanding: base
  head / understanding expert under INT8 (token agreement + logprob similarity
  + MAE).
* ``test_bf16_vs_int8_t2i_image_consistency`` — t2i (AR→DiT): drives the AR
  stage to emit visual tokens, which exercises the generation experts
  (``gen_mlp``, kept in BF16 by the checkpoint's ``ignore`` list) and the extra
  vocabulary/head (``gen_embed_tokens`` stays BF16; ``gen_head`` must either
  stay BF16 or carry a valid per-channel scale), and compares the decoded
  image.
* ``test_int8_checkpoint_quantization_scales`` — checkpoint-loading metadata +
  quantization scales: the serialized checkpoint declares compressed-tensors
  W8A8 (int8 weights, per-channel symmetric; dynamic per-token int8
  activations), carries a finite, positive per-channel ``weight_scale`` for
  every quantized layer, and keeps the ignored generation experts in BF16.

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

# BF16 baseline checkpoint (same one the FP8 A/B gate uses).
MODEL_PATH = os.environ.get("MAMMOTH_MODA2_MODEL", "bytedance-research/MammothModa2-Dev")

# Serialized INT8 W8A8 checkpoint (compressed-tensors). Override via env for CI
# or hub ids; the default points at the locally produced llm-compressor
# checkpoint.
INT8_MODEL_PATH = os.environ.get("MAMMOTH_MODA2_INT8_MODEL", "/root/autodl-fs/MammothModa2-Dev-W8A8")
# The checkpoint's ``config.json`` declares ``quant_method: compressed-tensors``;
# setting the same stage key makes the A/B arm explicit and deterministic
# instead of relying on auto-detection.
INT8_QUANTIZATION = "compressed-tensors"

# A/B cases: (checkpoint path, quantization stage key).
QUANTIZATION_CASES = [
    pytest.param(MODEL_PATH, None, id="bf16"),
    pytest.param(INT8_MODEL_PATH, INT8_QUANTIZATION, id="int8"),
]

# Hardware gate: any CUDA GPU. ``@hardware_test`` pins specific SKUs, so use a
# plain skip so local workstation cards can run the A/B gate too.
_CUDA_ONLY = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")


# NOTE: provisional — will be finalized from the value measured on the target
# GPU.
MIN_LOGPROB_COSINE = 0.90

# Decoded-image A/B thresholds (same seed + greedy AR + deterministic DiT).
MIN_IMAGE_COSINE = 0.98
MAX_IMAGE_REL_L2 = 0.10

_PROMPT = "Explain multimodal generation in three sentences."

pytestmark = [pytest.mark.slow]


# ---------------------------------------------------------------------------
# AR-only understanding A/B
# ---------------------------------------------------------------------------


_AR_STAGE_OVERRIDES = {"gpu_memory_utilization": 0.6, "skip_mm_profiling": True}


def _stage_config(quantization: str | None) -> str:
    """Return a patched AR deploy config for the requested quantization.

    ``None`` deletes the ``quantization`` key (BF16 baseline); otherwise the
    key is set to ``quantization``.
    """
    from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

    base = get_deploy_config_path("mammoth_moda2_ar.yaml")
    stage_updates = dict(_AR_STAGE_OVERRIDES)
    if quantization is None:
        return modify_stage_config(base, updates={"stages": {0: stage_updates}}, deletes={"stages": {0: ["quantization"]}})
    stage_updates["quantization"] = quantization
    return modify_stage_config(base, updates={"stages": {0: stage_updates}})


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
@pytest.mark.parametrize(("model", "quantization"), QUANTIZATION_CASES)
def test_ar_generation_smoke(model: str, quantization: str | None):
    """Each supported format loads and produces a non-empty greedy sequence."""
    token_ids, _ = _generate(model, quantization)
    assert token_ids, f"no tokens generated for model={model!r} quantization={quantization!r}"


@_CUDA_ONLY
@pytest.mark.omni
def test_bf16_vs_int8_generation_consistency():
    """A/B: BF16 vs INT8 — report token agreement + logprob similarity + MAE."""
    bf16_ids, bf16_lp = _generate(MODEL_PATH, None)
    int8_ids, int8_lp = _generate(INT8_MODEL_PATH, INT8_QUANTIZATION)

    assert bf16_ids, "BF16 baseline produced no tokens"
    assert int8_ids, "INT8 run produced no tokens"

    common = min(len(bf16_ids), len(int8_ids))
    token_agree = sum(a == b for a, b in zip(bf16_ids[:common], int8_ids[:common])) / common

    # Compare top-1 logprobs only while both runs stay on the same decoding
    # path. After the first greedy-token divergence each model continues from a
    # different context, so later logprobs are uncorrelated and would dominate
    # (and deflate) the cosine. Positional drift is reported via token_agree /
    # agree_prefix but is not gated (see the MIN_LOGPROB_COSINE comment).
    agree_prefix = next(
        (i for i, (a, b) in enumerate(zip(bf16_ids[:common], int8_ids[:common])) if a != b),
        common,
    )
    lp_len = min(agree_prefix, len(bf16_lp), len(int8_lp))
    logprob_cos = _cosine_sim(bf16_lp[:lp_len], int8_lp[:lp_len]) if lp_len > 1 else float("nan")
    logprob_mae = _mean_abs_diff(bf16_lp[:lp_len], int8_lp[:lp_len]) if lp_len > 0 else float("nan")

    print(
        f"[INT8 A/B] token_agreement={token_agree:.4f} "
        f"agree_prefix={agree_prefix}/{common} "
        f"logprob_cosine={logprob_cos:.4f} "
        f"logprob_mae={logprob_mae:.4f}"
    )

    if logprob_cos == logprob_cos:  # not NaN
        assert logprob_cos >= MIN_LOGPROB_COSINE, (
            f"BF16/INT8 logprob sequences diverge too much: cosine={logprob_cos:.4f} < "
            f"{MIN_LOGPROB_COSINE}"
        )


# ---------------------------------------------------------------------------
# t2i A/B — end-to-end trigger for generation experts + extra head
# ---------------------------------------------------------------------------
# The AR-only understanding test above never activates ``gen_mlp`` /
# ``gen_embed_tokens`` / ``gen_head``, because those only fire for image tokens
# (``input_ids >= gen_vocab_start_index == 152064``). The t2i task drives the AR
# stage to emit visual tokens, exercising the generation experts and the extra
# vocabulary/head under INT8 end to end. In the serialized W8A8 checkpoint the
# generation experts are listed under ``ignore`` (kept BF16) while the extra
# generation head carries its own scale — the image A/B covers both.

_AR_PATCH_SIZE = 16

# Dev (Qwen3-VL) and Preview (Qwen2.5-VL) share the same vision token ids.
_IMAGE_TOKEN_ID = 151655
_VIDEO_TOKEN_ID = 151656
_VISION_START_TOKEN_ID = 151652
_VISION_END_TOKEN_ID = 151653


def _load_t2i_gen_config(model: str) -> dict:
    """Load ``t2i_generation_config.json`` from a local dir or hub id.

    The t2i generation constants are model-family level, not weight level, and
    the serialized W8A8 checkpoint directory may omit this file, so fall back to
    the BF16 baseline checkpoint.
    """
    import json
    from pathlib import Path

    for candidate in (model, MODEL_PATH):
        local = Path(candidate) / "t2i_generation_config.json"
        if local.exists():
            return json.loads(local.read_text(encoding="utf-8"))

    from huggingface_hub import snapshot_download

    for candidate in (model, MODEL_PATH):
        try:
            weights_dir = Path(snapshot_download(candidate))
        except Exception:
            continue
        cfg_path = weights_dir / "t2i_generation_config.json"
        if cfg_path.exists():
            return json.loads(cfg_path.read_text(encoding="utf-8"))

    pytest.skip(f"t2i_generation_config.json not found for {model!r} or {MODEL_PATH!r}")


def _format_t2i_prompt(user_prompt: str, ar_width: int, ar_height: int) -> str:
    return (
        "<|im_start|>system\nYou are a helpful image generator.<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"<|image start|>{ar_width}*{ar_height}<|image token|>"
    )


def _t2i_stage_config(quantization: str | None) -> str:
    """Patch the AR→DiT deploy config: quantization + memory budget on AR stage 0.

    The DiT stage (stage 1) keeps its deploy default (0.3); the AR stage takes
    0.6 so both fit on a single 48 GiB device.
    """
    from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

    base = get_deploy_config_path("mammoth_moda2.yaml")
    stage_updates = dict(_AR_STAGE_OVERRIDES)
    if quantization is not None:
        stage_updates["quantization"] = quantization
    return modify_stage_config(base, updates={"stages": {0: stage_updates}})


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
def test_bf16_vs_int8_t2i_image_consistency():
    """A/B: BF16 vs INT8 t2i — decoded image must match (gen experts + head OK)."""
    bf16_img = _generate_t2i_image(MODEL_PATH, None)
    int8_img = _generate_t2i_image(INT8_MODEL_PATH, INT8_QUANTIZATION)

    assert bf16_img.shape == int8_img.shape, (bf16_img.shape, int8_img.shape)

    metrics = _image_metrics(bf16_img, int8_img)

    print("\n[INT8 t2i A/B] BF16 vs INT8 decoded image:")
    for name in ("psnr_db", "mae", "max_abs", "cosine", "rel_l2", "pixel_match"):
        print(f"  {name:12s} = {metrics[name]:.6f}")

    # Same seed + greedy AR + deterministic DiT: a correctly quantized
    # gen_head plus the BF16 gen_mlp/gen_embed_tokens must reproduce the same
    # visual tokens, hence the image.
    assert metrics["cosine"] >= MIN_IMAGE_COSINE, (
        f"BF16/INT8 t2i images diverge too much: cosine={metrics['cosine']:.6f} < {MIN_IMAGE_COSINE}"
    )
    assert metrics["rel_l2"] < MAX_IMAGE_REL_L2, (
        f"BF16/INT8 t2i images diverge too much: rel_l2={metrics['rel_l2']:.6f} >= {MAX_IMAGE_REL_L2}"
    )


# ---------------------------------------------------------------------------
# Checkpoint loading + quantization scales (no GPU required)
# ---------------------------------------------------------------------------

def _local_checkpoint_dir(model: str):
    """Return the local checkpoint directory for *model*, or ``None``."""
    from pathlib import Path

    path = Path(model)
    return path if path.is_dir() else None


def _read_tensor(ckpt_dir, weight_map: dict[str, str], key: str) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(str(ckpt_dir / weight_map[key]), framework="pt") as handle:
        return handle.get_tensor(key)


def _assert_int8_weight_and_scale(ckpt_dir, weight_map: dict[str, str], weight_key: str, scale_key: str) -> None:
    """Quantized weights are int8 and carry a finite, positive per-channel scale."""
    from safetensors import safe_open

    with safe_open(str(ckpt_dir / weight_map[weight_key]), framework="pt") as handle:
        dtype = handle.get_slice(weight_key).get_dtype()
    assert dtype == "I8", f"{weight_key} must be int8, got {dtype}"

    scale = _read_tensor(ckpt_dir, weight_map, scale_key)
    assert scale.ndim == 2 and scale.shape[-1] == 1, (
        f"{scale_key} must be a per-channel [out_features, 1] scale, got {tuple(scale.shape)}"
    )
    assert torch.isfinite(scale).all(), f"{scale_key} contains non-finite values"
    assert (scale > 0).all(), f"{scale_key} contains non-positive values"


@pytest.mark.cpu
def test_int8_checkpoint_quantization_scales():
    """Serialized INT8 W8A8 checkpoint: scheme metadata + per-channel scales.

    Validates that the checkpoint the A/B gate loads is a genuine W8A8
    compressed-tensors checkpoint and that its quantization scales cover exactly
    the layers it claims to quantize: the ignored generation experts stay in
    BF16, and the extra generation head is either BF16 or backed by a valid
    per-channel scale.
    """
    import json

    ckpt_dir = _local_checkpoint_dir(INT8_MODEL_PATH)
    if ckpt_dir is None:
        pytest.skip(f"INT8 W8A8 checkpoint not available locally: {INT8_MODEL_PATH}")

    config = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
    quant_config = config.get("quantization_config")
    assert quant_config is not None, f"{INT8_MODEL_PATH}/config.json has no quantization_config"
    assert quant_config["quant_method"] == "compressed-tensors"
    assert quant_config["quantization_status"] == "compressed"

    group = next(iter(quant_config["config_groups"].values()))
    weights, activations = group["weights"], group["input_activations"]
    # W8A8: 8-bit int weights (per-channel, symmetric) + 8-bit int activations
    # (dynamic per-token, symmetric).
    assert (weights["type"], weights["num_bits"]) == ("int", 8)
    assert weights["strategy"] == "channel" and weights["symmetric"] is True
    assert (activations["type"], activations["num_bits"]) == ("int", 8)
    assert activations["dynamic"] is True and activations["strategy"] == "token"

    weight_map = json.loads((ckpt_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]

    # 1. Quantized base linear layers carry a per-channel weight_scale.
    base_proj = "llm_model.model.language_model.layers.0.mlp.gate_proj"
    assert f"{base_proj}.weight" in weight_map
    assert f"{base_proj}.weight_scale" in weight_map
    _assert_int8_weight_and_scale(ckpt_dir, weight_map, f"{base_proj}.weight", f"{base_proj}.weight_scale")

    # 2. Generation experts (gen_mlp) are ignored and therefore stay BF16:
    #    weights present, no quantization scale.
    gen_expert_weights = sorted(key for key in weight_map if ".gen_mlp." in key)
    assert gen_expert_weights, "no generation-expert weights found in the checkpoint"
    assert not any(key.endswith(".weight_scale") for key in gen_expert_weights), (
        "generation experts must stay unquantized (listed under quant ignore)"
    )

    # 3. Extra vocabulary/head: gen_embed_tokens stays BF16 (embeddings are not
    #    quant targets). gen_head must be either kept in BF16 (the recipe lists
    #    it under ``ignore``) or quantized with a *valid* per-channel scale:
    #    an all-zero scale silently zeroes every visual-token logit and breaks
    #    t2i, so it is rejected explicitly here.
    embed_key = "llm_model.model.language_model.gen_embed_tokens.weight"
    assert embed_key in weight_map
    assert not any("gen_embed_tokens" in key and key.endswith(".weight_scale") for key in weight_map)
    assert "llm_model.gen_head.weight" in weight_map

    gen_head_scale_key = "llm_model.gen_head.weight_scale"
    if gen_head_scale_key in weight_map:
        _assert_int8_weight_and_scale(
            ckpt_dir,
            weight_map,
            "llm_model.gen_head.weight",
            gen_head_scale_key,
        )
