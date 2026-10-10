# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU end-to-end INT8 W8A8 A/B gates for the MammothModa2 AR stage.

The INT8 path loads serialized ``compressed-tensors`` W8A8 checkpoints.
Preview (Qwen2.5-VL) and Dev (Qwen3-VL) run as independent BF16/INT8 pairs.
The official BF16 and user-published INT8 checkpoints are pinned below for both
versions. INT8 overrides use ``MAMMOTH_MODA2_{PREVIEW,DEV}_INT8_MODEL`` and
``*_INT8_REVISION`` (full Hub commit SHA). Local checkpoint directories are
also supported and do not require a revision.
BF16 overrides use ``MAMMOTH_MODA2_{PREVIEW,DEV}_MODEL`` and ``*_REVISION``.
Missing checkpoint configuration or download errors fail the gates.

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

Checkpoint metadata checks run on CPU, using the same snapshots as inference.
"""

from __future__ import annotations

import json
import os
import re
from functools import cache
from pathlib import Path

import pytest
import torch

from tests.helpers.mark import hardware_test

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Immutable official BF16 baselines, stored as (repository ID, commit SHA).
_BF16_CHECKPOINTS = {
    "preview": (
        "bytedance-research/MammothModa2-Preview",
        "ef5a5e41dbf0de1ef6275586b7580f0d4248b4c6",
    ),
    "dev": (
        "bytedance-research/MammothModa2-Dev",
        "461ad0d7d846bd5fa944619b08213a936eee2e30",
    ),
}
# Immutable user-published W8A8 artifacts. Both pinned revisions contain the
# model config, safetensors weight index and all five weight shards.
_INT8_CHECKPOINTS = {
    "preview": (
        "wenjyanasd/MammothModa2-Preview-W8A8",
        "a745b457de0738524e4939a38fb3376a5cd03958",
    ),
    "dev": (
        "wenjyanasd/MammothModa2-Dev-W8A8",
        "17e114cdff9648d64c5d6dee19c134c320699d5a",
    ),
}
_MODEL_VERSIONS = [pytest.param("preview", id="preview"), pytest.param("dev", id="dev")]
# The checkpoint's ``config.json`` declares ``quant_method: compressed-tensors``;
# setting the same stage key makes the A/B arm explicit and deterministic
# instead of relying on auto-detection.
INT8_QUANTIZATION = "compressed-tensors"

# Each version runs both arms against its own architecture's baseline.
QUANTIZATION_CASES = [
    pytest.param(None, id="bf16"),
    pytest.param(INT8_QUANTIZATION, id="int8"),
]

# The CI lane uses one H100 (80 GiB). Keep local Ada/Hopper runs possible,
# while rejecting ROCm and Blackwell, which this W8A8 backend cannot run.
_INT8_CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires an NVIDIA CUDA GPU",
)


@pytest.fixture
def int8_cuda_device():
    capability = torch.cuda.get_device_capability()
    if not (75 <= capability[0] * 10 + capability[1] < 100):
        pytest.skip(f"INT8 W8A8 requires SM75-SM90; got compute capability {capability}")


@cache
def _checkpoint_dir(version: str, quantization: str | None) -> Path:
    """Resolve local paths or pinned Hub snapshots for both inference and metadata."""
    prefix = f"MAMMOTH_MODA2_{version.upper()}"
    if quantization is not None:
        prefix += "_INT8"
    checkpoints = _BF16_CHECKPOINTS if quantization is None else _INT8_CHECKPOINTS
    default_model, default_revision = checkpoints[version]
    model = os.environ.get(f"{prefix}_MODEL", default_model)
    revision = os.environ.get(f"{prefix}_REVISION")
    # Preserve the existing Dev workstation overrides.
    if version == "dev":
        legacy = "MAMMOTH_MODA2_INT8" if quantization is not None else "MAMMOTH_MODA2"
        model = os.environ.get(f"{prefix}_MODEL", os.environ.get(f"{legacy}_MODEL", default_model))
        revision = os.environ.get(f"{prefix}_REVISION", os.environ.get(f"{legacy}_REVISION"))
    assert model, f"Set {prefix}_MODEL to the published checkpoint ID or a local directory"
    path = Path(model)
    if path.is_dir():
        return path
    if revision is None and model == default_model:
        revision = default_revision
    assert revision and re.fullmatch(r"[0-9a-fA-F]{40}", revision), (
        f"Set {prefix}_REVISION to the full commit SHA for {model!r}"
    )
    from vllm_omni.transformers_utils.repo_utils import hf_api

    # snapshot_download reuses the loader's Hub cache. No skip on cache misses,
    # missing files, permission errors or network failures.
    return Path(hf_api().snapshot_download(repo_id=model, revision=revision))


@pytest.fixture(params=_MODEL_VERSIONS, scope="module")
def checkpoint_pair(request) -> tuple[str, str]:
    version = request.param
    int8 = _checkpoint_dir(version, INT8_QUANTIZATION)
    bf16 = _checkpoint_dir(version, None)
    configs = [json.loads((path / "config.json").read_text(encoding="utf-8")) for path in (bf16, int8)]
    for config in configs:
        _assert_model_version(config, version)
    # Ignore serialization-only fields (dtype, _name_or_path, etc.) while
    # requiring the same language architecture, vocabulary and routing setup.
    ar_configs = [config["llm_config"] for config in configs]
    assert ar_configs[0]["model_type"] == ar_configs[1]["model_type"]
    text_configs = [config.get("text_config", config) for config in ar_configs]
    for key in (
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "gen_vocab_size",
        "gen_vocab_start_index",
        "moe_type",
    ):
        bf16_value = text_configs[0].get(key, ar_configs[0].get(key))
        int8_value = text_configs[1].get(key, ar_configs[1].get(key))
        assert bf16_value == int8_value, f"{version}: AR config mismatch for {key}"
    assert not configs[0].get("quantization_config"), f"{version}: BF16 baseline is quantized"
    return str(bf16), str(int8)


# NOTE: provisional — will be finalized from the value measured on the target
# GPU.
MIN_LOGPROB_COSINE = 0.90

# Minimum number of aligned greedy steps required before the logprob gate can
# be scored (cosine needs at least two points). A shorter shared prefix means
# the two arms diverge almost immediately, which must fail rather than fall
# back to NaN and silently skip the gate.
MIN_SHARED_PREFIX = 2

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
        return modify_stage_config(
            base, updates={"stages": {0: stage_updates}}, deletes={"stages": {0: ["quantization"]}}
        )
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


@_INT8_CUDA_ONLY
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.usefixtures("int8_cuda_device")
@pytest.mark.omni
@pytest.mark.parametrize("quantization", QUANTIZATION_CASES)
def test_ar_generation_smoke(checkpoint_pair: tuple[str, str], quantization: str | None):
    """Each supported format loads and produces a non-empty greedy sequence."""
    model = checkpoint_pair[quantization is not None]
    token_ids, _ = _generate(model, quantization)
    assert token_ids, f"no tokens generated for model={model!r} quantization={quantization!r}"


@_INT8_CUDA_ONLY
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.usefixtures("int8_cuda_device")
@pytest.mark.omni
def test_bf16_vs_int8_generation_consistency(checkpoint_pair: tuple[str, str]):
    """A/B: BF16 vs INT8 — report token agreement + logprob similarity + MAE."""
    bf16_model, int8_model = checkpoint_pair
    bf16_ids, bf16_lp = _generate(bf16_model, None)
    int8_ids, int8_lp = _generate(int8_model, INT8_QUANTIZATION)

    assert bf16_ids, "BF16 baseline produced no tokens"
    assert int8_ids, "INT8 run produced no tokens"

    common = min(len(bf16_ids), len(int8_ids))
    token_agree = sum(a == b for a, b in zip(bf16_ids[:common], int8_ids[:common])) / common

    # Compare top-1 logprobs only while both runs stay on the same decoding
    # path. After the first greedy-token divergence each model continues from a
    # different context, so later logprobs are uncorrelated and would dominate
    # (and deflate) the cosine. Drift beyond the shared prefix is reported but
    # not scored; a shared prefix shorter than MIN_SHARED_PREFIX is gated.
    agree_prefix = next(
        (i for i, (a, b) in enumerate(zip(bf16_ids[:common], int8_ids[:common])) if a != b),
        common,
    )
    lp_len = min(agree_prefix, len(bf16_lp), len(int8_lp))

    # Diverging within the first step or two is the strongest possible failure
    # signal, so it must fail here instead of degrading the metrics to NaN and
    # turning the gate below into a no-op.
    assert lp_len >= MIN_SHARED_PREFIX, (
        f"BF16/INT8 agree on only {lp_len} step(s) before diverging "
        f"(agree_prefix={agree_prefix}/{common}); cannot score logprobs"
    )

    logprob_cos = _cosine_sim(bf16_lp[:lp_len], int8_lp[:lp_len])
    logprob_mae = _mean_abs_diff(bf16_lp[:lp_len], int8_lp[:lp_len])

    print(
        f"[INT8 A/B] token_agreement={token_agree:.4f} "
        f"agree_prefix={agree_prefix}/{common} "
        f"logprob_cosine={logprob_cos:.4f} "
        f"logprob_mae={logprob_mae:.4f}"
    )

    # NaN (unscorable logprobs) fails the comparison below rather than skipping
    # the gate, so a broken run cannot pass by returning non-finite metrics.
    assert logprob_cos >= MIN_LOGPROB_COSINE, (
        f"BF16/INT8 logprob sequences diverge too much: cosine={logprob_cos:.4f} < {MIN_LOGPROB_COSINE}"
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
# generation head either stays BF16 or carries its own scale.

_AR_PATCH_SIZE = 16

# Dev (Qwen3-VL) and Preview (Qwen2.5-VL) share the same vision token ids.
_IMAGE_TOKEN_ID = 151655
_VIDEO_TOKEN_ID = 151656
_VISION_START_TOKEN_ID = 151652
_VISION_END_TOKEN_ID = 151653


_T2I_GEN_CONFIG_FILE = "t2i_generation_config.json"


def _load_t2i_gen_config(model: str, baseline_model: str) -> dict:
    """Load ``t2i_generation_config.json`` from a local dir or hub id.

    The t2i generation constants are model-family level, not weight level, and
    the serialized W8A8 checkpoint directory may omit this file, so fall back to
    the matching BF16 baseline checkpoint. The normal callers pass resolved
    snapshot directories. If a Hub ID is supplied, only this single file is
    fetched; config lookup never downloads model weights.
    """
    from huggingface_hub.errors import RemoteEntryNotFoundError

    from vllm_omni.transformers_utils.repo_utils import hf_api

    missing = []
    for candidate in dict.fromkeys((model, baseline_model)):
        local_dir = Path(candidate)
        if local_dir.is_dir():
            cfg_path = local_dir / _T2I_GEN_CONFIG_FILE
            if cfg_path.is_file():
                return json.loads(cfg_path.read_text(encoding="utf-8"))
            missing.append(str(cfg_path))
            continue
        try:
            cfg_path = Path(hf_api().hf_hub_download(repo_id=candidate, filename=_T2I_GEN_CONFIG_FILE))
        except RemoteEntryNotFoundError as exc:
            # Only a genuinely absent file permits the matching BF16 fallback.
            # Authentication, repository, revision and network errors propagate.
            missing.append(f"{candidate}: {exc}")
            continue
        return json.loads(cfg_path.read_text(encoding="utf-8"))

    raise FileNotFoundError(f"Required {_T2I_GEN_CONFIG_FILE} missing from both checkpoints: {missing}")


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
    if quantization is None:
        return modify_stage_config(
            base, updates={"stages": {0: stage_updates}}, deletes={"stages": {0: ["quantization"]}}
        )
    stage_updates["quantization"] = quantization
    return modify_stage_config(base, updates={"stages": {0: stage_updates}})


def _generate_t2i_image(model: str, quantization: str | None, baseline_model: str) -> torch.Tensor:
    """Run t2i (AR→DiT) and return the decoded image tensor."""
    from vllm.sampling_params import SamplingParams

    from tests.helpers.runtime import OmniRunner

    gen_cfg = _load_t2i_gen_config(model, baseline_model)
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

    return _extract_image_tensor(outputs, expected_count=1, expected_size=(width, height))


def _assert_finite_image_payload(payload) -> None:
    """Reject NaN/Inf before the image helper clamps and casts tensors to uint8."""
    if isinstance(payload, torch.Tensor):
        assert torch.isfinite(payload).all(), "raw image tensor contains non-finite values"
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            _assert_finite_image_payload(item)


def _extract_image_tensor(outputs, *, expected_count: int, expected_size: tuple[int, int]) -> torch.Tensor:
    """Extract the decoded image as a ``(C, H, W)`` float tensor in ``[0, 1]``.

    Count every completion and request-level image before selecting the single
    requested image. Shared payload objects exposed through multiple aliases
    are counted once; repeated images within a batch or separate completions
    still count as separate outputs.
    """
    import numpy as np

    from vllm_omni.diffusion.utils.image_output import (
        _coerce_images,
        _image_values_from_mapping_like,
    )

    assert expected_count == 1, "this A/B helper compares exactly one requested image"
    images = []
    for output in outputs:
        completion_payload_ids: set[int] = set()

        def collect(payload, seen: set[int]) -> set[int]:
            _assert_finite_image_payload(payload)
            # Compare with earlier aliases, not earlier positions in this batch:
            # [image, image] explicitly contains two requested outputs.
            previous = seen.copy()
            found: set[int] = set()

            def visit(value) -> None:
                if isinstance(value, (list, tuple)):
                    for item in value:
                        visit(item)
                elif value is not None:
                    found.add(id(value))
                    if id(value) not in previous:
                        decoded = _coerce_images(value)
                        assert decoded, f"unsupported image payload: {type(value).__name__}"
                        images.extend(decoded)

            visit(payload)
            seen.update(found)
            return found

        for completion in getattr(output, "outputs", None) or []:
            seen: set[int] = set()
            for payload in _image_values_from_mapping_like(getattr(completion, "multimodal_output", None)):
                completion_payload_ids.update(collect(payload, seen))

        # OmniRequestOutput.multimodal_output returns only the first completion's
        # payload when present, so inspect its stored request-level mapping too.
        root_mapping = getattr(output, "_multimodal_output", None)
        if root_mapping is None:
            root_mapping = getattr(output, "multimodal_output", None)
        root_seen = completion_payload_ids.copy()
        collect(getattr(output, "images", None), root_seen)
        for payload in _image_values_from_mapping_like(root_mapping):
            collect(payload, root_seen)
    assert len(images) == expected_count, f"expected {expected_count} image(s), got {len(images)}"
    for image in images:
        assert image.size == expected_size, f"expected image size {expected_size}, got {image.size}"
        assert image.mode == "RGB", f"expected RGB image, got mode {image.mode!r}"

    arr = np.asarray(images[0], dtype=np.float32) / 255.0  # (H, W, C)
    return torch.from_numpy(arr).permute(2, 0, 1)  # (C, H, W)


@pytest.mark.cpu
@pytest.mark.parametrize("location", ["requests", "completions", "batch"])
def test_image_output_contract_rejects_extra_images(location: str):
    from types import SimpleNamespace

    from PIL import Image

    # Even identical image objects count twice when returned as two outputs.
    image = Image.new("RGB", (8, 8))
    completion = SimpleNamespace(multimodal_output={"image": image})
    if location == "requests":
        outputs = [SimpleNamespace(images=[image]), SimpleNamespace(images=[image])]
    elif location == "completions":
        outputs = [SimpleNamespace(outputs=[completion, completion])]
    else:
        outputs = [SimpleNamespace(images=[image, image])]
    with pytest.raises(AssertionError, match=r"expected 1 image\(s\), got 2"):
        _extract_image_tensor(outputs, expected_count=1, expected_size=(8, 8))


@pytest.mark.cpu
def test_image_output_contract_counts_aliases_once():
    from types import SimpleNamespace

    from PIL import Image

    image = Image.new("RGB", (8, 8))
    payload = {"image": image, "images": [image], "model_outputs": [image]}
    output = SimpleNamespace(
        images=[image],
        outputs=[SimpleNamespace(multimodal_output=payload)],
        _multimodal_output=payload,
    )
    tensor = _extract_image_tensor([output], expected_count=1, expected_size=(8, 8))
    assert tensor.shape == (3, 8, 8)


@pytest.mark.cpu
@pytest.mark.parametrize("invalid", ["size", "mode", "nan", "inf"])
def test_image_output_contract_rejects_invalid_payload(invalid: str):
    from types import SimpleNamespace

    from PIL import Image

    if invalid in {"nan", "inf"}:
        image = torch.zeros(3, 8, 8)
        image[0, 0, 0] = float(invalid)
        message = "raw image tensor contains non-finite values"
    elif invalid == "size":
        image = Image.new("RGB", (4, 8))
        message = "expected image size"
    else:
        image = Image.new("L", (8, 8))
        message = "expected RGB image"
    with pytest.raises(AssertionError, match=message):
        _extract_image_tensor([SimpleNamespace(images=[image])], expected_count=1, expected_size=(8, 8))


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


@_INT8_CUDA_ONLY
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.usefixtures("int8_cuda_device")
@pytest.mark.diffusion
def test_bf16_vs_int8_t2i_image_consistency(checkpoint_pair: tuple[str, str]):
    """A/B: BF16 vs INT8 t2i — decoded image must match (gen experts + head OK)."""
    bf16_model, int8_model = checkpoint_pair
    bf16_img = _generate_t2i_image(bf16_model, None, bf16_model)
    int8_img = _generate_t2i_image(int8_model, INT8_QUANTIZATION, bf16_model)

    assert bf16_img.shape == int8_img.shape, (bf16_img.shape, int8_img.shape)

    metrics = _image_metrics(bf16_img, int8_img)

    print("\n[INT8 t2i A/B] BF16 vs INT8 decoded image:")
    for name in ("psnr_db", "mae", "max_abs", "cosine", "rel_l2", "pixel_match"):
        print(f"  {name:12s} = {metrics[name]:.6f}")

    # Compare decoded images under the same seed and greedy AR sampling.
    # INT8 can change visual tokens, so exact token/pixel identity is not gated.
    assert metrics["cosine"] >= MIN_IMAGE_COSINE, (
        f"BF16/INT8 t2i images diverge too much: cosine={metrics['cosine']:.6f} < {MIN_IMAGE_COSINE}"
    )
    assert metrics["rel_l2"] < MAX_IMAGE_REL_L2, (
        f"BF16/INT8 t2i images diverge too much: rel_l2={metrics['rel_l2']:.6f} >= {MAX_IMAGE_REL_L2}"
    )


# ---------------------------------------------------------------------------
# Checkpoint loading + quantization scales (no GPU required)
# ---------------------------------------------------------------------------


def _assert_model_version(config: dict, version: str) -> None:
    assert "mammoth" in config["model_type"].lower(), "expected a MammothModa2 checkpoint"
    model_type = config["llm_config"]["model_type"].lower()
    assert ("qwen3" in model_type) == (version == "dev"), f"{version}: unexpected AR architecture {model_type!r}"


def _tensor_metadata(ckpt_dir: Path, weight_map: dict[str, str]) -> dict[str, tuple[str, list[int]]]:
    """Read dtype/shape from shard headers without materializing model weights."""
    from safetensors import safe_open

    metadata = {}
    shards: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        shards.setdefault(shard, []).append(key)
    for shard, keys in shards.items():
        with safe_open(str(ckpt_dir / shard), framework="pt") as handle:
            for key in keys:
                tensor = handle.get_slice(key)
                metadata[key] = (tensor.get_dtype(), tensor.get_shape())
    return metadata


def _matches_module(module: str, target: str) -> bool:
    return re.match(target[3:], module) is not None if target.startswith("re:") else module == target


def _quantized_weight_keys(quant_config: dict, metadata: dict[str, tuple[str, list[int]]]) -> set[str]:
    """Match checkpoint targets/ignore entries against MammothModa2 AR weights.

    Quantization applies only to ``llm_model.*``, matching the AR loader's scope.
    The checkpoint also stores the BF16 DiT and image tokenizer; their Linear
    modules are not targets of the AR recipe. Within the AR stage, embedding
    leaves and output heads are excluded from the Linear class target. Exact
    names and ``re:`` patterns are supported alongside ``Linear`` and
    ``ParallelLMHead``. Unknown classes fail rather than losing coverage.
    """
    embedding_leaves = {
        "embed_tokens",
        "gen_embed_tokens",
        "position_embedding",
        "position_embeddings",
        "pos_embed",
        "embed_positions",
        "word_embeddings",
        "token_embedding",
        "embedding",
        "embeddings",
    }
    # Both output heads are ParallelLMHead in Mammoth2ForCausalLM.
    lm_head_modules = {"llm_model.lm_head", "llm_model.gen_head"}
    targets = [target for group in quant_config["config_groups"].values() for target in group["targets"]]
    assert targets, "no quantization targets declared"
    for target in targets:
        assert target in {"Linear", "ParallelLMHead"} or target.startswith("re:") or "." in target, (
            f"unsupported quantization target class {target!r}"
        )
    quantized = set()
    for key, (_, shape) in metadata.items():
        if not key.startswith("llm_model.") or not key.endswith(".weight"):
            continue
        module = key.removesuffix(".weight")
        if any(_matches_module(module, ignored) for ignored in quant_config.get("ignore", [])):
            continue
        is_lm_head = module in lm_head_modules
        is_linear = len(shape) == 2 and module.rsplit(".", 1)[-1] not in embedding_leaves and not is_lm_head
        if any(
            (target == "Linear" and is_linear)
            or (target == "ParallelLMHead" and is_lm_head)
            or _matches_module(module, target)
            for target in targets
        ):
            quantized.add(key)
    assert quantized, "no checkpoint weights match the declared quantization targets"
    return quantized


@pytest.mark.cpu
@pytest.mark.parametrize(
    "targets,ignore,expected_modules",
    [
        (["Linear"], [], {"llm_model.model.language_model.layers.0.mlp.gate_proj"}),
        (["ParallelLMHead"], [], {"llm_model.lm_head", "llm_model.gen_head"}),
        (["ParallelLMHead"], ["llm_model.gen_head"], {"llm_model.lm_head"}),
        (["re:llm_model\\..*_head"], [], {"llm_model.lm_head", "llm_model.gen_head"}),
    ],
)
def test_quantization_target_classes(targets: list[str], ignore: list[str], expected_modules: set[str]):
    # Target discovery must not depend on the saved dtype: a declared target
    # incorrectly saved as BF16 still needs to reach the dtype/scale checks.
    metadata = {
        "llm_model.model.language_model.layers.0.mlp.gate_proj.weight": ("BF16", [16, 8]),
        "llm_model.model.language_model.embed_tokens.weight": ("BF16", [32, 8]),
        "llm_model.lm_head.weight": ("BF16", [32, 8]),
        "llm_model.gen_head.weight": ("BF16", [32, 8]),
        # These matrix-shaped weights belong to other stages, so the AR
        # recipe's Linear class target must not require INT8 weights/scales.
        "gen_transformer.time_caption_embed.image_embedder.layers.0.cross_attn.out_proj.weight": (
            "BF16",
            [16, 8],
        ),
        "gen_tokenizer.image_tokenizer.decoder.adaptive.0.gamma.weight": ("BF16", [16, 8]),
    }
    config = {"config_groups": {"group_0": {"targets": targets}}, "ignore": ignore}
    assert _quantized_weight_keys(config, metadata) == {f"{module}.weight" for module in expected_modules}


def _read_tensor(ckpt_dir, weight_map: dict[str, str], key: str) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(str(ckpt_dir / weight_map[key]), framework="pt") as handle:
        return handle.get_tensor(key)


def _assert_int8_weight_and_scale(
    ckpt_dir: Path,
    weight_map: dict[str, str],
    metadata: dict[str, tuple[str, list[int]]],
    weight_key: str,
    scale_key: str,
) -> None:
    """Quantized weights are int8 and carry a finite, positive per-channel scale."""
    assert weight_key in weight_map, f"missing quantized weight {weight_key}"
    assert scale_key in weight_map, f"missing per-channel scale {scale_key}"
    dtype, weight_shape = metadata[weight_key]
    assert dtype == "I8", f"{weight_key} must be int8, got {dtype}"
    assert len(weight_shape) == 2, f"{weight_key} must be a matrix, got {weight_shape}"

    scale = _read_tensor(ckpt_dir, weight_map, scale_key)
    assert scale.is_floating_point(), f"{scale_key} must have a floating-point dtype, got {scale.dtype}"
    assert tuple(scale.shape) == (weight_shape[0], 1), (
        f"{scale_key} must have shape {(weight_shape[0], 1)}, got {tuple(scale.shape)}"
    )
    assert torch.isfinite(scale).all(), f"{scale_key} contains non-finite values"
    assert (scale > 0).all(), f"{scale_key} contains non-positive values"


@pytest.mark.cpu
@pytest.mark.parametrize("version", _MODEL_VERSIONS)
def test_int8_checkpoint_quantization_scales(version: str):
    """Serialized INT8 W8A8 checkpoint: scheme metadata + per-channel scales.

    Validates that the checkpoint the A/B gate loads is a genuine W8A8
    compressed-tensors checkpoint and that its quantization scales cover exactly
    the layers it claims to quantize: the ignored generation experts stay in
    BF16, and the extra generation head is either BF16 or backed by a valid
    per-channel scale.
    """
    ckpt_dir = _checkpoint_dir(version, INT8_QUANTIZATION)

    config = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
    _assert_model_version(config, version)
    quant_config = config.get("quantization_config")
    assert quant_config is not None, f"{ckpt_dir}/config.json has no quantization_config"
    assert quant_config["quant_method"] == "compressed-tensors"
    assert quant_config["quantization_status"] == "compressed"

    assert quant_config["config_groups"], "no quantization scheme groups declared"
    for name, group in quant_config["config_groups"].items():
        weights, activations = group["weights"], group["input_activations"]
        assert (weights["type"], weights["num_bits"]) == ("int", 8), name
        assert weights["strategy"] == "channel" and weights["symmetric"] is True, name
        assert weights["dynamic"] is False, name
        assert (activations["type"], activations["num_bits"]) == ("int", 8), name
        assert activations["dynamic"] is True and activations["strategy"] == "token", name
        assert activations["symmetric"] is True, name

    weight_map = json.loads((ckpt_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]

    metadata = _tensor_metadata(ckpt_dir, weight_map)
    quantized_keys = _quantized_weight_keys(quant_config, metadata)
    # Validate every declared AR target, including missing scales and weights
    # incorrectly saved in full precision; checking only I8 keys misses those.
    for weight_key in sorted(quantized_keys):
        scale_key = weight_key.removesuffix(".weight") + ".weight_scale"
        _assert_int8_weight_and_scale(ckpt_dir, weight_map, metadata, weight_key, scale_key)
    # Check the reverse direction as well: no undeclared INT8 weight or orphan
    # scale is allowed, even if a target/ignore entry accidentally excluded it.
    int8_keys = {key for key, (dtype, _) in metadata.items() if key.endswith(".weight") and dtype == "I8"}
    assert int8_keys == quantized_keys, f"INT8 weights do not match declared targets: {int8_keys ^ quantized_keys}"
    expected_scales = {key.removesuffix(".weight") + ".weight_scale" for key in quantized_keys}
    actual_scales = {key for key in weight_map if key.endswith(".weight_scale")}
    assert actual_scales == expected_scales, f"scale coverage mismatch: {actual_scales ^ expected_scales}"

    # 2. Generation experts (gen_mlp) are ignored and therefore stay BF16:
    #    weights present, no quantization scale.
    gen_expert_weights = sorted(key for key in weight_map if ".gen_mlp." in key and key.endswith(".weight"))
    assert gen_expert_weights, "no generation-expert weights found in the checkpoint"
    assert not any(".gen_mlp." in key and key.endswith(".weight_scale") for key in weight_map), (
        "generation experts must stay unquantized (listed under quant ignore)"
    )
    for key in gen_expert_weights:
        assert metadata[key][0] == "BF16", f"{key} must remain BF16, got {metadata[key][0]}"

    # 3. Extra vocabulary/head: gen_embed_tokens stays BF16 (embeddings are not
    #    quant targets). gen_head must be either kept in BF16 (the recipe lists
    #    it under ``ignore``) or quantized with a *valid* per-channel scale:
    #    an all-zero scale silently zeroes every visual-token logit and breaks
    #    t2i, so it is rejected explicitly here.
    embed_key = "llm_model.model.language_model.gen_embed_tokens.weight"
    assert embed_key in weight_map
    assert not any("gen_embed_tokens" in key and key.endswith(".weight_scale") for key in weight_map)
    assert metadata[embed_key][0] == "BF16", f"{embed_key} must remain BF16, got {metadata[embed_key][0]}"
    assert "llm_model.gen_head.weight" in weight_map

    gen_head_scale_key = "llm_model.gen_head.weight_scale"
    if gen_head_scale_key in weight_map:
        _assert_int8_weight_and_scale(
            ckpt_dir,
            weight_map,
            metadata,
            "llm_model.gen_head.weight",
            gen_head_scale_key,
        )
    else:
        head_key = "llm_model.gen_head.weight"
        assert metadata[head_key][0] == "BF16", f"{head_key} must remain BF16, got {metadata[head_key][0]}"
