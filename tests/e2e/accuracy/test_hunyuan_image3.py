# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import gc
import importlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import requests
import torch
import yaml
from PIL import Image

from tests.e2e.accuracy.helpers import (
    CLIPScorer,
    SemanticSimilarityScorer,
    assert_similarity,
    compute_image_ssim_psnr,
    download_images,
    model_output_dir,
)
from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniRunner, OmniServer
from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
    build_prompt_tokens,
    resolve_stop_token_ids,
)

# CoT structural markers that MUST appear in AR output text.
# The AR trajectory under think_recaption mode is:
#   <think> ... </think> <recaption> ... <relation_1> ... </relation_1>
#   <relation_2> ... </relation_2> </recaption>
# Test deploy config sets include_stop_str_in_output=True and
# skip_special_tokens=False, so ALL opening and closing markers
# should be present in the detokenized text.
_THINK_CLOSE = "</" + "think>"  # avoid Python slash-t-as-TAB pitfall
_COT_REQUIRED_MARKERS = (
    "<recaption>",
    "</recaption>",
    "<relation_1>",
    "</relation_1>",
    "<relation_2>",
    "</relation_2>",
    _THINK_CLOSE,
)


def _assert_cot_structural_markers(cot_text: str, label: str) -> None:
    """Assert that the AR CoT output contains all required structural tags.

    Guards against regressions where detokenizer stop-token stripping
    or truncation logic loses CoT content.  Under the test deploy config
    (include_stop_str_in_output=True, skip_special_tokens=False),
    ALL markers should be present in the detokenized text.
    """
    missing = [m for m in _COT_REQUIRED_MARKERS if m not in cot_text]
    assert not missing, (
        f"[{label}] CoT text missing required structural markers: {missing}. "
        f"This typically indicates that stop-token stripping or "
        f"_truncate_at_cot_end truncation removed CoT content. "
        f"CoT text length={len(cot_text)}, first 100 chars: {repr(cot_text[:100])}"
    )


os.environ["DIFFUSION_ATTENTION_BACKEND"] = "TORCH_SDPA"
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

pytestmark = [pytest.mark.local_model, pytest.mark.diffusion]

# ============================================================================
# Configurable Parameters
# ============================================================================
# Comma-separated logical CUDA device ids per stage: split visible GPUs (0..n-1), first half -> AR, second -> DiT.


def _default_ar_dit_devices() -> tuple[str, str]:
    """First floor(n/2) logical devices -> AR, rest -> DiT. ``device_count`` respects ``CUDA_VISIBLE_DEVICES``."""
    n = torch.accelerator.device_count()
    if n < 2:
        return "0,1", "2,3"
    split = n // 2
    ar = ",".join(str(i) for i in range(split))
    dit = ",".join(str(i) for i in range(split, n))
    return ar, dit


AR_DEVICES, DIT_DEVICES = _default_ar_dit_devices()

# Model configuration based on HUNYUAN_MODEL_PATH
DISTIL_MODEL_NAME = "tencent/HunyuanImage-3.0-Instruct-Distil"
INSTRUCT_MODEL_NAME = "tencent/HunyuanImage-3.0-Instruct"
MODEL_PATH = os.environ.get("HUNYUAN_MODEL_PATH", DISTIL_MODEL_NAME)

# Determine model type and set parameters accordingly
IS_DISTIL = "distil" in MODEL_PATH.lower()
MODEL_NAME = DISTIL_MODEL_NAME if IS_DISTIL else INSTRUCT_MODEL_NAME
NUM_INFERENCE_STEPS = 8 if IS_DISTIL else 50
GUIDANCE_SCALE = 2.5

# Asset paths based on model type
ASSET_SUBDIR = "hunyuan_image3"
REF_IMAGE_NAME = "hunyuan_image_distill_ref.png" if IS_DISTIL else "hunyuan_image_instruct_ref.png"
COT_REF_NAME = "hunyuan_image_distill_cot_ref.txt" if IS_DISTIL else "hunyuan_image3_instruct_cot_ref.txt"
HUNYUAN_IMAGE_REF_PATH = get_asset_path(f"{ASSET_SUBDIR}/{REF_IMAGE_NAME}")
HUNYUAN_COT_REF_PATH = get_asset_path(f"{ASSET_SUBDIR}/{COT_REF_NAME}")

# ============================================================================
# Constants
# ============================================================================
# Test input
PROMPT = "基于图一的logo，参考图二中冰箱贴的材质，制作一个新的冰箱贴"
TEST_IMAGE_URLS = [
    "https://raw.githubusercontent.com/Tencent-Hunyuan/HunyuanImage-3.0/main/assets/demo_instruct_imgs/input_1_0.png",
    "https://raw.githubusercontent.com/Tencent-Hunyuan/HunyuanImage-3.0/main/assets/demo_instruct_imgs/input_1_1.png",
]
SEED = 42
AR_TP_SIZE = len(AR_DEVICES.split(","))
DIT_TP_SIZE = len(DIT_DEVICES.split(","))

# Precision thresholds organized by model variant and test type
THRESHOLDS = {
    "instruct": {
        # AR text comparison
        "text_prefix_match": 10,  # First 10 characters must match exactly
        "cot_semantic_sim": 0.9,  # Full CoT semantic similarity
        # Image comparison
        "clip_score": 90,  # CLIP image semantic similarity
        "ssim": 0.26,  # Structural similarity
        "psnr": 12.5,  # Peak signal-to-noise ratio (dB)
    },
    # The 8-step distilled model has more structural variance across GPU
    # runtimes; CLIP and PSNR retain the same semantic and pixel safeguards.
    "distil": {
        "text_prefix_match": 10,
        "cot_semantic_sim": 0.9,
        "clip_score": 90,
        "ssim": 0.23,
        "psnr": 12.5,
    },
    # Thresholds for quantized DiT accuracy tests (bf16 vs quant comparison)
    "quant": {
        "psnr": 10.0,
        "ssim": 0.20,
        "clip_score": 20.0,  # Minimum CLIP score for quantized output
        "clip_score_drop": float(
            os.environ.get("HUNYUAN_IMAGE3_QUANT_CLIP_SCORE_DROP_THRESHOLD", "5.0")
        ),  # Max allowed drop from bf16
    },
}


# Helper to get thresholds for current model variant
def _get_thresholds() -> dict:
    """Return appropriate thresholds based on test context."""
    return THRESHOLDS["distil" if IS_DISTIL else "instruct"]


def _get_quant_thresholds() -> dict:
    """Return thresholds for quantization accuracy tests."""
    return THRESHOLDS["quant"]


# Quant test configuration
QUANT_PROMPT = "A brown and white dog is running on the grass."
QUANT_HEIGHT, QUANT_WIDTH = 1024, 1024
QUANT_RUN_ENV = "HUNYUAN_IMAGE3_RUN_QUANT_ACCURACY"
QUANT_BF16_ENV = "HUNYUAN_IMAGE3_BF16_MODEL"
QUANT_FP8_ENV = "HUNYUAN_IMAGE3_FP8_MODEL"
QUANT_NVFP4_ENV = "HUNYUAN_IMAGE3_NVFP4_MODEL"
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
# fmt: off
_DEPLOY_CONFIG = {
    "pipeline": "hunyuan_image_3_moe",
    "async_chunk": False,
    "trust_remote_code": True,
    "connectors": {
        "shared_memory_connector": {
            "name": "SharedMemoryConnector",
        },
    },
    "stages": [
        {
            "stage_id": 0,
            "is_comprehension": False,
            "final_output": True,
            "final_output_type": "text",
            "max_num_seqs": 1,
            "gpu_memory_utilization": 0.95,
            "enforce_eager": True,
            "trust_remote_code": True,
            "max_num_batched_tokens": 32768,
            "devices": AR_DEVICES,
            "tensor_parallel_size": AR_TP_SIZE,
            "hf_overrides": {
                "rope_parameters": {"mrope_section": [0, 32, 32], "rope_type": "default"},
            },
            "output_connectors": {"to_stage_1": "shared_memory_connector"},
            "default_sampling_params": {
                "temperature": 0.0,
                "top_p": 1,
                "top_k": -1,
                "max_tokens": 8192,
                "detokenize": True,
                "skip_special_tokens": False,
                "include_stop_str_in_output": True,
            },
        },
        {
            "stage_id": 1,
            "max_num_seqs": 1,
            "enforce_eager": True,
            "trust_remote_code": True,
            "devices": DIT_DEVICES,
            "distributed_executor_backend": "mp",
            "parallel_config": {"tensor_parallel_size": DIT_TP_SIZE, "enable_expert_parallel": True},
            "input_connectors": {"from_stage_0": "shared_memory_connector"},
            "default_sampling_params": {
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "guidance_scale": GUIDANCE_SCALE,
            },
        },
    ],
    "edges": [{"from": 0, "to": 1}],
}
# fmt: on

_QUANT_DIT_CONFIG = {
    "pipeline": "hunyuan_image_3_moe",
    "async_chunk": False,
    "trust_remote_code": True,
    "stages": [
        {
            "stage_id": 0,
            "model_stage": "dit",
            "enforce_eager": True,
            "trust_remote_code": True,
            "devices": "0,1",
            "distributed_executor_backend": "mp",
            "force_cutlass_fp8": True,
            "moe_backend": "cutlass",
            "parallel_config": {
                "tensor_parallel_size": 2,
                "enable_expert_parallel": True,
            },
            "omni_kv_config": {"need_recv_cache": True},
            "final_output": True,
            "final_output_type": "image",
            "is_comprehension": False,
            "default_sampling_params": {"seed": SEED},
        }
    ],
}


@dataclass(frozen=True)
class _QuantAccuracyCase:
    name: str
    model_env: str
    nvfp4_backend: str | None = None


def _quant_accuracy_cases() -> list[pytest.ParameterSet]:
    cases = [
        _QuantAccuracyCase(name="fp8", model_env=QUANT_FP8_ENV),
        _QuantAccuracyCase(name="mixed_nvfp4", model_env=QUANT_NVFP4_ENV, nvfp4_backend="cutlass"),
    ]
    params: list[pytest.ParameterSet] = []
    run_quant_accuracy = os.environ.get(QUANT_RUN_ENV, "").lower() in _TRUE_ENV_VALUES
    for case in cases:
        marks = []
        if not run_quant_accuracy:
            marks.append(pytest.mark.skip(reason=f"Set {QUANT_RUN_ENV}=1 to run HunyuanImage3 quant accuracy."))
        if not os.environ.get(QUANT_BF16_ENV):
            marks.append(pytest.mark.skip(reason=f"Set {QUANT_BF16_ENV} to run HunyuanImage3 quant accuracy."))
        if not os.environ.get(case.model_env):
            marks.append(pytest.mark.skip(reason=f"Set {case.model_env} to run this quant accuracy case."))
        params.append(pytest.param(case, id=case.name, marks=marks))
    return params


def _make_config(enable_kv_reuse: bool, path: Path) -> None:
    config = copy.deepcopy(_DEPLOY_CONFIG)
    config["stages"][0]["omni_kv_config"] = {"need_send_cache": enable_kv_reuse}
    config["stages"][1]["omni_kv_config"] = {"need_recv_cache": enable_kv_reuse}
    path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))


def _quant_devices() -> str:
    return os.environ.get("HUNYUAN_IMAGE3_QUANT_DEVICES", "0,1")


def _quant_tensor_parallel_size() -> int:
    return int(os.environ.get("HUNYUAN_IMAGE3_QUANT_TP", str(len(_quant_devices().split(",")))))


def _make_quant_dit_config(path: Path) -> None:
    config = copy.deepcopy(_QUANT_DIT_CONFIG)
    config["stages"][0]["devices"] = _quant_devices()
    config["stages"][0]["parallel_config"]["tensor_parallel_size"] = _quant_tensor_parallel_size()
    path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))


def _run_offline(deploy_config_path: str, output_path: Path) -> tuple[Image.Image, str, float]:
    from transformers import AutoTokenizer

    from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType
    from vllm_omni.platforms import current_omni_platform

    build_kwargs: dict = {"task": "it2i", "bot_task": "think_recaption", "sys_type": "en_unified", "num_images": 2}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    result = build_prompt_tokens(
        PROMPT,
        tokenizer,
        **build_kwargs,
    )
    token_ids = result.token_ids
    system_prompt_type = result.system_prompt_type

    ar_stop_token_ids = resolve_stop_token_ids(task="it2i", bot_task="think_recaption", tokenizer=tokenizer)
    with OmniRunner(MODEL_PATH, deploy_config=deploy_config_path, trust_remote_code=True) as runner:
        params_list = list(runner.omni.default_sampling_params_list)
        for sp in params_list:
            if isinstance(sp, OmniDiffusionSamplingParams):
                sp.num_inference_steps = NUM_INFERENCE_STEPS
                sp.guidance_scale = GUIDANCE_SCALE
                sp.seed = SEED
                sp.generator = torch.Generator(device=current_omni_platform.device_type or "cuda").manual_seed(SEED)
            elif hasattr(sp, "stop_token_ids"):
                sp.stop_token_ids = ar_stop_token_ids

        images = download_images(TEST_IMAGE_URLS)
        prompts: list[OmniPromptType] = [
            {
                "prompt_token_ids": token_ids,
                "prompt": PROMPT,
                "use_system_prompt": system_prompt_type,
                "modalities": ["image"],
                "multi_modal_data": {"image": images},
            }
        ]
        t0 = time.perf_counter()
        outputs = list(runner.omni.generate(prompts=prompts, sampling_params_list=params_list))
        elapsed = time.perf_counter() - t0

    assert outputs, "Pipeline produced no outputs"
    images = None
    cot_text = ""
    for out in outputs:
        ro = out
        if ro and getattr(ro, "outputs", None):
            cot_text = "".join(getattr(o, "text", "") or "" for o in ro.outputs)
        if not cot_text:
            multimodal_output = getattr(out, "multimodal_output", {}) or {}
            metadata = multimodal_output.get("metadata", {}) if isinstance(multimodal_output, dict) else {}
            text_metadata = metadata.get("text", {}) if isinstance(metadata, dict) else {}
            ar_text = text_metadata.get("ar_generated_text") if isinstance(text_metadata, dict) else None
            if isinstance(ar_text, list):
                cot_text = "\n".join(text for text in ar_text if text)
            else:
                cot_text = ar_text or ""

        imgs = getattr(out, "images", None)
        if not imgs and ro and hasattr(ro, "images"):
            imgs = ro.images
        if imgs:
            images = imgs

    assert images, "Pipeline output had no images"
    cot_text = cot_text.lstrip("\n")

    image = images[0].convert("RGB")
    image.save(output_path / "image_offline.png")
    (output_path / "cot_offline.txt").write_text(cot_text, encoding="utf-8")
    gc.collect()
    if torch.accelerator.is_available():
        torch.accelerator.empty_cache()
    return image, cot_text, elapsed


def _run_online(deploy_config_path: str, output_path: Path) -> tuple[Image.Image, str, float]:
    from benchmarks.accuracy.common import decode_base64_image, pil_to_png_bytes

    server_args = [
        "--deploy-config",
        deploy_config_path,
        "--stage-init-timeout",
        "300",
        "--init-timeout",
        "900",
        "--trust-remote-code",
    ]
    try:
        with OmniServer(MODEL_PATH, server_args, use_omni=True) as omni_server:
            images = download_images(TEST_IMAGE_URLS)
            t0 = time.perf_counter()
            response = requests.post(
                f"http://{omni_server.host}:{omni_server.port}/v1/images/edits",
                data={
                    "model": omni_server.model,
                    "prompt": PROMPT,
                    "n": 1,
                    "response_format": "b64_json",
                    "num_inference_steps": NUM_INFERENCE_STEPS,
                    "guidance_scale": GUIDANCE_SCALE,
                    "seed": SEED,
                    "sys_type": "en_unified",
                    "bot_task": "think_recaption",
                    "size": "1280x720",
                },
                files=[
                    ("image", (f"image_{i}.png", pil_to_png_bytes(img), "image/png")) for i, img in enumerate(images)
                ],
                timeout=600,
            )
            elapsed = time.perf_counter() - t0
            if not response.ok:
                print(f"[ONLINE] HTTP {response.status_code} response body: {response.text}")
            response.raise_for_status()
            payload = response.json()
            assert len(payload["data"]) == 1
            image = decode_base64_image(payload["data"][0]["b64_json"])
            image.load()
            image.save(output_path / "image_online.png")
            cot_text = payload.get("cot_output") or ""
            (output_path / "cot_online.txt").write_text(cot_text, encoding="utf-8")
            return image, cot_text, elapsed
    finally:
        gc.collect()
        if torch.accelerator.is_available():
            torch.accelerator.empty_cache()


@pytest.mark.skipif(
    torch.accelerator.device_count() < AR_TP_SIZE + DIT_TP_SIZE,
    reason=f"Needs {AR_TP_SIZE + DIT_TP_SIZE}+ GPUs ({AR_TP_SIZE} AR + {DIT_TP_SIZE} DiT)",
)
def test_image_to_image_alignment_online(accuracy_artifact_root: Path) -> None:
    """Online API test: same pipeline, same seed as offline → PSNR >= 10 dB."""
    if importlib.util.find_spec("FlagEmbedding") is None:
        raise ImportError("Missing dependency: FlagEmbedding\nInstall with: pip install FlagEmbedding")
    from tabulate import tabulate

    output_dir = model_output_dir(accuracy_artifact_root, MODEL_NAME + "-online")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        _make_config(True, tmp / "online.yaml")
        online_image, online_cot, _ = _run_online(str(tmp / "online.yaml"), output_dir)

    online_cot = online_cot.lstrip("\n")
    scorer = SemanticSimilarityScorer()
    clip_scorer = CLIPScorer()
    cot_ref = HUNYUAN_COT_REF_PATH.read_text(encoding="utf-8")
    cot_results = scorer.text_similarity(online_cot, cot_ref)
    image_ref = Image.open(HUNYUAN_IMAGE_REF_PATH).convert("RGB")
    image_clip_score = clip_scorer.image_image_score(online_image, image_ref)
    ssim_value, psnr_value = compute_image_ssim_psnr(prediction=online_image, reference=image_ref, compare_mode="RGB")

    thresholds = _get_thresholds()
    table = [
        ["COT similarity to reference", f"{cot_results['cot_semantic_sim']:.4f}", thresholds["cot_semantic_sim"]],
        ["COT prefix match", f"{cot_results['text_prefix_match_count']:.4f}", thresholds["text_prefix_match"]],
        ["Image-Image similarity", f"{image_clip_score:.4f}", thresholds["clip_score"]],
        ["SSIM", f"{ssim_value:.4f}", thresholds["ssim"]],
        ["PSNR (dB)", f"{psnr_value:.2f}", thresholds["psnr"]],
    ]
    print("[ONLINE] " + tabulate(table, headers=["Metric", "Value", "Threshold"], tablefmt="grid"))

    _assert_cot_structural_markers(online_cot, "ONLINE")

    assert cot_results["cot_semantic_sim"] >= thresholds["cot_semantic_sim"], (
        f"[ONLINE] COT semantic similarity {cot_results['cot_semantic_sim']:.4f} below threshold {thresholds['cot_semantic_sim']}"
    )
    assert cot_results["text_prefix_match_count"] >= thresholds["text_prefix_match"], (
        f"[ONLINE] COT prefix match {cot_results['text_prefix_match_count']} below threshold {thresholds['text_prefix_match']}"
    )
    assert image_clip_score >= thresholds["clip_score"], (
        f"[ONLINE] Image-Image similarity {image_clip_score:.4f} below threshold {thresholds['clip_score']}"
    )
    assert ssim_value >= thresholds["ssim"], f"[ONLINE] SSIM {ssim_value:.4f} below threshold {thresholds['ssim']}"
    assert psnr_value >= thresholds["psnr"], (
        f"[ONLINE] PSNR {psnr_value:.2f} dB below threshold {thresholds['psnr']} dB"
    )


def _extract_image(outputs) -> Image.Image:
    assert outputs, "Pipeline produced no outputs"
    for output in outputs:
        images = getattr(output, "images", None)
        request_output = output
        if not images and request_output is not None:
            images = getattr(request_output, "images", None)
        if images:
            image = images[0].convert("RGB")
            image.load()
            return image
    raise AssertionError("Pipeline output had no images")


def _run_dit_model(
    model: str,
    deploy_config_path: str,
    output_path: Path,
    *,
    nvfp4_backend: str | None = None,
) -> tuple[Image.Image, float]:
    from tests.helpers.runtime import OmniRunner
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms import current_omni_platform

    old_backend = os.environ.get("VLLM_NVFP4_GEMM_BACKEND")
    if nvfp4_backend is not None:
        os.environ["VLLM_NVFP4_GEMM_BACKEND"] = nvfp4_backend

    try:
        with OmniRunner(
            model,
            deploy_config=deploy_config_path,
            mode="text-to-image",
            trust_remote_code=True,
        ) as runner:
            generator = torch.Generator(device=current_omni_platform.device_type or "cuda").manual_seed(SEED)
            params = OmniDiffusionSamplingParams(
                height=QUANT_HEIGHT,
                width=QUANT_WIDTH,
                seed=SEED,
                generator=generator,
                num_inference_steps=20,
                guidance_scale=4.0,
                guidance_scale_provided=True,
            )
            t0 = time.perf_counter()
            outputs = runner.omni.generate({"prompt": QUANT_PROMPT}, params)
            elapsed = time.perf_counter() - t0
            image = _extract_image(outputs)
            image.save(output_path)
            return image, elapsed
    finally:
        if nvfp4_backend is not None:
            if old_backend is None:
                os.environ.pop("VLLM_NVFP4_GEMM_BACKEND", None)
            else:
                os.environ["VLLM_NVFP4_GEMM_BACKEND"] = old_backend
        gc.collect()
        if torch.accelerator.is_available():
            torch.accelerator.empty_cache()


@hardware_test(res={"cuda": "H100"}, num_cards=8)
@pytest.mark.skipif(
    torch.accelerator.device_count() < AR_TP_SIZE + DIT_TP_SIZE,
    reason=f"Needs {AR_TP_SIZE + DIT_TP_SIZE}+ GPUs ({AR_TP_SIZE} AR + {DIT_TP_SIZE} DiT)",
)
def test_image_to_image_alignment(accuracy_artifact_root: Path) -> None:
    if importlib.util.find_spec("FlagEmbedding") is None:
        raise ImportError("Missing dependency: FlagEmbedding\nInstall with: pip install FlagEmbedding")
    from tabulate import tabulate  # lazy import

    """KV reuse ON vs OFF: same pipeline, same seed → PSNR >= 10 dB."""
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_NAME + "-offline-kv-reuse")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        _make_config(True, tmp / "on.yaml")
        omni_image, omni_cot, time_reuse = _run_offline(str(tmp / "on.yaml"), output_dir)

    scorer = SemanticSimilarityScorer()
    clip_scorer = CLIPScorer()
    cot_ref = HUNYUAN_COT_REF_PATH.read_text(encoding="utf-8")
    cot_results = scorer.text_similarity(omni_cot, cot_ref)
    image_ref = Image.open(HUNYUAN_IMAGE_REF_PATH).convert("RGB")
    image_clip_score = clip_scorer.image_image_score(omni_image, image_ref)
    ssim_value, psnr_value = compute_image_ssim_psnr(prediction=omni_image, reference=image_ref, compare_mode="RGB")

    thresholds = _get_thresholds()
    table = [
        ["COT similarity to reference", f"{cot_results['cot_semantic_sim']:.4f}", thresholds["cot_semantic_sim"]],
        ["COT prefix match", f"{cot_results['text_prefix_match_count']:.4f}", thresholds["text_prefix_match"]],
        ["Image-Image similarity", f"{image_clip_score:.4f}", thresholds["clip_score"]],
        ["SSIM", f"{ssim_value:.4f}", thresholds["ssim"]],
        ["PSNR (dB)", f"{psnr_value:.2f}", thresholds["psnr"]],
    ]

    print(tabulate(table, headers=["Metric", "Value", "Threshold"], tablefmt="grid"))

    _assert_cot_structural_markers(omni_cot, "OFFLINE")

    assert cot_results["cot_semantic_sim"] >= thresholds["cot_semantic_sim"], (
        f"COT semantic similarity {cot_results['cot_semantic_sim']:.4f} is below threshold {thresholds['cot_semantic_sim']}"
    )
    assert cot_results["text_prefix_match_count"] >= thresholds["text_prefix_match"], (
        f"COT prefix match count {cot_results['text_prefix_match_count']} is below threshold {thresholds['text_prefix_match']}"
    )
    assert image_clip_score >= thresholds["clip_score"], (
        f"Image-Image similarity{image_clip_score:.4f} is below threshold {thresholds['clip_score']}"
    )
    assert ssim_value >= thresholds["ssim"], f"SSIM {ssim_value:.4f} is below threshold {thresholds['ssim']}"
    assert psnr_value >= thresholds["psnr"], f"PSNR {psnr_value:.2f} dB is below threshold {thresholds['psnr']} dB"


@pytest.mark.parametrize("case", _quant_accuracy_cases())
@pytest.mark.skipif(torch.accelerator.device_count() < 2, reason="Needs 2+ GPUs for HunyuanImage3 DiT")
def test_quantized_dit_matches_bf16_accuracy(
    case: _QuantAccuracyCase,
    accuracy_artifact_root: Path,
) -> None:
    """Quantized DiT checkpoints should preserve prompt-aligned image quality."""
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_NAME + "-quant")
    bf16_model = os.environ[QUANT_BF16_ENV]
    quant_model = os.environ[case.model_env]

    with tempfile.TemporaryDirectory() as tmpdir:
        deploy_config_path = Path(tmpdir) / "hunyuan_image3_quant_dit.yaml"
        _make_quant_dit_config(deploy_config_path)

        bf16_image, bf16_time = _run_dit_model(
            bf16_model,
            str(deploy_config_path),
            output_dir / "bf16.png",
        )
        quant_image, quant_time = _run_dit_model(
            quant_model,
            str(deploy_config_path),
            output_dir / f"{case.name}.png",
            nvfp4_backend=case.nvfp4_backend,
        )

    ssim_score, psnr_score = compute_image_ssim_psnr(
        prediction=quant_image,
        reference=bf16_image,
    )
    quant_thresholds = _get_quant_thresholds()
    assert_similarity(
        model_name=f"{MODEL_NAME} {case.name} vs bf16",
        vllm_image=quant_image,
        diffusers_image=bf16_image,
        ssim_threshold=quant_thresholds["ssim"],
        psnr_threshold=quant_thresholds["psnr"],
        width=QUANT_WIDTH,
        height=QUANT_HEIGHT,
    )

    clip_scorer = CLIPScorer()
    bf16_clip_score = clip_scorer.score(bf16_image, QUANT_PROMPT)
    quant_clip_score = clip_scorer.score(quant_image, QUANT_PROMPT)
    clip_score_drop = bf16_clip_score - quant_clip_score

    metrics = {
        "case": case.name,
        "bf16_model": bf16_model,
        "quant_model": quant_model,
        "prompt": QUANT_PROMPT,
        "seed": SEED,
        "height": QUANT_HEIGHT,
        "width": QUANT_WIDTH,
        "num_inference_steps": 20,
        "guidance_scale": 4.0,
        "bf16_elapsed_s": bf16_time,
        "quant_elapsed_s": quant_time,
        "ssim": ssim_score,
        "psnr": psnr_score,
        "bf16_clip_score": bf16_clip_score,
        "quant_clip_score": quant_clip_score,
        "clip_score_drop": clip_score_drop,
    }
    metrics_path = output_dir / f"{case.name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\nHunyuanImage3 quant accuracy ({case.name})")
    print(f"  bf16 model:       {bf16_model}")
    print(f"  quant model:      {quant_model}")
    print(f"  BF16 CLIP score:  {bf16_clip_score:.4f}")
    print(f"  quant CLIP score: {quant_clip_score:.4f} threshold>={quant_thresholds['clip_score']:.4f}")
    print(f"  CLIP score drop:  {clip_score_drop:.4f} threshold<={quant_thresholds['clip_score_drop']:.4f}")
    print(f"  metrics:          {metrics_path}")

    assert quant_clip_score >= quant_thresholds["clip_score"], (
        f"{case.name} CLIP score below threshold: got {quant_clip_score:.4f}, "
        f"expected >= {quant_thresholds['clip_score']:.4f}"
    )
    assert clip_score_drop <= quant_thresholds["clip_score_drop"], (
        f"{case.name} CLIP score drop too large: got {clip_score_drop:.4f}, "
        f"expected <= {quant_thresholds['clip_score_drop']:.4f}"
    )
