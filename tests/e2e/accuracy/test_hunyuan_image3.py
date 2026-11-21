# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from PIL import Image

from tests.e2e.accuracy.helpers import (
    CLIPScorer,
    SemanticSimilarityScorer,
    SimilarityThresholds,
    assert_similarity_metrics,
    baseline_from_similarity_thresholds,
    download_images,
    measure_image_similarity,
    model_output_dir,
    resolve_device_profile,
    resolve_device_threshold,
    resolve_similarity_thresholds,
    write_accuracy_result,
)
from tests.helpers.client import OnlineOmniClient
from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniRunner, OmniServer, OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
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


pytestmark = [pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def _accuracy_env(monkeypatch):
    monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "DEBUG")


# ============================================================================
# Configurable Parameters
# ============================================================================
# Fixed four-card CI topology: AR TP=2 + DiT TP=2.
AR_DEVICES, DIT_DEVICES = "0,1", "2,3"

# Use the distilled checkpoint without quantization for accuracy comparisons.
MODEL_NAME = "tencent/HunyuanImage-3.0-Instruct-Distil"
MODEL_PATH = os.environ.get("HUNYUAN_MODEL_PATH", MODEL_NAME)
NUM_INFERENCE_STEPS = 8
GUIDANCE_SCALE = 2.5

ASSET_SUBDIR = "hunyuan_image3"
HUNYUAN_IMAGE_REF_PATH = get_asset_path(f"{ASSET_SUBDIR}/hunyuan_image_distill_ref.png")
# Official Distil direct-DiT IT2I references (CUDA vs NPU).
HUNYUAN_DIT_IMAGE_REF_PATH = get_asset_path(f"{ASSET_SUBDIR}/hunyuan_image3_dit_distill_it2i_ref.png")
HUNYUAN_DIT_IMAGE_REF_PATH_NPU = get_asset_path(f"{ASSET_SUBDIR}/hunyuan_image3_dit_distill_it2i_ref_npu.png")
HUNYUAN_COT_REF_PATH = get_asset_path(f"{ASSET_SUBDIR}/hunyuan_image_distill_cot_ref.txt")

# ============================================================================
# Constants
# ============================================================================
# Test input
_IT2I_INPUT = json.loads(get_asset_path(f"{ASSET_SUBDIR}/it2i.jsonl").read_text(encoding="utf-8"))
PROMPT = _IT2I_INPUT["prompt"]
TEST_IMAGE_URLS = _IT2I_INPUT["image_files"]
SEED = 42
IMAGE_WIDTH, IMAGE_HEIGHT = 1280, 720
IMAGE_SIZE = f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}"
AR_TP_SIZE = len(AR_DEVICES.split(","))
DIT_TP_SIZE = len(DIT_DEVICES.split(","))

# Calibrated device-specific rows can override the existing distilled defaults.
SIMILARITY_THRESHOLDS_BY_DEVICE: dict[str, SimilarityThresholds] = {
    "default": SimilarityThresholds(ssim=0.23, psnr=12.5),
}
COT_SEMANTIC_THRESHOLDS: dict[str, float] = {"default": 0.9}
COT_PREFIX_THRESHOLDS: dict[str, float] = {"default": 10}
IMAGE_SIMILARITY_THRESHOLDS: dict[str, float] = {"default": 90}


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
            # Leave headroom for the input images' VAE encoding.
            "gpu_memory_utilization": 0.85,
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


def _make_ar_dit_config() -> str:
    config = cast(dict[str, Any], copy.deepcopy(_DEPLOY_CONFIG))
    config["stages"][0]["omni_kv_config"] = {"need_send_cache": True}
    config["stages"][1]["omni_kv_config"] = {"need_recv_cache": True}
    config["stages"] = {stage.pop("stage_id"): stage for stage in config["stages"]}
    return modify_stage_config(
        get_deploy_config_path("hunyuan_image_3_moe.yaml"),
        updates=config,
        deletes={"platforms": None},
    )


_AR_DIT_DEPLOY = _make_ar_dit_config()
_AR_DIT_SERVER_PARAMS = OmniServerParams(
    model=MODEL_PATH,
    stage_config_path=_AR_DIT_DEPLOY,
    server_args=["--trust-remote-code", "--dtype", "bfloat16"],
    env_dict={"DIFFUSION_ATTENTION_BACKEND": "TORCH_SDPA"},
    stage_init_timeout=900,
    init_timeout=1800,
)
_AR_DIT_RUNNER_PARAMS = (
    MODEL_PATH,
    _AR_DIT_DEPLOY,
    {"trust_remote_code": True, "dtype": "bfloat16", "stage_init_timeout": 900, "init_timeout": 1800},
)


# DiT-only: CUDA vs hunyuan_image3_dit_distill_it2i_ref.png;
# NPU vs hunyuan_image3_dit_distill_it2i_ref_npu.png.
# Acceptance floors, not calibration means. CUDA five-run BF16 calibration:
# CUDA/H200: https://github.com/vllm-project/vllm-omni/pull/10408#issuecomment-6940430119
# A3: two BF16 runs on 2026-11-19 (21:24:18 and 21:31:07) both yielded
# SSIM 0.939406 / PSNR 26.196703 dB; floors allow for run-to-run variation.
DIT_SIMILARITY_THRESHOLDS_BY_DEVICE: dict[str, SimilarityThresholds] = {
    "default": SimilarityThresholds(ssim=0.95, psnr=29.0),
    "A3": SimilarityThresholds(ssim=0.92, psnr=25.0),
}


def _dit_it2i_reference_path() -> Path:
    from vllm_omni.platforms import current_omni_platform

    if current_omni_platform.is_npu():
        return HUNYUAN_DIT_IMAGE_REF_PATH_NPU
    return HUNYUAN_DIT_IMAGE_REF_PATH


_DIT_SERVER_PARAMS = OmniServerParams(
    model=os.environ.get("HUNYUAN_IMAGE3_MODEL", MODEL_NAME),
    stage_config_path=get_deploy_config_path("hunyuan_image3_dit.yaml"),
    server_args=["--trust-remote-code", "--dtype", "bfloat16"],
    env_dict={"DIFFUSION_ATTENTION_BACKEND": "TORCH_SDPA"},
    stage_init_timeout=900,
    init_timeout=1800,
)


def _run_offline(runner: OmniRunner, output_path: Path) -> tuple[Image.Image, str]:
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

    # Match online fixed-size generation: stop at </recaption> before AR predicts a ratio.
    ar_stop_token_ids = resolve_stop_token_ids(
        task="it2i", bot_task="think_recaption", tokenizer=tokenizer, image_size=IMAGE_SIZE
    )
    params_list = list(runner.omni.default_sampling_params_list)
    for sp in params_list:
        if isinstance(sp, OmniDiffusionSamplingParams):
            sp.width = IMAGE_WIDTH
            sp.height = IMAGE_HEIGHT
            sp.num_inference_steps = NUM_INFERENCE_STEPS
            sp.guidance_scale = GUIDANCE_SCALE
            sp.seed = SEED
            sp.generator = torch.Generator(device=current_omni_platform.device_type or "cuda").manual_seed(SEED)
        elif hasattr(sp, "stop_token_ids"):
            sp.stop_token_ids = ar_stop_token_ids
            sp.seed = SEED
            sp.extra_args = {**(sp.extra_args or {}), "target_h": IMAGE_HEIGHT, "target_w": IMAGE_WIDTH}

    images = download_images(TEST_IMAGE_URLS)
    prompts: list[OmniPromptType] = [
        {
            "prompt_token_ids": token_ids,
            "prompt": PROMPT,
            "use_system_prompt": system_prompt_type,
            "modalities": ["image"],
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "multi_modal_data": {"image": images},
        }
    ]
    outputs = list(runner.omni.generate(prompts=prompts, sampling_params_list=params_list))

    assert outputs, "Pipeline produced no outputs"
    output_images = None
    cot_text = ""
    for out in outputs:
        if out and getattr(out, "outputs", None):
            cot_text = "".join(getattr(o, "text", "") or "" for o in out.outputs)
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
        if imgs:
            output_images = imgs

    assert output_images, "Pipeline output had no images"
    cot_text = cot_text.lstrip("\n")

    image = output_images[0].convert("RGB")
    image.save(output_path / "image_offline.png")
    (output_path / "cot_offline.txt").write_text(cot_text, encoding="utf-8")
    return image, cot_text


def _run_online(
    omni_server: OmniServer, online_client: OnlineOmniClient, output_path: Path, *, bot_task: str
) -> tuple[Image.Image, str]:
    from benchmarks.accuracy.common import decode_base64_image, pil_to_png_bytes

    images = download_images(TEST_IMAGE_URLS)
    response = online_client.send_images_edits_http_request(
        {
            "data": {
                "model": omni_server.model,
                "prompt": PROMPT,
                "n": 1,
                "response_format": "b64_json",
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "guidance_scale": GUIDANCE_SCALE,
                "seed": SEED,
                "sys_type": "en_unified",
                "bot_task": bot_task,
                "size": IMAGE_SIZE,
            },
            "files": [
                ("image", (f"image_{i}.png", pil_to_png_bytes(img), "image/png")) for i, img in enumerate(images)
            ],
            "timeout": 600,
        }
    )[0]
    assert response.status_code == 200, response.error_message
    payload = response.json_body
    assert isinstance(payload, dict)
    assert len(payload["data"]) == 1
    image = decode_base64_image(payload["data"][0]["b64_json"])
    image.load()
    image.save(output_path / "image_online.png")
    cot_text = (payload.get("cot_output") or "").lstrip("\n")
    (output_path / "cot_online.txt").write_text(cot_text, encoding="utf-8")
    return image, cot_text


def _assert_accuracy(
    image: Image.Image,
    cot: str | None,
    label: str,
    similarity_thresholds: dict[str, SimilarityThresholds],
    *,
    reference_path: Path | None = None,
) -> None:
    ref_path = reference_path or HUNYUAN_IMAGE_REF_PATH
    device_profile = resolve_device_profile(profiles=similarity_thresholds)
    thresholds = resolve_similarity_thresholds(similarity_thresholds, device_profile)
    with Image.open(ref_path) as ref:
        assert ref.size == (IMAGE_WIDTH, IMAGE_HEIGHT), f"IT2I reference must be {IMAGE_SIZE}"
        reference_image = ref.convert("RGB")
    model_label = f"{MODEL_NAME} [{label}] vs reference"
    ssim, psnr = measure_image_similarity(
        model_name=model_label,
        vllm_image=image,
        diffusers_image=reference_image,
        compare_mode="RGB",
    )
    metrics = {"ssim": ssim, "psnr": psnr}
    resolved = {"ssim": thresholds.ssim, "psnr": thresholds.psnr}
    baseline = baseline_from_similarity_thresholds(similarity_thresholds)
    metric_profiles: dict[str, str] = {}

    # Only AR+DiT emits CoT; DiT-only checks the two image quality metrics.
    if cot is not None:
        if importlib.util.find_spec("FlagEmbedding") is None:
            raise ImportError("Missing dependency: FlagEmbedding\nInstall with: pip install FlagEmbedding")
        cot_ref = HUNYUAN_COT_REF_PATH.read_text(encoding="utf-8")
        cot_results = SemanticSimilarityScorer().text_similarity(cot, cot_ref)
        metrics.update(
            {
                "clip_score": CLIPScorer().image_image_score(image, reference_image),
                "cot_semantic_sim": cot_results["cot_semantic_sim"],
                "text_prefix_match_count": cot_results["text_prefix_match_count"],
            }
        )
        for metric, table in (
            ("cot_semantic_sim", COT_SEMANTIC_THRESHOLDS),
            ("text_prefix_match_count", COT_PREFIX_THRESHOLDS),
            ("clip_score", IMAGE_SIMILARITY_THRESHOLDS),
        ):
            profile, threshold = resolve_device_threshold(table, label=metric)
            metric_profiles[metric] = profile
            resolved[metric] = threshold
            for profile, value in table.items():
                baseline.setdefault(profile, {})[metric] = value

    write_accuracy_result(
        model_id=MODEL_NAME,
        dataset="image_similarity",
        metrics=metrics,
        baseline=baseline,
        extra={
            "label": label,
            "reference": str(ref_path),
            "quantization": "disabled",
            "device_profile": device_profile,
            "metric_profiles": metric_profiles,
        },
    )
    for name, value in metrics.items():
        print(f"[{label}] {name}: {value:.6f} (threshold >= {resolved[name]})")
    if cot is not None:
        _assert_cot_structural_markers(cot, label)
        for metric in ("clip_score", "cot_semantic_sim", "text_prefix_match_count"):
            assert metrics[metric] >= resolved[metric], (
                f"[{label}] {metric} {metrics[metric]:.4f} below threshold {resolved[metric]}"
            )
    assert_similarity_metrics(
        model_name=model_label,
        ssim_score=ssim,
        psnr_score=psnr,
        ssim_threshold=thresholds.ssim,
        psnr_threshold=thresholds.psnr,
    )


@hardware_test(res={"cuda": ["H100", "B200"], "npu": "A3"}, num_cards=4)
@pytest.mark.parametrize("omni_server_function", [_AR_DIT_SERVER_PARAMS], indirect=True)
@pytest.mark.local_model
def test_hunyuan_image3_ar_tp2_dit_tp2_it2i_online(
    accuracy_artifact_root: Path, omni_server_function: OmniServer, online_client_function: OnlineOmniClient
) -> None:
    """AR TP=2 + DiT TP=2: online image and CoT quality against the references."""
    label = f"ar_tp{AR_TP_SIZE}_dit_tp{DIT_TP_SIZE}_it2i_online"
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_NAME + "-" + label)
    image, cot = _run_online(omni_server_function, online_client_function, output_dir, bot_task="think_recaption")
    _assert_accuracy(image, cot, label, SIMILARITY_THRESHOLDS_BY_DEVICE)


@hardware_test(res={"cuda": ["H100", "B200"], "npu": "A3"}, num_cards=4)
@pytest.mark.parametrize("omni_runner_function", [_AR_DIT_RUNNER_PARAMS], indirect=True)
@pytest.mark.local_model
def test_hunyuan_image3_ar_tp2_dit_tp2_it2i_offline(
    accuracy_artifact_root: Path, omni_runner_function: OmniRunner
) -> None:
    """AR TP=2 + DiT TP=2: offline image and CoT quality against the references."""
    label = f"ar_tp{AR_TP_SIZE}_dit_tp{DIT_TP_SIZE}_it2i_offline"
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_NAME + "-" + label)
    image, cot = _run_offline(omni_runner_function, output_dir)
    _assert_accuracy(image, cot, label, SIMILARITY_THRESHOLDS_BY_DEVICE)


@hardware_test(res={"cuda": ["H100", "H200", "B200"], "npu": "A3"}, num_cards=4)
@pytest.mark.parametrize("omni_server_function", [_DIT_SERVER_PARAMS], indirect=True)
@pytest.mark.full_model
def test_hunyuan_image3_dit_tp4_it2i_online(
    accuracy_artifact_root: Path, omni_server_function: OmniServer, online_client_function: OnlineOmniClient
) -> None:
    """DiT TP=4: image quality against the official Distil direct-DiT IT2I reference."""
    label = "dit_tp4_it2i_online"
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_NAME + "-" + label)
    image, _ = _run_online(omni_server_function, online_client_function, output_dir, bot_task="none")
    _assert_accuracy(
        image,
        None,
        label,
        DIT_SIMILARITY_THRESHOLDS_BY_DEVICE,
        reference_path=_dit_it2i_reference_path(),
    )
