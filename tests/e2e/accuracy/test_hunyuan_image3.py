# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import gc
import importlib
import json
import logging
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
from tests.helpers.runtime import OmniRunner, OmniServer
from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import build_prompt_tokens, resolve_stop_token_ids

os.environ["DIFFUSION_ATTENTION_BACKEND"] = "TORCH_SDPA"
logger = logging.getLogger(__name__)

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


def _empty_accelerator_cache() -> None:
    from vllm_omni.platforms import current_omni_platform

    if not current_omni_platform.is_cpu():
        current_omni_platform.empty_cache()


AR_DEVICES, DIT_DEVICES = _default_ar_dit_devices()
MODEL_NAME = "tencent/HunyuanImage-3.0-Instruct"
NPU_MODEL_NAME = "tencent/HunyuanImage-3.0-Instruct-Distil"
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 2.5
NPU_NUM_INFERENCE_STEPS = 8
NPU_GUIDANCE_SCALE = 1.0

# ============================================================================
# Constants
# ============================================================================
MODEL_PATH = os.environ.get("HUNYUAN_MODEL_PATH", MODEL_NAME)
NPU_MODEL_PATH = os.environ.get("HUNYUAN_NPU_MODEL_PATH", NPU_MODEL_NAME)
# Test input
PROMPT = "基于图一的logo，参考图二中冰箱贴的材质，制作一个新的冰箱贴"
TEST_IMAGE_URLS = [
    "https://raw.githubusercontent.com/Tencent-Hunyuan/HunyuanImage-3.0/main/assets/demo_instruct_imgs/input_1_0.png",
    "https://raw.githubusercontent.com/Tencent-Hunyuan/HunyuanImage-3.0/main/assets/demo_instruct_imgs/input_1_1.png",
]
SEED = 42
AR_TP_SIZE = len(AR_DEVICES.split(","))
DIT_TP_SIZE = len(DIT_DEVICES.split(","))

# Precision thresholds
THRESHOLDS = {
    # AR text comparison
    "text_prefix_match": 10,  # First 10 characters must match exactly
    "cot_semantic_sim": 0.9,  # Full CoT semantic similarity
    # Image comparison
    "clip_score": 90,  # CLIP image semantic similarity
    "ssim": 0.26,  # Structural similarity
    "psnr": 12.5,  # Peak signal-to-noise ratio (dB)
}
NPU_THRESHOLDS = {
    **THRESHOLDS,
}

QUANT_PROMPT = "A brown and white dog is running on the grass."
QUANT_HEIGHT, QUANT_WIDTH = 1024, 1024
QUANT_PSNR_THRESHOLD = 10.0
QUANT_SSIM_THRESHOLD = 0.20
QUANT_CLIP_SCORE_THRESHOLD = 20.0
QUANT_CLIP_SCORE_DROP_THRESHOLD = float(os.environ.get("HUNYUAN_IMAGE3_QUANT_CLIP_SCORE_DROP_THRESHOLD", "5.0"))
QUANT_RUN_ENV = "HUNYUAN_IMAGE3_RUN_QUANT_ACCURACY"
QUANT_BF16_ENV = "HUNYUAN_IMAGE3_BF16_MODEL"
QUANT_FP8_ENV = "HUNYUAN_IMAGE3_FP8_MODEL"
QUANT_NVFP4_ENV = "HUNYUAN_IMAGE3_NVFP4_MODEL"
NPU_DIT_BF16_MODEL = os.environ.get("HUNYUAN_IMAGE3_NPU_DIT_BF16_MODEL", "tencent/HunyuanImage-3.0-Instruct-Distil")
NPU_DIT_QUANT_MODEL_ENV = "HUNYUAN_IMAGE3_NPU_DIT_QUANT_MODEL"
NPU_DIT_TP_ENV = "HUNYUAN_IMAGE3_NPU_DIT_TP"
NPU_DIT_EP_ENV = "HUNYUAN_IMAGE3_NPU_DIT_ENABLE_EP"
NPU_DIT_GPU_MEMORY_UTILIZATION_ENV = "HUNYUAN_IMAGE3_NPU_DIT_GPU_MEMORY_UTILIZATION"
NPU_DIT_NUM_INFERENCE_STEPS = 8
NPU_DIT_GUIDANCE_SCALE = 1.0
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
# fmt: off
_DEPLOY_CONFIG = {
    "pipeline": "hunyuan_image_3_moe",
    "async_chunk": False,
    "trust_remote_code": True,
    "connectors": {
        "shared_memory_connector": {
            "name": "SharedMemoryConnector",
            "extra": {"shm_threshold_bytes": 65536},
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
                "stop_token_ids": [128025],
                "detokenize": True,
                "skip_special_tokens": False,
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

# fmt: off
_NPU_DEPLOY_CONFIG = {
    "pipeline": "hunyuan_image_3_moe",
    "async_chunk": False,
    "trust_remote_code": True,
    "connectors": {
        "shared_memory_connector": {
            "name": "SharedMemoryConnector",
            "extra": {"shm_threshold_bytes": 65536},
        },
    },
    "stages": [
        {
            "stage_id": 0,
            "is_comprehension": True,
            "final_output": True,
            "final_output_type": "text",
            "max_num_seqs": 1,
            "gpu_memory_utilization": 0.8,
            "enforce_eager": True,
            "trust_remote_code": True,
            "enable_prefix_caching": False,
            "max_num_batched_tokens": 8192,
            "devices": "0,1,2,3",
            "tensor_parallel_size": 4,
            "hf_overrides": {
                "rope_parameters": {"mrope_section": [0, 32, 32], "rope_type": "default"},
            },
            "omni_kv_config": {"need_send_cache": True},
            "output_connectors": {"to_stage_1": "shared_memory_connector"},
            "default_sampling_params": {
                "temperature": 0.0,
                "top_p": 1,
                "top_k": -1,
                "max_tokens": 8192,
                "stop_token_ids": [128025],
                "detokenize": True,
                "skip_special_tokens": False,
            },
        },
        {
            "stage_id": 1,
            "max_num_seqs": 1,
            "gpu_memory_utilization": 0.65,
            "enforce_eager": True,
            "trust_remote_code": True,
            "devices": "4,5,6,7",
            "distributed_executor_backend": "mp",
            "max_num_batched_tokens": 8192,
            "omni_kv_config": {"need_recv_cache": True},
            "parallel_config": {
                "tensor_parallel_size": 4,
                "enable_expert_parallel": False,
                "sequence_parallel_size": 1,
                "ulysses_degree": 1,
            },
            "input_connectors": {"from_stage_0": "shared_memory_connector"},
            "default_sampling_params": {
                "num_inference_steps": NPU_NUM_INFERENCE_STEPS,
                "guidance_scale": NPU_GUIDANCE_SCALE,
            },
        },
    ],
    "edges": [{"from": 0, "to": 1, "window_size": -1, "max_inflight": 1}],
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

_NPU_DIT_CONFIG = {
    "pipeline": "hunyuan_image3_dit",
    "async_chunk": False,
    "trust_remote_code": True,
    "stages": [
        {
            "stage_id": 0,
            "gpu_memory_utilization": 0.65,
            "enforce_eager": True,
            "trust_remote_code": True,
            "devices": "0,1,2,3",
            "distributed_executor_backend": "mp",
            "max_num_batched_tokens": 32768,
            "parallel_config": {
                "tensor_parallel_size": 4,
                "enable_expert_parallel": True,
                "sequence_parallel_size": 1,
                "ulysses_degree": 1,
            },
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


# fmt: off
COT_REF = ("首先，我分析所有输入图像：图像1是一个圆形的logo，设计现代且抽象。它由不同色调的蓝色（深蓝、中蓝、浅蓝）和白色构成，这些色块以流畅的曲线相互交织，形成一个动态的、类似旋涡或波浪的图案。整个logo是扁平化的矢量图形，背景为纯黑色。图像2展示了四个并排摆放的卡通动物造型冰箱贴，"
           "它们被放置在灰色的织物背景上。这些冰箱贴的关键特征是其材质：它们具有光滑、高光的珐琅或烤漆质感，边缘有明显的金属包边，整体呈现出一种立体的、有厚度的实体感。用户的指令是“基于图一的logo，参考图二中冰箱贴的材质，制作一个新的冰箱贴”。这个指令要求将一个二维的平面设计（logo）"
           "转化为一个具有特定物理属性（材质和立体感）的三维物体。核心任务是保留logo的视觉识别性，同时赋予其冰箱贴的实体质感。为了构建答案图像，我会将图一的圆形logo作为基础形状。然后，我会将图二中冰箱贴的材质特性应用到这个logo上。具体来说，logo中的每一个色块（深蓝、中蓝、浅蓝、白色）"
           "都会被渲染成具有高光泽度的珐琅质感，表面会反射出柔和的环境光，形成自然的高光。logo中不同颜色区域之间的分界线，将被处理成纤细的、带有金属光泽的凸起边缘，这既能清晰地勾勒出图案，也符合珐琅工艺品的典型特征。整个冰箱贴会呈现出轻微的厚度和圆润的边缘，使其看起来像一个真实的、可触摸的物体。"
           "最后，将这个制作完成的冰箱贴放置在图二所示的灰色织物背景上，并为其添加一个微妙的、柔和的阴影，以增强其立体感和与背景的融合度，最终呈现出一个精致、逼真的产品展示图。</think><recaption>这幅图像以产品摄影的精致风格，呈现了一枚根据`image_1`标志定制的圆形珐琅冰箱贴。最终图像使用`image_2`的分辨率。"
           "冰箱贴居中放置在`image_2`的灰色织物背景上，其设计完美复刻了`image_1`中由深蓝、中蓝、浅蓝和白色构成的动态旋涡图案。整个冰箱贴被赋予了`image_2`中冰箱贴特有的高级质感：表面覆盖着一层光滑如镜的珐琅釉面，反射出柔和而清晰的高光；图案的每一个色块边缘都由纤细的抛光金属边框精确勾勒，增强了立体感。"
           "柔和的顶光在冰箱贴的弧形边缘上形成平滑的过渡，并在其下方投下淡淡的、轮廓模糊的阴影，使其与织物背景无缝融合，营造出一种真实、静谧的视觉效果。<relation_1>最终图像完整保留了`image_1`中标志的全部设计元素。这包括其完美的圆形轮廓，以及内部由深蓝、中蓝、浅蓝和白色组成的精确旋涡状图案布局、形状和色彩关系。"
           "</relation_1><relation_2>最终图像的分辨率、背景和材质均来自`image_2`。背景中灰色织物的纹理和质感被完整保留。冰箱贴的材质被完美重构，精确复刻了`image_2`中冰箱贴所展示的光滑珐琅质感、抛光金属边框的视觉效果，以及整体柔和、均匀的布光环境和由此产生的自然阴影。</relation_2></recaption><answer><boi>"
           "<img_size_1024><img_ratio_36><timestep>[<img>]{3600}<eoi></answer>")
# fmt: on


def _make_config(enable_kv_reuse: bool, path: Path) -> None:
    config = copy.deepcopy(_DEPLOY_CONFIG)
    config["stages"][0]["omni_kv_config"] = {"need_send_cache": enable_kv_reuse}
    config["stages"][1]["omni_kv_config"] = {"need_recv_cache": enable_kv_reuse}
    path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))


def _make_npu_config(path: Path) -> None:
    path.write_text(yaml.dump(_NPU_DEPLOY_CONFIG, default_flow_style=False, sort_keys=False))


def _quant_devices() -> str:
    return os.environ.get("HUNYUAN_IMAGE3_QUANT_DEVICES", "0,1")


def _quant_tensor_parallel_size() -> int:
    return int(os.environ.get("HUNYUAN_IMAGE3_QUANT_TP", str(len(_quant_devices().split(",")))))


def _make_quant_dit_config(path: Path) -> None:
    config = copy.deepcopy(_QUANT_DIT_CONFIG)
    config["stages"][0]["devices"] = _quant_devices()
    config["stages"][0]["parallel_config"]["tensor_parallel_size"] = _quant_tensor_parallel_size()
    path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))


def _npu_dit_devices() -> str:
    return os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")


def _npu_dit_tensor_parallel_size() -> int:
    return int(os.environ.get(NPU_DIT_TP_ENV, str(len(_npu_dit_devices().split(",")))))


def _npu_dit_enable_expert_parallel() -> bool:
    return os.environ.get(NPU_DIT_EP_ENV, "1").lower() in _TRUE_ENV_VALUES


def _npu_dit_gpu_memory_utilization() -> float:
    return float(os.environ.get(NPU_DIT_GPU_MEMORY_UTILIZATION_ENV, "0.65"))


def _make_npu_dit_config(path: Path) -> None:
    config = copy.deepcopy(_NPU_DIT_CONFIG)
    config["stages"][0]["gpu_memory_utilization"] = _npu_dit_gpu_memory_utilization()
    config["stages"][0]["devices"] = _npu_dit_devices()
    config["stages"][0]["parallel_config"]["tensor_parallel_size"] = _npu_dit_tensor_parallel_size()
    config["stages"][0]["parallel_config"]["enable_expert_parallel"] = _npu_dit_enable_expert_parallel()
    path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))


def _apply_offline_it2i_size_overrides(
    params_list: list[object],
    prompt: dict,
    *,
    height: int,
    width: int,
) -> None:
    prompt["height"] = height
    prompt["width"] = width
    prompt["mm_processor_kwargs"] = {"target_h": height, "target_w": width}

    for sp in params_list:
        if hasattr(sp, "height") and hasattr(sp, "width"):
            sp.height = height
            sp.width = width


def _run_offline(
    deploy_config_path: str,
    output_path: Path,
    *,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
    model_path: str = MODEL_PATH,
    output_size: tuple[int, int] | None = None,
) -> tuple[Image.Image, str, float]:
    from transformers import AutoTokenizer

    from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType
    from vllm_omni.platforms import current_omni_platform

    build_kwargs: dict = {"task": "it2i", "bot_task": "think_recaption", "sys_type": "en_unified", "num_images": 2}

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    result = build_prompt_tokens(
        PROMPT,
        tokenizer,
        **build_kwargs,
    )
    token_ids = result.token_ids
    system_prompt_type = result.system_prompt_type

    ar_stop_token_ids = resolve_stop_token_ids(task="it2i", bot_task="think_recaption", tokenizer=tokenizer)
    with OmniRunner(model_path, deploy_config=deploy_config_path) as runner:
        params_list = list(runner.omni.default_sampling_params_list)
        for sp in params_list:
            if isinstance(sp, OmniDiffusionSamplingParams):
                sp.num_inference_steps = num_inference_steps
                sp.guidance_scale = guidance_scale
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
        if output_size is not None:
            height, width = output_size
            _apply_offline_it2i_size_overrides(params_list, prompts[0], height=height, width=width)
        t0 = time.perf_counter()
        outputs = list(runner.omni.generate(prompts=prompts, sampling_params_list=params_list))
        elapsed = time.perf_counter() - t0

    assert outputs, "Pipeline produced no outputs"
    images = None
    cot_text = ""
    for out in outputs:
        ro = getattr(out, "request_output", None)
        if ro and getattr(ro, "outputs", None):
            cot_text = "".join(getattr(o, "text", "") or "" for o in ro.outputs)
        if not cot_text:
            ar_text = getattr(out, "custom_output", {}).get("ar_generated_text")
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
    _empty_accelerator_cache()
    return image, cot_text, elapsed


def _run_online(
    stage_configs_path: str,
    output_path: Path,
    *,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
    model_path: str = MODEL_PATH,
) -> tuple[Image.Image, str, float]:
    from benchmarks.accuracy.common import decode_base64_image, pil_to_png_bytes

    server_args = [
        "--stage-configs-path",
        stage_configs_path,
        "--stage-init-timeout",
        "300",
        "--init-timeout",
        "900",
    ]
    try:
        with OmniServer(model_path, server_args, use_omni=True) as omni_server:
            images = download_images(TEST_IMAGE_URLS)
            t0 = time.perf_counter()
            response = requests.post(
                f"http://{omni_server.host}:{omni_server.port}/v1/images/edits",
                data={
                    "model": omni_server.model,
                    "prompt": PROMPT,
                    "n": 1,
                    "response_format": "b64_json",
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": SEED,
                    "sys_type": "en_unified",
                    "use_system_prompt": "en_unified",
                    "bot_task": "think_recaption",
                    "size": "1280x720",
                    "vae_use_tiling": "false",
                    "enforce_eager": "false",
                },
                files=[
                    ("image", (f"image_{i}.png", pil_to_png_bytes(img), "image/png")) for i, img in enumerate(images)
                ],
                timeout=600,
            )
            elapsed = time.perf_counter() - t0
            if not response.ok:
                logger.error("[ONLINE] HTTP %s response body: %s", response.status_code, response.text)
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
        _empty_accelerator_cache()


@pytest.mark.skipif(
    torch.accelerator.device_count() < AR_TP_SIZE + DIT_TP_SIZE,
    reason=f"Needs {AR_TP_SIZE + DIT_TP_SIZE}+ GPUs ({AR_TP_SIZE} AR + {DIT_TP_SIZE} DiT)",
)
def test_image_to_image_alignment_online(accuracy_artifact_root: Path, accuracy_assets_root: Path) -> None:
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
    cot_results = scorer.text_similarity(online_cot, COT_REF)
    image_ref = Image.open(str(accuracy_assets_root / "hunyuan_image_ref.png")).convert("RGB")
    image_clip_score = clip_scorer.image_image_score(online_image, image_ref)
    ssim_value, psnr_value = compute_image_ssim_psnr(prediction=online_image, reference=image_ref, compare_mode="RGB")

    table = [
        ["COT similarity to reference", f"{cot_results['cot_semantic_sim']:.4f}", 0.9644],
        ["COT prefix match", f"{cot_results['text_prefix_match_count']:.4f}", 29],
        ["Image-Image similarity", f"{image_clip_score:.4f}", 94.5538],
        ["SSIM", f"{ssim_value:.4f}", 0.242],
        ["PSNR (dB)", f"{psnr_value:.2f}", 14.1],
    ]
    logger.info("[ONLINE] %s", tabulate(table, headers=["Metric", "Value", "L20x Reference"], tablefmt="grid"))

    assert cot_results["cot_semantic_sim"] >= THRESHOLDS["cot_semantic_sim"], (
        f"[ONLINE] COT semantic similarity {cot_results['cot_semantic_sim']:.4f} below threshold {THRESHOLDS['cot_semantic_sim']}"
    )
    assert cot_results["text_prefix_match_count"] >= THRESHOLDS["text_prefix_match"], (
        f"[ONLINE] COT prefix match {cot_results['text_prefix_match_count']} below threshold {THRESHOLDS['text_prefix_match']}"
    )
    assert image_clip_score >= THRESHOLDS["clip_score"], (
        f"[ONLINE] Image-Image similarity {image_clip_score:.4f} below threshold {THRESHOLDS['clip_score']}"
    )
    assert ssim_value >= THRESHOLDS["ssim"], f"[ONLINE] SSIM {ssim_value:.4f} below threshold {THRESHOLDS['ssim']}"
    assert psnr_value >= THRESHOLDS["psnr"], (
        f"[ONLINE] PSNR {psnr_value:.2f} dB below threshold {THRESHOLDS['psnr']} dB"
    )


@pytest.mark.parametrize(
    "case_name,runner",
    [
        pytest.param("offline", _run_offline, id="npu_offline"),
        pytest.param("online", _run_online, id="npu_online"),
    ],
)
@pytest.mark.npu
@pytest.mark.distributed_npu(num_cards=8)
@pytest.mark.skipif(torch.accelerator.device_count() < 8, reason="Needs 8+ NPUs (4 AR + 4 DiT)")
def test_image_to_image_alignment_npu(
    case_name: str,
    runner,
    accuracy_artifact_root: Path,
    accuracy_assets_root: Path,
) -> None:
    if importlib.util.find_spec("FlagEmbedding") is None:
        raise ImportError("Missing dependency: FlagEmbedding\nInstall with: pip install FlagEmbedding")
    from tabulate import tabulate

    output_dir = model_output_dir(accuracy_artifact_root, NPU_MODEL_NAME + f"-npu-{case_name}")

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "npu.yaml"
        _make_npu_config(config_path)
        runner_kwargs = {
            "num_inference_steps": NPU_NUM_INFERENCE_STEPS,
            "guidance_scale": NPU_GUIDANCE_SCALE,
            "model_path": NPU_MODEL_PATH,
        }
        if case_name == "offline":
            runner_kwargs["output_size"] = (720, 1280)
        npu_image, npu_cot, _ = runner(str(config_path), output_dir, **runner_kwargs)

    npu_cot = npu_cot.lstrip("\n")
    scorer = SemanticSimilarityScorer()
    clip_scorer = CLIPScorer()
    cot_results = scorer.text_similarity(npu_cot, COT_REF)
    image_ref = Image.open(str(accuracy_assets_root / "hunyuan_image_ref.png")).convert("RGB")
    image_clip_score = clip_scorer.image_image_score(npu_image, image_ref)
    ssim_value, psnr_value = compute_image_ssim_psnr(prediction=npu_image, reference=image_ref, compare_mode="RGB")

    table = [
        ["COT similarity to reference", f"{cot_results['cot_semantic_sim']:.4f}", NPU_THRESHOLDS["cot_semantic_sim"]],
        ["COT prefix match", f"{cot_results['text_prefix_match_count']:.4f}", NPU_THRESHOLDS["text_prefix_match"]],
        ["Image-Image similarity", f"{image_clip_score:.4f}", NPU_THRESHOLDS["clip_score"]],
        ["SSIM", f"{ssim_value:.4f}", NPU_THRESHOLDS["ssim"]],
        ["PSNR (dB)", f"{psnr_value:.2f}", NPU_THRESHOLDS["psnr"]],
    ]
    logger.info("[NPU][%s] %s", case_name, tabulate(table, headers=["Metric", "Value", "Threshold"], tablefmt="grid"))

    assert cot_results["cot_semantic_sim"] >= NPU_THRESHOLDS["cot_semantic_sim"], (
        f"[NPU][{case_name}] COT semantic similarity {cot_results['cot_semantic_sim']:.4f} "
        f"below threshold {NPU_THRESHOLDS['cot_semantic_sim']}"
    )
    assert cot_results["text_prefix_match_count"] >= NPU_THRESHOLDS["text_prefix_match"], (
        f"[NPU][{case_name}] COT prefix match {cot_results['text_prefix_match_count']} "
        f"below threshold {NPU_THRESHOLDS['text_prefix_match']}"
    )
    assert image_clip_score >= NPU_THRESHOLDS["clip_score"], (
        f"[NPU][{case_name}] Image-Image similarity {image_clip_score:.4f} "
        f"below threshold {NPU_THRESHOLDS['clip_score']}"
    )
    assert ssim_value >= NPU_THRESHOLDS["ssim"], (
        f"[NPU][{case_name}] SSIM {ssim_value:.4f} below threshold {NPU_THRESHOLDS['ssim']}"
    )
    assert psnr_value >= NPU_THRESHOLDS["psnr"], (
        f"[NPU][{case_name}] PSNR {psnr_value:.2f} dB below threshold {NPU_THRESHOLDS['psnr']} dB"
    )


def test_offline_it2i_size_overrides_align_with_online() -> None:
    from types import SimpleNamespace

    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    ar_params = SimpleNamespace(extra_args={}, stop_token_ids=[])
    dit_params = OmniDiffusionSamplingParams()
    prompt = {"prompt": PROMPT}

    _apply_offline_it2i_size_overrides([ar_params, dit_params], prompt, height=720, width=1280)

    assert prompt["height"] == 720
    assert prompt["width"] == 1280
    assert prompt["mm_processor_kwargs"] == {"target_h": 720, "target_w": 1280}
    assert ar_params.extra_args == {}
    assert dit_params.height == 720
    assert dit_params.width == 1280


def test_npu_it2i_config_uses_distil_params(tmp_path: Path) -> None:
    config_path = tmp_path / "npu.yaml"
    _make_npu_config(config_path)
    config = yaml.safe_load(config_path.read_text())

    assert config["stages"][1]["default_sampling_params"]["num_inference_steps"] == NPU_NUM_INFERENCE_STEPS
    assert config["stages"][1]["default_sampling_params"]["guidance_scale"] == NPU_GUIDANCE_SCALE


def _extract_image(outputs) -> Image.Image:
    assert outputs, "Pipeline produced no outputs"
    for output in outputs:
        images = getattr(output, "images", None)
        request_output = getattr(output, "request_output", None)
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
    num_inference_steps: int = 20,
    guidance_scale: float = 4.0,
    nvfp4_backend: str | None = None,
) -> tuple[Image.Image, float]:
    from tests.helpers.runtime import OmniRunner
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms import current_omni_platform

    old_backend = os.environ.get("VLLM_NVFP4_GEMM_BACKEND")
    if nvfp4_backend is not None:
        os.environ["VLLM_NVFP4_GEMM_BACKEND"] = nvfp4_backend

    try:
        logger.info("[NPU DiT] launching OmniRunner for model=%s", model)
        with OmniRunner(model, deploy_config=deploy_config_path, mode="text-to-image", log_stats=True) as runner:
            logger.info("[NPU DiT] OmniRunner ready, starting generation")
            generator = torch.Generator(device=current_omni_platform.device_type or "cuda").manual_seed(SEED)
            params = OmniDiffusionSamplingParams(
                height=QUANT_HEIGHT,
                width=QUANT_WIDTH,
                seed=SEED,
                generator=generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
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
        _empty_accelerator_cache()


def test_npu_dit_config_defaults_to_four_card_expert_parallel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv(NPU_DIT_TP_ENV, raising=False)
    monkeypatch.delenv(NPU_DIT_EP_ENV, raising=False)
    monkeypatch.delenv(NPU_DIT_GPU_MEMORY_UTILIZATION_ENV, raising=False)

    config_path = tmp_path / "npu_dit.yaml"
    _make_npu_dit_config(config_path)
    config = yaml.safe_load(config_path.read_text())
    stage = config["stages"][0]

    assert config["pipeline"] == "hunyuan_image3_dit"
    assert stage["gpu_memory_utilization"] == 0.65
    assert stage["devices"] == "0,1,2,3"
    assert stage["parallel_config"]["tensor_parallel_size"] == 4
    assert stage["parallel_config"]["enable_expert_parallel"] is True
    assert "force_cutlass_fp8" not in stage
    assert "moe_backend" not in stage


def test_npu_dit_config_accepts_env_parallelism(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv(NPU_DIT_TP_ENV, "2")
    monkeypatch.setenv(NPU_DIT_EP_ENV, "0")
    monkeypatch.setenv(NPU_DIT_GPU_MEMORY_UTILIZATION_ENV, "0.8")

    config_path = tmp_path / "npu_dit.yaml"
    _make_npu_dit_config(config_path)
    stage = yaml.safe_load(config_path.read_text())["stages"][0]

    assert stage["gpu_memory_utilization"] == 0.8
    assert stage["devices"] == "2,3"
    assert stage["parallel_config"]["tensor_parallel_size"] == 2
    assert stage["parallel_config"]["enable_expert_parallel"] is False


@hardware_test(res={"cuda": "H100"}, num_cards=8)
@pytest.mark.skipif(
    torch.accelerator.device_count() < AR_TP_SIZE + DIT_TP_SIZE,
    reason=f"Needs {AR_TP_SIZE + DIT_TP_SIZE}+ GPUs ({AR_TP_SIZE} AR + {DIT_TP_SIZE} DiT)",
)
def test_image_to_image_alignment(accuracy_artifact_root: Path, accuracy_assets_root: Path) -> None:
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
    cot_results = scorer.text_similarity(omni_cot, COT_REF)
    image_ref = Image.open(str(accuracy_assets_root / "hunyuan_image_ref.png")).convert("RGB")
    image_clip_score = clip_scorer.image_image_score(omni_image, image_ref)
    ssim_value, psnr_value = compute_image_ssim_psnr(prediction=omni_image, reference=image_ref, compare_mode="RGB")

    table = [
        ["COT similarity to reference", f"{cot_results['cot_semantic_sim']:.4f}", 0.9644],
        ["COT prefix match", f"{cot_results['text_prefix_match_count']:.4f}", 29],
        ["Image-Image similarity", f"{image_clip_score:.4f}", 94.5538],
        ["SSIM", f"{ssim_value:.4f}", 0.242],
        ["PSNR (dB)", f"{psnr_value:.2f}", 14.1],
    ]

    logger.info("%s", tabulate(table, headers=["Metric", "Value", "L20x Reference"], tablefmt="grid"))

    assert cot_results["cot_semantic_sim"] >= THRESHOLDS["cot_semantic_sim"], (
        f"COT semantic similarity {cot_results['cot_semantic_sim']:.4f} is below threshold {THRESHOLDS['cot_semantic_sim']}"
    )
    assert cot_results["text_prefix_match_count"] >= THRESHOLDS["text_prefix_match"], (
        f"COT prefix match count {cot_results['text_prefix_match_count']} is below threshold {THRESHOLDS['text_prefix_match']}"
    )
    assert image_clip_score >= THRESHOLDS["clip_score"], (
        f"Image-Image similarity{image_clip_score:.4f} is below threshold {THRESHOLDS['clip_score']}"
    )
    assert ssim_value >= THRESHOLDS["ssim"], f"SSIM {ssim_value:.4f} is below threshold {THRESHOLDS['ssim']}"
    assert psnr_value >= THRESHOLDS["psnr"], f"PSNR {psnr_value:.2f} dB is below threshold {THRESHOLDS['psnr']} dB"


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
    assert_similarity(
        model_name=f"{MODEL_NAME} {case.name} vs bf16",
        vllm_image=quant_image,
        diffusers_image=bf16_image,
        ssim_threshold=QUANT_SSIM_THRESHOLD,
        psnr_threshold=QUANT_PSNR_THRESHOLD,
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

    logger.info("HunyuanImage3 quant accuracy (%s)", case.name)
    logger.info("  bf16 model:       %s", bf16_model)
    logger.info("  quant model:      %s", quant_model)
    logger.info("  BF16 CLIP score:  %.4f", bf16_clip_score)
    logger.info(
        "  quant CLIP score: %.4f threshold>=%0.4f",
        quant_clip_score,
        QUANT_CLIP_SCORE_THRESHOLD,
    )
    logger.info(
        "  CLIP score drop:  %.4f threshold<=%.4f",
        clip_score_drop,
        QUANT_CLIP_SCORE_DROP_THRESHOLD,
    )
    logger.info("  metrics:          %s", metrics_path)

    assert quant_clip_score >= QUANT_CLIP_SCORE_THRESHOLD, (
        f"{case.name} CLIP score below threshold: got {quant_clip_score:.4f}, "
        f"expected >= {QUANT_CLIP_SCORE_THRESHOLD:.4f}"
    )
    assert clip_score_drop <= QUANT_CLIP_SCORE_DROP_THRESHOLD, (
        f"{case.name} CLIP score drop too large: got {clip_score_drop:.4f}, "
        f"expected <= {QUANT_CLIP_SCORE_DROP_THRESHOLD:.4f}"
    )


@pytest.mark.npu
@pytest.mark.distributed_npu(num_cards=4)
@pytest.mark.skipif(
    torch.accelerator.device_count() < _npu_dit_tensor_parallel_size(),
    reason="Needs enough NPUs for HunyuanImage3 NPU DiT tensor parallelism",
)
def test_npu_dit_distil_smoke_accuracy(accuracy_artifact_root: Path) -> None:
    """NPU DiT-only smoke test using the long-term Distil bf16 reference model."""
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_NAME + "-npu-dit")

    with tempfile.TemporaryDirectory() as tmpdir:
        deploy_config_path = Path(tmpdir) / "hunyuan_image3_npu_dit.yaml"
        _make_npu_dit_config(deploy_config_path)
        image, elapsed = _run_dit_model(
            NPU_DIT_BF16_MODEL,
            str(deploy_config_path),
            output_dir / "npu_dit_distil.png",
            num_inference_steps=NPU_DIT_NUM_INFERENCE_STEPS,
            guidance_scale=NPU_DIT_GUIDANCE_SCALE,
        )

    assert image.size == (QUANT_WIDTH, QUANT_HEIGHT)
    metrics = {
        "case": "npu_dit_distil",
        "model": NPU_DIT_BF16_MODEL,
        "prompt": QUANT_PROMPT,
        "seed": SEED,
        "height": QUANT_HEIGHT,
        "width": QUANT_WIDTH,
        "num_inference_steps": NPU_DIT_NUM_INFERENCE_STEPS,
        "guidance_scale": NPU_DIT_GUIDANCE_SCALE,
        "elapsed_s": elapsed,
        "devices": _npu_dit_devices(),
        "tensor_parallel_size": _npu_dit_tensor_parallel_size(),
        "enable_expert_parallel": _npu_dit_enable_expert_parallel(),
        "gpu_memory_utilization": _npu_dit_gpu_memory_utilization(),
    }
    metrics_path = output_dir / "npu_dit_distil_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("HunyuanImage3 NPU DiT smoke accuracy metrics: %s", metrics_path)


@pytest.mark.npu
@pytest.mark.distributed_npu(num_cards=4)
@pytest.mark.skipif(
    torch.accelerator.device_count() < _npu_dit_tensor_parallel_size(),
    reason="Needs enough NPUs for HunyuanImage3 NPU DiT tensor parallelism",
)
def test_npu_quantized_dit_matches_bf16_accuracy(
    accuracy_artifact_root: Path,
) -> None:
    """NPU quantized DiT checkpoints should preserve bf16 DiT image quality."""
    quant_model = os.environ.get(NPU_DIT_QUANT_MODEL_ENV)
    if not quant_model:
        pytest.skip(f"Set {NPU_DIT_QUANT_MODEL_ENV} to run HunyuanImage3 NPU DiT quant accuracy.")

    output_dir = model_output_dir(accuracy_artifact_root, MODEL_NAME + "-npu-dit-quant")

    with tempfile.TemporaryDirectory() as tmpdir:
        deploy_config_path = Path(tmpdir) / "hunyuan_image3_npu_dit.yaml"
        _make_npu_dit_config(deploy_config_path)
        bf16_image, bf16_time = _run_dit_model(
            NPU_DIT_BF16_MODEL,
            str(deploy_config_path),
            output_dir / "npu_bf16.png",
            num_inference_steps=NPU_DIT_NUM_INFERENCE_STEPS,
            guidance_scale=NPU_DIT_GUIDANCE_SCALE,
        )
        quant_image, quant_time = _run_dit_model(
            quant_model,
            str(deploy_config_path),
            output_dir / "npu_quant.png",
            num_inference_steps=NPU_DIT_NUM_INFERENCE_STEPS,
            guidance_scale=NPU_DIT_GUIDANCE_SCALE,
        )

    ssim_score, psnr_score = compute_image_ssim_psnr(
        prediction=quant_image,
        reference=bf16_image,
    )
    assert_similarity(
        model_name=f"{MODEL_NAME} npu quant vs bf16",
        vllm_image=quant_image,
        diffusers_image=bf16_image,
        ssim_threshold=QUANT_SSIM_THRESHOLD,
        psnr_threshold=QUANT_PSNR_THRESHOLD,
        width=QUANT_WIDTH,
        height=QUANT_HEIGHT,
    )

    clip_scorer = CLIPScorer()
    bf16_clip_score = clip_scorer.score(bf16_image, QUANT_PROMPT)
    quant_clip_score = clip_scorer.score(quant_image, QUANT_PROMPT)
    clip_score_drop = bf16_clip_score - quant_clip_score

    metrics = {
        "case": "npu_quant",
        "bf16_model": NPU_DIT_BF16_MODEL,
        "quant_model": quant_model,
        "prompt": QUANT_PROMPT,
        "seed": SEED,
        "height": QUANT_HEIGHT,
        "width": QUANT_WIDTH,
        "num_inference_steps": NPU_DIT_NUM_INFERENCE_STEPS,
        "guidance_scale": NPU_DIT_GUIDANCE_SCALE,
        "bf16_elapsed_s": bf16_time,
        "quant_elapsed_s": quant_time,
        "ssim": ssim_score,
        "psnr": psnr_score,
        "bf16_clip_score": bf16_clip_score,
        "quant_clip_score": quant_clip_score,
        "clip_score_drop": clip_score_drop,
        "devices": _npu_dit_devices(),
        "tensor_parallel_size": _npu_dit_tensor_parallel_size(),
        "enable_expert_parallel": _npu_dit_enable_expert_parallel(),
        "gpu_memory_utilization": _npu_dit_gpu_memory_utilization(),
    }
    metrics_path = output_dir / "npu_quant_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    logger.info("HunyuanImage3 NPU DiT quant accuracy")
    logger.info("  bf16 model:       %s", NPU_DIT_BF16_MODEL)
    logger.info("  quant model:      %s", quant_model)
    logger.info("  BF16 CLIP score:  %.4f", bf16_clip_score)
    logger.info(
        "  quant CLIP score: %.4f threshold>=%0.4f",
        quant_clip_score,
        QUANT_CLIP_SCORE_THRESHOLD,
    )
    logger.info(
        "  CLIP score drop:  %.4f threshold<=%.4f",
        clip_score_drop,
        QUANT_CLIP_SCORE_DROP_THRESHOLD,
    )
    logger.info("  metrics:          %s", metrics_path)

    assert quant_clip_score >= QUANT_CLIP_SCORE_THRESHOLD, (
        f"NPU quant CLIP score below threshold: got {quant_clip_score:.4f}, "
        f"expected >= {QUANT_CLIP_SCORE_THRESHOLD:.4f}"
    )
    assert clip_score_drop <= QUANT_CLIP_SCORE_DROP_THRESHOLD, (
        f"NPU quant CLIP score drop too large: got {clip_score_drop:.4f}, "
        f"expected <= {QUANT_CLIP_SCORE_DROP_THRESHOLD:.4f}"
    )
