# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import gc
import importlib
import os
import tempfile
import time
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from tests.e2e.accuracy.helpers import (
    CLIPScorer,
    SemanticSimilarityScorer,
    compute_image_ssim_psnr,
    download_images,
    model_output_dir,
)
from tests.helpers.runtime import OmniRunner
from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import build_prompt_tokens, resolve_stop_token_ids

os.environ["DIFFUSION_ATTENTION_BACKEND"] = "TORCH_SDPA"

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]

# ============================================================================
# Configurable Parameters
# ============================================================================
AR_DEVICES = "0,1"
DIT_DEVICES = "2,3"
MODEL_NAME = "tencent/HunyuanImage-3.0-Instruct"
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 2.5

# ============================================================================
# Constants
# ============================================================================
MODEL_PATH = os.environ.get("HUNYUAN_MODEL_PATH", MODEL_NAME)
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
    "clip_score": 85,  # CLIP image semantic similarity
    "ssim": 0.20,  # Structural similarity
    "psnr": 11.0,  # Peak signal-to-noise ratio (dB)
}
# fmt: off
_BASE_CONFIG = {
    "stage_args": [
        {
            "stage_id": 0, "stage_type": "llm",
            "runtime": {"process": True, "devices": AR_DEVICES, "max_batch_size": 1, "requires_multimodal_data": True},
            "engine_args": {
                "model_stage": "AR", "model_arch": "HunyuanImage3ForCausalMM",
                "worker_cls": "vllm_omni.worker.gpu_ar_worker.GPUARWorker",
                "scheduler_cls": "vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler",
                "gpu_memory_utilization": 0.95, "enforce_eager": True, "trust_remote_code": True,
                "engine_output_type": "latent", "enable_prefix_caching": False,
                "max_num_batched_tokens": 32768, "tensor_parallel_size": 2, "pipeline_parallel_size": 1,
                "hf_overrides": {"rope_parameters": {"mrope_section": [0, 32, 32], "rope_type": "default"}},
            },
            "is_comprehension": False, "final_output": True, "final_output_type": "text",
            "default_sampling_params": {
                "temperature": 0.0, "top_p": 1, "top_k": -1, "max_tokens": 8192,
                "stop_token_ids": [128025], "detokenize": True, "skip_special_tokens": False,
            },
            "output_connectors": {"to_stage_1": "shared_memory_connector"},
        },
        {
            "stage_id": 1, "stage_type": "diffusion",
            "runtime": {"process": True, "devices": DIT_DEVICES, "max_batch_size": 1, "requires_multimodal_data": True},
            "engine_args": {
                "model_stage": "dit", "model_arch": "HunyuanImage3ForCausalMM",
                "enforce_eager": True, "trust_remote_code": True, "distributed_executor_backend": "mp",
                "parallel_config": {"tensor_parallel_size": 2, "enable_expert_parallel": True},
            },
            "engine_input_source": [0],
            "custom_process_input_func": "vllm_omni.model_executor.stage_input_processors.hunyuan_image3.ar2diffusion",
            "final_output": True, "final_output_type": "image",
            "default_sampling_params": {"num_inference_steps": NUM_INFERENCE_STEPS, "guidance_scale": GUIDANCE_SCALE},
            "input_connectors": {"from_stage_0": "shared_memory_connector"},
        },
    ],
    "runtime": {
        "enabled": True,
        "connectors": {"shared_memory_connector": {
            "name": "SharedMemoryConnector",
            "extra": {"shm_threshold_bytes": 65536}
        }},
        "edges": [{"from": 0, "to": 1}],
    },
}
# fmt: on

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
    config = copy.deepcopy(_BASE_CONFIG)
    config["stage_args"][0]["engine_args"]["omni_kv_config"] = {"need_send_cache": enable_kv_reuse}
    config["stage_args"][1]["engine_args"]["omni_kv_config"] = {"need_recv_cache": enable_kv_reuse}
    path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))


def _run_offline(stage_configs_path: str, output_path: Path) -> tuple[Image.Image, str, float]:
    from transformers import AutoTokenizer

    from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType
    from vllm_omni.platforms import current_omni_platform

    build_kwargs: dict = {"task": "it2i", "bot_task": "think_recaption", "sys_type": "en_unified", "num_images": 2}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    result = build_prompt_tokens(
        PROMPT,
        tokenizer,
        **build_kwargs,
    )
    token_ids = result.token_ids
    system_prompt_type = result.system_prompt_type

    ar_stop_token_ids = resolve_stop_token_ids(task="it2i", bot_task="think_recaption", tokenizer=tokenizer)
    with OmniRunner(MODEL_NAME, stage_configs_path=stage_configs_path) as runner:
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
    if torch.accelerator.is_available():
        torch.accelerator.empty_cache()
    return image, cot_text, elapsed


@pytest.mark.skipif(torch.accelerator.device_count() < 4, reason="Needs 4+ GPUs (2 AR + 2 DiT)")
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

    print(tabulate(table, headers=["Metric", "Value", "L20x Reference"], tablefmt="grid"))

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
