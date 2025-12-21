"""
MammothModa2 文生图（AR -> DiT）离线推理示例，使用 vllm_omni.Omni + stage config 管线。

说明：
- Stage 0（AR）生成 gen tokens（视觉 token 序列），同时输出每个 token 的 hidden states（engine_output_type=latent）。
- Stage 1（DiT）消费 gen tokens 对应的 hidden states（通过 additional_information 传递），执行 diffusion + VAE decode 输出图像。

用法示例：
  uv run python examples/offline_inference/run_mammothmoda2_t2i.py \\
    --model /data/datasets/models-hf/MammothModa2-Preview \\
    --stage-config vllm_omni/model_executor/stage_configs/mammoth_moda2.yaml \\
    --prompt "一只戴着墨镜的柴犬，电影海报风格" \\
    --ar-width 32 --ar-height 32 \\
    --max-tokens 1056 \\
    --out out.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image

from vllm.sampling_params import SamplingParams
from vllm_omni import Omni


def _load_gen_vocab_range(model_dir: str) -> tuple[int, int]:
    """从模型目录的 `config.json` 读取 extra gen vocab 范围。

    vLLM V1 不支持 `SamplingParams.logits_processors`（callable），但支持
    `SamplingParams.allowed_token_ids`，因此用 gen vocab 范围约束采样。
    """
    cfg_path = Path(model_dir) / "config.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    llm_cfg = cfg.get("llm_config") or {}
    text_cfg = llm_cfg.get("text_config") or {}
    gen_vocab_start_index = text_cfg.get("gen_vocab_start_index", cfg.get("gen_vocab_start_index"))
    gen_vocab_size = text_cfg.get("gen_vocab_size", cfg.get("gen_vocab_size"))
    if gen_vocab_start_index is None or gen_vocab_size is None:
        raise ValueError(f"Missing gen vocab range in {cfg_path}")
    return int(gen_vocab_start_index), int(gen_vocab_size)

def _load_t2i_token_range(model_dir: str) -> tuple[int, int, int]:
    """读取 `t2i_generation_config.json`，返回 (eol, visual_start, visual_end_exclusive)。"""
    cfg_path = Path(model_dir) / "t2i_generation_config.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    eol = int(cfg["eol_token_id"])
    visual_start = int(cfg["visual_token_start_id"])
    visual_end = int(cfg["visual_token_end_id"])
    return eol, visual_start, visual_end


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MammothModa2 t2i (AR -> DiT) with vLLM-Omni.")
    p.add_argument(
        "--model",
        type=str,
        default="/data/datasets/models-hf/MammothModa2-Preview",
        help="模型路径（本地 HF 目录）",
    )
    p.add_argument(
        "--stage-config",
        type=str,
        default="vllm_omni/model_executor/stage_configs/mammoth_moda2.yaml",
        help="两阶段配置（AR + DiT）",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="一幅极具电影感与哲学意味的超现实主义画面："
                "在黄昏与夜晚交界的时间点，一座悬浮在半空中的巨大城市遗迹缓慢崩塌，城市由古典石质建筑与未来感金属结构交织而成，仿佛文明在不同时间维度中重叠。城市下方是一片无边无际的深色海洋，海面如镜，倒映着天空中翻涌的云层与破碎的建筑轮廓。"
                "天空呈现出层次分明的色彩，从地平线处的暗橙色、深紫色逐渐过渡到高空的靛蓝与黑色，厚重的云层中隐约透出冷色调的光束，像是来自未知天体的微弱照明。远处可以看到巨大的行星或月亮，占据画面一角，表面布满清晰可见的纹理与裂隙，暗示着时间与宇宙尺度的存在。"
                "画面中央站立着一个孤独的人物，背对观者，站在一块漂浮的岩石平台上。人物身披长风衣，衣摆随风飘动，剪影清晰但面部不可见，性别模糊，具有象征意义。人物手中握着一盏微弱发光的光源（可能是灯笼、能量核心或抽象的光体），成为画面中最温暖的光点，与周围冷色调环境形成强烈对比。"
                "建筑碎片与尘埃缓慢漂浮在空中，呈现出近乎静止的失重状态，强调时间被拉长或暂停的感觉。整体构图遵循电影级构图法则，前景、中景、远景层次分明，景深明显，具有强烈的纵深感与沉浸感。"
                "风格上融合超现实主义（surrealism）、科幻概念艺术（sci-fi concept art）、史诗电影分镜（cinematic storyboard）、油画质感与数字绘画细节。画面细节极其丰富，纹理清晰，材质真实，光影遵循物理逻辑但略带艺术夸张。氛围情绪：孤独、宏大、沉思、文明的终结与新生、时间与存在的哲学感。"
                "画质与技术要求：ultra-high resolution, ultra-detailed, 8k quality, sharp focus, global illumination, volumetric lighting, soft cinematic contrast, dramatic lighting, high dynamic range, no text, no watermark, no logo, masterpiece, best quality",
        help="文本提示",
    )
    p.add_argument("--ar-width", type=int, default=32, help="AR 生成网格宽（token 级）")
    p.add_argument("--ar-height", type=int, default=32, help="AR 生成网格高（token 级）")
    p.add_argument(
        "--max-tokens",
        type=int,
        default=1056,
        help="AR 生成 token 数（建议 ar_height * (ar_width + 1)，先跑通再调）",
    )
    p.add_argument("--out", type=str, default="out.png", help="输出图片路径")
    p.add_argument("--trust-remote-code", action="store_true", help="信任远端自定义代码（本地目录一般不需要）")
    return p.parse_args()


def _to_pil(image: torch.Tensor) -> Image.Image:
    # image: [3,H,W] or [1,3,H,W], range ~[-1,1]
    if image.ndim == 4:
        image = image[0]
    image = image.detach().to("cpu")
    image = (image / 2 + 0.5).clamp(0, 1)
    image = (image * 255).to(torch.uint8)
    image = image.permute(1, 2, 0).contiguous().numpy()
    return Image.fromarray(image)


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # 约束 AR stage 只生成视觉 token（额外允许 eol）：
    # vLLM V1 不支持 logits_processors(callable)，用 allowed_token_ids 实现同等效果。
    eol_token_id, visual_start, visual_end = _load_t2i_token_range(args.model)
    allowed_token_ids = [eol_token_id] + list(range(visual_start, visual_end))
    expected_grid_tokens = int(args.ar_height) * (int(args.ar_width) + 1)
    if int(args.max_tokens) != expected_grid_tokens:
        print(
            f"[warn] --max-tokens={int(args.max_tokens)} 与网格期望值 "
            f"ar_height*(ar_width+1)={expected_grid_tokens} 不一致；"
            "如需完整 2D 网格 token，建议使用期望值。"
        )

    prompt = (
        "<|im_start|>system\nYou are a helpful image generator.<|im_end|>\n"
        f"<|im_start|>user\n{args.prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"<|image start|>{int(args.ar_width)}*{int(args.ar_height)}<|image token|>"
    )

    omni = Omni(model=args.model, stage_configs_path=args.stage_config, trust_remote_code=args.trust_remote_code)
    try:
        ar_sampling = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=2048,
            max_tokens=max(1, int(args.max_tokens)),
            detokenize=False,
            allowed_token_ids=allowed_token_ids,
        )
        # DiT stage 使用 pooling 输出；SamplingParams 仅为满足 pipeline 接口
        dit_sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=1,
            detokenize=False,
        )

        # 通过 additional_information 将 t2i 网格信息透传给 AR 模型，
        # 使其在 compute_logits 阶段实现“每 (ar_width+1) 个 token 的最后一个必须为 eol”的动态约束。
        outputs = omni.generate(
            [
                {
                    "prompt": prompt,
                    "additional_information": {
                        "omni_task": "t2i",
                        "ar_width": int(args.ar_width),
                        "ar_height": int(args.ar_height),
                        "eol_token_id": int(eol_token_id),
                        "visual_token_start_id": int(visual_start),
                        "visual_token_end_id": int(visual_end),
                    },
                }
            ],
            [ar_sampling, dit_sampling],
        )
        ro = outputs[0].request_output
        if isinstance(ro, list):
            if not ro:
                raise RuntimeError("Empty request_output from final stage.")
            ro = ro[0]

        # final stage: image stored in multimodal_output
        mm = getattr(ro, "multimodal_output", None)
        if not isinstance(mm, dict) or "image" not in mm:
            raise RuntimeError(f"Unexpected final output payload: {type(mm)} {mm}")

        img_tensor = mm["image"]
        if not isinstance(img_tensor, torch.Tensor):
            raise TypeError(f"Expected image tensor, got {type(img_tensor)}")

        pil = _to_pil(img_tensor)
        pil.save(args.out)
        print("saved:", args.out)
    finally:
        omni.close()


if __name__ == "__main__":
    main()
