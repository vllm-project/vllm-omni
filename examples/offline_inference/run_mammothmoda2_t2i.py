"""
MammothModa2 文生图（AR -> DiT）离线推理示例，使用 vllm_omni.Omni + stage config 管线。

说明：
- Stage 0（AR）生成 gen tokens（视觉 token 序列），同时输出每个 token 的 hidden states（engine_output_type=latent）。
- Stage 1（DiT）消费 gen tokens 对应的 hidden states（通过 prompt_embeds 传递），执行 diffusion + VAE decode 输出图像。

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
        default="一只戴着墨镜的柴犬，电影海报风格",
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

    prompt = (
        "<|im_start|>system\nYou are a helpful image generator.<|im_end|>\n"
        f"<|im_start|>user\n{args.prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"<|image start|>{int(args.ar_width)}*{int(args.ar_height)}<|image token|>"
    )

    omni = Omni(
        model=args.model,
        stage_configs_path=args.stage_config,
        trust_remote_code=args.trust_remote_code,
    )
    try:
        ar_sampling = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=2048,
            max_tokens=max(1, int(args.max_tokens)),
            detokenize=False,
            stop_token_ids=[eol_token_id],
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

        outputs = omni.generate([{"prompt": prompt}], [ar_sampling, dit_sampling])
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
