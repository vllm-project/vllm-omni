"""
用 `vllm_omni.Omni`（即 OmniLLM pipeline）启动 MammothModa2 的 AR 单阶段推理。

Usage:
    uv run python examples/offline_inference/run_mammothmoda2.py \\
        --model /data/datasets/models-hf/MammothModa2-Preview \\
        --stage-config vllm_omni/model_executor/stage_configs/mammoth_moda2_single.yaml \\
        --prompt "你好，介绍一下你自己。" \\
        --image /path/to/image.png \\
        --max-tokens 64
"""

from __future__ import annotations

import argparse
import os

from vllm_omni import Omni
from vllm.sampling_params import SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MammothModa2 with vLLM-Omni.")
    parser.add_argument(
        "--model",
        type=str,
        default="/data/datasets/models-hf/MammothModa2-Preview",
        help="HuggingFace 模型路径或名称，例如 /data/datasets/models-hf/MammothModa2-Preview",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="是否信任远端自定义代码（必须开启以加载 MammothModa2）。",
    )
    parser.add_argument(
        "--stage-config",
        type=str,
        default="vllm_omni/model_executor/stage_configs/mammoth_moda2_single.yaml",
        help="Stage config 路径，默认使用仓库内 mammoth_moda2_single.yaml",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="你好，介绍一下你自己。然后描述你看到的图片。",
        help="文本提示（若传入 --image，会作为图像问题/指令）",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="mammothmoda/doc/example0.png",
        help="可选：输入图片路径（本地文件）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="最大生成 token 数（达到 EOS 也会提前停止）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    omni = Omni(
        model=args.model,
        stage_configs_path=args.stage_config,
        trust_remote_code=args.trust_remote_code,
    )
    try:
        sampling_params_list = [
            SamplingParams(
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                max_tokens=max(1, int(args.max_tokens)),
                detokenize=True,
            )
        ]
        engine_prompt: dict[str, object]
        if args.image:
            if not os.path.exists(args.image):
                raise FileNotFoundError(f"Image file not found: {args.image}")
            from PIL import Image

            image_data = Image.open(args.image).convert("RGB")
            # Qwen2.5-VL 风格：图片占位符由 chat_template.jinja 渲染为
            # `<|vision_start|><|image_pad|><|vision_end|>`。
            prompt = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n"
                "<|vision_start|><|image_pad|><|vision_end|>"
                f"{args.prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            engine_prompt = {
                "prompt": prompt,
                "multi_modal_data": {"image": image_data},
            }
        else:
            # NOTE: 当前 omni pipeline 的 stage worker 侧只会把 `dict` / `list` 输入加入 batch，
            # 直接传 `str` 会导致 batch 为空，从而不会触发模型 forward，最终得到空输出列表。
            engine_prompt = {"prompt": args.prompt}

        omni_outputs = omni.generate([engine_prompt], sampling_params_list)

        # OmniLLM 的 stage 输出在单 batch 场景下通常是 `list[RequestOutput]`。
        request_output = omni_outputs[0].request_output
        if isinstance(request_output, list):
            if not request_output:
                raise RuntimeError(
                    "Stage 返回了空的 request_output 列表：这通常意味着 stage 没有实际执行 generate/forward。"
                )
            request_output = request_output[0]

        gen0 = request_output.outputs[0]
        print("generated_text:", gen0.text)
        print("generated_token_ids:", gen0.token_ids)
    finally:
        omni.close()


if __name__ == "__main__":
    main()
