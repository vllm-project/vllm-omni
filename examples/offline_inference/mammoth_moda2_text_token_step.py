"""
最小里程碑 1：文本 token -> 下一步 token（并打印 top-k logprobs）。

目标：
- 不走多模态输入，只用 prompt_token_ids 驱动 AR stage；
- 验证：模型能完成一次 prefill/decoding，并产出下一 token（间接证明 logits 正常）。

用法示例：
    TMPDIR=/home/dsc/vllm-omni/.tmp \\
    uv run python examples/offline_inference/mammoth_moda2_text_token_step.py \\
      --model /data/datasets/models-hf/MammothModa2-Preview \\
      --text "你好，介绍一下你自己。" \\
      --max-tokens 1
"""

from __future__ import annotations

import argparse
import os
from typing import cast

from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams

from vllm_omni.entrypoints.omni_llm import OmniStageLLM
from vllm_omni.model_executor.models.mammoth_moda2.tokenization_mammothmoda2_qwen2_5_vl import (
    MammothUTokenizer,
)


def _ensure_tmpdir() -> None:
    tmpdir = os.environ.get("TMPDIR")
    if tmpdir and os.path.isdir(tmpdir) and os.access(tmpdir, os.W_OK):
        return
    tmpdir = os.path.join(os.path.expanduser("~"), ".cache", "vllm_omni", "tmp")
    os.makedirs(tmpdir, exist_ok=True)
    os.environ["TMPDIR"] = tmpdir


def _parse_token_ids(s: str) -> list[int]:
    s = s.strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    return [int(p) for p in parts if p]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MammothModa2 文本 token 单步生成验证")
    parser.add_argument(
        "--model",
        type=str,
        default="/data/datasets/models-hf/MammothModa2-Preview",
        help="模型路径（本地 HF 目录）",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="你好，介绍一下你自己。",
        help="用于编码成 prompt_token_ids 的文本（当 --prompt-token-ids 未提供时生效）",
    )
    parser.add_argument(
        "--prompt-token-ids",
        type=str,
        default="",
        help="逗号分隔的 token ids（优先级高于 --text），例如: 151643, 42, 17",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1,
        help="生成 token 数（建议先用 1）",
    )
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=5,
        help="打印 top-k logprobs",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="是否信任远端自定义代码（离线本地目录一般不需要，但保持与主流程一致）",
    )
    return parser.parse_args()


def main() -> None:
    _ensure_tmpdir()
    args = parse_args()

    # vLLM 0.11 的平台探测是硬前置：没有识别到 CUDA/ROCm/TPU/XPU/CPU 平台时会直接报错。
    # 这个脚本的目标是跑通 token->token，因此这里提前给出明确提示。
    from vllm.platforms import current_platform
    from vllm.platforms.interface import UnspecifiedPlatform

    if isinstance(current_platform, UnspecifiedPlatform):
        raise RuntimeError(
            "vLLM 未检测到可用平台（current_platform=UnspecifiedPlatform）。\n"
            "- 如果你要用 GPU：请在有 CUDA 驱动/设备的环境运行。\n"
            "- 如果你要用 CPU：需要安装 vLLM 的 CPU build（版本字符串通常包含 'cpu'）。"
        )

    # NOTE: 为了便于验收，这里始终加载 tokenizer 用于 decode 输出文本。
    tok = MammothUTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

    if args.prompt_token_ids:
        prompt_token_ids = _parse_token_ids(args.prompt_token_ids)
    else:
        prompt_token_ids = cast(list[int], tok.encode(args.text, add_special_tokens=False))

    if not prompt_token_ids:
        raise ValueError("prompt_token_ids 为空，请检查 --prompt-token-ids 或 --text")

    llm = OmniStageLLM(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        # 走 vllm-omni 的 AR worker/runner（dummy/profile 也会解包 OmniOutput）
        model_arch="Mammothmoda2Model",
        model_stage="ar",
        worker_cls="vllm_omni.worker.gpu_ar_worker.GPUARWorker",
        scheduler_cls="vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler",
        # 单卡/单进程优先，避免引入额外变量
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=max(256, len(prompt_token_ids) + args.max_tokens + 8),
        enforce_eager=True,
        disable_log_stats=True,
    )

    sampling_params = SamplingParams(
        max_tokens=max(1, int(args.max_tokens)),
        temperature=0.0,
        top_p=1.0,
        logprobs=max(0, int(args.top_logprobs)),
    )

    outputs = llm.generate(
        prompts=[TokensPrompt(prompt_token_ids=prompt_token_ids)],
        sampling_params=sampling_params,
    )

    out = outputs[0]
    gen = out.outputs[0]
    print("output_len:", len(gen.token_ids))
    print(
        "generated_text(skip_special_tokens=True):",
        tok.decode(gen.token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
    )
    print(
        "generated_text(skip_special_tokens=False):",
        tok.decode(gen.token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
    )
    if gen.logprobs:
        print("top_logprobs(first_token):")
        first = gen.logprobs[0]
        for token_id, logprob in first.items():
            print(f"  {token_id}: {logprob}")


if __name__ == "__main__":
    main()
