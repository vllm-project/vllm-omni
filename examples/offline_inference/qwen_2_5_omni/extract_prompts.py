#!/usr/bin/env python3
import argparse
from typing import Optional
import os
import torch


def extract_prompt(line: str) -> Optional[str]:
    # 提取第一个 '|' 与第二个 '|' 之间的内容
    i = line.find('|')
    if i == -1:
        return None
    j = line.find('|', i + 1)
    if j == -1:
        return None
    return line[i + 1:j].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="输入 .lst 文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出文件路径")
    parser.add_argument("--topk", "-k", type=int, default=100, help="提取前 K 个 prompt（默认 100）")
    parser.add_argument("--pt-output", "-p", type=str, default=None, help="可选：输出 .pt 文件路径（默认与 --output 同名 .pt）")
    args = parser.parse_args()

    prompts = []
    with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if len(prompts) >= args.topk:
                break
            p = extract_prompt(line.rstrip("\n"))
            if p:
                prompts.append(p)

    with open(args.output, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(p + "\n")

    # 另存为 PyTorch 序列化的 .pt，内容为字符串列表
    pt_path = args.pt_output if args.pt_output else os.path.splitext(args.output)[0] + ".pt"
    torch.save(prompts, pt_path)


if __name__ == "__main__":
    main()


