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
    parser.add_argument("--input", "-i", required=True, help="Input .lst file path")
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument("--topk", "-k", type=int, default=100, help="Extract the top K prompts (default: 100)")
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


if __name__ == "__main__":
    main()


