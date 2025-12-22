#!/usr/bin/env python3
import argparse


def extract_prompt(line: str) -> str | None:
    # Extract the content between the first '|' and the second '|'
    i = line.find("|")
    if i == -1:
        return None
    j = line.find("|", i + 1)
    if j == -1:
        return None
    return line[i + 1 : j].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input .lst file path")
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument(
        "--topk",
        "-k",
        type=int,
        default=100,
        help="Extract the top K prompts (default: 100)",
    )
    args = parser.parse_args()

    prompts = []
    with open(args.input, encoding="utf-8", errors="ignore") as f:
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
