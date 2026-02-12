#!/usr/bin/env python3
"""
Extract prompts from meta.lst and save them to a txt file.

Each line in meta.lst has the format:
ID|prompt_text|audio_path|target_text

This script extracts the prompt_text (second field) from the first N lines.
"""

import argparse
from pathlib import Path


def extract_prompts(input_file: str, output_file: str, num_lines: int) -> None:
    """
    Extract prompts from meta.lst and save to output file.

    Args:
        input_file: Path to the meta.lst file
        output_file: Path to the output txt file
        num_lines: Number of lines to process
    """
    prompts = []

    with open(input_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_lines:
                break

            line = line.strip()
            if not line:  # Skip empty lines
                continue

            parts = line.split("|")
            if len(parts) >= 2:
                prompt = parts[1]  # The prompt is the second field
                prompts.append(prompt)

    # Write prompts to output file
    with open(output_file, "w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(prompt + "\n")

    # Print result stats
    print(f"Extracted {len(prompts)} prompts from first {num_lines} lines")
    print(f"Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract prompts from meta.lst file")
    parser.add_argument(
        "-i", "--input", type=str, default="meta.lst", help="Input meta.lst file path (default: meta.lst)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="prompts.txt", help="Output txt file path (default: prompts.txt)"
    )
    parser.add_argument(
        "-n", "--num-lines", type=int, required=True, help="Number of lines to extract from the beginning"
    )

    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' not found")
        return

    extract_prompts(args.input, args.output, args.num_lines)


if __name__ == "__main__":
    main()
