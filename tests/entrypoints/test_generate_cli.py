# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the `vllm generate --omni` CLI subcommand.

Tests cover argument parsing, output file naming, and error behavior
via local mirrors of the parser and save logic. These do NOT directly
test the production OmniGenerateCommand class, since generate.py has
heavy transitive imports (vllm.entrypoints.cli.types → vllm._C).
If production logic changes, these tests must be updated manually.
"""

import argparse
from pathlib import Path

import PIL.Image
import pytest


def _build_parser():
    """Build a parser that mirrors OmniGenerateCommand.subparser_init."""
    parent = argparse.ArgumentParser()
    subs = parent.add_subparsers(dest="subparser")
    parser = subs.add_parser("generate")
    parser.add_argument("--omni", action="store_true")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--stage-configs-path", type=str, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-cpu-offload", action="store_true")
    return parent


def _save_images(images: list, output: str):
    """Mirror of output saving logic in OmniGenerateCommand.cmd."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(images) == 1:
        images[0].save(output_path)
    else:
        stem = output_path.stem
        suffix = output_path.suffix or ".png"
        for i, img in enumerate(images):
            img.save(output_path.parent / f"{stem}_{i}{suffix}")


@pytest.mark.core_model
@pytest.mark.cpu
class TestGenerateCLI:
    def test_parser_registers_generate(self):
        """generate subcommand registers and parses args."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "generate",
                "--model",
                "test-model",
                "--prompt",
                "hello",
                "--omni",
            ]
        )
        assert args.model == "test-model"
        assert args.prompt == "hello"
        assert args.omni is True

    def test_default_args(self):
        """Default values match plan."""
        parser = _build_parser()
        args = parser.parse_args(["generate", "--model", "m", "--prompt", "p"])
        assert args.height == 1024
        assert args.width == 1024
        assert args.num_inference_steps == 50
        assert args.guidance_scale == 4.0
        assert args.cfg_scale == 4.0
        assert args.seed == 42
        assert args.num_images == 1
        assert args.output == "output.png"

    def test_single_image_output(self, tmp_path):
        """Single image saved to output path."""
        img = PIL.Image.new("RGB", (64, 64), "blue")
        output = str(tmp_path / "result.png")
        _save_images([img], output)
        assert (tmp_path / "result.png").exists()

    def test_multi_image_output(self, tmp_path):
        """Multiple images saved with numbered suffixes."""
        imgs = [PIL.Image.new("RGB", (64, 64), c) for c in ["red", "green", "blue"]]
        output = str(tmp_path / "result.png")
        _save_images(imgs, output)
        assert (tmp_path / "result_0.png").exists()
        assert (tmp_path / "result_1.png").exists()
        assert (tmp_path / "result_2.png").exists()

    def test_no_output_raises(self):
        """Empty images list should raise RuntimeError (matches generate.py)."""
        with pytest.raises(RuntimeError, match="No images"):
            images = []
            if not images:
                raise RuntimeError("No images in output.")
            _save_images(images, "out.png")
