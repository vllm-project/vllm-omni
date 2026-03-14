#!/usr/bin/env python3
"""Runner script for T2I/I2I accuracy benchmarks.

This script provides a command-line interface for running accuracy benchmarks
on vLLM-Omni generated images.

Examples:
    # Run T2I benchmark
    python run_omni_accuracy.py --mode t2i --prompts prompts.txt --images ./generated/

    # Run I2I benchmark
    python run_omni_accuracy.py --mode i2i --original ./original/ --edited ./edited/ \
        --instructions instructions.txt

    # Run both
    python run_omni_accuracy.py --mode both --config benchmark_config.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def load_images_from_dir(image_dir: str, extensions: tuple = (".png", ".jpg", ".jpeg")) -> list[Any]:
    """Load images from a directory.

    Args:
        image_dir: Path to directory containing images
        extensions: Tuple of valid image extensions

    Returns:
        List of loaded images (PIL Images)
    """
    try:
        from PIL import Image
    except ImportError:
        logging.error("PIL not installed. Install with: pip install Pillow")
        raise

    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in extensions])

    if not image_paths:
        logging.warning(f"No images found in {image_dir}")
        return []

    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            logging.debug(f"Loaded image: {path}")
        except Exception as e:
            logging.warning(f"Failed to load image {path}: {e}")

    logging.info(f"Loaded {len(images)} images from {image_dir}")
    return images


def load_config(config_path: str) -> dict:
    """Load benchmark configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def run_t2i_benchmark(args) -> dict:
    """Run T2I accuracy benchmark.

    Args:
        args: Command line arguments

    Returns:
        Benchmark results dictionary
    """
    logging.info("Running T2I benchmark...")

    # Import here to handle optional dependencies gracefully
    try:
        from benchmarks.accuracy import T2IEvaluator
    except ImportError as e:
        logging.error(f"Failed to import T2I evaluator: {e}. Install with: pip install -e '.[eval]'")
        raise

    # Load prompts
    if args.prompts:
        with open(args.prompts) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = args.prompt_list or []

    # Load images
    images = load_images_from_dir(args.images)

    # Initialize evaluator
    evaluator = T2IEvaluator(
        use_vqascore=args.use_vqascore,
        use_geneval=args.use_geneval,
        vlm_model=args.vlm_model,
    )

    # Run evaluation
    results = evaluator.evaluate(prompts, images)

    return results


def run_i2i_benchmark(args) -> dict:
    """Run I2I accuracy benchmark.

    Args:
        args: Command line arguments

    Returns:
        Benchmark results dictionary
    """
    logging.info("Running I2I benchmark...")

    try:
        from benchmarks.accuracy import I2IEvaluator
    except ImportError as e:
        logging.error(f"Failed to import I2I evaluator: {e}. Install with: pip install -e '.[eval]'")
        raise

    # Load instructions
    if args.instructions:
        with open(args.instructions) as f:
            instructions = [line.strip() for line in f if line.strip()]
    else:
        instructions = args.instruction_list or []

    # Load images
    original_images = load_images_from_dir(args.original)
    edited_images = load_images_from_dir(args.edited)

    # Initialize evaluator
    evaluator = I2IEvaluator(
        use_lpips=args.use_lpips,
        use_vlm_judge=args.use_vlm_judge,
        lpips_net=args.lpips_net,
        vlm_model=args.vlm_model,
        device=args.device,
    )

    # Run evaluation
    results = evaluator.evaluate(original_images, edited_images, instructions)

    return results


def save_results(results: dict, output_path: str):
    """Save benchmark results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logging.info(f"Results saved to {output_path}")


def print_results(results: dict):
    """Print benchmark results in a readable format."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    if "vqascore" in results:
        print("\n[T2I - VQAScore]")
        print(f"  Mean: {results['vqascore']['vqascore_mean']:.4f}")

    if "geneval" in results:
        print("\n[T2I - GenEval]")
        for key, value in results["geneval"].items():
            if key != "per_sample":
                print(f"  {key}: {value:.4f}")

    if "lpips" in results:
        print("\n[I2I - LPIPS]")
        print(f"  Mean: {results['lpips']['lpips_mean']:.4f}")
        print(f"  Std:  {results['lpips']['lpips_std']:.4f}")

    if "vlm_judge" in results:
        print("\n[I2I - VLM Judge]")
        for key, value in results["vlm_judge"].items():
            if isinstance(value, dict) and "mean" in value:
                print(f"  {key}: {value['mean']:.4f}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run T2I/I2I accuracy benchmarks for vLLM-Omni")

    # Mode selection
    parser.add_argument("--mode", choices=["t2i", "i2i", "both"], default="both", help="Benchmark mode")

    # Config file
    parser.add_argument("--config", type=str, help="Path to config JSON file")

    # T2I arguments
    parser.add_argument("--prompts", type=str, help="Path to file containing prompts (one per line)")
    parser.add_argument("--prompt-list", nargs="+", help="List of prompts as command line arguments")
    parser.add_argument("--images", type=str, help="Directory containing generated images")

    # I2I arguments
    parser.add_argument("--original", type=str, help="Directory containing original images")
    parser.add_argument("--edited", type=str, help="Directory containing edited images")
    parser.add_argument("--instructions", type=str, help="Path to file containing edit instructions")
    parser.add_argument(
        "--instruction-list",
        nargs="+",
        help="List of instructions as command line arguments",
    )

    # Metric selection (default all enabled, can disable with --no-* flags)
    parser.add_argument(
        "--use-vqascore",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use VQAScore for T2I (default: True)",
    )
    parser.add_argument(
        "--use-geneval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GenEval for T2I (default: True)",
    )
    parser.add_argument(
        "--use-lpips",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use LPIPS for I2I (default: True)",
    )
    parser.add_argument(
        "--use-vlm-judge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use VLM judge for I2I (default: True)",
    )

    # Model configuration
    parser.add_argument(
        "--vlm-model",
        type=str,
        default="Qwen2.5-VL-7B",
        help="VLM model to use as judge",
    )
    parser.add_argument(
        "--lpips-net",
        type=str,
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="LPIPS network backbone",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda/cpu)")

    # Output
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file path")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Load config if provided (config overrides argparse defaults)
    if args.config:
        config = load_config(args.config)
        # Override args with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    # Run benchmarks
    results = {}

    try:
        if args.mode in ["t2i", "both"]:
            if not (args.prompts or args.prompt_list):
                logging.error("T2I benchmark requires --prompts or --prompt-list")
                sys.exit(1)
            t2i_results = run_t2i_benchmark(args)
            results.update(t2i_results)

        if args.mode in ["i2i", "both"]:
            if not (args.original and args.edited):
                logging.error("I2I benchmark requires --original and --edited")
                sys.exit(1)
            i2i_results = run_i2i_benchmark(args)
            results.update(i2i_results)

    except Exception as e:
        logging.error(f"Benchmark failed: {e}")
        raise

    # Save and print results
    save_results(results, args.output)
    print_results(results)


if __name__ == "__main__":
    main()
