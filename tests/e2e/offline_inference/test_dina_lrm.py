"""Numerical equivalence test: DRMInferencer vs DiNaLRMPipeline.

Verifies that the official diffusion_rm inference path (DRMInferencer) and
the vLLM-Omni integration path (DiNaLRMPipeline) produce numerically identical
reward scores when given the same inputs, the same checkpoint, and the same
random seed for noise generation.

Prerequisites
─────────────
  pip install -e /path/to/diffusion-rm   # diffusion_rm package
  pip install vllm-omni                  # for DiffusionOutput in forward()

Usage
─────
  python tests/test_vllm_omni_equivalence.py \
      --rm-model  liuhuohuo/DiNa-LRM-SD35M-12layers \
      --sd3-model stabilityai/stable-diffusion-3.5-medium \
      [--image-path path/to/image.png] \
      [--u 0.1] \
      [--device cuda:0] \
      [--atol 5e-3]
"""

from __future__ import annotations

import argparse
import gc
import importlib
import os
import sys

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Dependency checks  (run before anything else so the error is clear)
# ──────────────────────────────────────────────────────────────────────────────


def _check_dependencies() -> None:
    """Verify that required packages are importable and print their versions."""
    errors: list[str] = []

    # ── diffusion_rm ──────────────────────────────────────────────────────────
    try:
        import diffusion_rm  # noqa: F401

        dr_version = getattr(diffusion_rm, "__version__", "unknown")
        print(f"  [OK] diffusion_rm       version = {dr_version}")
    except ImportError:
        errors.append("diffusion_rm is NOT installed.\n  Fix: pip install -e /path/to/diffusion-rm")

    # ── vllm_omni ─────────────────────────────────────────────────────────────
    try:
        import vllm_omni  # noqa: F401

        vo_version = getattr(vllm_omni, "__version__", "unknown")
        print(f"  [OK] vllm_omni          version = {vo_version}")
    except ImportError:
        errors.append("vllm_omni is NOT installed.\n  Fix: pip install vllm-omni")

    # ── other required packages ───────────────────────────────────────────────
    _required = [
        ("diffusers", "diffusers"),
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("omegaconf", "omegaconf"),
        ("yaml", "pyyaml"),
        ("huggingface_hub", "huggingface_hub"),
        ("PIL", "pillow"),
    ]
    for module_name, pip_name in _required:
        try:
            mod = importlib.import_module(module_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  [OK] {module_name:<20s} version = {ver}")
        except ImportError:
            errors.append(f"{module_name} is NOT installed.\n  Fix: pip install {pip_name}")

    if errors:
        print("\n[ABORT] Dependency check failed:\n")
        for err in errors:
            print(f"  ✗ {err}\n")
        sys.exit(1)

    print("\nAll dependencies satisfied. Proceeding with test.\n")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _make_test_image(size: tuple[int, int] = (512, 512)) -> Image.Image:
    """Create a deterministic synthetic RGB image."""
    rng = np.random.RandomState(seed=0)
    arr = rng.randint(0, 256, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _image_to_latent(pipe, image: Image.Image, device, dtype) -> torch.Tensor:
    """Encode a PIL image to SD3 latent space using the provided pipeline VAE."""
    transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
    img_t = transform(image).unsqueeze(0).to(device, dtype=dtype)
    with torch.no_grad():
        latents = pipe.vae.encode(img_t).latent_dist.sample()
        latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    return latents


def _print_score(label: str, scores: torch.Tensor) -> None:
    vals = scores.float().cpu().tolist()
    print(f"  {label:<30s} {', '.join(f'{v:.6f}' for v in vals)}")


# ──────────────────────────────────────────────────────────────────────────────
# Path A – official DRMInferencer
# ──────────────────────────────────────────────────────────────────────────────


def run_official(args, prompts, image, device, dtype) -> torch.Tensor:
    """Run inference using the official DRMInferencer from diffusion_rm.

    Returns raw reward scores as a float32 CPU tensor of shape (B,).
    """
    from diffusers import StableDiffusion3Pipeline
    from diffusion_rm.infer.inference import DRMInferencer
    from diffusion_rm.models.sd3_rm import encode_prompt

    print("[A] Loading SD3.5-M pipeline (official path) …")
    local = os.path.exists(args.sd3_model)
    pipe = StableDiffusion3Pipeline.from_pretrained(args.sd3_model, torch_dtype=dtype, local_files_only=local)
    for comp in [
        pipe.vae,
        pipe.text_encoder,
        pipe.text_encoder_2,
        pipe.text_encoder_3,
        pipe.transformer,
    ]:
        comp.to(device, dtype=dtype)

    print("[A] Loading DRMInferencer …")
    scorer = DRMInferencer(
        pipeline=pipe,
        config_path=None,
        model_path=args.rm_model,
        device=device,
        model_dtype=dtype,
        load_from_disk=os.path.exists(args.rm_model),
    )

    text_encoders = [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3]
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3]
    with torch.no_grad():
        prompt_embeds, pooled_embeds = encode_prompt(text_encoders, tokenizers, prompts, max_sequence_length=256)
    prompt_embeds = prompt_embeds.to(device)
    pooled_embeds = pooled_embeds.to(device)

    latents = _image_to_latent(pipe, image, device, dtype)

    torch.manual_seed(42)
    with torch.no_grad():
        scores = scorer.reward(
            text_conds={
                "encoder_hidden_states": prompt_embeds,
                "pooled_projections": pooled_embeds,
            },
            latents=latents,
            u=args.u,
        )

    result = scores[0].float().cpu()
    _print_score("DRMInferencer (official)", result)

    del scorer, pipe
    gc.collect()
    torch.cuda.empty_cache()

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Path B – OmniDiffusion.generate (vLLM-Omni integration)
# ──────────────────────────────────────────────────────────────────────────────


def run_vllm_omni(args, prompts, image, device, dtype) -> torch.Tensor:
    """Run inference using OmniDiffusion.generate (the vLLM-Omni integration path).

    Returns raw reward scores as a float32 CPU tensor of shape (B,).
    """
    from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    dtype_str = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }.get(dtype, "bfloat16")

    print("[B] Loading DiNaLRMPipeline via OmniDiffusion …")
    client = OmniDiffusion(model=args.rm_model, dtype=dtype_str)

    request_prompts = {"prompt": prompts[0], "multi_modal_data": {"image": image}}
    sampling_params = OmniDiffusionSamplingParams(
        extra_args={"noise_level": args.u},
    )

    torch.manual_seed(42)
    outputs = client.generate(request_prompts, sampling_params)

    scores = outputs[0].latents[0].float().cpu()
    _print_score("OmniDiffusion (vllm-omni)", scores)

    del client
    gc.collect()
    torch.cuda.empty_cache()

    return scores


# ──────────────────────────────────────────────────────────────────────────────
# Test
# ──────────────────────────────────────────────────────────────────────────────


def test_same_noise_seed(args, device, dtype, image) -> None:
    """Both paths use add_noise=True with the same random seed (42).
    Scores must match within the specified floating-point tolerance.
    """
    prompts = [args.prompts[0]]

    score_official = run_official(args, prompts, image, device, dtype)
    score_vllm = run_vllm_omni(args, prompts, image, device, dtype)

    max_diff = (score_official - score_vllm).abs().max().item()
    passed = torch.allclose(score_official, score_vllm, atol=args.atol, rtol=0.0)
    status = "PASS ✓" if passed else "FAIL ✗"

    print(f"\n  [{status}] max_abs_diff = {max_diff:.3e}  (atol = {args.atol:.1e})")

    if not passed:
        print(f"  official  = {score_official.tolist()}")
        print(f"  vllm-omni = {score_vllm.tolist()}")
        raise AssertionError(f"Equivalence check FAILED: max_abs_diff = {max_diff:.3e}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Equivalence test: DRMInferencer vs vLLM-Omni",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--rm-model",
        default="liuhuohuo/DiNa-LRM-SD35M-12layers",
        help="DiNa-LRM HF repo ID or local checkpoint directory.",
    )
    p.add_argument(
        "--sd3-model",
        default="stabilityai/stable-diffusion-3.5-medium",
        help="SD3.5-M base model used by DRMInferencer.",
    )
    p.add_argument(
        "--image-path",
        default=None,
        help="Path to an image file to score.  Defaults to a synthetic 512×512 image.",
    )
    p.add_argument(
        "--prompts",
        nargs="+",
        default=["A girl walking in the street"],
        help="Text prompt(s) to score the image against.",
    )
    p.add_argument(
        "--u",
        type=float,
        default=0.1,
        help="Noise sigma passed to the reward model.",
    )
    p.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    p.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model dtype.",
    )
    p.add_argument(
        "--atol",
        type=float,
        default=5e-3,
        help=("Absolute tolerance for score comparison.  bfloat16 precision is ~1e-2; 5e-3 is a reasonable default."),
    )
    return p.parse_args()


def main() -> None:
    print("=" * 64)
    print("Dependency check")
    print("=" * 64)
    _check_dependencies()

    args = parse_args()
    device = torch.device(args.device)
    dtype = _DTYPE_MAP[args.dtype]

    if args.image_path and os.path.exists(args.image_path):
        image = Image.open(args.image_path).convert("RGB")
        print(f"[setup] Using image: {args.image_path}")
    else:
        image = _make_test_image()
        print("[setup] Using synthetic test image (512×512).")

    print("\n" + "=" * 64)
    print("test_same_noise_seed  (add_noise=True, seed=42)")
    print("=" * 64)

    try:
        test_same_noise_seed(args, device, dtype, image)
    except AssertionError as e:
        print(f"\nRESULT: FAILED — {e}")
        sys.exit(1)

    print("\nRESULT: PASSED ✓")
    print("DRMInferencer and DiNaLRMPipeline produce numerically equivalent scores.")


if __name__ == "__main__":
    main()