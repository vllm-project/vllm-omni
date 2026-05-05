"""
Pipeline comparison: reference (moonvalley_ai) vs vllm-omni Marey inference.

Isolates each pipeline component and compares outputs between the two
implementations using deterministic inputs and fixed seeds.

Run with:
    PYTHONPATH=/home/david/repos/moonvalley_ai \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run --project /home/david/repos/vllm-omni \
    python /home/david/repos/vllm-omni/examples/offline_inference/marey/compare_pipelines.py

Tests:
  1. Guidance schedule (pure math, no GPU)
  2. Timestep schedule (pure math, no GPU)
  3. Text encoding (GPU)
  4. Transformer single forward pass (GPU)
  5. Multi-step denoising loop (GPU)
  6. VAE decode (GPU)
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = "/app/wlam/models/checkpoints/marey/distilled-0001/config.yaml"
DEFAULT_TRANSFORMER_WEIGHTS = (
    "/app/wlam/models/checkpoints/marey/distilled-0001/epoch0-global_step5000_distilled/ema_inference_ckpt.safetensors"
)
DEFAULT_VAE_WEIGHTS = "/app/wlam/models/checkpoints/marey/vae/epoch_4_step2819000.ckpt"

DEFAULT_PROMPT = (
    "Detailed Description: A majestic, aged eagle with mottled golden-brown "
    "feathers soars gracefully through a vast, ancient indoor chamber."
)
DEFAULT_NEGATIVE_PROMPT = (
    "<synthetic> <scene cut> gopro, bright, contrast, static, overexposed, bright, "
    "vignette, artifacts, still, noise, texture, scanlines, videogame, 360 camera, "
    "VR, transition, flare, saturation, distorted, warped, wide angle, contrast, "
    "saturated, vibrant, glowing, cross dissolve, texture, videogame, saturation, "
    "cheesy, ugly hands, mutated hands, mutant, disfigured, extra fingers, blown out, "
    "horrible, blurry, worst quality, bad, transition, dissolve, cross-dissolve, melt, "
    "fade in, fade out, wobbly, weird, low quality, plastic, stock footage, video camera, "
    "boring, static"
)


def parse_args():
    p = argparse.ArgumentParser(description="Compare vllm-omni vs reference Marey inference.")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--transformer-weights", default=DEFAULT_TRANSFORMER_WEIGHTS)
    p.add_argument("--vae-weights", default=DEFAULT_VAE_WEIGHTS)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--num-frames", type=int, default=33)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-steps", type=int, default=4)
    p.add_argument("--cooldown-steps", type=int, default=18)
    p.add_argument("--guidance-every-n-steps", type=int, default=2)
    p.add_argument("--flow-shift", type=float, default=3.0)
    p.add_argument(
        "--skip", nargs="*", default=[], help="Tests to skip: guidance timesteps encoding forward denoise vae"
    )
    p.add_argument("--only", nargs="*", default=[], help="Run only these tests (same names as --skip)")
    return p.parse_args()


# ============================================================================
# Utilities
# ============================================================================


def _banner(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _compare_tensors(name: str, a: torch.Tensor, b: torch.Tensor, rtol=1e-3, atol=1e-4):
    """Print detailed comparison between two tensors."""
    a_f = a.float().cpu()
    b_f = b.float().cpu()
    diff = (a_f - b_f).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    a_std = a_f.std().item()
    b_std = b_f.std().item()
    cos_sim = torch.nn.functional.cosine_similarity(a_f.flatten().unsqueeze(0), b_f.flatten().unsqueeze(0)).item()

    match = torch.allclose(a_f, b_f, rtol=rtol, atol=atol)
    status = "MATCH" if match else "MISMATCH"
    print(f"  [{status}] {name}:")
    print(f"    shape: {list(a.shape)} vs {list(b.shape)}")
    print(f"    mean:  {a_f.mean().item():.6f} vs {b_f.mean().item():.6f}")
    print(f"    std:   {a_std:.6f} vs {b_std:.6f}")
    print(f"    max_diff: {max_diff:.8f}  mean_diff: {mean_diff:.8f}")
    print(f"    cosine_sim: {cos_sim:.8f}")
    if not match:
        rel_diff = diff / (a_f.abs() + 1e-8)
        print(f"    max_rel_diff: {rel_diff.max().item():.8f}")
        # Show first few differing elements
        flat_diff = diff.flatten()
        topk = min(5, flat_diff.numel())
        vals, idxs = flat_diff.topk(topk)
        print(f"    top-{topk} diffs: {vals.tolist()}")
    return match


def _should_run(test_name: str, args) -> bool:
    if args.only:
        return test_name in args.only
    return test_name not in args.skip


# ============================================================================
# Test 1: Guidance Schedule
# ============================================================================


def reference_guidance_schedule(
    guidance_scale: float,
    num_steps: int,
    warmup_steps: int,
    cooldown_steps: int,
    guidance_every_n_steps: int,
    guidance_during_warmup: bool = True,
    guidance_during_cooldown: bool = False,
) -> list[float]:
    """Exact copy of opensora create_guidance_schedule."""
    guidance_schedule = np.ones(num_steps)
    if warmup_steps > 0 and guidance_during_warmup:
        guidance_schedule[:warmup_steps] = guidance_scale
    if cooldown_steps > 0 and guidance_during_cooldown:
        guidance_schedule[-cooldown_steps:] = guidance_scale
    main_phase_steps = num_steps - warmup_steps - cooldown_steps
    if main_phase_steps > 0:
        oscillation = np.ones(main_phase_steps)
        start_idx = 1 if guidance_during_warmup else 0
        oscillation[start_idx::guidance_every_n_steps] = guidance_scale
        guidance_schedule[warmup_steps : warmup_steps + main_phase_steps] = oscillation
    return guidance_schedule.tolist()


def vllm_guidance_schedule(
    num_steps: int,
    guidance_scale: float,
    warmup_steps: int,
    cooldown_steps: int,
    guidance_every_n_steps: int,
) -> list[float]:
    """Exact copy of vllm-omni build_guidance_schedule (fixed version)."""
    cooldown_start = num_steps - cooldown_steps
    schedule = []
    for i in range(num_steps):
        if i < warmup_steps:
            schedule.append(guidance_scale)
        elif i >= cooldown_start:
            schedule.append(1.0)
        else:
            middle_idx = i - warmup_steps
            if middle_idx > 0 and (middle_idx - 1) % guidance_every_n_steps == 0:
                schedule.append(guidance_scale)
            else:
                schedule.append(1.0)
    return schedule


def test_guidance_schedule(args):
    _banner("Test 1: Guidance Schedule Comparison")

    ref = reference_guidance_schedule(
        args.guidance_scale,
        args.steps,
        args.warmup_steps,
        args.cooldown_steps,
        args.guidance_every_n_steps,
    )
    vllm = vllm_guidance_schedule(
        args.steps,
        args.guidance_scale,
        args.warmup_steps,
        args.cooldown_steps,
        args.guidance_every_n_steps,
    )

    ref_active = sum(1 for g in ref if g > 1.0)
    vllm_active = sum(1 for g in vllm if g > 1.0)
    print(f"  Reference: {ref_active}/{len(ref)} steps with guidance active")
    print(f"  vllm-omni: {vllm_active}/{len(vllm)} steps with guidance active")

    mismatches = []
    for i, (r, v) in enumerate(zip(ref, vllm)):
        if abs(r - v) > 1e-6:
            mismatches.append((i, r, v))

    if not mismatches:
        print("  [MATCH] Guidance schedules are identical")
    else:
        print(f"  [MISMATCH] {len(mismatches)} steps differ:")
        for i, r, v in mismatches[:20]:
            phase = (
                "warmup" if i < args.warmup_steps else ("cooldown" if i >= args.steps - args.cooldown_steps else "main")
            )
            mid = i - args.warmup_steps if phase == "main" else None
            print(f"    step {i:3d} ({phase}, mid_idx={mid}): ref={r:.1f}  vllm={v:.1f}")
        if len(mismatches) > 20:
            print(f"    ... and {len(mismatches) - 20} more")

        print(
            "\n  ROOT CAUSE: Reference oscillation uses numpy slicing "
            "`oscillation[1::n] = gs` (1-indexed from main start),"
        )
        print("  while vllm-omni uses `middle_idx > 0 and middle_idx % n == 0` (2-indexed from main start).")
        print("  The first active guidance step in the main phase is off by 1.")

    return len(mismatches) == 0


# ============================================================================
# Test 2: Timestep Schedule
# ============================================================================


def reference_timesteps(num_steps, shift, tmin=0.001, tmax=1.0, teacher_steps=100, num_timesteps=1000):
    """Reference RFLOW draw_time + distilled subsampling."""
    timesteps_sigma = torch.linspace(tmax, tmin, teacher_steps)
    timesteps_raw = timesteps_sigma * num_timesteps  # shape [teacher_steps]
    if shift is not None and shift > 0:
        # timestep_shift: t/1000, shift, *1000
        t_norm = timesteps_raw / num_timesteps
        timesteps_raw = (shift * t_norm / (1.0 + (shift - 1.0) * t_norm)) * num_timesteps
    num_distilled_steps = teacher_steps // num_steps
    return [timesteps_raw[i * num_distilled_steps].unsqueeze(0) for i in range(num_steps)]


def vllm_timesteps(num_steps, shift, tmin=0.001, tmax=1.0, teacher_steps=100, num_timesteps=1000):
    """vllm-omni create_flow_timesteps."""
    sigmas = torch.linspace(tmax, tmin, teacher_steps)
    if shift is not None and shift > 0:
        sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    timesteps_all = sigmas * num_timesteps
    stride = max(1, teacher_steps // num_steps)
    return [timesteps_all[i * stride].unsqueeze(0) for i in range(num_steps)]


def test_timestep_schedule(args):
    _banner("Test 2: Timestep Schedule Comparison")

    ref = reference_timesteps(args.steps, args.flow_shift)
    vllm = vllm_timesteps(args.steps, args.flow_shift)

    print(f"  Reference: {len(ref)} timesteps, first={ref[0].item():.4f}, last={ref[-1].item():.4f}")
    print(f"  vllm-omni: {len(vllm)} timesteps, first={vllm[0].item():.4f}, last={vllm[-1].item():.4f}")

    max_diff = 0.0
    for i, (r, v) in enumerate(zip(ref, vllm)):
        d = abs(r.item() - v.item())
        max_diff = max(max_diff, d)

    match = max_diff < 1e-4
    status = "MATCH" if match else "MISMATCH"
    print(f"  [{status}] Max timestep difference: {max_diff:.8f}")
    if not match:
        for i, (r, v) in enumerate(zip(ref, vllm)):
            d = abs(r.item() - v.item())
            if d > 1e-4:
                print(f"    step {i}: ref={r.item():.6f}  vllm={v.item():.6f}  diff={d:.6f}")
    return match


# ============================================================================
# Test 3: Text Encoding
# ============================================================================


def _load_reference_text_encoder(config, device, dtype):
    """Load the reference MQTextEncoder by directly calling the factory.

    Avoids using the opensora registry which triggers heavyweight imports
    (datasets, dask, etc.).
    """
    import types

    # Stub out modules that the opensora import chain would pull in
    for mod_name in (
        "opensora.datasets",
        "opensora.datasets.utils",
        "opensora.datasets.video_transforms",
        "opensora.datasets.datasets",
        "dask",
        "dask.dataframe",
    ):
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            stub.__path__ = []
            stub.__package__ = mod_name
            if mod_name == "opensora.datasets":
                stub.IMG_FPS = 120
            sys.modules[mod_name] = stub

    from opensora.models.text_encoder.custom_encoders import ul2_clip_quote

    te_cfg = config.get("text_encoder", {})
    return ul2_clip_quote(
        ul2_pretrained=te_cfg.get("ul2_pretrained", "google/ul2"),
        clip_pretrained=te_cfg.get("clip_pretrained", "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"),
        byt5_pretrained=te_cfg.get("byt5_pretrained", "google/byt5-large"),
        ul2_vars=te_cfg.get("ul2_vars", "seq"),
        clip_vars=te_cfg.get("clip_vars", "vector"),
        ul2_max_length=te_cfg.get("ul2_max_length", 300),
        clip_max_length=te_cfg.get("clip_max_length", 77),
        byt5_max_length=te_cfg.get("byt5_max_length", 70),
        device=str(device),
        dtype=dtype,
    )


def _extract_quotes_reference(prompt: str) -> str:
    """Reference spaCy-based quote extraction."""
    import spacy

    nlp = spacy.load("en_core_web_sm")
    tokens = nlp(prompt)
    text_quotes = ""
    for sentence in tokens.sents:
        find_quotes = [pos for pos, char in enumerate(sentence) if char == '"']
        if len(find_quotes) % 2 == 0:
            text_quotes = sentence
            break
    return str(text_quotes)


def _extract_quotes_vllm(text: str) -> str:
    """vllm-omni regex-based quote extraction."""
    import re

    matches = re.findall(r'["\u201c\u201d](.*?)["\u201c\u201d]', text)
    return " ".join(matches) if matches else text


def test_text_encoding(args, config, device, dtype):
    _banner("Test 3: Text Encoding Comparison")

    # --- Quote extraction ---
    ref_quote = _extract_quotes_reference(args.prompt)
    vllm_quote = _extract_quotes_vllm(args.prompt)
    print(f"  Reference quote: {repr(ref_quote[:80])}...")
    print(f"  vllm-omni quote: {repr(vllm_quote[:80])}...")
    if ref_quote == vllm_quote:
        print("  [MATCH] Quote extraction results are identical")
    else:
        print("  [MISMATCH] Quote extraction differs!")
        print(f"    ref len={len(ref_quote)}, vllm len={len(vllm_quote)}")

    # --- Load reference encoder ---
    print("\n  Loading reference text encoder (opensora MQTextEncoder)...")
    ref_encoder = _load_reference_text_encoder(config, device, dtype)

    # --- Load vllm-omni encoder ---
    print("  Loading vllm-omni text encoders...")
    from text_to_video import encode_text, load_text_encoders

    vllm_encoders = load_text_encoders(config, device, dtype)

    # --- Encode with reference ---
    print("  Encoding prompt with reference...")
    ref_cond = ref_encoder.encode([args.prompt], [ref_quote])
    ref_seq = ref_cond.seq_cond  # list of tensors
    ref_mask = ref_cond.seq_cond_mask  # [B, total_len]
    ref_vec = ref_cond.vector_cond  # list of tensors or None

    # --- Encode with vllm-omni ---
    print("  Encoding prompt with vllm-omni...")
    vllm_seq, vllm_masks, vllm_vec = encode_text(args.prompt, vllm_encoders, device, dtype)

    # --- Compare seq embeddings ---
    all_match = True
    print("\n  Sequence embeddings:")
    if isinstance(ref_seq, list):
        for i, (rs, vs) in enumerate(zip(ref_seq, vllm_seq)):
            name = ["UL2", "ByT5"][i] if i < 2 else f"enc_{i}"
            m = _compare_tensors(f"seq_cond[{i}] ({name})", rs, vs)
            all_match = all_match and m
    else:
        m = _compare_tensors(
            "seq_cond", ref_seq, torch.cat(vllm_seq, dim=1) if isinstance(vllm_seq, list) else vllm_seq
        )
        all_match = all_match and m

    # --- Compare masks ---
    print("\n  Attention masks:")
    if isinstance(vllm_masks, list):
        vllm_mask_cat = torch.cat(vllm_masks, dim=1)
    else:
        vllm_mask_cat = vllm_masks
    m = _compare_tensors("seq_cond_mask", ref_mask.float(), vllm_mask_cat.float())
    all_match = all_match and m

    # --- Compare vector embeddings ---
    print("\n  Vector embeddings (CLIP):")
    if ref_vec is not None:
        if isinstance(ref_vec, list):
            ref_v = ref_vec[0]
        else:
            ref_v = ref_vec
        m = _compare_tensors("vector_cond (CLIP)", ref_v, vllm_vec)
        all_match = all_match and m
    else:
        print("  Reference vector_cond is None, skipping")

    # --- Encode negative prompt ---
    print("\n  Negative prompt encoding:")
    ref_null = ref_encoder.encode([DEFAULT_NEGATIVE_PROMPT], [ref_quote])
    vllm_neg_seq, vllm_neg_masks, vllm_neg_vec = encode_text(
        DEFAULT_NEGATIVE_PROMPT,
        vllm_encoders,
        device,
        dtype,
        quote_override=vllm_quote,
    )

    if isinstance(ref_null.seq_cond, list):
        for i, (rs, vs) in enumerate(zip(ref_null.seq_cond, vllm_neg_seq)):
            name = ["UL2", "ByT5"][i] if i < 2 else f"enc_{i}"
            m = _compare_tensors(f"neg_seq[{i}] ({name})", rs, vs)
            all_match = all_match and m

    # Clean up
    del ref_encoder, vllm_encoders
    torch.cuda.empty_cache()

    return all_match


# ============================================================================
# Test 4: Transformer Single Forward Pass
#
# Loads each model sequentially (they don't fit on one GPU simultaneously).
# Runs one forward pass, saves the output to CPU, frees GPU, loads the next.
# ============================================================================


def _build_reference_model_args(config, args, device, dtype):
    """Build model_args dict matching reference prepare_multi_resolution_info + extra features."""
    model_args = dict(
        height=torch.tensor([args.height], device=device, dtype=dtype),
        width=torch.tensor([args.width], device=device, dtype=dtype),
        num_frames=torch.tensor([args.num_frames], device=device, dtype=dtype),
        ar=torch.tensor([args.height / args.width], device=device, dtype=dtype),
        fps=torch.tensor([24.0], device=device, dtype=dtype),
    )

    model_cfg = config.get("model", {})
    extra_cfg = model_cfg.get("extra_features_embedders", {})

    def _to_batched_tensor(x):
        return torch.tensor([x], dtype=torch.int32, device=device)

    for feat, feat_cfg in extra_cfg.items():
        if "eval_value" in feat_cfg:
            eval_value = feat_cfg["eval_value"]
            if isinstance(eval_value, (tuple, list)):
                eval_value, eval_value_upper_bound = eval_value
                model_args[f"{feat}_upper_bound"] = _to_batched_tensor(eval_value_upper_bound)
            model_args[feat] = _to_batched_tensor(eval_value)

    for key in ["dover_technical", "aesthetics_score_total"]:
        if key in model_args:
            model_args[key] = torch.ones_like(model_args[key]) * 0

    model_args["bitdepth_class"] = _to_batched_tensor(1)
    return model_args


def test_transformer_forward(args, config, device, dtype):
    _banner("Test 4: Transformer Single Forward Pass")

    vae_cfg = config.get("vae", {})
    in_channels = len(vae_cfg.get("scaling_factor", [0] * 16))
    vae_ds = tuple(vae_cfg.get("downsample_factors", (4, 16, 16)))
    vae_t, vae_s = vae_ds[0], vae_ds[1]
    time_pad = 0 if args.num_frames % vae_t == 0 else (vae_t - args.num_frames % vae_t)
    num_latent_frames = (args.num_frames + time_pad) // vae_t
    latent_h = math.ceil(args.height / vae_s)
    latent_w = math.ceil(args.width / vae_s)

    # Deterministic input
    torch.manual_seed(args.seed)
    z = torch.randn(1, in_channels, num_latent_frames, latent_h, latent_w, device=device, dtype=dtype)
    t = torch.tensor([750.0], device=device, dtype=dtype)

    # ---- Phase 1: reference model ----
    print("  Loading reference text encoder...")
    ref_encoder = _load_reference_text_encoder(config, device, dtype)

    # Grab metadata we need for model construction before freeing encoder
    ref_caption_channels = ref_encoder.output_dim
    ref_vector_cond_channels = ref_encoder.vector_dim
    ref_model_max_length = ref_encoder.model_max_length

    ref_quote = _extract_quotes_reference(args.prompt)
    ref_text_cond = ref_encoder.encode([args.prompt], [ref_quote])
    del ref_encoder
    torch.cuda.empty_cache()

    print("  Loading reference model...")
    import safetensors.torch
    from omegaconf import OmegaConf
    from opensora.model_utils import build_model_from_config
    from opensora.utils.ckpt_utils import maybe_update_model_cfg
    from opensora.utils.config_utils import to_container

    model_cfg_om = OmegaConf.create(config.get("model", {}))
    updated_model_cfg = maybe_update_model_cfg(model_cfg_om)

    ref_model = build_model_from_config(
        to_container(updated_model_cfg),
        caption_channels=ref_caption_channels,
        vector_cond_channels=ref_vector_cond_channels,
        model_max_length=ref_model_max_length,
        in_channels=in_channels,
    )
    state_dict = safetensors.torch.load_file(args.transformer_weights)
    ref_model.load_state_dict(state_dict, assign=True)
    del state_dict
    ref_model.to(device, dtype).eval()
    torch.cuda.empty_cache()

    model_args = _build_reference_model_args(config, args, device, dtype)

    print("  Running reference model forward...")
    with torch.inference_mode():
        ref_out = ref_model(z, t, text_cond=ref_text_cond, **model_args)
        ref_v = ref_out.chunk(2, dim=1)[0].cpu()

    del ref_model, ref_text_cond, model_args
    torch.cuda.empty_cache()
    print(f"  Reference output saved (shape={list(ref_v.shape)})")

    # ---- Phase 2: vllm-omni model ----
    print("  Loading vllm-omni text encoders...")
    from text_to_video import (
        build_extra_features,
        build_transformer,
        encode_text,
        load_text_encoders,
        load_transformer_weights,
    )

    vllm_encoders = load_text_encoders(config, device, dtype)
    vllm_seq, vllm_masks, vllm_vec = encode_text(args.prompt, vllm_encoders, device, dtype)
    del vllm_encoders
    torch.cuda.empty_cache()

    caption_channels = [p.shape[-1] for p in vllm_seq]
    vector_cond_channels = vllm_vec.shape[-1]
    print("  Loading vllm-omni model...")
    vllm_model = build_transformer(config, in_channels, caption_channels, vector_cond_channels)
    load_transformer_weights(vllm_model, args.transformer_weights, device, dtype)

    extra_features = build_extra_features(config, args.height, args.width, device, dtype)
    for qkey in ("dover_technical", "aesthetics_score_total"):
        if qkey in extra_features:
            extra_features[qkey] = torch.tensor([0], device=device, dtype=torch.long)

    height_t = torch.tensor([args.height], device=device, dtype=dtype)
    width_t = torch.tensor([args.width], device=device, dtype=dtype)
    fps_t = torch.tensor([24.0], device=device, dtype=dtype)

    print("  Running vllm-omni model forward...")
    with torch.inference_mode():
        vllm_out = vllm_model(
            hidden_states=z,
            timestep=t,
            encoder_hidden_states=vllm_seq,
            encoder_hidden_states_mask=vllm_masks,
            vector_cond=vllm_vec,
            height=height_t,
            width=width_t,
            fps=fps_t,
            extra_features=extra_features,
            return_dict=False,
        )[0]
        if vllm_out.shape[1] != in_channels:
            vllm_v = vllm_out[:, :in_channels].cpu()
        else:
            vllm_v = vllm_out.cpu()

    del vllm_model, vllm_out, vllm_seq, vllm_masks, vllm_vec, extra_features
    torch.cuda.empty_cache()
    # ---- Compare ----
    print(f"  vllm-omni output saved (shape={list(vllm_v.shape)})")
    match = _compare_tensors("v_pred (cond forward)", ref_v, vllm_v, rtol=1e-2, atol=1e-3)
    return match


# ============================================================================
# Test 5: Multi-step Denoising (schedule + loop math only)
#
# Since Test 4 verifies the model forward pass matches, this test verifies
# that the denoising loop logic (CFG, DDPM step, clipping) produces
# identical results when given the same model. Uses a single model instance.
# ============================================================================


def test_denoising_loop(args, config, device, dtype):
    _banner("Test 5: Multi-step Denoising Loop (3 steps, single model)")

    vae_cfg = config.get("vae", {})
    in_channels = len(vae_cfg.get("scaling_factor", [0] * 16))
    vae_ds = tuple(vae_cfg.get("downsample_factors", (4, 16, 16)))

    from text_to_video import (
        _extract_quotes,
        build_extra_features,
        build_transformer,
        encode_text,
        load_text_encoders,
        load_transformer_weights,
    )

    vllm_encoders = load_text_encoders(config, device, dtype)
    vllm_seq, vllm_masks, vllm_vec = encode_text(args.prompt, vllm_encoders, device, dtype)
    vllm_neg_seq, vllm_neg_masks, vllm_neg_vec = encode_text(
        DEFAULT_NEGATIVE_PROMPT,
        vllm_encoders,
        device,
        dtype,
        quote_override=_extract_quotes(args.prompt),
    )
    del vllm_encoders
    torch.cuda.empty_cache()

    caption_channels = [p.shape[-1] for p in vllm_seq]
    vector_cond_channels = vllm_vec.shape[-1]
    model = build_transformer(config, in_channels, caption_channels, vector_cond_channels)
    load_transformer_weights(model, args.transformer_weights, device, dtype)

    extra_features = build_extra_features(config, args.height, args.width, device, dtype)
    uncond_extra_features = dict(extra_features)
    for qkey in ("dover_technical", "aesthetics_score_total"):
        if qkey in extra_features:
            uncond_extra_features[qkey] = torch.tensor([9], device=device, dtype=torch.long)
            extra_features[qkey] = torch.tensor([0], device=device, dtype=torch.long)

    vae_t, vae_s = vae_ds[0], vae_ds[1]
    time_pad = 0 if args.num_frames % vae_t == 0 else (vae_t - args.num_frames % vae_t)
    num_latent_frames = (args.num_frames + time_pad) // vae_t
    latent_h = math.ceil(args.height / vae_s)
    latent_w = math.ceil(args.width / vae_s)
    shape = (1, in_channels, num_latent_frames, latent_h, latent_w)

    num_test_steps = 3
    ref_ts = reference_timesteps(args.steps, args.flow_shift)[:num_test_steps]
    vllm_ts = vllm_timesteps(args.steps, args.flow_shift)[:num_test_steps]

    ref_sched = reference_guidance_schedule(
        args.guidance_scale,
        args.steps,
        args.warmup_steps,
        args.cooldown_steps,
        args.guidance_every_n_steps,
    )
    vllm_sched = vllm_guidance_schedule(
        args.steps,
        args.guidance_scale,
        args.warmup_steps,
        args.cooldown_steps,
        args.guidance_every_n_steps,
    )

    height_t = torch.tensor([args.height], device=device, dtype=dtype)
    width_t = torch.tensor([args.width], device=device, dtype=dtype)
    fps_t = torch.tensor([24.0], device=device, dtype=dtype)
    num_train_timesteps = 1000

    def _forward(z_in, t_in, text_emb, vec, text_mask, ef, mdl):
        raw = mdl(
            hidden_states=z_in.to(dtype),
            timestep=t_in,
            encoder_hidden_states=text_emb,
            encoder_hidden_states_mask=text_mask,
            vector_cond=vec,
            height=height_t,
            width=width_t,
            fps=fps_t,
            extra_features=ef,
            return_dict=False,
        )[0]
        if raw.shape[1] != in_channels:
            raw = raw[:, :in_channels]
        return raw

    torch.manual_seed(args.seed)
    z_ref = torch.randn(shape, device=device, dtype=dtype)
    z_vllm = z_ref.clone()

    all_match = True
    with torch.inference_mode():
        for i in range(num_test_steps):
            t_ref = ref_ts[i].to(device)
            t_vllm = vllm_ts[i].to(device)
            gs_ref = ref_sched[i]
            gs_vllm = vllm_sched[i]

            print(
                f"\n  Step {i}: t_ref={t_ref.item():.2f} t_vllm={t_vllm.item():.2f} "
                f"gs_ref={gs_ref:.1f} gs_vllm={gs_vllm:.1f}"
            )

            pred_cond = _forward(z_ref, t_ref.expand(1), vllm_seq, vllm_vec, vllm_masks, extra_features, model)

            use_uncond = gs_ref > 1.0
            if use_uncond:
                pred_uncond = _forward(
                    z_ref,
                    t_ref.expand(1),
                    vllm_neg_seq,
                    vllm_neg_vec,
                    vllm_neg_masks,
                    uncond_extra_features,
                    model,
                )
                v_pred = pred_uncond + gs_ref * (pred_cond - pred_uncond)
            else:
                v_pred = pred_cond

            # DDPM step
            sigma_t = (t_ref / num_train_timesteps).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            x0 = z_ref - sigma_t * v_pred
            x0 = torch.clamp(x0, -10.0, 10.0)
            v_pred = (z_ref - x0) / sigma_t

            if i < num_test_steps - 1:
                sigma_s = (
                    (ref_ts[i + 1].to(device) / num_train_timesteps)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
                alpha_t = 1.0 - sigma_t
                alpha_s = 1.0 - sigma_s
                alpha_ts = alpha_t / alpha_s
                alpha_ts_sq = alpha_ts**2
                sigma_s_div_t_sq = (sigma_s / sigma_t) ** 2
                sigma_ts_div_t_sq = 1.0 - alpha_ts_sq * sigma_s_div_t_sq
                mean = alpha_ts * sigma_s_div_t_sq * z_ref + alpha_s * sigma_ts_div_t_sq * x0
                variance = sigma_ts_div_t_sq * sigma_s**2
                torch.manual_seed(args.seed + i + 1000)
                noise = torch.randn_like(z_ref)
                z_ref = mean + torch.sqrt(variance) * noise
            else:
                z_ref = x0

            # Same thing with vllm schedule (should be identical now)
            pred_cond2 = _forward(z_vllm, t_vllm.expand(1), vllm_seq, vllm_vec, vllm_masks, extra_features, model)
            use_uncond2 = gs_vllm > 1.0
            if use_uncond2:
                pred_uncond2 = _forward(
                    z_vllm,
                    t_vllm.expand(1),
                    vllm_neg_seq,
                    vllm_neg_vec,
                    vllm_neg_masks,
                    uncond_extra_features,
                    model,
                )
                v_pred2 = pred_uncond2 + gs_vllm * (pred_cond2 - pred_uncond2)
            else:
                v_pred2 = pred_cond2

            sigma_t2 = (t_vllm / num_train_timesteps).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            x02 = z_vllm - sigma_t2 * v_pred2
            x02 = torch.clamp(x02, -10.0, 10.0)
            v_pred2 = (z_vllm - x02) / sigma_t2

            if i < num_test_steps - 1:
                sigma_s2 = (
                    (vllm_ts[i + 1].to(device) / num_train_timesteps)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
                alpha_t2 = 1.0 - sigma_t2
                alpha_s2 = 1.0 - sigma_s2
                alpha_ts2 = alpha_t2 / alpha_s2
                alpha_ts_sq2 = alpha_ts2**2
                sigma_s_div_t_sq2 = (sigma_s2 / sigma_t2) ** 2
                sigma_ts_div_t_sq2 = 1.0 - alpha_ts_sq2 * sigma_s_div_t_sq2
                mean2 = alpha_ts2 * sigma_s_div_t_sq2 * z_vllm + alpha_s2 * sigma_ts_div_t_sq2 * x02
                variance2 = sigma_ts_div_t_sq2 * sigma_s2**2
                torch.manual_seed(args.seed + i + 1000)
                noise2 = torch.randn_like(z_vllm)
                z_vllm = mean2 + torch.sqrt(variance2) * noise2
            else:
                z_vllm = x02

            m = _compare_tensors(f"step {i} z (after DDPM)", z_ref, z_vllm, rtol=1e-2, atol=1e-2)
            all_match = all_match and m

    del model
    torch.cuda.empty_cache()
    return all_match


# ============================================================================
# Test 6: VAE Decode
# ============================================================================


def test_vae_decode(args, config, device, dtype):
    _banner("Test 6: VAE Decode Comparison")

    vae_cfg = config.get("vae", {})
    in_channels = len(vae_cfg.get("scaling_factor", [0] * 16))
    vae_ds = tuple(vae_cfg.get("downsample_factors", (4, 16, 16)))

    # Create deterministic latents
    vae_t, vae_s = vae_ds[0], vae_ds[1]
    time_pad = 0 if args.num_frames % vae_t == 0 else (vae_t - args.num_frames % vae_t)
    num_latent_frames = (args.num_frames + time_pad) // vae_t
    latent_h = math.ceil(args.height / vae_s)
    latent_w = math.ceil(args.width / vae_s)

    torch.manual_seed(args.seed)
    latents = torch.randn(1, in_channels, num_latent_frames, latent_h, latent_w, device=device, dtype=dtype)

    # Load VAE (both use the same opensora VAE)
    print("  Loading VAE...")
    from text_to_video import load_vae

    vae = load_vae(vae_cfg, device, dtype)
    if vae is None:
        print("  [SKIP] VAE not available")
        return True

    # Compute num_pixel_frames
    vae_t_ds = vae.downsample_factors[0]
    num_pixel_frames = num_latent_frames * vae_t_ds
    chunk = vae.frame_chunk_len
    if num_latent_frames <= 1:
        num_pixel_frames = 1
    elif chunk is not None and num_pixel_frames % chunk != 0:
        num_pixel_frames = (num_pixel_frames // chunk) * chunk

    # Decode twice with same input to verify determinism
    print("  Running VAE decode (pass 1)...")
    with torch.no_grad():
        out1 = vae.decode(latents.to(dtype), num_frames=num_pixel_frames, spatial_size=(args.height, args.width))
    if isinstance(out1, tuple):
        out1 = out1[0]

    print("  Running VAE decode (pass 2)...")
    with torch.no_grad():
        out2 = vae.decode(latents.to(dtype), num_frames=num_pixel_frames, spatial_size=(args.height, args.width))
    if isinstance(out2, tuple):
        out2 = out2[0]

    match = _compare_tensors("VAE decode determinism", out1, out2)

    del vae
    torch.cuda.empty_cache()
    return match


# ============================================================================
# Main
# ============================================================================


def main():
    args = parse_args()

    import yaml

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # Add text_to_video.py directory to path
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    # Ensure moonvalley_ai is importable
    moonvalley_dir = str(Path(__file__).resolve().parents[3] / "moonvalley_ai")
    if moonvalley_dir not in sys.path:
        sys.path.insert(0, moonvalley_dir)

    # Initialize vllm distributed state (needed for MareyTransformer's parallel layers)
    needs_vllm = any(_should_run(t, args) for t in ("forward", "denoise"))
    if needs_vllm:
        from text_to_video import cleanup_distributed, init_distributed

        init_distributed(tp_size=1)

    results = {}

    # Test 1: Guidance schedule (no GPU)
    if _should_run("guidance", args):
        try:
            results["guidance"] = test_guidance_schedule(args)
        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            results["guidance"] = False

    # Test 2: Timestep schedule (no GPU)
    if _should_run("timesteps", args):
        try:
            results["timesteps"] = test_timestep_schedule(args)
        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            results["timesteps"] = False

    # Test 3: Text encoding (GPU)
    if _should_run("encoding", args):
        try:
            results["encoding"] = test_text_encoding(args, config, device, dtype)
        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            results["encoding"] = False

    # Test 4: Transformer forward (GPU)
    if _should_run("forward", args):
        try:
            results["forward"] = test_transformer_forward(args, config, device, dtype)
        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            results["forward"] = False

    # Test 5: Multi-step denoising (GPU)
    if _should_run("denoise", args):
        try:
            results["denoise"] = test_denoising_loop(args, config, device, dtype)
        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            results["denoise"] = False

    # Test 6: VAE decode (GPU)
    if _should_run("vae", args):
        try:
            results["vae"] = test_vae_decode(args, config, device, dtype)
        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            results["vae"] = False

    # Summary
    _banner("SUMMARY")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:12s}: {status}")
        all_pass = all_pass and passed

    if not all_pass:
        print("\n  KNOWN DISCREPANCIES:")
        print("  1. Guidance schedule oscillation is off-by-1 in main phase")
        print("     Fix: align vllm-omni build_guidance_schedule with reference")
        print("     Impact: HIGH — 77/100 steps have wrong guidance scale")
        print("  2. CLIP vector embedding: vllm-omni uses CLIPModel.get_text_features()")
        print("     which applies an extra text_projection layer. Reference uses")
        print("     CLIPTextModel with raw pooler_output (no text_projection).")
        print("     Fix: use CLIPTextModel + CLIPTokenizer, read .pooler_output")
        print("     Impact: HIGH — cosine_sim ~0.05 (essentially uncorrelated)")
        print("  3. Quote extraction method differs (spaCy vs regex)")
        print("     Impact: low for prompts without quoted text")

    if needs_vllm:
        cleanup_distributed()

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
