"""
End-to-end comparison of reference (moonvalley_ai) vs vllm-omni Marey inference.

Runs BOTH denoising loop implementations side-by-side using a single model
instance, with the actual distilled-model parameters from the README test
commands.  Compares at every denoising step to find exactly where divergence
starts.

Run with:
    PYTHONPATH=/home/david/repos/moonvalley_ai \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run --project /home/david/repos/vllm-omni \
    python /home/david/repos/vllm-omni/examples/offline_inference/marey/compare_e2e.py
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
MOONVALLEY_DIR = str(Path(__file__).resolve().parents[3] / "moonvalley_ai")
if MOONVALLEY_DIR not in sys.path:
    sys.path.insert(0, MOONVALLEY_DIR)

DEFAULT_CONFIG = "/app/wlam/models/checkpoints/marey/distilled-0001/config.yaml"
DEFAULT_TRANSFORMER_WEIGHTS = (
    "/app/wlam/models/checkpoints/marey/distilled-0001/epoch0-global_step5000_distilled/ema_inference_ckpt.safetensors"
)

DEFAULT_PROMPT = (
    "Detailed Description: A majestic, aged eagle with mottled golden-brown "
    "feathers soars gracefully through a vast, ancient indoor chamber. Its "
    "expansive wings barely flap, catching the air as it glides effortlessly "
    "between towering stone pillars adorned with glinting metallic accents. "
    "Beams of morning light pierce the gloom, filtering through a cracked "
    "skylight high above and illuminating swirling dust motes in their path. "
    "The camera pans smoothly, following the eagle's silent flight as it "
    "navigates the cavernous space, its sharp eyes scanning the stone floor "
    "below, creating a scene of serene power and timeless solitude. Background: "
    "The far reaches of the chamber fade into deep shadow, with the silhouettes "
    "of distant pillars barely visible. High above, a cracked skylight serves "
    "as the primary light source, its fractured glass creating distinct rays of "
    "light. Middleground: The aged eagle glides on a steady path, its mottled "
    "golden-brown wings spread wide. It passes through the dramatic beams of "
    "light, which highlight the intricate details of its feathers and the dust "
    "particles dancing in the air. Foreground: The camera looks up from a low "
    "angle, tracking the eagle's movement across the expansive stone floor, "
    "which is patterned with the bright shafts of light and deep shadows cast "
    "by the pillars."
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
    p = argparse.ArgumentParser(description="E2E comparison of reference vs vllm-omni.")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--transformer-weights", default=DEFAULT_TRANSFORMER_WEIGHTS)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--num-frames", type=int, default=33)
    p.add_argument("--steps", type=int, default=33)
    p.add_argument("--guidance-scale", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-steps", type=int, default=4)
    p.add_argument("--cooldown-steps", type=int, default=18)
    p.add_argument("--guidance-every-n-steps", type=int, default=2)
    p.add_argument("--flow-shift", type=float, default=3.0)
    p.add_argument("--clip-value", type=float, default=10.0)
    p.add_argument(
        "--force-same-timesteps",
        action="store_true",
        help="Force both paths to use the same (ref) timesteps to isolate timestep drift.",
    )
    return p.parse_args()


def _banner(title):
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def _compare(name, a, b, rtol=1e-3, atol=1e-4, verbose=True):
    af, bf = a.float().cpu(), b.float().cpu()
    diff = (af - bf).abs()
    max_d = diff.max().item()
    mean_d = diff.mean().item()
    cos = torch.nn.functional.cosine_similarity(af.flatten().unsqueeze(0), bf.flatten().unsqueeze(0)).item()
    match = torch.allclose(af, bf, rtol=rtol, atol=atol)
    tag = "MATCH" if match else "MISMATCH"
    if verbose or not match:
        print(
            f"  [{tag}] {name}: max_diff={max_d:.8f} mean_diff={mean_d:.8f} "
            f"cos={cos:.8f} a_std={af.std().item():.6f} b_std={bf.std().item():.6f}"
        )
    return match


# ============================================================================
# Reference-style functions (matching opensora RFLOW exactly)
# ============================================================================


def ref_guidance_schedule(gs, n, warmup, cooldown, every_n):
    sched = np.ones(n)
    if warmup > 0:
        sched[:warmup] = gs
    main = n - warmup - cooldown
    if main > 0:
        osc = np.ones(main)
        osc[1::every_n] = gs
        sched[warmup : warmup + main] = osc
    return sched.tolist()


def ref_timesteps(num_steps, shift, tmin=0.001, tmax=1.0, teacher=100, nt=1000, device=None):
    """Exact replica of RFLOW.draw_time + RFlowScheduler.timestep_shift."""
    sigmas_f64 = torch.linspace(tmax, tmin, teacher).tolist()
    ts = [torch.tensor([t * nt], device=device) for t in sigmas_f64]
    if shift and shift > 0:
        for i, t in enumerate(ts):
            t_norm = t / nt
            ts[i] = (shift * t_norm / (1.0 + (shift - 1.0) * t_norm)) * nt
    stride = teacher // num_steps
    return [ts[i * stride] for i in range(num_steps)]


# ============================================================================
# vllm-omni-style functions (from text_to_video.py)
# ============================================================================


def vllm_guidance_schedule(n, gs, warmup, cooldown, every_n):
    from text_to_video import build_guidance_schedule

    return build_guidance_schedule(n, gs, warmup, cooldown, every_n)


def vllm_timesteps(num_steps, shift, tmin=0.001, tmax=1.0, teacher=100, nt=1000, device=None):
    from text_to_video import create_flow_timesteps

    return create_flow_timesteps(num_steps, shift=shift, tmin=tmin, tmax=tmax, teacher_steps=teacher, device=device)


# ============================================================================
# Main comparison
# ============================================================================


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # ---------------------------------------------------------------
    # Phase 0: Compare schedules
    # ---------------------------------------------------------------
    _banner("Phase 0: Schedule comparison")

    ref_sched = ref_guidance_schedule(
        args.guidance_scale, args.steps, args.warmup_steps, args.cooldown_steps, args.guidance_every_n_steps
    )
    vllm_sched = vllm_guidance_schedule(
        args.steps, args.guidance_scale, args.warmup_steps, args.cooldown_steps, args.guidance_every_n_steps
    )

    sched_diffs = [(i, r, v) for i, (r, v) in enumerate(zip(ref_sched, vllm_sched)) if abs(r - v) > 1e-6]
    if sched_diffs:
        print(f"  [MISMATCH] Guidance schedule: {len(sched_diffs)} steps differ")
        for i, r, v in sched_diffs[:10]:
            print(f"    step {i}: ref={r:.1f} vllm={v:.1f}")
    else:
        print(f"  [MATCH] Guidance schedules identical ({sum(1 for g in ref_sched if g > 1)} active)")

    ref_ts = ref_timesteps(args.steps, args.flow_shift, device=device)
    vllm_ts = vllm_timesteps(args.steps, args.flow_shift, device=device)
    ts_max_diff = max(abs(r.item() - v.item()) for r, v in zip(ref_ts, vllm_ts))
    print(f"  Timestep max diff: {ts_max_diff:.10f} ({'MATCH' if ts_max_diff < 1e-4 else 'MISMATCH'})")
    for i in range(min(10, len(ref_ts))):
        d = abs(ref_ts[i].item() - vllm_ts[i].item())
        if d > 0:
            print(f"    step {i}: ref={ref_ts[i].item():.10f} vllm={vllm_ts[i].item():.10f} diff={d:.10f}")

    if args.force_same_timesteps:
        print("  [FORCED] Using reference timesteps for both paths")
        vllm_ts = ref_ts

    # ---------------------------------------------------------------
    # Phase 1: Text encoding
    # ---------------------------------------------------------------
    _banner("Phase 1: Text encoding")
    from text_to_video import _extract_quotes, encode_text, load_text_encoders

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    encoders = load_text_encoders(config, device, dtype)
    pos_seq, pos_masks, pos_vec = encode_text(args.prompt, encoders, device, dtype)
    pos_quote = _extract_quotes(args.prompt)
    neg_seq, neg_masks, neg_vec = encode_text(
        DEFAULT_NEGATIVE_PROMPT, encoders, device, dtype, quote_override=pos_quote
    )
    del encoders
    torch.cuda.empty_cache()

    print(f"  pos_seq shapes: {[s.shape for s in pos_seq]}")
    print(f"  neg_seq shapes: {[s.shape for s in neg_seq]}")
    print(f"  pos_vec shape:  {pos_vec.shape}")
    print(f"  quote text:     {repr(pos_quote[:60])}...")

    # ---------------------------------------------------------------
    # Phase 2: Build model & extra features
    # ---------------------------------------------------------------
    _banner("Phase 2: Build model")

    from text_to_video import (
        build_extra_features,
        build_transformer,
        cleanup_distributed,
        init_distributed,
        load_transformer_weights,
    )

    init_distributed(tp_size=1)

    vae_cfg = config.get("vae", {})
    in_channels = len(vae_cfg.get("scaling_factor", [0] * 16))

    caption_channels = [s.shape[-1] for s in pos_seq]
    vec_channels = pos_vec.shape[-1]
    model = build_transformer(config, in_channels, caption_channels, vec_channels)
    load_transformer_weights(model, args.transformer_weights, device, dtype)

    extra_features = build_extra_features(config, args.height, args.width, device, dtype)
    uncond_ef = dict(extra_features)
    for qk in ("dover_technical", "aesthetics_score_total"):
        if qk in extra_features:
            uncond_ef[qk] = torch.tensor([9], device=device, dtype=torch.long)
            extra_features[qk] = torch.tensor([0], device=device, dtype=torch.long)

    print(f"  extra_features: {list(extra_features.keys())}")
    for k, v in extra_features.items():
        print(f"    cond  {k}: {v.tolist()}")
    for k, v in uncond_ef.items():
        if not torch.equal(v, extra_features.get(k, v)):
            print(f"    uncond {k}: {v.tolist()}")

    # ---------------------------------------------------------------
    # Phase 3: Prepare latent shape & noise
    # ---------------------------------------------------------------
    vae_ds = tuple(vae_cfg.get("downsample_factors", (4, 16, 16)))
    vae_t, vae_s = vae_ds[0], vae_ds[1]
    time_pad = 0 if args.num_frames % vae_t == 0 else (vae_t - args.num_frames % vae_t)
    num_latent_frames = (args.num_frames + time_pad) // vae_t
    latent_h = math.ceil(args.height / vae_s)
    latent_w = math.ceil(args.width / vae_s)
    shape = (1, in_channels, num_latent_frames, latent_h, latent_w)

    _banner(f"Phase 3: Denoising loop ({args.steps} steps, shape={shape})")

    height_t = torch.tensor([args.height], device=device, dtype=dtype)
    width_t = torch.tensor([args.width], device=device, dtype=dtype)
    fps_t = torch.tensor([24.0], device=device, dtype=dtype)
    nt = 1000

    def fwd(z_in, t_in, seq, vec, mask, ef):
        raw = model(
            hidden_states=z_in.to(dtype),
            timestep=t_in,
            encoder_hidden_states=seq,
            encoder_hidden_states_mask=mask,
            vector_cond=vec,
            height=height_t,
            width=width_t,
            fps=fps_t,
            extra_features=ef,
            return_dict=False,
        )[0]
        return raw[:, :in_channels] if raw.shape[1] != in_channels else raw

    # Generate initial noise identically
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    z_ref = torch.randn(shape, device=device, dtype=dtype)
    # vllm-omni path: zeros then randn_like
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    z_zeros = torch.zeros(shape, device=device, dtype=dtype)
    z_vllm = torch.randn_like(z_zeros)
    del z_zeros

    print(f"  Initial noise match: {_compare('z_init', z_ref, z_vllm)}")

    # ---------------------------------------------------------------
    # Phase 4: Step-by-step denoising comparison
    # ---------------------------------------------------------------
    all_match = True
    first_mismatch_step = None

    with torch.inference_mode():
        for i in range(args.steps):
            t_ref = ref_ts[i].to(device)
            t_vllm = vllm_ts[i].to(device)
            gs_ref = ref_sched[i]
            gs_vllm = vllm_sched[i]

            # Reference logic
            use_uncond_ref = gs_ref > 1.0
            pred_cond_ref = fwd(z_ref, t_ref.expand(1), pos_seq, pos_vec, pos_masks, extra_features)
            if use_uncond_ref:
                pred_uncond_ref = fwd(z_ref, t_ref.expand(1), neg_seq, neg_vec, neg_masks, uncond_ef)
                v_ref = pred_uncond_ref + gs_ref * (pred_cond_ref - pred_uncond_ref)
            else:
                v_ref = pred_cond_ref

            sigma_ref = (t_ref / nt).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            x0_ref = z_ref - sigma_ref * v_ref
            x0_ref = torch.clamp(x0_ref, -args.clip_value, args.clip_value)
            v_ref = (z_ref - x0_ref) / sigma_ref

            # vllm-omni logic
            use_uncond_vllm = gs_vllm > 1.0
            pred_cond_vllm = fwd(z_vllm, t_vllm.expand(1), pos_seq, pos_vec, pos_masks, extra_features)
            if use_uncond_vllm:
                pred_uncond_vllm = fwd(z_vllm, t_vllm.expand(1), neg_seq, neg_vec, neg_masks, uncond_ef)
                v_vllm = pred_uncond_vllm + gs_vllm * (pred_cond_vllm - pred_uncond_vllm)
            else:
                v_vllm = pred_cond_vllm

            sigma_vllm = (t_vllm / nt).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            x0_vllm = z_vllm - sigma_vllm * v_vllm
            x0_vllm = torch.clamp(x0_vllm, -args.clip_value, args.clip_value)
            v_vllm = (z_vllm - x0_vllm) / sigma_vllm

            # DDPM step
            if i < args.steps - 1:
                sigma_s_ref = (ref_ts[i + 1].to(device) / nt).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                sigma_s_vllm = (vllm_ts[i + 1].to(device) / nt).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                def ddpm_step(z, x0, sigma_t, sigma_s, seed_offset):
                    a_t = 1.0 - sigma_t
                    a_s = 1.0 - sigma_s
                    a_ts = a_t / a_s
                    a_ts2 = a_ts**2
                    ss_div_t2 = (sigma_s / sigma_t) ** 2
                    sts_div_t2 = 1.0 - a_ts2 * ss_div_t2
                    mean = a_ts * ss_div_t2 * z + a_s * sts_div_t2 * x0
                    var = sts_div_t2 * sigma_s**2
                    torch.manual_seed(args.seed + i + 1000)
                    noise = torch.randn_like(z)
                    return mean + torch.sqrt(var) * noise

                z_ref = ddpm_step(z_ref, x0_ref, sigma_ref, sigma_s_ref, i)
                z_vllm = ddpm_step(z_vllm, x0_vllm, sigma_vllm, sigma_s_vllm, i)
            else:
                z_ref = x0_ref
                z_vllm = x0_vllm

            step_match = _compare(
                f"step {i:2d} z (t={t_ref.item():.1f}, gs_ref={gs_ref:.1f} gs_vllm={gs_vllm:.1f})",
                z_ref,
                z_vllm,
                rtol=1e-5,
                atol=1e-6,
                verbose=True,
            )
            if not step_match and first_mismatch_step is None:
                first_mismatch_step = i
            all_match = all_match and step_match

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    _banner("SUMMARY")
    if all_match:
        print("  ALL STEPS MATCH - denoising logic is identical.")
        print("  If outputs still differ, the issue is in:")
        print("    - Multi-GPU sequence parallelism (numerical non-determinism)")
        print("    - VAE decode path")
        print("    - Video encoding/saving")
    else:
        print(f"  DIVERGENCE at step {first_mismatch_step}")
        if sched_diffs:
            print(f"  ROOT CAUSE: Guidance schedule mismatch at {len(sched_diffs)} steps")
        elif ts_max_diff > 1e-4:
            print(f"  ROOT CAUSE: Timestep schedule mismatch (max_diff={ts_max_diff:.8f})")
        else:
            print("  Guidance and timesteps match - issue may be in noise generation or accumulation")

    cleanup_distributed()
    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
