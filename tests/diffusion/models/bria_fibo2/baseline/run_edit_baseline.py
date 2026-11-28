#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fibo-2.0 Dev edit-path baseline: fixed-input generation (reference data).

Editing = instruction + image to edit (context-latents conditioning). Phase A
replays the __call__ edit branch and saves all intermediates; Phase B re-runs
the official pipe on the same inputs and compares.

Usage: python run_edit_baseline.py            # capture (saves ref-data/edit1)
       python run_edit_baseline.py --verify   # independent re-run comparison
"""

# ruff: noqa
# ^ Snapshot of the edit-baseline generator (works for Dev and Edit Turbo
#   via --model/--steps/--guidance); runs in the external diffusers
#   environment, not under this repo's lint policy.

import argparse
import hashlib
import json
import os
import subprocess
import time

import numpy as np
import torch
from PIL import Image
from diffusers import BriaFibo2Pipeline
from diffusers.pipelines.bria_fibo2.pipeline_bria_fibo2 import (
    DEFAULT_NEGATIVE_PROMPT,
    calculate_shift,
    retrieve_timesteps,
)
from safetensors.torch import load_file, save_file

EDIT_INSTRUCTION = "Make the background a snowy forest"
SOURCE_IMAGE = "/root/autodl-fs/fibo-baseline/ref-data/run1/image.png"  # the baseline T2I red-bicycle capture


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sync():
    torch.accelerator.synchronize()


def load_pipeline(model_id):
    pipe = BriaFibo2Pipeline.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def vae_decode(pipe, latents):
    lat = pipe._patchify_latents(latents)
    bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(lat.device, lat.dtype)
    bn_scale = (1 / torch.sqrt(pipe.vae.bn.running_var + pipe.vae.config.batch_norm_eps)).view(1, -1, 1, 1)
    lat = lat / bn_scale.to(lat.device, lat.dtype) + bn_mean
    lat = pipe._unpatchify_latents(lat)
    image = pipe.vae.decode(lat.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def capture_run(pipe, args):
    device = pipe._execution_device
    stats = {
        "steps": args.steps,
        "guidance": args.guidance,
        "seed": args.seed,
        "instruction": EDIT_INSTRUCTION,
        "source_image": SOURCE_IMAGE,
    }

    image = [Image.open(SOURCE_IMAGE).convert("RGB")]

    # 2. Preprocess: the VAE crops + the Qwen3-VL copies
    default_area = (pipe.default_sample_size * pipe.vae_scale_factor) ** 2
    width, height = pipe.image_processor.get_image_size(image[0].width / image[0].height, area=default_area)
    stats["height"], stats["width"] = height, width
    vae_images, text_encoder_images = pipe.image_processor.preprocess(
        image,
        height=height,
        width=width,
        area=default_area,
        text_encoder_image_size=pipe.text_encoder_image_size,
    )

    # 3. Encode: the edit prompt (with images) + the null caption (without).
    # diffusers' encode_prompt re-assembles via _get_edit_prompt internally, so
    # pass the raw instruction here; the assembled form is saved for the vLLM side
    edit_prompt = pipe._get_edit_prompt(EDIT_INSTRUCTION, len(image))
    stats["edit_prompt"] = edit_prompt
    t0 = time.perf_counter()
    prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(EDIT_INSTRUCTION, text_encoder_images, device, 1)
    sync()
    stats["encode_cond_s"] = time.perf_counter() - t0
    if args.guidance > 1.0:
        neg_embeds, neg_embeds_mask = pipe.encode_prompt(DEFAULT_NEGATIVE_PROMPT, None, device, 1)
    else:
        neg_embeds, neg_embeds_mask = None, None

    # 4. Initial noise (fp32) + context latents (BN-normalized)
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    latents = pipe.prepare_latents(1, pipe.transformer.config.in_channels, height, width, device, gen)
    initial_latents = latents.clone()
    image_latents = pipe.prepare_image_latents(vae_images, latents.shape[0], device, pipe.transformer.dtype)

    # 5. timesteps
    patch_size = pipe.transformer.config.patch_size
    image_seq_len = (latents.shape[2] // patch_size) * (latents.shape[3] // patch_size)
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.base_image_seq_len,
        pipe.scheduler.config.max_image_seq_len,
        pipe.scheduler.config.base_shift,
        pipe.scheduler.config.max_shift,
    )
    sigmas = np.linspace(1.0, 1 / args.steps, args.steps)
    timesteps, num_steps = retrieve_timesteps(pipe.scheduler, args.steps, device, sigmas=sigmas, mu=mu)
    stats["timesteps"] = timesteps.tolist()
    stats["mu"] = float(mu)

    # 6. Denoise loop (two independent forwards, both with context_latents)
    capture_at = sorted(set([0] + [s for s in (args.steps // 2, args.steps - 1) if 0 < s < args.steps]))
    first, captured_steps, trajectory = {}, {}, {}
    step_times = []
    for i, t in enumerate(timesteps):
        t0 = time.perf_counter()
        timestep = t.expand(latents.shape[0]) / 1000
        latent_model_input = latents.to(pipe.transformer.dtype)
        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_embeds_mask,
            context_latents=image_latents,
            return_dict=False,
        )[0].float()
        noise_pred_uncond = None
        if args.guidance > 1.0:  # Edit Turbo (guidance 1.0) runs a single pass
            noise_pred_uncond = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=neg_embeds,
                encoder_attention_mask=neg_embeds_mask,
                context_latents=image_latents,
                return_dict=False,
            )[0].float()
        if i in capture_at:
            step_capture = {
                "latent_model_input": latent_model_input.clone(),
                "timestep": timestep.clone(),
                "noise_pred_cond": noise_pred.clone(),
            }
            if noise_pred_uncond is not None:
                step_capture["noise_pred_uncond"] = noise_pred_uncond.clone()
            captured_steps[i] = step_capture
        if noise_pred_uncond is not None:
            guided = noise_pred_uncond + args.guidance * (noise_pred - noise_pred_uncond)
        else:
            guided = noise_pred
        if i == 0:
            first = {
                "noise_pred_cond": noise_pred.clone(),
                "noise_pred_guided": guided.clone(),
                "latent_model_input": latent_model_input.clone(),
            }
            if noise_pred_uncond is not None:
                first["noise_pred_uncond"] = noise_pred_uncond.clone()
        latents = pipe.scheduler.step(guided, t, latents, return_dict=False)[0]
        sync()
        step_times.append(time.perf_counter() - t0)
        trajectory[f"latents_step_{i:03d}"] = latents.clone()
    stats["loop_s"] = float(sum(step_times))
    final_latents = latents.clone()

    edited = vae_decode(pipe, final_latents)

    return {
        "prompt_embeds": prompt_embeds,
        "prompt_embeds_mask": prompt_embeds_mask,
        "neg_embeds": neg_embeds,
        "neg_embeds_mask": neg_embeds_mask,
        "initial_latents": initial_latents,
        "image_latents": image_latents[0],
        "edit_prompt": edit_prompt,
        "first": first,
        "captured_steps": captured_steps,
        "trajectory": trajectory,
        "final_latents": final_latents,
        "image": edited,
        "stats": stats,
    }


def official_run(pipe, cap, args):
    # The official __call__ assembles the edit prompt itself (_get_edit_prompt);
    # feeding it the already-assembled JSON would stack vision markers.
    out = pipe(
        prompt=EDIT_INSTRUCTION,
        image=[Image.open(SOURCE_IMAGE).convert("RGB")],
        height=cap["stats"]["height"],
        width=cap["stats"]["width"],
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
        output_type="latent",
        return_dict=False,
    )
    return out[0]


def save_artifacts(cap, official, out_dir, args):
    tensors_dir = os.path.join(out_dir, "tensors")
    os.makedirs(tensors_dir, exist_ok=True)
    save_file(
        {"initial_latents": cap["initial_latents"].contiguous()},
        os.path.join(tensors_dir, "initial_latents.safetensors"),
    )
    save_file(
        {"prompt_embeds": cap["prompt_embeds"].contiguous()}, os.path.join(tensors_dir, "prompt_embeds.safetensors")
    )
    if cap["neg_embeds"] is not None:
        save_file(
            {"negative_prompt_embeds": cap["neg_embeds"].contiguous()},
            os.path.join(tensors_dir, "negative_prompt_embeds.safetensors"),
        )
    save_file(
        {"image_latents": cap["image_latents"].contiguous()}, os.path.join(tensors_dir, "image_latents.safetensors")
    )
    save_file({k: v.contiguous() for k, v in cap["first"].items()}, os.path.join(tensors_dir, "first_step.safetensors"))
    sel = {f"step_{i:03d}_{kk}": v.contiguous() for i, d in cap["captured_steps"].items() for kk, v in d.items()}
    save_file(sel, os.path.join(tensors_dir, "selected_steps.safetensors"))
    save_file(
        {k: v.contiguous() for k, v in cap["trajectory"].items()}, os.path.join(tensors_dir, "trajectory.safetensors")
    )
    save_file(
        {"final_latents": cap["final_latents"].contiguous(), "official_final_latents": official.contiguous()},
        os.path.join(tensors_dir, "final_latents.safetensors"),
    )
    cap["image"].save(os.path.join(out_dir, "edited_image.png"))
    with open(os.path.join(out_dir, "edit_instruction.txt"), "w") as f:
        f.write(cap["edit_prompt"] + "\n")

    same = torch.equal(cap["final_latents"], official)
    meta = {
        "stats": cap["stats"],
        "manual_vs_official": {
            "final_latents_bitwise_equal": bool(same),
            "max_abs_diff": (cap["final_latents"] - official).abs().max().item(),
        },
    }
    import diffusers, transformers

    meta["versions"] = {
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "gpu": getattr(torch.cuda, "get_device_name")(0),  # snapshot runs in the external diffusers env,
    }
    for fn in sorted(os.listdir(tensors_dir)):
        meta.setdefault("artifacts_sha256", {})[fn] = sha256_file(os.path.join(tensors_dir, fn))
    meta["artifacts_sha256"]["edited_image.png"] = sha256_file(os.path.join(out_dir, "edited_image.png"))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="briaai/Fibo-2.0-dev")
    p.add_argument("--out", default="/root/autodl-fs/fibo-baseline/ref-data/edit1")
    p.add_argument("--mode", choices=["capture", "verify"], default="capture")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.mode == "verify":
        ref_dir = args.out
        pipe = load_pipeline(args.model)
        with torch.no_grad():
            cap = capture_run(pipe, args)
        report = {}
        for fn in (
            "initial_latents",
            "prompt_embeds",
            "negative_prompt_embeds",
            "image_latents",
            "first_step",
            "selected_steps",
            "trajectory",
            "final_latents",
        ):
            if not os.path.exists(os.path.join(ref_dir, "tensors", f"{fn}.safetensors")):
                continue
            ref = load_file(os.path.join(ref_dir, "tensors", f"{fn}.safetensors"))
            now = {
                "initial_latents": {"initial_latents": cap["initial_latents"]},
                "prompt_embeds": {"prompt_embeds": cap["prompt_embeds"]},
                "negative_prompt_embeds": {"negative_prompt_embeds": cap["neg_embeds"]},
                "image_latents": {"image_latents": cap["image_latents"]},
                "first_step": cap["first"],
                "selected_steps": {
                    f"step_{i:03d}_{kk}": v for i, d in cap["captured_steps"].items() for kk, v in d.items()
                },
                "trajectory": cap["trajectory"],
                "final_latents": {"final_latents": cap["final_latents"]},
            }[fn]
            ok = all(torch.equal(now[k].cpu(), ref[k].cpu()) for k in ref if k in now)
            report[fn] = {"pass": bool(ok)}
        ref_img = np.asarray(Image.open(os.path.join(ref_dir, "edited_image.png")))
        img_equal = bool(np.array_equal(np.asarray(cap["image"]), ref_img))
        report["edited_image_pixels"] = {"pass": img_equal}
        all_pass = all(v["pass"] for v in report.values())
        print(json.dumps(report, indent=2))
        print(f"ALL_PASS={all_pass}")
        raise SystemExit(0 if all_pass else 1)

    pipe = load_pipeline(args.model)
    with torch.no_grad():  # the official __call__ is no-grad; the replay must match or the autograd graph OOMs
        cap = capture_run(pipe, args)
        official = official_run(pipe, cap, args)
    meta = save_artifacts(cap, official, args.out, args)
    print(
        json.dumps(
            {
                "manual_vs_official": meta["manual_vs_official"],
                "height": meta["stats"]["height"],
                "width": meta["stats"]["width"],
                "loop_s": meta["stats"]["loop_s"],
            },
            indent=2,
        )
    )
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
