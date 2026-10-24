# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in real-weight validation; see the SANA-Video 2.0 recipe for commands."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import imageio.v3 as iio
import numpy as np
import torch
import yaml
from PIL import Image

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

PROMPT = "A small red boat sailing across a calm lake."
NEGATIVE = "blur, distortion, abrupt cuts"
REQUEST = dict(height=64, width=96, num_frames=9, num_inference_steps=50, guidance_scale=8.0)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def record_video(video, path):
    video = np.asarray(video)
    while video.ndim > 4 and video.shape[0] == 1:
        video = video[0]
    assert video.shape == (9, 64, 96, 3), video.shape
    assert np.isfinite(video).all() and np.ptp(video) > 0
    np.save(path.with_suffix(".npy"), video)
    pixels = (video.clip(0, 1) * 255).round().astype(np.uint8)
    iio.imwrite(path, pixels, plugin="FFMPEG", fps=24, codec="libx264", macro_block_size=1)
    decoded = iio.imread(path, plugin="FFMPEG")
    assert decoded.shape == (9, 64, 96, 3), decoded.shape
    return dict(
        shape=list(video.shape),
        raw_sha256=digest(video.tobytes()),
        mp4_sha256=digest(path.read_bytes()),
        mp4_bytes=path.stat().st_size,
        decoded_shape=list(decoded.shape),
        minimum=float(video.min()),
        maximum=float(video.max()),
    )


def component_options(args):
    return dict(
        vae_model=str(args.assets / "vae"),
        text_encoder_model=str(args.assets / "text"),
        tokenizer_model=str(args.assets / "text"),
    )


def compare(actual, expected):
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    assert actual.shape == expected.shape
    error = actual - expected
    result = dict(
        shape=list(actual.shape),
        max_abs=error.abs().max().item(),
        mean_abs=error.abs().mean().item(),
        bit_equal=torch.equal(actual, expected),
    )
    assert result["bit_equal"], result
    return result


class LoadingReceipt:
    model_runner: "DiffusionModelRunner"

    def sana_loading_receipt(self):
        pipeline = self.model_runner.pipeline
        report = asdict(pipeline.checkpoint_report)
        report["loaded_key_count"] = len(report.pop("loaded_keys"))
        return {"pipeline": type(pipeline).__name__, "checkpoint": report}


def offline(args, receipt):
    from vllm_omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    # Deliberately omit model_class_name: exercise native checkpoint discovery.
    engine = Omni(
        model=str(args.assets / "model"),
        model_config=component_options(args),
        dtype="bfloat16",
        enforce_eager=True,
        distributed_executor_backend="uni",
        cache_backend="none",
        max_num_seqs=1,
        worker_extension_cls="tests.e2e.accuracy.sana_video2.validate.LoadingReceipt",
    )
    try:
        receipt["native_dispatch"] = engine.engine.collective_rpc(
            method="sana_loading_receipt", stage_ids=[0], timeout=120
        )
        save(args, receipt)
        for task in ("t2v", "ti2v"):
            prompt: dict = dict(prompt=PROMPT, negative_prompt=NEGATIVE)
            if task == "ti2v":
                prompt["multi_modal_data"] = {"image": Image.open(args.output / "reference.png").convert("RGB")}
            start = time.perf_counter()
            output = engine.generate(
                prompt,
                OmniDiffusionSamplingParams(
                    **REQUEST, seed=42, fps=24, output_type="np", extra_args={"motion_score": 10, "flow_shift": 12.0}
                ),
                use_tqdm=False,
            )[0]
            elapsed = time.perf_counter() - start
            receipt[task] = {**record_video(output.images, args.output / f"offline-{task}.mp4"), "elapsed_s": elapsed}
            save(args, receipt)
    finally:
        engine.close()


@torch.no_grad()
def alignment(args, receipt):
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKLLTX2Video
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import center_crop, resize

    from tests.e2e.accuracy.sana_video2.reference import load_reference
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.sana_video2.pipeline_sana_video2 import SanaVideo2Pipeline

    config = OmniDiffusionConfig.from_kwargs(
        model=str(args.assets / "model"),
        dtype=torch.bfloat16,
        model_config=component_options(args),
        enforce_eager=True,
        cache_backend="none",
    )
    with set_current_diffusion_config(config), torch.device("cuda"):
        native = SanaVideo2Pipeline(od_config=config)
    report = asdict(native.checkpoint_report)
    report["loaded_key_count"] = len(report.pop("loaded_keys"))
    receipt["checkpoint"] = report
    upstream, dpm, euler = load_reference(args.reference)
    receipt["reference_sha"] = subprocess.check_output(
        ["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True
    ).strip()
    # Keep upstream non-parameter RoPE caches real; only parameters start on meta.
    with init_empty_weights():
        reference = upstream.SanaVideo2_5B()
    reference.to_empty(device="cuda").to(dtype=torch.bfloat16)
    state = torch.load(
        args.assets / "model/checkpoints/SANA_Video_2.0_5B_720p.pth", map_location="cpu", weights_only=True, mmap=True
    )["state_dict"]
    # Legacy positional/dropout buffers are unused in eval; all learned keys must match.
    legacy = {"pos_embed", "y_embedder.y_embedding"}
    receipt["reference_ignored_buffers"] = sorted(set(state) & legacy)
    missing, unexpected = reference.load_state_dict(
        {key: value for key, value in state.items() if key not in legacy}, strict=False
    )
    assert not unexpected and set(missing) <= legacy, (missing, unexpected)
    reference.eval().requires_grad_(False)
    reference.set_cross_attention_xformers(False)
    for module in reference.modules():
        module.fp32_attention = True
    vae_config = AutoencoderKLLTX2Video.load_config(str(args.assets / "vae"), subfolder="vae")
    overrides = (
        {} if "upsample_type" in vae_config else {"upsample_type": tuple(reversed(vae_config["decoder_upsample_type"]))}
    )
    reference_vae = (
        AutoencoderKLLTX2Video.from_pretrained(
            str(args.assets / "vae"), subfolder="vae", torch_dtype=torch.bfloat16, **overrides
        )
        .to("cuda")
        .eval()
    )
    release = yaml.safe_load((args.assets / "model/config.yaml").read_text())
    instruction = "\n".join(release["text_encoder"]["chi_prompt"])
    # Upstream builder.get_tokenizer_and_text_encoder explicitly uses right padding.
    native.tokenizer.padding_side = "right"
    indices = [0] + list(range(-299, 0))

    def text(prompt, prefix):
        length = len(native.tokenizer.encode(prefix)) + 298 if prefix else 300
        tokens = native.tokenizer(
            [prefix + prompt], max_length=length, padding="max_length", truncation=True, return_tensors="pt"
        ).to("cuda")
        embedding = native.text_encoder(tokens.input_ids, tokens.attention_mask, use_cache=False)[0]
        return embedding[:, indices], tokens.attention_mask[:, indices].bool()

    positive, pos_mask = text(PROMPT + " motion score: 10.", instruction)
    negative, neg_mask = text(NEGATIVE, "")
    native_conditioning = native.encode_prompt(PROMPT, NEGATIVE, 8.0, 10)
    receipt["conditioning"] = [
        compare(a, b) for a, b in zip(native_conditioning, (positive, pos_mask, negative, neg_mask))
    ]
    mask = torch.cat([neg_mask, pos_mask])
    for task in ("t2v", "ti2v"):
        initial = torch.randn(
            (1, 128, 2, 2, 3),
            device="cuda",
            dtype=torch.float32,
            generator=torch.Generator(device="cuda").manual_seed(42),
        )
        image = Image.open(args.output / "reference.png").convert("RGB") if task == "ti2v" else None
        native_steps = []
        raw = native.generate(
            PROMPT,
            negative_prompt=NEGATIVE,
            **REQUEST,
            image=image,
            latents=initial,
            callback=lambda _index, _time, x: native_steps.append(x.detach().cpu().clone()),
        )
        z = initial.clone()
        if image is not None:
            # Independent upstream resize-to-fill + torchvision center crop.
            w, h = image.size
            scale = max(64 / h, 96 / w)
            image = center_crop(
                resize(image, [round(h * scale), round(w * scale)], InterpolationMode.BICUBIC), [64, 96]
            )
            pixels = torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float() / 255
            pixels = ((pixels - 0.5) / 0.5)[None, :, None].to("cuda", torch.bfloat16)
            latent = reference_vae.encode(pixels).latent_dist.mode()
            mean = reference_vae.latents_mean.view(1, -1, 1, 1, 1).to(latent)
            std = reference_vae.latents_std.view(1, -1, 1, 1, 1).to(latent)
            z[:, :, :1] = (latent - mean) * reference_vae.config.scaling_factor / std
        kwargs = dict(mask=mask, data_info={})
        if task == "t2v":
            solver = dpm.DPMS(
                reference,
                condition=positive[:, None],
                uncondition=negative[:, None],
                cfg_scale=8.0,
                model_type="flow",
                schedule="FLOW",
                model_kwargs=kwargs,
            )
            ref_latent, steps = solver.sample(
                z,
                steps=50,
                order=2,
                skip_type="time_uniform_flow",
                method="multistep",
                flow_shift=12.0,
                return_intermediate=True,
            )
            steps = steps[1:]  # upstream includes x_T; native callbacks are after each update.
        else:
            kwargs["data_info"]["condition_frame_info"] = {0: 0.0}
            with torch.device("cpu"):
                solver = euler.LTXFlowEuler(reference, positive[:, None], negative[:, None], 8.0, 12.0, kwargs)
            # Observe the actual upstream loop's post-update latent at the next model invocation.
            observed = []
            hook = reference.register_forward_pre_hook(lambda _m, a: observed.append(a[0][:1].detach().cpu().clone()))
            try:
                ref_latent = solver.sample(z, steps=50)
            finally:
                hook.remove()
            steps = observed[1:] + [ref_latent.cpu()]
        assert len(native_steps) == len(steps) == 50
        mean = reference_vae.latents_mean.view(1, -1, 1, 1, 1).to(reference_vae.dtype).to("cuda")
        std = reference_vae.latents_std.view(1, -1, 1, 1, 1).to(reference_vae.dtype).to("cuda")
        denorm = ref_latent.to(reference_vae.dtype) * std / reference_vae.config.scaling_factor + mean
        ref_video = reference_vae.decode(denorm, temb=None, return_dict=False)[0]
        receipt[task] = dict(
            steps=[compare(a, b) for a, b in zip(native_steps, steps)], decoded=compare(raw, ref_video)
        )
        np.savez(
            args.output / f"alignment-{task}.npz",
            native_steps=torch.stack(native_steps).float().numpy(),
            reference_steps=torch.stack([x.cpu() for x in steps]).float().numpy(),
            native_decoded=raw.float().cpu().numpy(),
            reference_decoded=ref_video.float().cpu().numpy(),
        )
        receipt[task]["artifact_sha256"] = digest((args.output / f"alignment-{task}.npz").read_bytes())
        save(args, receipt)


def http(args, receipt):
    import requests

    for task in ("t2v", "ti2v"):
        data = {
            **REQUEST,
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE,
            "seed": 42,
            "fps": 24,
            "extra_params": json.dumps({"motion_score": 10, "flow_shift": 12}),
        }
        with (args.output / "reference.png").open("rb") as image:
            files = {"input_reference": ("reference.png", image, "image/png")} if task == "ti2v" else None
            response = requests.post(args.url + "/v1/videos", data=data, files=files, timeout=600)
        response.raise_for_status()
        created = response.json()
        deadline = time.monotonic() + 600
        while True:
            status = requests.get(args.url + "/v1/videos/" + created["id"], timeout=30)
            status.raise_for_status()
            payload = status.json()
            if payload["status"] == "completed":
                break
            assert payload["status"] != "failed", payload
            assert time.monotonic() < deadline, payload
            time.sleep(1)
        content = requests.get(args.url + "/v1/videos/" + created["id"] + "/content", timeout=60)
        content.raise_for_status()
        path = args.output / f"http-{task}.mp4"
        path.write_bytes(content.content)
        video = iio.imread(path, plugin="FFMPEG")
        assert video.shape == (9, 64, 96, 3) and np.ptp(video) > 0, video.shape
        receipt[task] = dict(
            created=created,
            completed=payload,
            mp4_sha256=digest(content.content),
            mp4_bytes=len(content.content),
            decoded_shape=list(video.shape),
        )
        save(args, receipt)


def serve(args, receipt):
    from urllib.parse import urlparse

    from vllm_omni.entrypoints.cli.main import main as cli_main

    url = urlparse(args.url)
    save(args, receipt)
    sys.argv = [
        "vllm",
        "serve",
        str(args.assets / "model"),
        "--omni",
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--host",
        url.hostname,
        "--port",
        str(url.port or 8091),
        "--stage-overrides",
        json.dumps({"0": {"model_config": component_options(args), "distributed_executor_backend": "uni"}}),
    ]
    cli_main()


def save(args, receipt):
    (args.output / f"{args.mode}-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["alignment", "offline", "http", "serve"])
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8091")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[..., 0] = np.arange(120, dtype=np.uint8)[None] * 2
    image[..., 1] = np.arange(80, dtype=np.uint8)[:, None] * 3
    image[..., 2] = 80
    Image.fromarray(image).save(args.output / "reference.png")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "vllm_omni"], check=True)
    receipt = dict(
        source_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        hostname=platform.node(),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        python=platform.python_version(),
        runner_sha256=digest(Path(__file__).read_bytes()),
        packages={
            name: importlib.metadata.version(name) for name in ("torch", "vllm", "diffusers", "transformers", "triton")
        },
        components={
            name: json.loads((args.assets / name / "validated_revision.json").read_text())
            for name in ("model", "text", "vae")
        },
        request=REQUEST,
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        seed=42,
        motion_score=10,
        flow_shift=12.0,
        input_image_sha256=digest((args.output / "reference.png").read_bytes()),
    )
    if args.mode == "alignment":
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.utils.network_utils import get_open_port

        from vllm_omni.diffusion.distributed.parallel_state import (
            destroy_distributed_environment,
            init_distributed_environment,
        )

        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(
                world_size=1, rank=0, local_rank=0, distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}"
            )
            try:
                alignment(args, receipt)
            finally:
                destroy_distributed_environment()
    else:
        globals()[args.mode](args, receipt)
    save(args, receipt)


if __name__ == "__main__":
    main()
