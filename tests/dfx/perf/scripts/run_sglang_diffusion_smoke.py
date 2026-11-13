# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Generate and decode one sample using the comparison image's SGLang environment."""

import argparse
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

from tests.dfx.perf.scripts.experiment_metadata import comparison_environment

ROOT = Path(__file__).resolve().parents[4]
PROMPT = "A red ceramic teapot sits on a wooden table beside a sunlit window."


@dataclass
class Case:
    model: str
    num_gpus: int
    size: tuple[int, int]
    steps: int
    frames: int = 1
    server: dict = field(default_factory=dict)
    sampling: dict = field(default_factory=dict)


CASES = {
    "qwen-image": Case(
        "Qwen/Qwen-Image-2.1",
        1,
        (512, 512),
        20,
        server={"attention_backend": "torch_sdpa"},
        sampling={"guidance_scale": 1.0},
    ),
    "wan22-i2v": Case(
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        1,
        (832, 480),
        4,
        81,
        server={"dit_cpu_offload": True, "text_encoder_cpu_offload": True},
        sampling={"fps": 16, "negative_prompt": "Negative prompt for benchmarking diffusion models"},
    ),
    "hunyuan-image3": Case("tencent/HunyuanImage-3.0-Instruct", 4, (1024, 1024), 8, server={"tp_size": 4}),
    "minimax-h3": Case(
        "MiniMaxAI/MiniMax-H3",
        2,
        (1344, 768),
        8,
        107,
        # Same 2-H100 T2VA deployment as SGLang 0.5.22's gpu_cases.py.
        server={
            "tp_size": 2,
            "ulysses_degree": 1,
            "model_variant": "fl2va",
            "performance_mode": "memory",
            "layerwise_offload_components": "dit,text_encoder,vae",
            "layerwise_resident_layers": {"video_vae": 36},
            "dit_offload_prefetch_size": 1,
            "dit_layerwise_resident_layers": 20,
            "enable_torch_compile": False,
        },
        sampling={
            "task": "t2va",
            "conditions": [],
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
            "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 4.0},
        },
    ),
    "cosmos3-nano": Case(
        "nvidia/Cosmos3-Nano",
        1,
        (1024, 1024),
        4,
        sampling={"guidance_scale": 7.0, "flow_shift": 3.0, "use_guardrails": False},
    ),
    "hunyuanvideo15": Case(
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        1,
        (832, 480),
        4,
        33,
        server={"backend": "diffusers", "dit_cpu_offload": True},
        sampling={"fps": 24, "negative_prompt": "Negative prompt for benchmarking diffusion models"},
    ),
}


def inspect_output(path: Path, case: Case, require_audio: bool) -> dict:
    if case.frames == 1:
        from PIL import Image

        with Image.open(path) as img:
            img.load()
            size = img.size
            frames = 1
            media = {"format": img.format, "mode": img.mode}
    else:
        import av

        with av.open(str(path)) as container:
            video = container.streams.video[0]
            size = (video.width, video.height)
            frames = sum(1 for _ in container.decode(video))
            media = {"format": video.codec_context.name, "fps": float(video.average_rate)}
        if require_audio:
            with av.open(str(path)) as container:
                audio = next(container.decode(audio=0))
                media["audio_sample_rate"] = audio.sample_rate
    if size != case.size or frames != case.frames:
        raise ValueError(f"Expected {case.size} with {case.frames} frames, decoded {size} with {frames}")
    return {**media, "width": size[0], "height": size[1], "frames": frames, "bytes": path.stat().st_size}


def run(name: str, case: Case, record: dict, workdir: Path) -> None:
    from PIL import Image
    from sglang.multimodal_gen import DiffGenerator
    from sglang.multimodal_gen.registry import get_model_info

    backend = case.server.get("backend", "sglang")
    server = {
        "model_path": case.model,
        "num_gpus": case.num_gpus,
        "backend": backend,
        "warmup_mode": "off",
        **case.server,
    }
    sampling = {"prompt": PROMPT, "seed": 42, "num_inference_steps": case.steps, **case.sampling}
    # H3 derives its canvas and 17n+5 frame count from its canonical target.
    if name != "minimax-h3":
        sampling.update(width=case.size[0], height=case.size[1], num_frames=case.frames)
    if name == "wan22-i2v":
        source = Path("tests/assets/hunyuan/hunyuan_image_ref.png")
        record["source_image"] = str(source)
        with Image.open(ROOT / source) as image:
            image.resize(case.size).save(workdir / "input.png")
        sampling["image_path"] = str(workdir / "input.png")
        sampling["prompt"] = "A slow camera pan across the scene in the reference image."
    if name == "cosmos3-nano":
        os.environ["SGLANG_DISABLE_COSMOS3_GUARDRAILS"] = "1"

    record.update(server=server, sampling=sampling)
    info = get_model_info(case.model, backend=backend)
    pipeline = info.pipeline_cls.__name__
    record["pipeline"] = pipeline
    if (pipeline == "DiffusersPipeline") != (backend == "diffusers"):
        raise RuntimeError(f"Unexpected pipeline for backend {backend}: {pipeline}")

    with DiffGenerator.from_pretrained(**server) as generator:
        start = time.monotonic()
        result = generator.generate({**sampling, "output_path": str(workdir), "output_file_name": "sample"})
        record["cold_generate_seconds"] = time.monotonic() - start
        if result is None or isinstance(result, list) or not result.output_file_path:
            raise ValueError("Expected exactly one saved generation")
        record["media"] = inspect_output(Path(result.output_file_path), case, require_audio=name == "minimax-h3")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    args = parser.parse_args()
    case = CASES[args.case]
    result_dir = Path(os.environ.get("BENCHMARK_DIR", ROOT / "tests/dfx/perf/results"))
    result_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "purpose": "single_request_offline_smoke",
        "case": args.case,
        "model": case.model,
        "num_gpus": case.num_gpus,
        "build_id": os.getenv("BUILDKITE_BUILD_ID"),
        "job_id": os.getenv("BUILDKITE_JOB_ID"),
        "image_digest": os.getenv("COMPARISON_IMAGE_DIGEST"),
        "runner_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    try:
        record.update(comparison_environment(record["runner_commit"]))
        record["versions"] = {name: version(name) for name in ("sglang", "diffusers", "torch", "transformers")}
        lock = record.get("environment_lock")
        expected = (
            {name: lock["installed_packages"]["sglang"][name] for name in record["versions"]}
            if lock
            else {"sglang": "0.5.22", "diffusers": "0.37.0"}
        )
        if any(record["versions"][name] != value for name, value in expected.items()):
            raise RuntimeError(f"Expected the pinned comparison environment, found {record['versions']}")
        import torch

        cuda = torch.get_device_module("cuda")
        record["visible_gpus"] = [
            {"name": props.name, "total_memory_bytes": props.total_memory}
            for props in (cuda.get_device_properties(i) for i in range(cuda.device_count()))
        ]
        if len(record["visible_gpus"]) != case.num_gpus:
            raise RuntimeError(f"Expected {case.num_gpus} GPUs, found {record['visible_gpus']}")
        with tempfile.TemporaryDirectory() as directory:
            run(args.case, case, record, Path(directory))
        record["status"] = "passed"
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (result_dir / f"smoke_{args.case}.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
