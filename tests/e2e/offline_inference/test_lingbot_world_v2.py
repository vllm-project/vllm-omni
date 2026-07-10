# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline example contract and opt-in LingBot-World v2 GPU smoke test."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_ROOT = Path(__file__).parents[3]
_EXAMPLE_PATH = _ROOT / "examples/offline_inference/diffusion/lingbot_world_v2.py"
_RUN_E2E_ENV = "VLLM_OMNI_RUN_LINGBOT_WORLD_V2_E2E"
_MODEL_ENV = "VLLM_OMNI_LINGBOT_WORLD_V2_CHECKPOINT_PATH"
_IMAGE_ENV = "VLLM_OMNI_LINGBOT_WORLD_V2_IMAGE_PATH"
_ACTION_ENV = "VLLM_OMNI_LINGBOT_WORLD_V2_ACTION_DIR"


def _load_example():
    assert _EXAMPLE_PATH.exists(), "LingBot-World v2 offline example has not been implemented"
    spec = importlib.util.spec_from_file_location("_lingbot_world_v2_example_under_test", _EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_assets(tmp_path: Path) -> tuple[Path, Path]:
    image = tmp_path / "first-frame.png"
    image.write_bytes(b"test image placeholder")
    action_dir = tmp_path / "trusted-actions" / "forward"
    action_dir.mkdir(parents=True)
    np.save(action_dir / "poses.npy", np.eye(4, dtype=np.float32)[None])
    np.save(action_dir / "intrinsics.npy", np.ones((1, 4), dtype=np.float32))
    return image, action_dir


def test_parse_args_exposes_lingbot_generation_controls() -> None:
    module = _load_example()

    args = module.parse_args(
        [
            "--prompt",
            "move forward",
            "--image",
            "frame.png",
            "--action-dir",
            "actions/forward",
            "--height",
            "64",
            "--width",
            "96",
            "--num-frames",
            "9",
            "--seed",
            "7",
            "--tensor-parallel-size",
            "2",
            "--output",
            "result.mp4",
            "--model",
            "/models/lingbot",
            "--flow-shift",
            "6.0",
            "--fps",
            "12",
            "--enforce-eager",
        ]
    )

    assert vars(args) == {
        "action_dir": "actions/forward",
        "enforce_eager": True,
        "flow_shift": 6.0,
        "fps": 12,
        "height": 64,
        "image": "frame.png",
        "model": "/models/lingbot",
        "num_frames": 9,
        "output": "result.mp4",
        "prompt": "move forward",
        "seed": 7,
        "tensor_parallel_size": 2,
        "width": 96,
    }


def test_resolve_paths_builds_a_canonical_trusted_action_root(tmp_path: Path) -> None:
    module = _load_example()
    image, action_dir = _make_assets(tmp_path)
    args = module.parse_args(
        [
            "--prompt",
            "move forward",
            "--image",
            str(image),
            "--action-dir",
            str(action_dir),
            "--output",
            str(tmp_path / "outputs/clip.mp4"),
        ]
    )

    paths = module.resolve_cli_paths(args)

    assert paths.image == image.resolve()
    assert paths.action_dir == action_dir.resolve()
    assert paths.action_root == action_dir.parent.resolve()
    assert paths.action_relative == Path("forward")
    assert paths.output == (tmp_path / "outputs/clip.mp4").resolve()


def test_resolve_paths_requires_both_camera_arrays(tmp_path: Path) -> None:
    module = _load_example()
    image, action_dir = _make_assets(tmp_path)
    (action_dir / "intrinsics.npy").unlink()
    args = module.parse_args(
        [
            "--prompt",
            "move forward",
            "--image",
            str(image),
            "--action-dir",
            str(action_dir),
        ]
    )

    with pytest.raises(ValueError, match="poses.npy and intrinsics.npy"):
        module.resolve_cli_paths(args)


def test_build_omni_kwargs_uses_tp_and_the_canonical_action_root(tmp_path: Path) -> None:
    module = _load_example()
    image, action_dir = _make_assets(tmp_path)
    args = module.parse_args(
        [
            "--prompt",
            "move forward",
            "--image",
            str(image),
            "--action-dir",
            str(action_dir),
            "--tensor-parallel-size",
            "2",
            "--flow-shift",
            "6.0",
            "--enforce-eager",
        ]
    )
    paths = module.resolve_cli_paths(args)
    parallel_config = object()

    kwargs = module.build_omni_kwargs(args, paths, parallel_config=parallel_config)

    assert kwargs == {
        "model": "robbyant/lingbot-world-v2-14b-causal-fast-diffusers",
        "flow_shift": 6.0,
        "parallel_config": parallel_config,
        "enforce_eager": True,
        "model_config": {"lingbot_action_root": str(action_dir.parent.resolve())},
    }
    assert "model_class_name" not in kwargs


def test_build_request_uses_fixed_dmd_and_text_contract(tmp_path: Path) -> None:
    module = _load_example()
    image, action_dir = _make_assets(tmp_path)
    args = module.parse_args(
        [
            "--prompt",
            "move forward",
            "--image",
            str(image),
            "--action-dir",
            str(action_dir),
            "--height",
            "64",
            "--width",
            "96",
            "--num-frames",
            "9",
            "--seed",
            "7",
            "--flow-shift",
            "6.0",
            "--fps",
            "12",
        ]
    )
    paths = module.resolve_cli_paths(args)

    prompt, sampling_kwargs = module.build_request(args, paths)

    assert prompt == {
        "prompt": "move forward",
        "multi_modal_data": {"image": str(image.resolve())},
    }
    assert sampling_kwargs == {
        "height": 64,
        "width": 96,
        "num_frames": 9,
        "num_inference_steps": 4,
        "max_sequence_length": 512,
        "seed": 7,
        "fps": 12,
        "extra_args": {"action_path": "forward", "flow_shift": 6.0},
    }


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--height", "65", "divisible by 16"),
        ("--num-frames", "13", "three-frame latent blocks"),
        ("--num-frames", "129", "117"),
    ],
)
def test_build_request_rejects_invalid_geometry_before_engine_start(
    tmp_path: Path, flag: str, value: str, message: str
) -> None:
    module = _load_example()
    image, action_dir = _make_assets(tmp_path)
    args = module.parse_args(
        [
            "--prompt",
            "move forward",
            "--image",
            str(image),
            "--action-dir",
            str(action_dir),
            flag,
            value,
        ]
    )

    with pytest.raises(ValueError, match=message):
        module.build_request(args, module.resolve_cli_paths(args))


def test_extract_video_array_unwraps_omni_diffusion_output() -> None:
    module = _load_example()
    video = np.zeros((1, 9, 8, 8, 3), dtype=np.float32)
    outputs = [SimpleNamespace(images=[video], request_output=None)]

    frames = module.extract_video_array(outputs)

    assert frames.shape == (9, 8, 8, 3)
    assert frames.dtype == np.float32


def _required_e2e_path(env_name: str, *, directory: bool) -> Path:
    raw_path = os.environ.get(env_name)
    if not raw_path:
        pytest.skip(f"{env_name} is required for the opt-in LingBot-World v2 E2E")
    path = Path(raw_path).expanduser().resolve()
    if directory and not path.is_dir():
        pytest.skip(f"{env_name} must point to an available directory")
    if not directory and not path.is_file():
        pytest.skip(f"{env_name} must point to an available file")
    return path


@pytest.mark.full_model
@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.H100
@pytest.mark.skipif(
    os.environ.get(_RUN_E2E_ENV) != "1",
    reason=f"set {_RUN_E2E_ENV}=1 and the LingBot asset-path variables to run",
)
def test_lingbot_world_v2_real_checkpoint_one_block(tmp_path: Path) -> None:
    """Auto-discover the official class and generate one 3-latent/9-raw-frame block."""

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("LingBot-World v2 E2E requires CUDA")

    model = _required_e2e_path(_MODEL_ENV, directory=True)
    image = _required_e2e_path(_IMAGE_ENV, directory=False)
    action_dir = _required_e2e_path(_ACTION_ENV, directory=True)
    module = _load_example()

    from diffusers.utils import export_to_video

    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.model_extras import get_model_class_name

    output_path = tmp_path / "lingbot-world-v2-one-block.mp4"
    args = module.parse_args(
        [
            "--model",
            str(model),
            "--prompt",
            "The camera moves slowly forward through the scene.",
            "--image",
            str(image),
            "--action-dir",
            str(action_dir),
            "--height",
            "64",
            "--width",
            "64",
            "--num-frames",
            "9",
            "--output",
            str(output_path),
            "--enforce-eager",
        ]
    )
    paths = module.resolve_cli_paths(args)
    parallel_config = DiffusionParallelConfig(tensor_parallel_size=args.tensor_parallel_size)
    omni = Omni(**module.build_omni_kwargs(args, paths, parallel_config=parallel_config))
    try:
        assert get_model_class_name(omni) == "LingBotWorldCausalDMDPipeline"
        prompt, sampling_kwargs = module.build_request(args, paths)
        outputs = omni.generate(
            prompt,
            OmniDiffusionSamplingParams(**sampling_kwargs),
            use_tqdm=False,
        )
        frames = module.extract_video_array(outputs)
        assert frames.shape == (9, 64, 64, 3)
        assert np.isfinite(frames).all()
        export_to_video(frames, str(output_path), fps=args.fps)
        assert output_path.is_file() and output_path.stat().st_size > 0
        peak_memory_mb = max(float(getattr(output, "peak_memory_mb", 0.0)) for output in outputs)
        print(f"LingBot-World v2 E2E artifact={output_path} shape={frames.shape} peak_memory_mb={peak_memory_mb:.2f}")
    finally:
        omni.close()
