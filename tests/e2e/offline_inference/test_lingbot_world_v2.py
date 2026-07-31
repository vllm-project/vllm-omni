# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline example contract and opt-in LingBot-World v2 GPU tests.

The CUDA matrix uses the repository's H100 resource class, which also covers
H200. It includes a compiled default-resolution TP=1 run, eager TP=2 sharding, request
determinism/camera sensitivity, and an 81-frame run that crosses the
checkpoint's 18-latent-frame sliding window. Set
``VLLM_OMNI_RUN_LINGBOT_WORLD_V2_E2E=1`` together with checkpoint, image,
primary-action, and alternate-action path variables defined below. The two
action directories must be siblings under the same trusted root.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.helpers.mark import hardware_test

_ROOT = Path(__file__).parents[3]
_EXAMPLE_PATH = _ROOT / "examples/offline_inference/diffusion/lingbot_world_v2.py"
_REALTIME_EXAMPLE_PATH = _ROOT / "examples/offline_inference/diffusion/lingbot_world_v2_realtime.py"
_RUN_E2E_ENV = "VLLM_OMNI_RUN_LINGBOT_WORLD_V2_E2E"
_MODEL_ENV = "VLLM_OMNI_LINGBOT_WORLD_V2_CHECKPOINT_PATH"
_IMAGE_ENV = "VLLM_OMNI_LINGBOT_WORLD_V2_IMAGE_PATH"
_ACTION_ENV = "VLLM_OMNI_LINGBOT_WORLD_V2_ACTION_DIR"
_ALTERNATE_ACTION_ENV = "VLLM_OMNI_LINGBOT_WORLD_V2_ALTERNATE_ACTION_DIR"


def _load_example():
    assert _EXAMPLE_PATH.exists(), "LingBot-World v2 offline example has not been implemented"
    spec = importlib.util.spec_from_file_location("_lingbot_world_v2_example_under_test", _EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_realtime_example():
    assert _REALTIME_EXAMPLE_PATH.exists(), "LingBot-World v2 realtime example has not been implemented"
    spec = importlib.util.spec_from_file_location(
        "_lingbot_world_v2_realtime_example_under_test", _REALTIME_EXAMPLE_PATH
    )
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
    np.save(action_dir / "poses.npy", np.repeat(np.eye(4, dtype=np.float32)[None], 117, axis=0))
    np.save(action_dir / "intrinsics.npy", np.ones((117, 4), dtype=np.float32))
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


def test_realtime_example_loads_prompt_and_action_events(tmp_path: Path) -> None:
    module = _load_realtime_example()
    events = tmp_path / "events.jsonl"
    events.write_text(
        '{"event_id":1,"frames":[["j"],[],[]]}\n{"event_id":2,"prompt":"snowy valley","frames":[["w"],["w"],["w"]]}\n'
    )

    assert module._load_events(events) == [
        {"event_id": 1, "prompt": None, "frames": [["j"], [], []]},
        {
            "event_id": 2,
            "prompt": "snowy valley",
            "frames": [["w"], ["w"], ["w"]],
        },
    ]


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
def test_realtime_example_uses_event_side_camera_script() -> None:
    module = _load_realtime_example()
    frames = [["j"], [], []]

    from vllm_omni.diffusion.models.lingbot_world.actions import (
        LINGBOT_CAMERA_ACTION_SCHEMA,
        LingBotCameraControlReducer,
    )
    from vllm_omni.experimental.ar_diffusion.session import (
        ARDiffusionSessionEvent,
    )
    from vllm_omni.experimental.ar_diffusion.tick_protocol import (
        ARDiffusionControlInput,
    )

    event_data = module._camera_event_data(frames)
    prepared = LingBotCameraControlReducer().prepare(
        current_controls={},
        events=(
            ARDiffusionSessionEvent(
                event_id=1,
                controls=(
                    ARDiffusionControlInput(
                        track="camera",
                        schema=LINGBOT_CAMERA_ACTION_SCHEMA,
                        data=event_data,
                    ),
                ),
            ),
        ),
        chunk_index=0,
    )

    assert event_data == {"mode": "script", "frames": frames}
    assert prepared.controls[0].data == {
        "mode": "frames",
        "frames": (("j",), (), ()),
    }


def test_realtime_example_rejects_non_monotonic_event_ids(tmp_path: Path) -> None:
    module = _load_realtime_example()
    events = tmp_path / "events.jsonl"
    events.write_text('{"event_id":2,"prompt":"a"}\n{"event_id":1,"prompt":"b"}\n')

    with pytest.raises(ValueError, match="strictly increasing"):
        module._load_events(events)


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
def test_realtime_example_rejects_more_than_ten_ticks(tmp_path: Path) -> None:
    module = _load_realtime_example()
    events = tmp_path / "events.jsonl"
    events.write_text(
        "".join(json.dumps({"event_id": event_id, "prompt": f"scene {event_id}"}) + "\n" for event_id in range(1, 12))
    )

    with pytest.raises(ValueError, match="at most 10 events.*117 pixel frames"):
        module._load_events(events)


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
    assert paths.camera_frames == 117
    assert paths.output == (tmp_path / "outputs/clip.mp4").resolve()


def test_resolve_paths_accepts_official_trajectory_longer_than_one_request(tmp_path: Path) -> None:
    module = _load_example()
    image, action_dir = _make_assets(tmp_path)
    np.save(action_dir / "poses.npy", np.repeat(np.eye(4, dtype=np.float32)[None], 269, axis=0))
    np.save(action_dir / "intrinsics.npy", np.ones((269, 4), dtype=np.float32))
    args = module.parse_args(
        [
            "--prompt",
            "move forward",
            "--image",
            str(image),
            "--action-dir",
            str(action_dir),
            "--num-frames",
            "81",
        ]
    )

    paths = module.resolve_cli_paths(args)
    _prompt, sampling_kwargs = module.build_request(args, paths)

    assert paths.camera_frames == 269
    assert sampling_kwargs["num_frames"] == 81


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


def test_resolve_paths_rejects_invalid_camera_shape_before_engine_start(tmp_path: Path) -> None:
    module = _load_example()
    image, action_dir = _make_assets(tmp_path)
    np.save(action_dir / "poses.npy", np.zeros((9, 3, 4), dtype=np.float32))
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

    with pytest.raises(ValueError, match="poses.npy.*frames, 4, 4"):
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


def _e2e_case(
    module,
    *,
    model: Path,
    image: Path,
    action_dir: Path,
    output_path: Path,
    num_frames: int,
    tensor_parallel_size: int,
    height: int = 64,
    width: int = 64,
    enforce_eager: bool = False,
):
    argv = [
        "--model",
        str(model),
        "--prompt",
        "The camera moves slowly forward through the scene.",
        "--image",
        str(image),
        "--action-dir",
        str(action_dir),
        "--height",
        str(height),
        "--width",
        str(width),
        "--num-frames",
        str(num_frames),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--output",
        str(output_path),
    ]
    if enforce_eager:
        argv.append("--enforce-eager")
    args = module.parse_args(argv)
    return args, module.resolve_cli_paths(args)


def _peak_memory_mb(outputs) -> float:
    peaks = []
    for output in outputs:
        request_output = getattr(output, "request_output", None)
        value = getattr(request_output, "peak_memory_mb", None) if request_output is not None else None
        if value is None:
            value = getattr(output, "peak_memory_mb", 0.0)
        peaks.append(float(value))
    return max(peaks, default=0.0)


def _generate(omni, module, args, paths):
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    prompt, sampling_kwargs = module.build_request(args, paths)
    outputs = omni.generate(
        prompt,
        OmniDiffusionSamplingParams(**sampling_kwargs),
        use_tqdm=False,
    )
    return module.extract_video_array(outputs), _peak_memory_mb(outputs)


def _assert_non_degenerate_video(
    frames: np.ndarray,
    *,
    num_frames: int,
    height: int = 64,
    width: int = 64,
) -> None:
    assert frames.shape == (num_frames, height, width, 3)
    assert np.issubdtype(frames.dtype, np.number)
    assert np.isfinite(frames).all()
    values = frames.astype(np.float32, copy=False)
    assert float(np.ptp(values)) > 1e-6, "generated video is spatially constant"
    assert float(np.mean(np.abs(np.diff(values, axis=0)))) > 1e-6, "generated video is temporally constant"


def _run_one_block(
    tmp_path: Path,
    *,
    tensor_parallel_size: int,
    height: int,
    width: int,
    enforce_eager: bool,
) -> None:
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
    from vllm_omni.model_extras import get_model_class_name

    output_path = tmp_path / f"lingbot-world-v2-tp{tensor_parallel_size}-{height}x{width}-one-block.mp4"
    args, paths = _e2e_case(
        module,
        model=model,
        image=image,
        action_dir=action_dir,
        output_path=output_path,
        num_frames=9,
        tensor_parallel_size=tensor_parallel_size,
        height=height,
        width=width,
        enforce_eager=enforce_eager,
    )
    parallel_config = DiffusionParallelConfig(tensor_parallel_size=tensor_parallel_size)
    omni = Omni(**module.build_omni_kwargs(args, paths, parallel_config=parallel_config))
    try:
        assert get_model_class_name(omni) == "LingBotWorldCausalDMDPipeline"
        frames, peak_memory_mb = _generate(omni, module, args, paths)
        _assert_non_degenerate_video(frames, num_frames=9, height=height, width=width)
        assert peak_memory_mb > 0.0
        export_to_video(frames, str(output_path), fps=args.fps)
        assert output_path.is_file() and output_path.stat().st_size > 0
        print(
            f"LingBot-World v2 TP={tensor_parallel_size} artifact={output_path} "
            f"shape={frames.shape} peak_memory_mb={peak_memory_mb:.2f}"
        )
    finally:
        omni.close()


@pytest.mark.full_model
@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.skipif(
    os.environ.get(_RUN_E2E_ENV) != "1",
    reason=f"set {_RUN_E2E_ENV}=1 and the LingBot asset-path variables to run",
)
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_lingbot_world_v2_real_checkpoint_tp1_one_block(tmp_path: Path) -> None:
    """Auto-discover and generate one default-resolution block on one GPU."""

    _run_one_block(
        tmp_path,
        tensor_parallel_size=1,
        height=480,
        width=832,
        enforce_eager=False,
    )


@pytest.mark.full_model
@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.skipif(
    os.environ.get(_RUN_E2E_ENV) != "1",
    reason=f"set {_RUN_E2E_ENV}=1 and the LingBot asset-path variables to run",
)
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_lingbot_world_v2_real_checkpoint_tp2_one_block(tmp_path: Path) -> None:
    """Load, shard, and generate one block with tensor parallel size two."""

    _run_one_block(
        tmp_path,
        tensor_parallel_size=2,
        height=64,
        width=64,
        enforce_eager=True,
    )


@pytest.mark.full_model
@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.skipif(
    os.environ.get(_RUN_E2E_ENV) != "1",
    reason=f"set {_RUN_E2E_ENV}=1 and the LingBot asset-path variables to run",
)
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_lingbot_world_v2_multi_block_determinism_and_camera_sensitivity(tmp_path: Path) -> None:
    """Exercise multi-block cache reuse and request isolation on real CUDA."""

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("LingBot-World v2 E2E requires CUDA")

    model = _required_e2e_path(_MODEL_ENV, directory=True)
    image = _required_e2e_path(_IMAGE_ENV, directory=False)
    action_dir = _required_e2e_path(_ACTION_ENV, directory=True)
    alternate_action_dir = _required_e2e_path(_ALTERNATE_ACTION_ENV, directory=True)
    if alternate_action_dir.parent != action_dir.parent:
        pytest.skip(f"{_ACTION_ENV} and {_ALTERNATE_ACTION_ENV} must be sibling directories under one trusted root")
    module = _load_example()

    from diffusers.utils import export_to_video

    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.entrypoints.omni import Omni

    primary_args, primary_paths = _e2e_case(
        module,
        model=model,
        image=image,
        action_dir=action_dir,
        output_path=tmp_path / "lingbot-world-v2-primary.mp4",
        num_frames=21,
        tensor_parallel_size=1,
        enforce_eager=True,
    )
    alternate_args, alternate_paths = _e2e_case(
        module,
        model=model,
        image=image,
        action_dir=alternate_action_dir,
        output_path=tmp_path / "lingbot-world-v2-alternate.mp4",
        num_frames=21,
        tensor_parallel_size=1,
        enforce_eager=True,
    )
    parallel_config = DiffusionParallelConfig(tensor_parallel_size=1)
    omni = Omni(**module.build_omni_kwargs(primary_args, primary_paths, parallel_config=parallel_config))
    try:
        primary, primary_peak = _generate(omni, module, primary_args, primary_paths)
        repeated, repeated_peak = _generate(omni, module, primary_args, primary_paths)
        alternate, alternate_peak = _generate(omni, module, alternate_args, alternate_paths)
    finally:
        omni.close()

    for frames in (primary, repeated, alternate):
        _assert_non_degenerate_video(frames, num_frames=21)
    assert min(primary_peak, repeated_peak, alternate_peak) > 0.0
    np.testing.assert_allclose(repeated, primary, rtol=1e-3, atol=1e-3)
    camera_mae = float(np.mean(np.abs(alternate.astype(np.float32) - primary.astype(np.float32))))
    assert camera_mae > 1e-5, "changing the camera action did not change the generated video"

    export_to_video(primary, str(primary_paths.output), fps=primary_args.fps)
    export_to_video(alternate, str(alternate_paths.output), fps=alternate_args.fps)
    assert primary_paths.output.stat().st_size > 0
    assert alternate_paths.output.stat().st_size > 0
    print(
        f"LingBot-World v2 multi-block primary={primary_paths.output} alternate={alternate_paths.output} "
        f"camera_mae={camera_mae:.6f} peaks_mb={(primary_peak, repeated_peak, alternate_peak)}"
    )


@pytest.mark.full_model
@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.skipif(
    os.environ.get(_RUN_E2E_ENV) != "1",
    reason=f"set {_RUN_E2E_ENV}=1 and the LingBot asset-path variables to run",
)
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_lingbot_world_v2_real_checkpoint_crosses_sliding_window(tmp_path: Path) -> None:
    """Generate seven blocks so the 18-latent-frame cache must evict history."""

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

    args, paths = _e2e_case(
        module,
        model=model,
        image=image,
        action_dir=action_dir,
        output_path=tmp_path / "lingbot-world-v2-sliding-window.mp4",
        num_frames=81,
        tensor_parallel_size=1,
        enforce_eager=True,
    )
    parallel_config = DiffusionParallelConfig(tensor_parallel_size=1)
    omni = Omni(**module.build_omni_kwargs(args, paths, parallel_config=parallel_config))
    try:
        frames, peak_memory_mb = _generate(omni, module, args, paths)
    finally:
        omni.close()

    _assert_non_degenerate_video(frames, num_frames=81)
    assert peak_memory_mb > 0.0
    export_to_video(frames, str(paths.output), fps=args.fps)
    assert paths.output.is_file() and paths.output.stat().st_size > 0
    print(
        f"LingBot-World v2 sliding-window artifact={paths.output} "
        f"shape={frames.shape} peak_memory_mb={peak_memory_mb:.2f}"
    )
