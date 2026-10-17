# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from pathlib import Path
from typing import Any

import pytest

from examples.offline_inference.diffusion import ar_diffusion_streaming_decode as streaming_example
from examples.offline_inference.diffusion import lingbot_world_v2 as example

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_offline_ulysses_argument(tmp_path):
    argv = ["--image", "frame.jpg", "--action-dir", "forward", "--prompt", "A lake"]
    paths = example.LingBotPaths(
        tmp_path / "frame.jpg", tmp_path / "forward", tmp_path, Path("forward"), 9, tmp_path / "out.mp4"
    )
    assert example.parse_args(argv).ulysses_degree == 1
    args = example.parse_args([*argv, "--ulysses-degree", "4"])
    kwargs = example.build_omni_kwargs(args, paths)
    assert kwargs["ulysses_degree"] == 4 and kwargs["tensor_parallel_size"] == 1
    args.ulysses_degree = 0
    with pytest.raises(ValueError, match="--ulysses-degree"):
        example.build_omni_kwargs(args, paths)


@pytest.mark.parametrize(("num_blocks", "num_frames"), [(10, 117), (11, 129)])
def test_streaming_action_script_accepts_long_rollouts(tmp_path, num_blocks, num_frames):
    action_script = tmp_path / "actions.json"
    action_script.write_text(json.dumps([[["w"], [], ["a"]] for _ in range(num_blocks)]))

    script = streaming_example._load_action_script(action_script)

    assert len(script) == num_blocks
    assert (
        streaming_example._validate_against_checkpoint(script, frames_per_block=3, temporal_compression=4) == num_frames
    )


def test_streaming_action_script_rejects_wrong_frame_count_in_eleventh_block(tmp_path):
    action_script = tmp_path / "actions.json"
    action_script.write_text(json.dumps([[[], [], []] for _ in range(10)] + [[[], []]]))
    script = streaming_example._load_action_script(action_script)

    with pytest.raises(ValueError, match="block 10 holds 2 frame action lists"):
        streaming_example._validate_against_checkpoint(script, frames_per_block=3, temporal_compression=4)


def test_streaming_native_geometry_does_not_require_diffusers_config(tmp_path):
    assert streaming_example._checkpoint_geometry(str(tmp_path), frames_per_block=3, temporal_compression=4) == (3, 4)
    assert (
        streaming_example._validate_against_checkpoint(
            [[[], [], []] for _ in range(35)], frames_per_block=3, temporal_compression=4
        )
        == 417
    )


@pytest.mark.parametrize("geometry", [{"frames_per_block": 3}, {"frames_per_block": 0, "temporal_compression": 4}])
def test_streaming_native_geometry_rejects_incomplete_or_invalid_contract(tmp_path, geometry):
    with pytest.raises(ValueError):
        streaming_example._checkpoint_geometry(str(tmp_path), **geometry)


def test_streaming_checkpoint_geometry_remains_config_driven(tmp_path):
    (tmp_path / "transformer").mkdir()
    (tmp_path / "vae").mkdir()
    (tmp_path / "transformer/config.json").write_text(json.dumps({"num_frames_per_block": 2}))
    (tmp_path / "vae/config.json").write_text(json.dumps({"scale_factor_temporal": 8}))
    assert streaming_example._checkpoint_geometry(str(tmp_path)) == (2, 8)


@pytest.mark.parametrize("config", ["[]", "not-json"])
def test_streaming_model_overrides_require_json_object(tmp_path, config):
    image, actions = tmp_path / "image.png", tmp_path / "actions.json"
    image.touch()
    actions.write_text("[[[],[],[]]]")
    args = streaming_example.parse_args(
        [
            "--image",
            str(image),
            "--action-script",
            str(actions),
            "--prompt",
            "A lake",
            "--output-dir",
            str(tmp_path / "out"),
            "--model-config",
            config,
        ]
    )
    with pytest.raises(ValueError, match="--model-config"):
        streaming_example._validate_args(args)


def test_streaming_pixels_reject_nonfinite_values():
    import numpy as np

    with pytest.raises(RuntimeError, match="non-finite"):
        streaming_example._as_frames(np.full((9, 2, 2, 3), np.nan))


@pytest.mark.parametrize("failure", [None, "error", "shape", "missing"])
def test_shared_streaming_runner_native_request_and_fail_closed(tmp_path, monkeypatch, failure):
    import asyncio
    import sys
    from types import SimpleNamespace

    import numpy as np

    image, actions = tmp_path / "image.png", tmp_path / "actions.json"
    image.touch()
    actions.write_text(json.dumps([[[], []], [[], []]]))
    calls: dict[str, Any] = {}

    class Engine:
        def __init__(self, **kwargs):
            calls["engine"] = kwargs

        async def generate(self, prompt, sampling, request_id):
            calls["sampling"] = sampling
            calls["requests"] = calls.get("requests", 0) + 1
            for index, frames in enumerate([9, 16]):
                if failure == "missing" and index == 1:
                    return
                yield SimpleNamespace(
                    error="decode failed" if failure == "error" else None,
                    images=[np.zeros((1 if failure == "shape" else frames, 16, 16, 3), dtype=np.uint8)],
                    multimodal_output={"metadata": {"ar_diffusion": {"chunk_index": index}}},
                )

        def shutdown(self):
            calls["shutdown"] = True

    monkeypatch.setitem(sys.modules, "vllm_omni.entrypoints.async_omni", SimpleNamespace(AsyncOmni=Engine))
    argv = [
        "--model",
        str(tmp_path),
        "--model-class-name",
        "NativeTestPipeline",
        "--model-config",
        '{"decoder":"test"}',
        "--image",
        str(image),
        "--action-script",
        str(actions),
        "--prompt",
        "A lake",
        "--output-dir",
        str(tmp_path / "out"),
        "--height",
        "16",
        "--width",
        "16",
        "--frames-per-block",
        "2",
        "--temporal-compression",
        "8",
        "--enforce-eager",
    ]
    if failure:
        with pytest.raises(RuntimeError):
            asyncio.run(streaming_example.run(argv))
        assert not (tmp_path / "out/summary.json").exists()
    else:
        summary = json.loads(asyncio.run(streaming_example.run(argv)).read_text())
        assert summary["chunks_delivered"] == 2
        assert summary["frames_delivered"] == summary["frames_expected"] == 25
        assert calls["sampling"].num_frames == 25
        assert calls["engine"]["model_class_name"] == "NativeTestPipeline"
        assert calls["engine"]["model_config"]["decoder"] == "test"
        assert calls["engine"]["model_config"]["ar_diffusion_kv_config"]["warmup_cudagraph"] is False
        assert calls["engine"]["step_execution"] is True
    assert calls["requests"] == 1
    assert calls["shutdown"] is True
