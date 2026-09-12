# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]

REPO_ROOT = Path(__file__).parents[3]


@pytest.fixture(scope="module")
def cosmos3_multiview() -> ModuleType:
    path = REPO_ROOT / "examples/offline_inference/multiview_video/cosmos3_multiview.py"
    spec = importlib.util.spec_from_file_location("cosmos3_multiview_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_requests_accepts_json_and_jsonl(cosmos3_multiview: ModuleType, tmp_path: Path) -> None:
    first = {"name": "first", "prompt": "one"}
    second = {"name": "second", "prompt": "two"}

    json_path = tmp_path / "input.json"
    json_path.write_text(json.dumps(first))
    assert cosmos3_multiview._load_requests(json_path) == [first]

    jsonl_path = tmp_path / "input.jsonl"
    jsonl_path.write_text(f"{json.dumps(first)}\n\n{json.dumps(second)}\n")
    assert cosmos3_multiview._load_requests(jsonl_path) == [first, second]


def test_request_fields_control_mode_resolution_and_seed(cosmos3_multiview: ModuleType) -> None:
    views = [{"camera_key": "front", "control_path": "control.mp4", "vision_path": "vision.mp4"}]
    request = {"model_mode": "image2video", "resolution": "480", "seed": 123}

    assert cosmos3_multiview._resolve_model_mode(request, views) == "image2video"
    assert cosmos3_multiview._resolve_resolution(request, {}) == ("480", 832, 480)
    assert cosmos3_multiview._resolve_seed(request, base_seed=42, sample_index=7) == 123
    assert cosmos3_multiview._resolve_seed({}, base_seed=42, sample_index=7) == 49


def test_model_mode_must_match_per_view_vision_inputs(cosmos3_multiview: ModuleType) -> None:
    i2v_view = [{"camera_key": "front", "control_path": "control.mp4", "vision_path": "vision.mp4"}]
    t2v_view = [{"camera_key": "front", "control_path": "control.mp4"}]

    with pytest.raises(ValueError, match="must not include"):
        cosmos3_multiview._resolve_model_mode({"model_mode": "text2video"}, i2v_view)
    with pytest.raises(ValueError, match="requires vision"):
        cosmos3_multiview._resolve_model_mode({"model_mode": "image2video"}, t2v_view)


def test_run_request_forwards_imaginaire_fields_and_writes_manifest(
    cosmos3_multiview: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class FakeOmni:
        def generate(self, prompt, sampling_params):
            captured["prompt"] = prompt
            captured["sampling_params"] = sampling_params
            return {
                "payload": {"video": np.zeros((1, 1, 2, 2, 3), dtype=np.float32)},
                "metadata": {"multiview": {"cameras": ["front"], "frames_per_view": 1, "fps": 10}},
            }

    monkeypatch.setattr(cosmos3_multiview, "export_to_video", lambda *args, **kwargs: None)
    request = {
        "name": "sample",
        "model_mode": "image2video",
        "resolution": "480",
        "num_frames": 1,
        "prompt": '{"views": []}',
        "negative_prompt": "bad video",
        # Alias field names accepted alongside the vLLM-Omni ones.
        "guidance": 3.0,
        "num_steps": 20,
        "shift": 5.0,
        "wsm": {},
        "multiview": {
            "views": [
                {
                    "camera_key": "front",
                    "control_path": "control.mp4",
                    "vision_path": "vision.mp4",
                }
            ],
            "condition_video_as_image": True,
        },
    }

    manifest = cosmos3_multiview._run_request(
        FakeOmni(),
        request,
        output_dir=tmp_path,
        seed=45,
        fallback_negative_prompt=None,
    )

    sampling_params = captured["sampling_params"]
    assert sampling_params.seed == 45
    assert (sampling_params.width, sampling_params.height) == (832, 480)
    assert sampling_params.extra_args["resolution"] == "480"
    assert sampling_params.extra_args["multiview"]["resolution"] == "480"
    assert sampling_params.guidance_scale == 3.0
    assert sampling_params.num_inference_steps == 20
    assert sampling_params.extra_args["flow_shift"] == 5.0
    # No fps in the record: leave it to the pipeline's training-rate default.
    assert sampling_params.fps is None
    assert sampling_params.num_frames == 1
    assert captured["prompt"]["prompt"] == request["prompt"]
    assert captured["prompt"]["negative_prompt"] == "bad video"
    assert manifest["seed"] == 45
    assert manifest["model_mode"] == "image2video"
    assert manifest["resolution"] == "480"
    assert json.loads((tmp_path / "sample_outputs.json").read_text())["seed"] == 45


def test_extract_payload_reads_metadata_from_request_output(cosmos3_multiview: ModuleType) -> None:
    from types import SimpleNamespace

    frames = np.zeros((6, 2, 2, 3), dtype=np.float32)
    metadata = {"multiview": {"cameras": ["a", "b"], "frames_per_view": 3, "fps": 30.0}}
    request_output = SimpleNamespace(images=[frames], multimodal_output={"metadata": metadata})

    video, extracted = cosmos3_multiview._extract_payload([request_output])
    assert extracted == metadata
    assert cosmos3_multiview._frame_list(video)[0].shape == (2, 2, 3)

    # A request output without metadata still yields the frames.
    bare = SimpleNamespace(images=[frames], multimodal_output={})
    _, extracted = cosmos3_multiview._extract_payload(bare)
    assert extracted == {}


def test_frames_per_view_uses_actual_frame_count_not_requested(cosmos3_multiview: ModuleType) -> None:
    resolve = cosmos3_multiview._resolve_frames_per_view
    frames = list(range(2211))
    cameras = [f"cam{i}" for i in range(11)]

    # 200 requested frames were rounded to 201 by the pipeline.
    assert resolve(frames, cameras, {"multiview": {"frames_per_view": 201}}) == 201
    assert resolve(frames, cameras, {}) == 201
    with pytest.raises(ValueError, match="Expected 2200"):
        resolve(frames, cameras, {"multiview": {"frames_per_view": 200}})
    with pytest.raises(ValueError, match="evenly"):
        resolve(frames[:-1], cameras, {})


def test_run_request_splits_rounded_output_without_metadata(
    cosmos3_multiview: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    exported: list[int] = []

    class FakeOmni:
        def generate(self, prompt, sampling_params):
            # Two cameras, 5 frames each: the pipeline rounded the requested 4 up.
            return {"payload": {"video": np.zeros((1, 10, 2, 2, 3), dtype=np.float32)}}

    monkeypatch.setattr(
        cosmos3_multiview, "export_to_video", lambda frames, *args, **kwargs: exported.append(len(frames))
    )
    request = {
        "name": "sample",
        "prompt": "a drive",
        "num_frames": 4,
        "wsm": {},
        "multiview": {
            "views": [
                {"camera_key": "front", "control_path": "c0.mp4"},
                {"camera_key": "rear", "control_path": "c1.mp4"},
            ]
        },
    }

    manifest = cosmos3_multiview._run_request(
        FakeOmni(), request, output_dir=tmp_path, seed=1, fallback_negative_prompt=None
    )
    assert manifest["frames_per_view"] == 5
    assert exported == [5, 5]
    assert manifest["multiview_cameras"] == ["front", "rear"]


def test_vllm_omni_field_names_win_over_aliases(cosmos3_multiview: ModuleType) -> None:
    request = {"guidance_scale": 6.0, "guidance": 3.0, "num_steps": 20, "shift": 5.0, "flow_shift": 10.0}
    assert cosmos3_multiview._first_present(request, "guidance_scale", "guidance", default=1.0) == 6.0
    assert cosmos3_multiview._first_present(request, "num_inference_steps", "num_steps", default=35) == 20
    assert cosmos3_multiview._first_present(request, "flow_shift", "shift") == 10.0
    assert cosmos3_multiview._first_present({}, "flow_shift", "shift") is None
    assert cosmos3_multiview._first_present({"fps": None}, "fps", default=30.0) == 30.0


def test_run_request_cli_overrides_win_over_record_values(
    cosmos3_multiview: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class FakeOmni:
        def generate(self, prompt, sampling_params):
            captured["sampling_params"] = sampling_params
            return {
                "payload": {"video": np.zeros((1, 1, 2, 2, 3), dtype=np.float32)},
                "metadata": {"multiview": {"cameras": ["front"], "frames_per_view": 1, "fps": 30.0}},
            }

    monkeypatch.setattr(cosmos3_multiview, "export_to_video", lambda *args, **kwargs: None)
    request = {
        "name": "sample",
        "prompt": "a drive",
        "fps": 10,
        "wsm": {},
        "multiview": {
            "num_frames": 93,
            "views": [{"camera_key": "front", "control_path": "control.mp4"}],
        },
    }

    # Record values apply when the CLI is silent...
    cosmos3_multiview._run_request(FakeOmni(), request, output_dir=tmp_path, seed=1, fallback_negative_prompt=None)
    sampling_params = captured["sampling_params"]
    assert sampling_params.fps == 10.0
    assert sampling_params.num_frames == 93

    # ...and CLI overrides win over them.
    manifest = cosmos3_multiview._run_request(
        FakeOmni(),
        request,
        output_dir=tmp_path,
        seed=1,
        fallback_negative_prompt=None,
        fps_override=30.0,
        num_frames_override=200,
    )
    sampling_params = captured["sampling_params"]
    assert sampling_params.fps == 30.0
    assert sampling_params.num_frames == 200
    assert sampling_params.extra_args["multiview"]["num_frames"] == 200
    assert manifest["fps"] == 30.0
