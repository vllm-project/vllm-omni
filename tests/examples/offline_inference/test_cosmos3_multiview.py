# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import copy
import importlib.util
import json
import sys
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
    assert cosmos3_multiview._resolve_resolution(request, {}) == "480"
    assert cosmos3_multiview._resolve_seed(request, base_seed=42, sample_index=7) == 123
    assert cosmos3_multiview._resolve_seed({}, base_seed=42, sample_index=7) == 49


def test_model_mode_must_match_per_view_vision_inputs(cosmos3_multiview: ModuleType) -> None:
    i2v_view = [{"camera_key": "front", "control_path": "control.mp4", "vision_path": "vision.mp4"}]
    t2v_view = [{"camera_key": "front", "control_path": "control.mp4"}]

    with pytest.raises(ValueError, match="must not include"):
        cosmos3_multiview._resolve_model_mode({"model_mode": "text2video"}, i2v_view)
    with pytest.raises(ValueError, match="requires at least one camera vision"):
        cosmos3_multiview._resolve_model_mode({"model_mode": "image2video"}, t2v_view)


@pytest.mark.parametrize(
    "resolution,width,height", [(480, 832, 480), ("480", 832, 480), (720, 1280, 720), ("720", 1280, 720)]
)
def test_resolution_manifest_fields(cosmos3_multiview, resolution, width, height):
    resolve = cosmos3_multiview._resolve_resolution
    assert resolve({"resolution": resolution}, {}) == str(resolution)
    assert resolve({}, {"resolution": resolution}) == str(resolution)
    assert resolve({"resolution": str(resolution)}, {"resolution": resolution}) == str(resolution)
    assert resolve({}, {}) is None


def test_resolution_conflicts_and_invalid_buckets(cosmos3_multiview):
    resolve = cosmos3_multiview._resolve_resolution
    with pytest.raises(ValueError, match="Conflicting"):
        resolve({"resolution": "480"}, {"resolution": "720"})
    assert resolve({"resolution": "480"}, {"resolution": "720"}, "720") == "720"
    for resolution in (256, 704, "1080", "720p", True):
        with pytest.raises(ValueError, match="Unsupported"):
            resolve({"resolution": resolution}, {})


def test_aspect_ratio_manifest_fields_and_overrides(cosmos3_multiview):
    resolve = cosmos3_multiview._resolve_aspect_ratio
    assert resolve({}, {}) == "auto"
    assert resolve({"aspect_ratio": "auto"}, {}) == "auto"
    assert resolve({"aspect_ratio": "16:9"}, {"aspect_ratio": "32,18"}) == "16,9"
    assert resolve({}, {"aspect_ratio": "9:16"}) == "9,16"
    with pytest.raises(ValueError, match="Conflicting"):
        resolve({"aspect_ratio": "1:1"}, {"aspect_ratio": "auto"})
    assert resolve({"aspect_ratio": "1:1"}, {"aspect_ratio": "9:16"}, "auto") == "auto"
    with pytest.raises(ValueError, match="Unsupported"):
        resolve({"aspect_ratio": "21:9"}, {})


@pytest.mark.parametrize("override,expected", [(None, (640, None)), ("auto", (None, None)), ("3:4", (544, 736))])
def test_dimensions_are_constraints_until_geometry_override(cosmos3_multiview, tmp_path, override, expected):
    class StopBeforeGenerationError(Exception):
        pass

    class FakeOmni:
        def generate(self, prompt, sp):
            assert (sp.width, sp.height) == expected
            raise StopBeforeGenerationError

    request = {
        "width": 640,
        "resolution": "480",
        "multiview": {"views": [{"camera_key": "front", "control_path": "wsm.mp4"}]},
    }
    with pytest.raises(StopBeforeGenerationError):
        cosmos3_multiview._run_request(
            FakeOmni(),
            request,
            output_dir=tmp_path,
            seed=42,
            fallback_negative_prompt=None,
            aspect_ratio_override=override,
        )
    if override is None:
        request["aspect_ratio"] = "9:16"
        with pytest.raises(ValueError, match="requires width=480"):
            cosmos3_multiview._run_request(
                FakeOmni(),
                request,
                output_dir=tmp_path,
                seed=42,
                fallback_negative_prompt=None,
            )


@pytest.mark.parametrize("aspect_override", [None, "auto", "9:16", "1:1"])
def test_cli_resolution_overrides_every_jsonl_record(cosmos3_multiview, monkeypatch, tmp_path, aspect_override):
    records = [
        {"name": "a", "resolution": "480", "multiview": {"resolution": "720", "views": [{"camera_key": "front"}]}},
        {"name": "b", "multiview": {"views": [{"camera_key": "rear"}]}},
    ]
    before = copy.deepcopy(records)
    input_path = tmp_path / "requests.jsonl"
    input_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    captured = []
    expected_ratio = (aspect_override or "auto").replace(":", ",")
    # The fake pipeline detects portrait inputs when the client leaves sizing open.
    output_ratio = "9,16" if expected_ratio == "auto" else expected_ratio
    output_width, output_height = cosmos3_multiview.SUPPORTED_RESOLUTIONS["720"][output_ratio]

    class FakeOmni:
        def generate(self, prompt, sampling_params):
            captured.append(sampling_params)
            # Actual-sized frames ensure export never receives a 480p canvas.
            return {"payload": {"video": np.zeros((1, 1, output_height, output_width, 3), dtype=np.float32)}}

    exported_shapes = []
    monkeypatch.setattr(cosmos3_multiview, "Omni", lambda **kwargs: FakeOmni())
    monkeypatch.setattr(
        cosmos3_multiview, "export_to_video", lambda frames, *args, **kwargs: exported_shapes.append(frames[0].shape)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "example",
            "--model",
            "test",
            "--input",
            str(input_path),
            "--output-dir",
            str(tmp_path),
            "--resolution",
            "720",
        ]
        + (["--aspect-ratio", aspect_override] if aspect_override is not None else []),
    )
    cosmos3_multiview.main()
    assert len(captured) == 2
    for sp in captured:
        assert (sp.width, sp.height) == ((None, None) if expected_ratio == "auto" else (output_width, output_height))
        assert sp.extra_args["multiview"]["aspect_ratio"] == expected_ratio
        assert sp.extra_args["resolution"] == sp.extra_args["multiview"]["resolution"] == "720"
    assert exported_shapes == [(output_height, output_width, 3)] * 2
    manifests = [json.loads(line) for line in (tmp_path / "sample_outputs.jsonl").read_text().splitlines()]
    assert [manifest["resolution"] for manifest in manifests] == ["720", "720"]
    assert [manifest["aspect_ratio"] for manifest in manifests] == [output_ratio, output_ratio]
    assert cosmos3_multiview._load_requests(input_path) == before


@pytest.mark.parametrize(
    "resolution,ratio,width,height",
    [
        ("480", "1,1", 640, 640),
        ("480", "4,3", 736, 544),
        ("480", "3,4", 544, 736),
        ("480", "16,9", 832, 480),
        ("480", "9,16", 480, 832),
        ("720", "1,1", 960, 960),
        ("720", "4,3", 1104, 832),
        ("720", "3,4", 832, 1104),
        ("720", "16,9", 1280, 720),
        ("720", "9,16", 720, 1280),
    ],
)
@pytest.mark.parametrize("automatic", [False, True])
def test_run_request_forwards_imaginaire_fields_and_writes_manifest(
    cosmos3_multiview: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resolution,
    ratio,
    width,
    height,
    automatic,
) -> None:
    captured: dict[str, object] = {}

    class FakeOmni:
        def generate(self, prompt, sampling_params):
            captured["prompt"] = prompt
            captured["sampling_params"] = sampling_params
            return {
                "payload": {"video": np.broadcast_to(np.zeros(3, dtype=np.float32), (1, 1, height, width, 3))},
                "metadata": {
                    "multiview": {
                        "cameras": ["front"],
                        "frames_per_view": 1,
                        "fps": 10,
                        "resolution": resolution,
                        "aspect_ratio": ratio,
                        "width": width,
                        "height": height,
                    }
                },
            }

    monkeypatch.setattr(cosmos3_multiview, "export_to_video", lambda *args, **kwargs: None)
    request = {
        "name": "sample",
        "model_mode": "image2video",
        "resolution": resolution,
        "aspect_ratio": "auto" if automatic else ratio,
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
    assert (sampling_params.width, sampling_params.height) == ((None, None) if automatic else (width, height))
    assert sampling_params.extra_args["resolution"] == resolution
    assert sampling_params.extra_args["multiview"]["resolution"] == resolution
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
    assert manifest["resolution"] == resolution
    assert (manifest["aspect_ratio"], manifest["width"], manifest["height"]) == (ratio, width, height)
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
            return {"payload": {"video": np.broadcast_to(np.zeros(3, dtype=np.float32), (1, 10, 736, 544, 3))}}

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
                "payload": {"video": np.broadcast_to(np.zeros(3, dtype=np.float32), (1, 1, 480, 832, 3))},
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


def test_joint_manifest_paths_resolve_like_http_client(cosmos3_multiview: ModuleType, tmp_path: Path) -> None:
    request = {
        "extra_params": {
            "wsm": {},
            "lidar": {"control_path": "map.safetensors"},
            "multiview": {"views": [{"camera_key": "front", "prompt": "Raw.", "control_path": "camera.mp4"}]},
        }
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    loaded = cosmos3_multiview._load_requests(path)[0]["extra_params"]
    assert loaded["lidar"]["control_path"] == str(tmp_path / "map.safetensors")
    assert loaded["multiview"]["views"][0]["control_path"] == str(tmp_path / "camera.mp4")
    assert loaded["multiview"]["views"][0]["prompt"] == "Raw."


@pytest.mark.parametrize("aspect_ratio", ["auto", "16:9"])
def test_omitted_geometry_uses_checkpoint_defaults(cosmos3_multiview, monkeypatch, tmp_path, aspect_ratio):
    class FakeOmni:
        def generate(self, prompt, sp):
            assert "resolution" not in sp.extra_args
            assert "resolution" not in sp.extra_args["multiview"]
            assert sp.width is None and sp.height is None and sp.fps is None
            return {
                "payload": {"video": np.broadcast_to(np.zeros(3, dtype=np.float32), (1, 1, 720, 1280, 3))},
                "metadata": {
                    "multiview": {
                        "cameras": ["front"],
                        "frames_per_view": 1,
                        "fps": 10,
                        "resolution": "720",
                        "aspect_ratio": "16,9",
                    }
                },
            }

    monkeypatch.setattr(cosmos3_multiview, "export_to_video", lambda *args, **kwargs: None)
    request = {"multiview": {"aspect_ratio": aspect_ratio, "views": [{"camera_key": "front"}]}}
    manifest = cosmos3_multiview._run_request(
        FakeOmni(), request, output_dir=tmp_path, seed=42, fallback_negative_prompt=None
    )
    assert manifest["resolution"] == "720" and manifest["fps"] == 10
    assert (manifest["width"], manifest["height"]) == (1280, 720)


@pytest.mark.parametrize("wrapped", [False, True])
def test_offline_saves_numeric_lidar_without_video_conversion(cosmos3_multiview, monkeypatch, tmp_path, wrapped):
    from types import SimpleNamespace

    import torch
    from safetensors.torch import load_file

    lidar = torch.ones(1, 3, 2, 128, 1800)
    lidar[:, 0] = 60
    video = np.zeros((1, 1, 480, 832, 3), dtype=np.float32)
    metadata = {
        "multiview": {
            "cameras": ["front"],
            "frames_per_view": 1,
            "fps": 30,
            "resolution": "480",
            "aspect_ratio": "16,9",
        },
        "lidar": {"fps": 10},
    }

    class FakeOmni:
        def generate(self, prompt, sp):
            assert sp.extra_args["lidar"]["return_output"] is True
            if wrapped:
                return [SimpleNamespace(images=[video], multimodal_output={"lidar": lidar, "metadata": metadata})]
            return {"payload": {"video": video, "lidar": lidar}, "metadata": metadata}

    monkeypatch.setattr(cosmos3_multiview, "export_to_video", lambda *args, **kwargs: None)
    request = {
        "lidar": {"control_path": "map.safetensors", "return_output": True},
        "multiview": {"views": [{"camera_key": "front"}]},
    }
    manifest = cosmos3_multiview._run_request(
        FakeOmni(), request, output_dir=tmp_path, seed=42, fallback_negative_prompt=None
    )
    assert manifest["lidar"]["shape"] == [3, 2, 128, 1800]
    assert manifest["lidar"]["fps"] == 10 and manifest["fps"] == 30
    torch.testing.assert_close(load_file(tmp_path / "lidar.safetensors")["frames"], lidar[0], rtol=0, atol=0)
