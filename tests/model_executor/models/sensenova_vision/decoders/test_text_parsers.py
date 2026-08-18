# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SenseNova-Vision structured-text parsers."""

from __future__ import annotations

import pytest
from vllm_omni.model_executor.models.sensenova.decoders.camera_pose import parse_camera_pose
from vllm_omni.model_executor.models.sensenova.decoders.text_parsers import (
    parse_bbox,
    parse_keypoints,
    parse_points,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_parse_bbox_single():
    text = "<p>dog</p><bbox>[0.1,0.2,0.8,0.9]</bbox>"
    parsed = parse_bbox(text)
    assert parsed == {"dog": [[0.1, 0.2, 0.8, 0.9]]}


def test_parse_bbox_multiple_and_clipping():
    text = "<p>cat</p><bbox>[-0.1,0.2,1.5,0.9]</bbox><bbox>[0.0,0.0,0.3,0.3]</bbox>"
    parsed = parse_bbox(text)
    assert parsed["cat"][0] == [0.0, 0.2, 0.999, 0.9]
    assert parsed["cat"][1] == [0.0, 0.0, 0.3, 0.3]


def test_parse_bbox_normalizes_category():
    text = "<p>person-1</p><bbox>[0.0,0.0,1.0,1.0]</bbox>"
    parsed = parse_bbox(text)
    assert "person 1" in parsed


def test_parse_bbox_empty():
    assert parse_bbox("") == {}
    assert parse_bbox("no tags here") == {}


def test_parse_points():
    text = "<p>traffic light</p><point>[0.4,0.6]</point>"
    parsed = parse_points(text)
    traffic_light = " ".join(parsed.keys()).lower()
    assert traffic_light == "traffic light"


def test_parse_points_multiple():
    text = "<p>pole</p><point>[0.1,0.2]</point><point>[0.3,0.4]</point>"
    parsed = parse_points(text)
    pole_key = " ".join(parsed.keys()).lower()
    assert parsed[pole_key] == [[0.1, 0.2], [0.3, 0.4]]


def test_parse_points_clipping():
    text = "<p>x</p><point>[1.2,-0.3]</point>"
    parsed = parse_points(text)
    key = " ".join(parsed.keys()).lower()
    assert parsed[key] == [[0.999, 0.0]]


def test_parse_keypoints_basic():
    text = (
        "<p>person</p><bbox>[0.0,0.0,0.5,0.5]</bbox>left shoulder<kpt>[0.1,0.2]</kpt>right shoulder<kpt>[0.3,0.4]</kpt>"
    )
    parsed = parse_keypoints(text)
    person_key = " ".join(parsed.keys()).lower()
    assert person_key == "person"
    instance = parsed[person_key][0]
    assert instance["bbox"] == [0.0, 0.0, 0.5, 0.5]
    assert instance["keypoints"]["left shoulder"] == [0.1, 0.2]
    assert instance["keypoints"]["right shoulder"] == [0.3, 0.4]


def test_parse_keypoints_invisible_and_ins():
    text = "<p>person</p><ins>1</ins>left eye<kpt>unvisible</kpt>right eye<kpt>[0.5,0.5]</kpt></ins>"
    parsed = parse_keypoints(text)
    person_key = " ".join(parsed.keys()).lower()
    kps = parsed[person_key][0]["keypoints"]
    # The official parser strips <ins> tags but keeps their text, so the
    # keypoint name includes the instance prefix "1".
    assert kps["1left eye"] == [-1.0, -1.0]
    assert kps["right eye"] == [0.5, 0.5]


def test_parse_camera_pose_frames():
    pose = (
        "<frame><quat>[0,0,0,1]</quat><offset>[100,0,0]</offset>"
        "<scale>200</scale></frame>"
        "<frame><quat>[1,0,0,0]</quat><offset>[0,-100,50]</offset>"
        "<scale>100</scale></frame>"
    )
    parsed = parse_camera_pose(pose)
    assert parsed is not None
    assert len(parsed["rotation"]) == 2
    assert len(parsed["translation"]) == 2
    # Values inside tags are milli-units: int/1000 for quat and offset, then
    # translation = offset * scale / 100.0 (official parser semantics).
    # offset [100, 0, 0] * scale 200 / 100.0 -> [0.2, 0.0, 0.0]
    assert parsed["translation"][0] == pytest.approx([0.2, 0.0, 0.0])
    # offset [0, -100, 50] * scale 100 / 100.0 -> [0.0, -0.1, 0.05]
    assert parsed["translation"][1] == pytest.approx([0.0, -0.1, 0.05])
    # quats are milli-units / 1000
    assert parsed["rotation"][1] == pytest.approx([0.001, 0.0, 0.0, 0.0])


def test_parse_camera_pose_no_frame_tags():
    pose = "<quat>[0,0,0,1]</quat><offset>[100,0,0]</offset><scale>100</scale>"
    parsed = parse_camera_pose(pose)
    assert parsed is not None
    # (100/1000) * 100 / 100.0 == 0.1
    assert parsed["translation"][0] == pytest.approx([0.1, 0.0, 0.0])


def test_parse_camera_pose_invalid():
    assert parse_camera_pose("no tags") is None
    assert parse_camera_pose("<offset>[0,0,0]</offset>") is None


def test_parse_camera_pose_non_string():
    with pytest.raises(TypeError):
        parse_camera_pose(123)
