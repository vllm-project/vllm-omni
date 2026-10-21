# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
from comfyui_vllm_omni.lingbot_geometry import LingBotGeometry
from comfyui_vllm_omni.lingbot_moves import (
    CameraMove,
    LingBotSequence,
    Shot,
    compile_sequence,
    moves_from_camera_script,
    parse_moves,
    sequence_payload,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
GEOMETRY = LingBotGeometry()


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_geometry_derives_the_nine_plus_twelve_grid():
    assert GEOMETRY.first_chunk_frames == 9
    assert GEOMETRY.chunk_frames == 12
    assert [GEOMETRY.frames_for(chunks) for chunks in (1, 2, 3)] == [9, 21, 33]
    assert [GEOMETRY.chunks_for(frames) for frames in (9, 21, 33, 189)] == [1, 2, 3, 16]


@pytest.mark.parametrize("frames", [0, 8, 10, 20, 32, 190])
def test_geometry_rejects_frames_off_the_grid(frames):
    with pytest.raises(ValueError, match="9 \\+ 12k|at least 9"):
        GEOMETRY.chunks_for(frames)


def test_geometry_samples_each_latent_frame_inside_its_chunk():
    assert [GEOMETRY.sample_frame(0, latent) for latent in range(3)] == [0, 4, 8]
    assert [GEOMETRY.sample_frame(1, latent) for latent in range(3)] == [9, 13, 17]
    assert GEOMETRY.chunk_start_frame(15) == 177
    with pytest.raises(ValueError, match="latent must be"):
        GEOMETRY.sample_frame(0, 3)


def test_only_the_first_shot_may_open_a_scene():
    assert GEOMETRY.shot_frames(0, 4) == 45
    assert GEOMETRY.shot_frames(1, 7) == 84
    assert GEOMETRY.shot_chunks(1, 84) == 7
    with pytest.raises(ValueError, match="whole chunks of 12 frames"):
        GEOMETRY.shot_chunks(1, 45)


# ---------------------------------------------------------------------------
# Moves text
# ---------------------------------------------------------------------------


def test_moves_parse_lane_value_and_frame_forms():
    moves = parse_moves("move forward 0 end\nlook_h right 21 45", shot_frames=45)
    assert moves == (
        CameraMove("move", "forward", 0, 45),
        CameraMove("look_h", "right", 21, 24),
    )
    assert parse_moves("forward 0f 45f") == (CameraMove("move", "forward", 0, 45),)
    assert parse_moves("up 0 12\ndown 12 24") == (
        CameraMove("look_v", "up", 0, 12),
        CameraMove("look_v", "down", 12, 12),
    )
    assert parse_moves("# only a comment\n\n") == ()


def test_moves_reject_a_bare_side_that_could_be_a_strafe_or_a_look():
    with pytest.raises(ValueError, match="can be a strafe or a look move"):
        parse_moves("left 0 24", shot_frames=24)


def test_moves_reject_an_unknown_lane_and_an_overlap():
    with pytest.raises(ValueError, match="unknown lane"):
        parse_moves("zoom forward 0 24", shot_frames=24)
    with pytest.raises(ValueError, match="Two move moves overlap"):
        parse_moves("move forward 0 24\nmove back 12 36", shot_frames=36)


def test_moves_reject_a_start_after_the_shot():
    with pytest.raises(ValueError, match="after the shot's"):
        parse_moves("move forward 48 60", shot_frames=48)


def test_legacy_per_chunk_json_becomes_moves():
    moves = moves_from_camera_script('[[["w"],["w"],["w"]],[["w","l"],["w","l"],["w","l"]],[[],["a"],[]]]')
    assert moves == (
        CameraMove("move", "forward", 0, 6),
        CameraMove("look_h", "right", 3, 3),
        CameraMove("strafe", "left", 7, 1),
    )


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def _three_shot_sequence() -> LingBotSequence:
    return LingBotSequence(
        (
            Shot("A", 45, parse_moves("move forward 0 end\nlook_h right 21 end", shot_frames=45)),
            Shot("B", 84, parse_moves("move forward 0 end\nstrafe left 0 36", shot_frames=84)),
            Shot("C", 60, parse_moves("look_v up 24 end", shot_frames=60)),
        )
    )


def test_compile_places_each_move_on_its_chunk_grid():
    compiled = compile_sequence(_three_shot_sequence(), GEOMETRY)
    assert compiled.total_chunks == 16
    assert compiled.total_frames == 189
    assert compiled.shot_start_chunks == (0, 4, 11)
    assert compiled.prompt_switches == ((4, "B"), (11, "C"))
    assert compiled.warnings == ()
    script = compiled.camera_script()
    assert len(script) == 16
    assert all(len(chunk) == 3 for chunk in script)
    assert script[0] == [["w"], ["w"], ["w"]]
    # The yaw starts at frame 21, which is the first sample of chunk 2.
    assert script[1] == [["w"], ["w"], ["w"]]
    assert script[2] == [["w", "l"]] * 3
    assert script[4] == [["w", "a"]] * 3
    assert script[7] == [["w"], ["w"], ["w"]]
    assert script[11] == [[], [], []]
    assert script[13] == [["i"]] * 3
    assert script[15] == [["i"]] * 3


def test_compile_reports_a_move_cut_at_its_shot_end():
    sequence = LingBotSequence((Shot("A", 33, parse_moves("move forward 0 99", shot_frames=33)),))
    compiled = compile_sequence(sequence, GEOMETRY)
    assert compiled.warnings == ("Shot 0: the move move is cut to 33 frames at the shot's end.",)
    assert compiled.camera_script()[-1] == [["w"]] * 3


def test_compile_widens_a_move_too_short_to_cover_a_latent_frame():
    sequence = LingBotSequence((Shot("A", 21, parse_moves("move forward 0 2", shot_frames=21)),))
    compiled = compile_sequence(sequence, GEOMETRY)
    assert "widened to one latent frame" in compiled.warnings[0]
    # The widened move covers the chunk's first latent sample only.
    assert compiled.camera_script()[0] == [["w"], [], []]


def test_compile_rejects_a_sequence_off_the_grid():
    with pytest.raises(ValueError, match="whole chunks of 12 frames"):
        compile_sequence(LingBotSequence((Shot("A", 21), Shot("B", 20))), GEOMETRY)


def test_compiled_script_is_what_the_server_parses():
    actions = pytest.importorskip("vllm_omni.diffusion.models.lingbot_world.actions")
    compiled = compile_sequence(_three_shot_sequence(), GEOMETRY)
    parsed = actions.parse_lingbot_camera_action_script(
        compiled.camera_script(), frames_per_chunk=GEOMETRY.latent_frames_per_chunk
    )
    assert len(parsed) == compiled.total_chunks


def test_visualizer_payload_describes_every_shot():
    sequence = _three_shot_sequence()
    payload = sequence_payload(sequence, compile_sequence(sequence, GEOMETRY), GEOMETRY, fps=16)
    assert payload["total_frames"] == 189
    assert [shot["chunks"] for shot in payload["shots"]] == [4, 7, 5]
    assert payload["shots"][1]["start_frame"] == 45
    assert payload["prompt_switches"] == [[4, "B"], [11, "C"]]
    assert payload["fps"] == 16
