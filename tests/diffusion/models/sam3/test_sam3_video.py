# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the SAM3 video port: tracker modules, association logic, mask utilities, pipeline wiring.

The Transformers SAM3 tracker is the parity oracle for the tracker modules: a
tiny random config is built for both, the reference weights are loaded into the
port and outputs must match in float32.

    pytest tests/diffusion/models/sam3/test_sam3_video.py -v
"""

from __future__ import annotations

from collections import defaultdict

import av
import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.sam3 import pipeline_sam3
from vllm_omni.diffusion.models.sam3.pipeline_sam3 import _video_frames, parse_video_prompts, rle_counts
from vllm_omni.diffusion.models.sam3.sam3_tracker import Sam3Tracker, Sam3TrackerConfig
from vllm_omni.diffusion.models.sam3.sam3_video import (
    Sam3VideoConfig,
    Sam3VideoFrameOutput,
    Sam3VideoSegmenter,
    Sam3VideoState,
    connected_component_areas,
    fill_holes_in_mask_scores,
    generic_nms,
    mask_iou,
    nms_masks,
)
from vllm_omni.entrypoints.openai.images.helpers import _coco_rle

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

transformers = pytest.importorskip("transformers")

# Transformers fixes the memory cross-attention key/value width at 64 channels, as released.
HIDDEN = 128
TINY_TRACKER = {
    "image_size": 112,
    "vision_config": {"fpn_hidden_size": HIDDEN, "backbone_feature_sizes": [[32, 32], [16, 16], [8, 8]]},
    "prompt_encoder_config": {"hidden_size": HIDDEN, "image_size": 112, "patch_size": 14, "mask_input_channels": 8},
    "mask_decoder_config": {
        "hidden_size": HIDDEN,
        "mlp_dim": 64,
        "num_attention_heads": 2,
        "iou_head_hidden_dim": 32,
    },
    "memory_attention_hidden_size": HIDDEN,
    "memory_attention_num_layers": 2,
    "memory_attention_feed_forward_hidden_size": 64,
    "memory_attention_rope_feat_sizes": [8, 8],
    "memory_encoder_hidden_size": HIDDEN,
    "memory_encoder_output_channels": 64,
    "mask_downsampler_embed_dim": HIDDEN,
    "memory_fuser_embed_dim": HIDDEN,
    "memory_fuser_intermediate_dim": 64,
}


def _tracker_pair():
    from transformers import Sam3TrackerVideoConfig, Sam3TrackerVideoModel

    torch.manual_seed(0)
    reference = Sam3TrackerVideoModel(Sam3TrackerVideoConfig(**TINY_TRACKER), remove_vision_encoder=True).eval()
    # Zero-initialized tensors would make the comparison vacuous.
    with torch.no_grad():
        for param in reference.parameters():
            param.normal_(0, 0.2)
    port = Sam3Tracker(Sam3TrackerConfig.from_dict(reference.config.to_dict())).eval()
    port.load_state_dict(reference.state_dict(), strict=True)
    return reference, port


def _features(batch: int = 1):
    s0 = torch.randn(batch, HIDDEN // 8, 32, 32)
    s1 = torch.randn(batch, HIDDEN // 4, 16, 16)
    top = torch.randn(batch, HIDDEN, 8, 8)
    return s0, s1, top


@torch.no_grad()
def test_tracker_heads_match_transformers():
    reference, port = _tracker_pair()
    s0, s1, top = _features()
    expected = reference._single_frame_forward(image_embeddings=[s0, s1, top], multimask_output=True)
    actual = port._sam_heads([s0, s1, top], None)
    for key in ("pred_masks", "high_res_masks", "object_pointer", "object_score_logits"):
        torch.testing.assert_close(actual[key], getattr(expected, key), rtol=0, atol=1e-5)


@torch.no_grad()
def test_mask_as_output_matches_transformers():
    reference, port = _tracker_pair()
    s0, s1, top = _features()
    mask = (torch.rand(1, 1, 32, 32) > 0.5).float()
    expected = reference._use_mask_as_output(top, [s0, s1], mask)
    actual = port._use_mask_as_output(top, [s0, s1], mask)
    for key in ("pred_masks", "high_res_masks", "object_pointer", "object_score_logits"):
        torch.testing.assert_close(actual[key], getattr(expected, key), rtol=0, atol=1e-5)


@torch.no_grad()
def test_memory_attention_and_encoder_match_transformers():
    reference, port = _tracker_pair()
    vision = torch.randn(64, 1, HIDDEN)
    vision_pos = torch.randn(64, 1, HIDDEN)
    # Two spatial memories plus four object-pointer tokens that skip RoPE.
    memory = torch.randn(2 * 64 + 4, 1, 64)
    memory_pos = torch.randn(2 * 64 + 4, 1, 64)
    expected = reference.memory_attention(
        current_vision_features=vision,
        memory=memory,
        current_vision_position_embeddings=vision_pos,
        memory_posision_embeddings=memory_pos,
        num_object_pointer_tokens=4,
    )
    actual = port.memory_attention(vision, memory, vision_pos, memory_pos, num_object_pointer_tokens=4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)

    masks = torch.randn(3, 1, 32, 32)
    scores = torch.tensor([1.0, -1.0, 2.0]).view(3, 1, 1)  # object-score logits as the tracker emits them
    expected = reference._encode_new_memory(vision.expand(-1, 3, -1), masks, scores, is_mask_from_pts=False)
    actual = port.encode_memory(vision.expand(-1, 3, -1), masks, scores, is_mask_from_pts=False)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=1e-5)


def test_tracker_config_reads_the_released_layout():
    config = Sam3TrackerConfig.from_dict(
        {
            "num_maskmem": 7,
            "vision_config": {"fpn_hidden_size": 256, "backbone_feature_sizes": [[288, 288], [144, 144], [72, 72]]},
            "mask_decoder_config": {"num_attention_heads": 8, "mlp_dim": 2048},
            "prompt_encoder_config": {"mask_input_channels": 16, "patch_size": 14},
        }
    )
    assert config.hidden_dim == 256 and config.num_maskmem == 7
    assert config.mask_decoder_num_attention_heads == 8 and config.image_embedding_size == (72, 72)
    assert config.mask_input_size == (288, 288)


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------


def test_connected_component_areas_use_eight_connectivity():
    skimage_measure = pytest.importorskip("skimage.measure")
    rng = np.random.default_rng(0)
    planes = rng.random((3, 24, 20)) > 0.6
    areas = connected_component_areas(torch.from_numpy(planes).unsqueeze(1)).squeeze(1).numpy()
    for plane, got in zip(planes, areas):
        labels = skimage_measure.label(plane, connectivity=2)
        expected = np.zeros_like(got)
        for label in range(1, labels.max() + 1):
            expected[labels == label] = (labels == label).sum()
        np.testing.assert_array_equal(got, expected)


def test_fill_holes_fills_small_holes_and_drops_sprinkles():
    mask = torch.full((1, 1, 12, 12), -5.0)
    mask[..., 2:9, 2:9] = 5.0
    mask[..., 5, 5] = -5.0  # a one-pixel hole
    mask[..., 11, 11] = 5.0  # a one-pixel sprinkle
    filled = fill_holes_in_mask_scores(mask, max_area=4)
    assert filled[0, 0, 5, 5] == 0.1 and filled[0, 0, 11, 11] == -0.1
    assert torch.equal(filled[..., 2:9, 2:9] > 0, torch.ones(1, 1, 7, 7, dtype=torch.bool))
    assert torch.equal(fill_holes_in_mask_scores(mask, max_area=0), mask)


def test_nms_keeps_the_best_of_overlapping_masks():
    masks = torch.full((4, 8, 8), -1.0)
    masks[0, :4] = 1.0
    masks[1, :5] = 1.0  # overlaps 0 (IoU 0.8)
    masks[2, 6:] = 1.0
    masks[3, :4] = 1.0  # below the score threshold
    probs = torch.tensor([0.6, 0.9, 0.7, 0.2])
    keep = nms_masks(probs, masks, prob_threshold=0.5, iou_threshold=0.5)
    assert keep.tolist() == [False, True, True, False]
    ious = mask_iou(masks > 0, masks > 0)
    assert generic_nms(ious[:3, :3], probs[:3], 0.5).tolist() == [1, 2]


def test_rle_counts_match_the_image_endpoint_encoding():
    rng = np.random.default_rng(1)
    masks = rng.random((3, 7, 9)) > 0.5
    masks[1] = False
    masks[2, 0, 0] = True
    for mask, counts in zip(masks, rle_counts(torch.from_numpy(masks))):
        assert counts == _coco_rle(mask).counts


def test_video_prompts_are_deduplicated_and_bounded():
    assert parse_video_prompts(" person ", {"prompts": ["dog", "person", " "]}) == ["person", "dog"]
    with pytest.raises(ValueError, match="at least one"):
        parse_video_prompts("", {})
    with pytest.raises(ValueError, match="list of strings"):
        parse_video_prompts("x", {"prompts": "dog"})
    with pytest.raises(ValueError, match="at most"):
        parse_video_prompts("x", {"prompts": [f"p{i}" for i in range(20)]})


def _write_video(path, container_format: str, num_frames: int) -> str:
    with av.open(str(path), mode="w", format=container_format) as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = 16, 16, "yuv420p"
        for index in range(num_frames):
            frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), index * 40, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return str(path)


def test_video_frames_decode_a_file_within_the_frame_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_sam3, "MAX_VIDEO_FRAMES", 5)
    frames, fps = _video_frames(_write_video(tmp_path / "clip.mp4", "mp4", 5))
    decoded = list(frames)
    assert fps == 24.0 and len(decoded) == 5
    assert decoded[0].shape == (3, 16, 16) and decoded[0].dtype == torch.uint8


@pytest.mark.parametrize(
    ("name", "container_format"),
    [
        ("clip.mp4", "mp4"),  # frame count in the header: rejected before any frame is decoded
        ("clip.mkv", "matroska"),  # no frame count: rejected once decoding passes the cap
    ],
)
def test_video_frames_reject_videos_over_the_frame_cap(tmp_path, monkeypatch, name, container_format):
    path = _write_video(tmp_path / name, container_format, 5)
    monkeypatch.setattr(pipeline_sam3, "MAX_VIDEO_FRAMES", 4)
    with pytest.raises(ValueError, match="at most 4 frames"):
        frames, _ = _video_frames(path)
        list(frames)
    with pytest.raises(ValueError, match="at most 4 frames"):
        _video_frames(np.zeros((5, 16, 16, 3), dtype=np.uint8))


def test_video_frames_reject_files_without_a_video_stream(tmp_path):
    path = str(tmp_path / "tone.wav")
    with av.open(path, mode="w", format="wav") as container:
        stream = container.add_stream("pcm_s16le", rate=8000)
        frame = av.AudioFrame.from_ndarray(np.zeros((1, 800), dtype=np.int16), format="s16", layout="mono")
        frame.sample_rate = 8000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with pytest.raises(ValueError, match="no video stream"):
        _video_frames(path)


# ---------------------------------------------------------------------------
# Association and hot-start
# ---------------------------------------------------------------------------


def _segmenter(**overrides) -> Sam3VideoSegmenter:
    segmenter = Sam3VideoSegmenter.__new__(Sam3VideoSegmenter)
    segmenter.config = Sam3VideoConfig(**overrides)
    return segmenter


def _state(**fields) -> Sam3VideoState:
    return Sam3VideoState(num_frames=100, video_height=8, video_width=8, dtype=torch.float32, prompts=["x"], **fields)


def _box_mask(rows: slice, cols: slice) -> torch.Tensor:
    mask = torch.full((8, 8), -1.0)
    mask[rows, cols] = 1.0
    return mask


def test_association_finds_new_unmatched_and_high_confidence_matches():
    segmenter = _segmenter()
    det = torch.stack([_box_mask(slice(0, 4), slice(0, 4)), _box_mask(slice(5, 8), slice(5, 8))])
    trk = torch.stack(
        [_box_mask(slice(0, 4), slice(0, 4)), _box_mask(slice(0, 2), slice(6, 8)), torch.full((8, 8), -1.0)]
    )
    new, unmatched, det_to_trk, high_conf, empty = segmenter._associate(
        det,
        torch.tensor([0.9, 0.8]),
        trk,
        [10, 11, 12],
        torch.zeros(2, dtype=torch.long),
        torch.zeros(3, dtype=torch.long),
    )
    assert new == [1]  # the second detection overlaps no masklet and clears new_det_thresh
    assert unmatched == [11] and empty == [12]
    assert det_to_trk == {0: [10], 1: []} and high_conf == {10: 0}


def test_association_never_matches_across_prompts():
    segmenter = _segmenter()
    mask = _box_mask(slice(0, 4), slice(0, 4))
    new, unmatched, det_to_trk, _, _ = segmenter._associate(
        mask[None], torch.tensor([0.9]), mask[None], [3], torch.tensor([1]), torch.tensor([0])
    )
    assert new == [0] and unmatched == [3] and det_to_trk == {0: []}


def test_hotstart_removes_masklets_unmatched_within_the_window():
    segmenter = _segmenter(hotstart_delay=4, hotstart_unmatch_thresh=2)
    state = _state()
    segmenter._hotstart(state, 0, {}, [0, 1], [], [])
    assert segmenter._hotstart(state, 1, {0: [0]}, [], [], [1]) == set()
    assert segmenter._hotstart(state, 2, {0: [0]}, [], [], [1]) == {1}
    assert state.removed_obj_ids == {1} and state.trk_keep_alive[0] == 32 - 2  # capped at max_trk_keep_alive
    # Past the hot-start window an object is no longer removed for being unmatched.
    late = _state(obj_first_frame_idx={5: 0}, trk_keep_alive={5: 30}, unmatched_frame_inds=defaultdict(list))
    assert segmenter._hotstart(late, 10, {}, [], [], [5]) == set()
    assert segmenter._hotstart(late, 11, {}, [], [], [5]) == set()


def test_hotstart_removes_the_later_of_duplicated_masklets():
    segmenter = _segmenter(hotstart_delay=4, hotstart_dup_thresh=2)
    state = _state()
    segmenter._hotstart(state, 0, {}, [0], [], [])
    segmenter._hotstart(state, 1, {}, [1], [], [])
    assert segmenter._hotstart(state, 2, {0: [0, 1]}, [], [], []) == set()
    assert segmenter._hotstart(state, 3, {0: [1, 0]}, [], [], []) == {1}


def test_postprocess_hides_removed_objects_and_resolves_overlaps_per_prompt():
    segmenter = _segmenter()
    state = _state(obj_id_to_prompt_id={0: 0, 1: 0, 2: 1}, hotstart_removed_obj_ids={3})
    shared = _box_mask(slice(0, 4), slice(0, 8))
    out = Sam3VideoFrameOutput(
        frame_idx=0,
        obj_id_to_mask={0: shared[None], 1: shared[None], 2: shared[None], 3: shared[None]},
        obj_id_to_score={0: 0.9, 1: 0.8, 2: 0.7, 3: 0.6},
        obj_id_to_tracker_score={0: 0.4, 1: 0.6, 2: 0.5},
        removed_obj_ids=set(),
        suppressed_obj_ids=set(),
    )
    post = segmenter.postprocess(state, out)
    assert post["object_ids"].tolist() == [0, 1, 2]
    # Objects 0 and 1 share a prompt, so the higher tracker score (1) takes the overlap;
    # object 2 is another prompt's concept and keeps its own mask.
    assert not post["masks"][0].any() and post["masks"][1].sum() == 32 and post["masks"][2].sum() == 32
    assert post["boxes"][1].tolist() == [0.0, 0.0, 7.0, 3.0]
