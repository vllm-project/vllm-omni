# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""A conditioning stream arrives wrapped; a decoded video is a frame sequence.

``multi_modal_data[modality]`` holds the request's media, so a single reference
arrives as a one-item list, while ``normalize_decoded_video_frames`` treats a
list or tuple as a video's frames.  The two are told apart by what the single
item is.
"""

from PIL import Image

from vllm_omni.diffusion.models.lance.pipeline_lance import (
    unwrap_single_media_stream,
    unwrap_single_video_stream,
)


def test_single_item_wrapper_is_unwrapped():
    frame = Image.new("RGB", (2, 2))
    assert unwrap_single_media_stream([frame]) is frame
    assert unwrap_single_media_stream(["/work/ref.mp4"]) == "/work/ref.mp4"
    assert unwrap_single_video_stream(["/work/ref.mp4"]) == "/work/ref.mp4"
    assert unwrap_single_video_stream([["/work/f0.png", "/work/f1.png"]]) == ["/work/f0.png", "/work/f1.png"]


def test_longer_containers_are_left_alone():
    frames = [Image.new("RGB", (2, 2))] * 3
    assert unwrap_single_media_stream(frames) is frames
    assert unwrap_single_video_stream(frames) is frames


def test_one_frame_sequence_stays_a_sequence():
    # normalize_decoded_video_frames requires PIL frames, so a single image in a
    # list is a one-frame video rather than the transport's wrapper.
    sequence = [Image.new("RGB", (2, 2))]
    assert unwrap_single_video_stream(sequence) is sequence
