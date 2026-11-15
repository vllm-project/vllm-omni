# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Guard the PyAV color-conversion dependency used when saving video."""

import av
import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("width", [32, 34], ids=["aligned", "padded-rows"])
def test_bt709_metadata_controls_yuv_to_rgb(width: int) -> None:
    frame = av.VideoFrame(width, 24, "yuv420p")
    frame.colorspace = 1  # AVCOL_SPC_BT709
    frame.color_range = 1  # AVCOL_RANGE_MPEG (limited)
    for plane, value in zip(frame.planes, [105, 111, 214]):
        plane.update(bytes([value]) * plane.buffer_size)

    # The save paths use the default conversion, which must honor frame tags.
    # PyAV 16.1 instead uses BT.601 coefficients here, yielding [240, 39, 68].
    rgb = frame.reformat(format="rgb24")
    plane = rgb.planes[0]
    center = 12 * plane.line_size + (width // 2) * 3
    pixel = list(bytes(plane)[center : center + 3])
    assert pixel == pytest.approx([255, 60, 67], abs=2)
