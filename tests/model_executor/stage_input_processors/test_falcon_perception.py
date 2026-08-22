# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Falcon Perception thinker -> segmentation stage bridge.

The row indexing is the subtle part and the easiest thing to get silently
wrong: ``hidden[i]`` is the state that *produced* token ``i + 1``, so the row
behind ``output_token_ids[i]`` is ``hidden[len(prompt) - 1 + i]``. An off-by-one
here still runs and still produces masks — just of the wrong instances — so it
is asserted directly against hand-computed indices.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.falcon_perception import (
    build_segmentation_payload,
    thinker2segmentation_token_only,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

IMG_ID = 227
SEG_ID = 262
COORD_ID = 240
SIZE_ID = 241
HIDDEN = 8


def _source_output(prompt_ids, output_ids, latent, xy=None, hw=None, finished=True):
    """Mimic a stage-0 RequestOutput carrying its latent trajectory."""
    mm = {"latent": latent}
    if xy is not None:
        mm["geometry"] = {"xy": xy, "hw": hw}
    completion = SimpleNamespace(cumulative_token_ids=output_ids, multimodal_output=mm)
    return SimpleNamespace(finished=finished, prompt_token_ids=prompt_ids, outputs=[completion])


def test_selects_image_rows_and_seg_rows_at_the_right_offsets():
    # prompt: [text, cls, img, img, img, eoi]  -> image rows at 2, 3, 4
    prompt = [1, 244, IMG_ID, IMG_ID, IMG_ID, 230]
    # output: [<object>, <coord>, <size>, <seg>, <seg>]
    output = [239, COORD_ID, SIZE_ID, SEG_ID, SEG_ID]
    n_prompt = len(prompt)

    # Row value == row index, so the assertions read directly.
    hidden = torch.arange(n_prompt + len(output), dtype=torch.float32).unsqueeze(1).repeat(1, HIDDEN)

    payload = build_segmentation_payload(hidden, prompt, output)

    image_features = payload["hidden_states"]["image_features"]
    assert image_features.shape == (3, HIDDEN)
    assert image_features[:, 0].tolist() == [2.0, 3.0, 4.0]

    # <seg> is at output indices 3 and 4 -> hidden rows 6-1+3=8 and 6-1+4=9.
    seg_features = payload["hidden_states"]["seg_features"]
    assert seg_features.shape == (2, HIDDEN)
    assert seg_features[:, 0].tolist() == [8.0, 9.0]

    assert bool(payload["meta"]["finished"])
    assert int(payload["meta"]["num_seg_tokens"]) == 2
    assert int(payload["meta"]["num_image_tokens"]) == 3


def test_no_seg_tokens_yields_empty_features_not_a_failure():
    """A detection-only answer is legitimate and must still hand over the image."""
    prompt = [1, IMG_ID, IMG_ID]
    output = [239, COORD_ID, SIZE_ID]
    hidden = torch.randn(len(prompt) + len(output), HIDDEN)

    payload = build_segmentation_payload(hidden, prompt, output)
    assert payload is not None
    assert payload["hidden_states"]["seg_features"].shape == (0, HIDDEN)
    assert payload["hidden_states"]["image_features"].shape == (2, HIDDEN)


def test_geometry_rows_follow_the_same_offset_rule():
    prompt = [1, IMG_ID, IMG_ID]
    output = [239, COORD_ID, SIZE_ID, SEG_ID]
    n_rows = len(prompt) + len(output)
    hidden = torch.randn(n_rows, HIDDEN)
    xy = torch.arange(n_rows, dtype=torch.float32).unsqueeze(1).repeat(1, 2)
    hw = xy.clone()

    payload = build_segmentation_payload(hidden, prompt, output, xy=xy, hw=hw)
    # <coord> at output idx 1 -> row 3-1+1 = 3 ; <size> at idx 2 -> row 4.
    assert payload["hidden_states"]["box_xy"][:, 0].tolist() == [3.0]
    assert payload["hidden_states"]["box_hw"][:, 0].tolist() == [4.0]


def test_accepted_coordinate_deltas_override_the_raw_argmax_rows():
    prompt = [1, IMG_ID, IMG_ID]
    output = [COORD_ID, SIZE_ID, SEG_ID, COORD_ID, SIZE_ID, SEG_ID]
    n_rows = len(prompt) + len(output)
    hidden = torch.randn(n_rows, HIDDEN)
    raw_xy = torch.full((n_rows, 2), 0.125)
    hw = torch.arange(n_rows, dtype=torch.float32).unsqueeze(1).repeat(1, 2)
    selected_xy = torch.tensor([[0.25, 0.75], [0.5, 1.0]])

    payload = build_segmentation_payload(
        hidden,
        prompt,
        output,
        xy=raw_xy,
        hw=hw,
        selected_xy=selected_xy,
    )

    assert torch.equal(payload["hidden_states"]["box_xy"], selected_xy)


@pytest.mark.parametrize(
    ("hidden", "prompt", "output"),
    [
        (torch.randn(2, HIDDEN), [], [SEG_ID]),  # no prompt ids
        (torch.randn(1, HIDDEN), [1, IMG_ID, IMG_ID], [SEG_ID]),  # trajectory too short
        (torch.randn(4, HIDDEN), [1, 2, 3], [SEG_ID]),  # no image tokens
        (torch.randn(4), [1, IMG_ID], [SEG_ID]),  # not 2-D
    ],
)
def test_unusable_inputs_return_none_rather_than_a_wrong_payload(hidden, prompt, output):
    assert build_segmentation_payload(hidden, prompt, output) is None


def test_token_only_builds_one_input_per_finished_request_with_the_payload():
    prompt = [1, IMG_ID, IMG_ID, 230]
    output = [COORD_ID, SIZE_ID, SEG_ID]
    hidden = torch.randn(len(prompt) + len(output), HIDDEN)
    # A real image: the hook processes it for real now that a processing failure
    # propagates instead of being swallowed into a maskless payload.
    image = _rgb(64, 48)

    finished = _source_output(prompt, output, hidden)
    unfinished = _source_output(prompt, output, hidden, finished=False)

    inputs = thinker2segmentation_token_only(
        [finished, unfinished, finished],
        prompt={"multi_modal_data": {"image": image}},
    )
    assert len(inputs) == 2
    for item in inputs:
        assert len(item["prompt_token_ids"]) == 1
        # AnyUp guides upsampling with the original pixels, so the image must
        # reach stage 1 alongside the sliced features.
        assert item["multi_modal_data"]["image"] is image
        info = item["additional_information"]
        assert info["hidden_states"]["seg_features"].shape == (1, HIDDEN)
        assert info["hidden_states"]["image_features"].shape == (2, HIDDEN)


def test_token_only_ships_processed_pixels_that_match_the_image_feature_grid():
    """The mask head gets its pixels through the payload, not model kwargs.

    The generation runner has no multimodal plumbing, so
    ``requires_multimodal_data`` never reaches stage 1's forward. This asserts
    the pixels travel in ``additional_information`` *and* that their patch grid
    matches the number of image feature rows — a mismatch there makes the mask
    head silently skip every request.
    """
    import numpy as np
    from PIL import Image

    from vllm_omni.model_executor.models.falcon_perception.configuration_falcon_perception import (
        FalconPerceptionConfig,
    )
    from vllm_omni.model_executor.models.falcon_perception.processing_falcon_perception import (
        FalconPerceptionImageProcessor,
    )

    config = FalconPerceptionConfig()
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (414, 738, 3), dtype=np.uint8))
    gh, gw = FalconPerceptionImageProcessor(config).target_grid(414, 738)

    prompt_ids = [1] + [IMG_ID] * (gh * gw) + [230]
    output_ids = [COORD_ID, SIZE_ID, SEG_ID]
    hidden = torch.randn(len(prompt_ids) + len(output_ids), HIDDEN)

    inputs = thinker2segmentation_token_only(
        [_source_output(prompt_ids, output_ids, hidden)],
        prompt={"multi_modal_data": {"image": image}},
    )
    info = inputs[0]["additional_information"]

    pixels = info["hidden_states"]["pixel_values"]
    assert pixels.ndim == 3 and pixels.shape[2] == 3
    assert pixels.dtype == torch.float32, "bfloat16 cannot cross the stage process boundary"
    # The grid the mask head derives from the pixels must match the rows it got.
    patch = config.spatial_patch_size
    assert (pixels.shape[0] // patch) * (pixels.shape[1] // patch) == info["hidden_states"]["image_features"].shape[0]


@pytest.mark.parametrize("nested", [True, False])
def test_geometry_is_read_from_either_nested_or_flat_dotted_payloads(nested):
    """Exercises the real runtime shape of stage-0's multimodal output.

    Guards a specific trap: resolving these with ``a or b`` raises
    "Boolean value of Tensor with more than one value is ambiguous",
    which only shows up once real (multi-element) tensors flow through.
    """
    prompt = [1, IMG_ID, IMG_ID]
    output = [COORD_ID, SIZE_ID, SEG_ID]
    n = len(prompt) + len(output)
    hidden = torch.randn(n, HIDDEN)
    xy = torch.arange(n, dtype=torch.float32).unsqueeze(1).repeat(1, 2)

    mm = (
        {"latent": hidden, "hidden_states": {"box_xy": xy, "box_hw": xy}}
        if nested
        else {"latent": hidden, "hidden_states.box_xy": xy, "hidden_states.box_hw": xy}
    )
    completion = SimpleNamespace(cumulative_token_ids=output, multimodal_output=mm)
    source = SimpleNamespace(finished=True, prompt_token_ids=prompt, outputs=[completion])

    info = thinker2segmentation_token_only([source], prompt=None)[0]["additional_information"]
    # <coord> at output idx 0 -> hidden row 3-1+0 = 2.
    assert info["hidden_states"]["box_xy"][:, 0].tolist() == [2.0]


def test_payload_tensors_are_float32_for_cross_process_transfer():
    prompt = [1, IMG_ID, IMG_ID]
    output = [COORD_ID, SIZE_ID, SEG_ID]
    n = len(prompt) + len(output)
    payload = build_segmentation_payload(
        torch.randn(n, HIDDEN, dtype=torch.bfloat16),
        prompt,
        output,
        xy=torch.randn(n, 2, dtype=torch.bfloat16),
        hw=torch.randn(n, 2, dtype=torch.bfloat16),
    )
    assert payload["hidden_states"]["image_features"].dtype == torch.float32
    assert payload["hidden_states"]["seg_features"].dtype == torch.float32
    assert payload["hidden_states"]["box_xy"].dtype == torch.float32


def test_token_only_still_allocates_a_slot_when_the_latent_is_missing():
    """Stage 1 must not hang just because stage 0 shipped nothing usable."""
    completion = SimpleNamespace(cumulative_token_ids=[SEG_ID], multimodal_output=None)
    source = SimpleNamespace(finished=True, prompt_token_ids=[1, IMG_ID], outputs=[completion])

    inputs = thinker2segmentation_token_only([source], prompt=None)
    assert len(inputs) == 1
    assert inputs[0]["additional_information"] == {}
    assert inputs[0].get("multi_modal_data") is None


# --------------------------------------------------------------------------
# Image key: lets stage 1 skip AnyUp for an image it has already upsampled.
# AnyUp is the most expensive step of a request (~560 ms) and depends only on
# the image, so the key must track image *content* and nothing else.
# --------------------------------------------------------------------------


def _rgb(width, height, colour=(10, 20, 30)):
    from PIL import Image

    return Image.new("RGB", (width, height), colour)


def test_image_key_is_stable_for_the_same_image_and_differs_for_others():
    from vllm_omni.model_executor.stage_input_processors.falcon_perception import _image_key

    a = _image_key({"image": _rgb(64, 48)})
    same_content = _image_key({"image": _rgb(64, 48)})
    other_colour = _image_key({"image": _rgb(64, 48, (10, 20, 31))})
    other_size = _image_key({"image": _rgb(48, 64)})

    assert a is not None
    # Equal content must hit, or the cache never fires for the workload it exists for.
    assert int(a) == int(same_content)
    # A one-channel-value difference must miss, or masks come from the wrong image.
    assert int(a) != int(other_colour)
    assert int(a) != int(other_size)
    # Crosses a process boundary as a tensor, and negative ids would be a nuisance.
    assert a.dtype == torch.long and int(a) >= 0


def test_image_key_absent_rather_than_wrong_when_there_is_no_image():
    from vllm_omni.model_executor.stage_input_processors.falcon_perception import _image_key

    # None means "do not cache", which is safe. A fabricated key would collide.
    assert _image_key({}) is None
    assert _image_key({"image": None}) is None
    assert _image_key("not-a-dict") is None


def test_payload_carries_the_image_key_for_the_mask_stage():
    prompt = [1, IMG_ID, IMG_ID, 2]
    output = [SEG_ID]
    latent = torch.randn(len(prompt) + len(output), HIDDEN)
    inputs = thinker2segmentation_token_only(
        [_source_output(prompt, output, latent)],
        prompt={"multi_modal_data": {"image": _rgb(64, 48)}},
    )
    meta = inputs[0]["additional_information"]["meta"]
    assert "image_key" in meta, "stage 1 cannot cache AnyUp without an image identity"
    assert meta["image_key"].dtype == torch.long
