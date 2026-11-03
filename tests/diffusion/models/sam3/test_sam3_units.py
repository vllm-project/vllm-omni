# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the SAM3 detector port and pipeline.

The Transformers SAM3 implementation is the parity oracle: a tiny random
config is built for both, the reference weights are loaded into the port, and
outputs must match exactly in float32.

    pytest tests/diffusion/models/sam3/test_sam3_units.py -v
"""

from __future__ import annotations

import numpy as np
import PIL.Image
import pytest
import torch

from vllm_omni.diffusion.models.sam3.pipeline_sam3 import Sam3Pipeline, parse_prompt_boxes
from vllm_omni.diffusion.models.sam3.sam3_model import Sam3Detector, Sam3DetectorConfig, load_sam3_weights
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

transformers = pytest.importorskip("transformers")

IMAGE_SIZE = 112
TINY_CONFIG = {
    "vision_config": {
        "backbone_config": {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "image_size": IMAGE_SIZE,
            "patch_size": 14,
            "window_size": 4,
            "global_attn_indexes": [1],
            "pretrain_image_size": 56,
        },
        "fpn_hidden_size": 32,
        "backbone_feature_sizes": [[32, 32], [16, 16], [8, 8]],
    },
    "text_config": {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "projection_dim": 32,
        "vocab_size": 128,
        "max_position_embeddings": 32,
    },
    "geometry_encoder_config": {"hidden_size": 32, "num_layers": 1, "num_attention_heads": 2, "intermediate_size": 64},
    "detr_encoder_config": {"hidden_size": 32, "num_layers": 1, "num_attention_heads": 2, "intermediate_size": 64},
    "detr_decoder_config": {
        "hidden_size": 32,
        "num_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 64,
        "num_queries": 8,
    },
    "mask_decoder_config": {"hidden_size": 32, "num_attention_heads": 2},
}


@pytest.fixture(scope="module")
def reference():
    torch.manual_seed(0)
    config = transformers.Sam3Config(**TINY_CONFIG)
    config._attn_implementation = "sdpa"
    return transformers.Sam3Model(config).eval()


@pytest.fixture(scope="module")
def detector(reference):
    port = Sam3Detector(Sam3DetectorConfig.from_dict(TINY_CONFIG)).eval()
    # The combined image/video checkpoint layout, including tracker tensors to drop.
    weights = [("detector_model." + name, tensor) for name, tensor in reference.state_dict().items()]
    weights.append(("tracker_model.memory_encoder.weight", torch.zeros(1)))
    loaded = load_sam3_weights(port, weights)
    assert loaded == {name for name, _ in port.named_parameters()}
    return port


def _text_inputs(batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.zeros(batch, 32, dtype=torch.long)
    input_ids[:, :5] = torch.tensor([1, 17, 42, 9, 2])
    attention_mask = (input_ids != 0).long()
    attention_mask[:, 0] = 1
    return input_ids, attention_mask


@pytest.mark.parametrize("with_boxes", [False, True], ids=["text", "text+boxes"])
def test_detector_matches_transformers(reference, detector, with_boxes):
    torch.manual_seed(1)
    pixel_values = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    input_ids, attention_mask = _text_inputs()
    boxes = labels = None
    if with_boxes:
        boxes = torch.tensor([[[0.4, 0.4, 0.3, 0.5], [0.7, 0.3, 0.2, 0.2]]])
        labels = torch.tensor([[1, 0]])
    with torch.no_grad():
        expected = reference(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_boxes=boxes,
            input_boxes_labels=labels,
        )
        actual = detector(pixel_values, input_ids, attention_mask, boxes, labels)
    for name in ("pred_masks", "pred_boxes", "pred_logits", "presence_logits", "semantic_seg"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=1e-5, msg=name)


def test_config_accepts_combined_video_checkpoint_layout():
    flat = Sam3DetectorConfig.from_dict(TINY_CONFIG)
    nested = Sam3DetectorConfig.from_dict({"model_type": "sam3_video", "detector_config": TINY_CONFIG})
    assert nested == flat
    assert flat.vision_config.backbone_config.global_attn_indexes == [1]
    assert Sam3DetectorConfig.from_dict({}).vision_config.backbone_config.global_attn_indexes == [7, 15, 23, 31]


def test_prompt_box_validation():
    boxes, labels = parse_prompt_boxes({"boxes": [[1, 2, 30, 40]], "box_labels": [0]}, 64, 48)
    assert boxes == [[1.0, 2.0, 30.0, 40.0]] and labels == [0]
    assert parse_prompt_boxes({"boxes": [[0, 0, 5, 5]]}, 8, 8)[1] == [1]
    for bad in (
        {"boxes": [[0, 0, 70, 10]]},
        {"boxes": [[10, 0, 5, 10]]},
        {"boxes": [[0, 0, 5]]},
        {"boxes": [[0, 0, 5, 5]], "box_labels": [2]},
        {"boxes": [[0, 0, 5, 5]], "box_labels": [1, 1]},
    ):
        with pytest.raises(ValueError):
            parse_prompt_boxes(bad, 64, 48)


class _Tokenizer:
    def __call__(self, text, **kwargs):
        input_ids, attention_mask = _text_inputs()
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _pipeline(detector) -> Sam3Pipeline:
    pipeline = Sam3Pipeline.__new__(Sam3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.detector = detector
    pipeline.tokenizer = _Tokenizer()
    pipeline.image_size = IMAGE_SIZE
    return pipeline


def _image(width: int = 160, height: int = 120) -> PIL.Image.Image:
    rng = np.random.default_rng(0)
    return PIL.Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))


def test_preprocessing_and_postprocessing_match_reference_processor(detector):
    pipeline = _pipeline(detector)
    image = _image()
    processor = transformers.Sam3ImageProcessor(size={"height": IMAGE_SIZE, "width": IMAGE_SIZE})
    expected_pixels = processor(images=image, return_tensors="pt")["pixel_values"]
    torch.testing.assert_close(pipeline._pixel_values(image), expected_pixels, rtol=0, atol=1e-5)

    boxes, labels = pipeline._prompt_boxes([[16, 12, 80, 72]], [1], 160, 120)
    torch.testing.assert_close(boxes, torch.tensor([[[0.3, 0.35, 0.4, 0.5]]]))
    assert labels.tolist() == [[1]]

    result = pipeline.segment(image, "cat", [], [], score_threshold=0.0, mask_threshold=0.5)
    input_ids, attention_mask = _text_inputs()
    with torch.no_grad():
        outputs = detector(expected_pixels, input_ids, attention_mask)
    [reference] = processor.post_process_instance_segmentation(outputs, threshold=0.0, target_sizes=[(120, 160)])
    count, height, width = result["mask_shape"].tolist()
    masks = np.unpackbits(result["masks"], axis=1, count=height * width).reshape(count, height, width)
    assert (count, height, width) == (len(reference["scores"]), 120, 160)
    np.testing.assert_array_equal(masks, reference["masks"].numpy())
    np.testing.assert_allclose(result["scores"], reference["scores"].numpy(), atol=1e-6)
    np.testing.assert_allclose(result["boxes"], reference["boxes"].numpy(), atol=1e-4)


def test_forward_returns_segmentation_payload_and_rejects_bad_requests(detector):
    pipeline = _pipeline(detector)

    def request(prompt, **extra):
        single = OmniDiffusionRequest(
            prompt=prompt, request_id="seg", sampling_params=OmniDiffusionSamplingParams(extra_args=extra)
        )
        return DiffusionRequestBatch(requests=[single])

    image = _image(64, 48)
    output = pipeline(request({"prompt": "cat", "multi_modal_data": {"image": image}}, score_threshold=1.0))
    payload = output.output["segmentation"]
    # Nothing clears a threshold of 1: an empty result keeps the image geometry.
    assert payload["mask_shape"].tolist() == [0, 48, 64]
    assert payload["masks"].shape[0] == payload["boxes"].shape[0] == payload["scores"].shape[0] == 0

    boxes_only = pipeline(
        request({"prompt": "", "multi_modal_data": {"image": image}}, boxes=[[4, 4, 30, 30]], score_threshold=0.0)
    )
    assert boxes_only.error is None and boxes_only.output["segmentation"]["mask_shape"][0] > 0

    for bad in (
        request({"prompt": "cat"}),
        request({"prompt": "", "multi_modal_data": {"image": image}}),
        request({"prompt": "cat", "multi_modal_data": {"image": image}}, score_threshold=2),
        request({"prompt": "cat", "multi_modal_data": {"image": image}}, boxes=[[0, 0, 100, 10]]),
    ):
        failed = pipeline(bad)
        assert failed.error and failed.error_status_code == 400
