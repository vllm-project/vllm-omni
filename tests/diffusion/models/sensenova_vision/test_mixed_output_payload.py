# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the SenseNovaVision mixed text+image output contract.

SenseNovaVision ``caption_generate`` / ``think_generate`` produce an image together
with its caption/reasoning text.  Phase 3 wires that mixed payload through the
existing ``TEXT | IMAGE`` output-modality contract: the pipeline exposes the
text under ``payload["text"]`` alongside ``payload["image"]``, the diffusion
formatter preserves it in ``multimodal_output``, and the serving layer
serializes it as a leading ``{type: text}`` OpenAI content part.  No
SenseNovaVision-specific output modality keys are introduced.

These tests are CPU-only and construct the payload dict exactly as the
pipeline produces it (via :func:`build_sensenova_vision_diffusion_output`); no model
weights are loaded and no GPU is required.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from vllm_omni.diffusion import output_formatter
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.sensenova_vision.pipeline_sensenova_vision import (
    SenseNovaVisionPipeline,
    build_sensenova_vision_diffusion_output,
)
from vllm_omni.diffusion.output_formatter import (
    format_diffusion_outputs,
    normalize_diffusion_postprocess_output,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs.mm_outputs import MultimodalPayload

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]

_CAPTION = "a red car parked in front of a brick wall"


def _image() -> Image.Image:
    return Image.new("RGB", (8, 8), color=(200, 40, 40))


def _request() -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt="generate an image",
        request_id="req-mixed",
        sampling_params=OmniDiffusionSamplingParams(
            num_inference_steps=1,
            num_outputs_per_prompt=1,
            resolution=512,
        ),
    )


def _config() -> SimpleNamespace:
    return SimpleNamespace(model_class_name="SenseNovaVisionPipeline")


def test_build_sensenova_vision_diffusion_output_carries_text_and_image() -> None:
    """The canonical envelope exposes both text and image payload keys."""
    image = _image()
    output = build_sensenova_vision_diffusion_output(
        text=_CAPTION,
        image=image,
        think_text="thinking before caption",
        stage_durations={"execute": 1.25},
    )

    assert output.output["payload"]["text"] == _CAPTION
    assert output.output["payload"]["image"] is image
    assert output.output["metadata"]["text"] == {
        "text_output": _CAPTION,
        "think_text": "thinking before caption",
    }
    assert output.stage_durations == {"execute": 1.25}


def test_mixed_payload_formats_to_image_with_text_multimodal_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mixed payload serializes as an image output that keeps its caption.

    The formatter infers ``image`` as the primary key (the shared contract is
    ``TEXT | IMAGE``), so ``result.images`` carries the image while
    ``result.multimodal_output["text"]`` keeps the caption for serving.
    """
    monkeypatch.setattr(output_formatter, "supports_audio_output", lambda _: False)
    image = _image()
    diffusion_output = build_sensenova_vision_diffusion_output(
        text=_CAPTION,
        image=image,
        think_text="thinking before caption",
    )
    postprocess_output = normalize_diffusion_postprocess_output(diffusion_output.output)

    assert postprocess_output.primary_key == "image"
    assert postprocess_output.outputs == {"text": _CAPTION, "image": image}

    [result] = format_diffusion_outputs(
        request=_request(),
        od_config=_config(),
        diffusion_output=diffusion_output,
        output_data=diffusion_output.output,
        postprocess_output=postprocess_output,
    )

    assert result.images == [image]
    assert result.final_output_type == "image"
    assert result.multimodal_output["text"] == _CAPTION
    assert result.multimodal_output["metadata"]["text"] == {
        "text_output": _CAPTION,
        "think_text": "thinking before caption",
    }
    # The multimodal output must not gain any SenseNovaVision-specific modality keys.
    assert not ({"depth", "normal", "segmentation", "camera_pose", "point_map"} & set(result.multimodal_output))


def test_multimodal_payload_from_dict_splits_text_and_image() -> None:
    """Non-tensor payload values (PIL image, caption str) land in metadata."""
    payload = MultimodalPayload.from_dict({"text": _CAPTION, "image": _image()})
    assert payload is not None
    assert payload.metadata["text"] == _CAPTION
    assert payload.metadata["image"].size == (8, 8)
    assert payload.tensors == {}


def test_sensenova_vision_forward_merges_think_text_into_image_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``forward`` lifts the caption/reasoning text into the image payload."""
    image = _image()
    super_output = DiffusionOutput(
        output={
            "payload": {"image": image},
            "metadata": {"text": {"think_text": _CAPTION}},
        },
        stage_durations={"execute": 0.5},
    )

    def fake_super_forward(_req):
        return super_output

    pipeline = object.__new__(SenseNovaVisionPipeline)
    monkeypatch.setattr(pipeline, "_apply_mode_defaults", lambda _req: None)
    monkeypatch.setattr(pipeline, "forward", fake_super_forward)
    req = DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                prompt="generate",
                request_id="req-mixed",
                sampling_params=OmniDiffusionSamplingParams(
                    num_inference_steps=1,
                    extra_args={"caption": "ignored because metadata text wins"},
                ),
            )
        ]
    )

    merged = pipeline._merge_mixed_task_text(req, super_output)

    assert merged.output["payload"]["text"] == _CAPTION
    assert merged.output["payload"]["image"] is image
    assert merged.output["metadata"]["text"]["text_output"] == _CAPTION
    assert merged.output["metadata"]["text"]["think_text"] == _CAPTION


def test_merge_is_additive_when_text_already_present() -> None:
    """Outputs that already carry a text payload are returned unchanged."""
    pipeline = _make_pipeline()
    existing = DiffusionOutput(
        output={
            "payload": {"text": "keep me", "image": _image()},
            "metadata": {"text": {"text_output": "keep me"}},
        }
    )
    req = DiffusionRequestBatch(requests=[_request()])

    merged = pipeline._merge_mixed_task_text(req, existing)

    assert merged is existing
    assert merged.output["payload"]["text"] == "keep me"


def test_diffusion_text_content_part_serializes_caption() -> None:
    """The serving helper emits an OpenAI ``{type: text}`` content part."""
    part = OmniOpenAIServingChat._diffusion_text_content_part({"text": _CAPTION, "image": _image()})
    assert part == {"type": "text", "text": _CAPTION}

    assert OmniOpenAIServingChat._diffusion_text_content_part({}) is None
    assert OmniOpenAIServingChat._diffusion_text_content_part({"text": "   "}) is None
    assert OmniOpenAIServingChat._diffusion_text_content_part(None) is None


def test_create_image_choice_emits_text_then_image_parts() -> None:
    """A mixed output serializes to ``[text, image_url]`` content parts."""
    image = _image()
    omni_outputs = SimpleNamespace(
        request_output=None,
        stage_durations={"diffusion": 0.25},
        peak_memory_mb=10.0,
        images=[image],
        outputs=[],
        multimodal_output={
            "text": _CAPTION,
            "image": image,
            "metadata": {"text": {"text_output": _CAPTION}},
        },
    )

    choices = OmniOpenAIServingChat._create_image_choice(  # type: ignore[misc]
        None,
        omni_outputs=omni_outputs,
        role="assistant",
        request=SimpleNamespace(return_token_ids=False),
    )

    assert len(choices) == 1
    content = choices[0].message.content
    assert isinstance(content, list)
    assert [part["type"] for part in content] == ["text", "image_url"]
    assert content[0] == {"type": "text", "text": _CAPTION}
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def _make_pipeline() -> SenseNovaVisionPipeline:
    """Build a SenseNovaVisionPipeline instance without loading weights."""
    pipeline = object.__new__(SenseNovaVisionPipeline)
    pipeline.bagel = None
    pipeline._stage_durations = {"execute": 0.5}
    pipeline._profiler_lock = None
    pipeline.scheduler = None
    pipeline.scheduler_kwargs = None
    pipeline.new_token_ids = {}
    pipeline.device = None
    return pipeline


def _merged_output(think_text: str | None = None) -> DiffusionOutput:
    payload: dict[str, object] = {"image": _image()}
    metadata: dict[str, object] = {}
    if think_text is not None:
        metadata["text"] = {"think_text": think_text}
    return DiffusionOutput(output={"payload": payload, "metadata": metadata}, stage_durations={"execute": 0.5})


def test_merge_uses_extra_args_when_metadata_has_no_text() -> None:
    """``forward`` falls back to request extra_args (caption/text_output)."""
    output = _merged_output()
    req = DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                prompt="generate",
                request_id="req-mixed",
                sampling_params=OmniDiffusionSamplingParams(
                    num_inference_steps=1,
                    extra_args={"caption": "caption from extra args"},
                ),
            )
        ]
    )
    merged = _make_pipeline()._merge_mixed_task_text(req, output)

    assert merged.output["payload"]["text"] == "caption from extra args"
    assert merged.output["metadata"]["text"]["text_output"] == "caption from extra args"


def test_merge_leaves_payload_unchanged_without_text() -> None:
    """No caption available anywhere -> the image-only payload is unchanged."""
    output = _merged_output()
    req = DiffusionRequestBatch(requests=[_request()])

    merged = _make_pipeline()._merge_mixed_task_text(req, output)

    assert merged is output
    assert "text" not in merged.output["payload"]
    assert merged.output["metadata"] == {}


def test_diffusion_text_content_part_unwraps_single_item_list() -> None:
    """Single-element list text values (producer convention) are unwrapped."""
    part = OmniOpenAIServingChat._diffusion_text_content_part({"text": [_CAPTION]})
    assert part == {"type": "text", "text": _CAPTION}
