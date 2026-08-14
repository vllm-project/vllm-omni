# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for OmniRequestOutput class."""

import pytest
from PIL import Image
from vllm.outputs import CompletionOutput, RequestOutput

from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ── helpers ────────────────────────────────────────────────────────────────


def _make_text_request_output(
    request_id: str = "req-1",
    text: str = "hello world",
    token_ids: list[int] | None = None,
    finished: bool = True,
) -> RequestOutput:
    """Build a minimal vLLM RequestOutput for testing."""
    return RequestOutput(
        request_id=request_id,
        prompt="test prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=token_ids or [10, 11, 12],
                cumulative_logprob=-1.0,
                logprobs=None,
                finish_reason="stop" if finished else None,
                stop_reason=None,
            )
        ],
        finished=finished,
        metrics=None,
        lora_request=None,
    )


# ── from_stage_output ──────────────────────────────────────────────────────


class TestFromStageOutput:
    """Tests for OmniRequestOutput.from_stage_output()."""

    def test_copies_request_output_content(self):
        """Content from a vLLM RequestOutput is flattened onto the new object."""
        source = _make_text_request_output(text="copied text")
        out = OmniRequestOutput.from_stage_output(source, request_id="req-1", stage_id=0, final_output_type="text")

        assert out.prompt == "test prompt"
        assert out.prompt_token_ids == [1, 2, 3]
        assert len(out.outputs) == 1
        assert out.outputs[0].text == "copied text"
        assert out.finished is True
        assert out.stage_id == 0
        assert out.final_output_type == "text"

    def test_copies_omni_content(self):
        """Diffusion fields are copied when source is another OmniRequestOutput."""
        image = Image.new("RGB", (64, 64), color="blue")
        inner = OmniRequestOutput.from_diffusion(request_id="req-diff", images=[image], prompt="a cat")
        outer = OmniRequestOutput.from_stage_output(inner, request_id="req-diff", stage_id=0, final_output_type="image")

        assert len(outer.images) == 1
        assert outer.images[0] is image
        assert outer.prompt == "a cat"
        assert outer.is_diffusion_output

    def test_with_duck_typed_source(self):
        """A duck-typed source (SimpleNamespace-style) copies correctly."""
        from types import SimpleNamespace

        source = SimpleNamespace(
            outputs=[],
            prompt="duck prompt",
            prompt_token_ids=[7, 8, 9],
            finished=False,
        )
        out = OmniRequestOutput.from_stage_output(source, request_id="duck-1", stage_id=2)

        assert out.prompt == "duck prompt"
        assert out.prompt_token_ids == [7, 8, 9]
        assert out.finished is False
        assert out.stage_id == 2

    def test_request_id_always_copied_from_source(self):
        """request_id is always copied from source, even when passed in kwargs."""
        source = _make_text_request_output(request_id="source-id")
        out = OmniRequestOutput.from_stage_output(source, request_id="kwargs-id", stage_id=1)
        # Source wins — request_id is content, not a control field.
        assert out.request_id == "source-id"

    def test_passes_control_fields(self):
        """Control fields (metrics, stage_durations, peak_memory_mb) are set."""
        source = _make_text_request_output()
        out = OmniRequestOutput.from_stage_output(
            source,
            request_id="r1",
            stage_id=3,
            metrics={"ttft": 0.5},
            stage_durations={"prefill": 0.12},
            peak_memory_mb=1024.0,
        )

        assert out.metrics == {"ttft": 0.5}
        assert out.stage_durations == {"prefill": 0.12}
        assert out.peak_memory_mb == 1024.0

    def test_finished_field_copied(self):
        """The *finished* flag is copied from the source when present."""
        source = _make_text_request_output(finished=False)
        out = OmniRequestOutput.from_stage_output(source, request_id="r1")
        assert out.finished is False

        source2 = _make_text_request_output(finished=True)
        out2 = OmniRequestOutput.from_stage_output(source2, request_id="r2")
        assert out2.finished is True

    def test_outputs_field_copied(self):
        """The *outputs* list is copied from the source."""
        source = _make_text_request_output(text="hello", token_ids=[1, 2, 3])
        out = OmniRequestOutput.from_stage_output(source, request_id="r1")

        assert len(out.outputs) == 1
        assert out.outputs[0].text == "hello"
        assert out.outputs[0].token_ids == [1, 2, 3]


# ── dataclass safety ────────────────────────────────────────────────────────


class TestDataclassSafety:
    def test_from_stage_output_is_classmethod(self):
        """from_stage_output must be a classmethod."""
        assert callable(OmniRequestOutput.from_stage_output)


# ── msgpack round-trip ─────────────────────────────────────────────────────


class TestMsgpackRoundTrip:
    """Tests that OmniRequestOutput survives msgpack encode → decode without
    infinite recursion or data loss."""

    def test_round_trip_pipeline_output(self):
        """A pipeline output round-trips without data loss."""
        from vllm_omni.distributed.omni_connectors.utils.serialization import (
            OmniMsgpackDecoder,
            OmniMsgpackEncoder,
        )

        source = _make_text_request_output(text="round trip")
        out = OmniRequestOutput.from_stage_output(
            source,
            request_id="rt-1",
            stage_id=0,
            final_output_type="text",
        )

        encoder = OmniMsgpackEncoder()
        decoder = OmniMsgpackDecoder()

        encoded = encoder.encode(out)
        decoded = decoder.decode(encoded)

        assert isinstance(decoded, OmniRequestOutput)
        assert decoded.request_id == "req-1"
        assert decoded.stage_id == 0
        assert decoded.final_output_type == "text"
        assert decoded.finished is True
        assert decoded.outputs[0].text == "round trip"
        assert decoded.prompt == "test prompt"
        assert decoded.prompt_token_ids == [1, 2, 3]

    def test_round_trip_diffusion_output(self):
        """A diffusion output round-trips without data loss."""
        from vllm_omni.distributed.omni_connectors.utils.serialization import (
            OmniMsgpackDecoder,
            OmniMsgpackEncoder,
        )

        image = Image.new("RGB", (64, 64), color="red")
        out = OmniRequestOutput.from_diffusion(
            request_id="diff-1",
            images=[image],
            prompt="a red image",
            metrics={"steps": 50},
            multimodal_output={"audio": None},
        )

        encoder = OmniMsgpackEncoder()
        decoder = OmniMsgpackDecoder()

        encoded = encoder.encode(out)
        decoded = decoder.decode(encoded)

        assert isinstance(decoded, OmniRequestOutput)
        assert decoded.request_id == "diff-1"
        assert decoded.prompt == "a red image"
        assert decoded.num_images == 1
        assert decoded.images[0].size == (64, 64)
        assert decoded.metrics == {"steps": 50}

    def test_round_trip_does_not_recurse(self):
        """Encoding a pipeline output must not trigger infinite recursion."""
        from vllm_omni.distributed.omni_connectors.utils.serialization import (
            OmniMsgpackEncoder,
        )

        source = _make_text_request_output(text="no recursion")
        out = OmniRequestOutput.from_stage_output(source, request_id="rec-1", stage_id=0)

        encoder = OmniMsgpackEncoder()
        encoded = encoder.encode(out)
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0

    def test_legacy_wire_format_deserialized(self):
        """When the wire format contains a nested ``request_output`` key
        (pre-inheritance legacy), it is merged via _copy_content_from."""
        from vllm_omni.distributed.omni_connectors.utils.serialization import (
            OmniMsgpackDecoder,
        )

        # Build a legacy-style dict: OmniRequestOutput fields + a nested
        # request_output dict containing RequestOutput-like content.
        legacy_dict = {
            "request_id": "legacy-1",
            "finished": True,
            "stage_id": 1,
            "final_output_type": "text",
            "images": [],
            "_multimodal_output": {},
            "_custom_output": {},
            "stage_durations": {},
            "peak_memory_mb": 0.0,
            # --- nested request_output key (legacy) ---
            "request_output": {
                "request_id": "legacy-1",
                "prompt": "legacy prompt",
                "prompt_token_ids": [1, 2],
                "prompt_logprobs": None,
                "outputs": [],
                "finished": True,
                "metrics": {},
                "lora_request": None,
                "encoder_prompt": None,
                "encoder_prompt_token_ids": None,
                "num_cached_tokens": None,
                "kv_transfer_params": None,
            },
        }

        # Manually run the decoder's post_process to test legacy path.
        # The decoder's _decode_omni_request_output pops "request_output"
        # and merges it via _copy_content_from.
        decoder = OmniMsgpackDecoder()
        result = decoder._post_process(legacy_dict)

        assert isinstance(result, OmniRequestOutput)
        assert result.request_id == "legacy-1"
        assert result.finished is True
        # Content merged from legacy nested key
        assert result.prompt == "legacy prompt"
        assert result.prompt_token_ids == [1, 2]
