# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the StagePayload transport envelope (#4948 DiffusionV2Atoms).

``StagePayload`` (in ``interface.py``) is the four-dict envelope; its transport
safety (tensor sanitization, non-transportable-value rejection) is delegated to
the torch-free helpers in ``stage_payload.py``. Both modules load torch-free by
file path (see conftest); the ``interface_mod`` fixture pre-registers the
``stage_payload`` leaf under its real dotted name so ``StagePayload``'s lazy
``_transport()`` resolves without importing torch.
"""

from __future__ import annotations

import pytest


def test_create_valid_payload(interface_mod):
    p = interface_mod.StagePayload.create(
        request_id="req-1",
        boundary=interface_mod.StageBoundary.ENCODE_TO_DIT,
        scalar_fields={"session_id": "s0"},
        private_scalar_fields={"num_steps": 4, "cfg": 1.5, "shape": [1, 16, 4]},
    )
    assert p.payload_version == interface_mod.STAGE_PAYLOAD_SCHEMA_VERSION
    assert p.request_id == "req-1"
    assert p.boundary is interface_mod.StageBoundary.ENCODE_TO_DIT
    assert p.scalar_fields["session_id"] == "s0"
    assert p.private_scalar_fields["num_steps"] == 4


def test_payload_version_validation(interface_mod):
    p = interface_mod.StagePayload(
        request_id="r",
        boundary=interface_mod.StageBoundary.ENCODE_TO_DIT,
        payload_version=999,
    )
    with pytest.raises(Exception, match="payload_version"):
        p.validate()


def test_empty_request_id_rejected(interface_mod):
    with pytest.raises(Exception, match="request_id"):
        interface_mod.StagePayload.create(
            request_id="",
            boundary=interface_mod.StageBoundary.ENCODE_TO_DIT,
        )


def test_boundary_must_be_enum(interface_mod):
    p = interface_mod.StagePayload(request_id="r", boundary="encode_to_dit")
    with pytest.raises(Exception, match="boundary"):
        p.validate()


def test_expect_boundary(interface_mod):
    p = interface_mod.StagePayload.create(
        request_id="keep-me",
        boundary=interface_mod.StageBoundary.DIT_TO_DECODE,
    )
    p.expect_boundary(interface_mod.StageBoundary.DIT_TO_DECODE)
    assert p.request_id == "keep-me"
    with pytest.raises(Exception, match="expected"):
        p.expect_boundary(interface_mod.StageBoundary.ENCODE_TO_DIT)


def test_boundary_from_roles(interface_mod):
    SB = interface_mod.StageBoundary
    assert interface_mod.boundary_from_roles("encode", "denoise") is SB.ENCODE_TO_DIT
    assert interface_mod.boundary_from_roles("denoise", "decode") is SB.DIT_TO_DECODE
    with pytest.raises(Exception):
        interface_mod.boundary_from_roles("encode", "decode")


def test_roundtrip_to_from_dict(interface_mod, fake_tensor_cls):
    p = interface_mod.StagePayload.create(
        request_id="r",
        boundary=interface_mod.StageBoundary.ENCODE_TO_DIT,
        scalar_fields={"session_id": "s0"},
        private_tensor_fields={"prompt_embeds": fake_tensor_cls()},
        private_scalar_fields={"steps": 4},
    )
    d = p.to_dict()
    assert d["boundary"] == "encode_to_dit"  # enum flattens to its str value
    p2 = interface_mod.StagePayload.from_dict(d)
    assert p2.boundary is interface_mod.StageBoundary.ENCODE_TO_DIT
    assert p2.scalar_fields["session_id"] == "s0"
    assert p2.private_scalar_fields["steps"] == 4
    assert "prompt_embeds" in p2.private_tensor_fields


def test_tensor_dtype_shape_preserved(interface_mod, fake_tensor_cls):
    t = fake_tensor_cls(shape=(1, 16, 4, 22, 40), dtype="bfloat16")
    p = interface_mod.StagePayload.create(
        request_id="r",
        boundary=interface_mod.StageBoundary.ENCODE_TO_DIT,
        private_tensor_fields={"latents": t},
    )
    out = p.private_tensor_fields["latents"]
    assert out.shape == (1, 16, 4, 22, 40)
    assert out.dtype == "bfloat16"


def test_sanitize_detaches_and_moves_to_host(stage_payload, fake_tensor_cls):
    t = fake_tensor_cls(data_ptr=500)
    # sanitize path: detach -> cpu -> contiguous; clone only if ptr unchanged.
    out = stage_payload.sanitize_transport_tensor(t)
    assert "detach" in t.calls and "cpu" in t.calls and "contiguous" in t.calls
    # data_ptr is unchanged in the fake, so a clone must have been forced.
    assert out is not t


def test_non_tensor_in_tensor_fields_rejected(interface_mod):
    with pytest.raises(Exception, match="tensor"):
        interface_mod.StagePayload.create(
            request_id="r",
            boundary=interface_mod.StageBoundary.ENCODE_TO_DIT,
            private_tensor_fields={"bad": [1, 2, 3]},
        )


def test_tensor_in_scalar_fields_rejected(interface_mod, fake_tensor_cls):
    with pytest.raises(Exception, match="tensor"):
        interface_mod.StagePayload.create(
            request_id="r",
            boundary=interface_mod.StageBoundary.ENCODE_TO_DIT,
            private_scalar_fields={"latents": fake_tensor_cls()},
        )


def test_scheduler_object_in_scalar_fields_rejected(interface_mod):
    class FlowUniPCMultistepScheduler:  # name matches the non-transportable guard
        pass

    with pytest.raises(Exception, match="scheduler"):
        interface_mod.StagePayload.create(
            request_id="r",
            boundary=interface_mod.StageBoundary.ENCODE_TO_DIT,
            private_scalar_fields={"sched": FlowUniPCMultistepScheduler()},
        )


def test_callable_in_scalar_fields_rejected(interface_mod):
    with pytest.raises(Exception):
        interface_mod.StagePayload.create(
            request_id="r",
            boundary=interface_mod.StageBoundary.ENCODE_TO_DIT,
            private_scalar_fields={"fn": lambda x: x},
        )


def test_generator_like_object_rejected(interface_mod):
    class _FakeTorchGenerator:
        pass

    _FakeTorchGenerator.__module__ = "torch"
    _FakeTorchGenerator.__name__ = "Generator"

    with pytest.raises(Exception, match="Generator"):
        interface_mod.StagePayload.create(
            request_id="r",
            boundary=interface_mod.StageBoundary.ENCODE_TO_DIT,
            private_scalar_fields={"gen": _FakeTorchGenerator()},
        )


def test_process_local_state_object_rejected(interface_mod):
    class DreamZeroState:
        pass

    with pytest.raises(Exception, match="process-local"):
        interface_mod.StagePayload.create(
            request_id="r",
            boundary=interface_mod.StageBoundary.ENCODE_TO_DIT,
            private_scalar_fields={"state": DreamZeroState()},
        )


def test_nested_scalar_fields_allowed(interface_mod):
    p = interface_mod.StagePayload.create(
        request_id="r",
        boundary=interface_mod.StageBoundary.ENCODE_TO_DIT,
        private_scalar_fields={"nested": {"a": [1, 2, {"b": None, "c": b"bytes"}], "d": True}},
    )
    assert p.private_scalar_fields["nested"]["d"] is True


def test_require_tensors(interface_mod, fake_tensor_cls):
    p = interface_mod.StagePayload.create(
        request_id="r",
        boundary=interface_mod.StageBoundary.ENCODE_TO_DIT,
        private_tensor_fields={"latents": fake_tensor_cls()},
    )
    p.require_tensors("latents")
    with pytest.raises(Exception, match="missing required"):
        p.require_tensors("latents", "prompt_embeds")


def test_summary_has_no_tensor_contents(interface_mod, fake_tensor_cls):
    p = interface_mod.StagePayload.create(
        request_id="r",
        boundary=interface_mod.StageBoundary.ENCODE_TO_DIT,
        private_tensor_fields={"latents": fake_tensor_cls(shape=(1, 16), dtype="bfloat16")},
        scalar_fields={"session_id": "s0"},
    )
    s = p.summary()
    assert "latents" in s and "(1, 16)" in s and "session_id" in s
