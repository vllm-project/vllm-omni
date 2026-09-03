# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Parity tests: async-chunk prewarm placeholder vs sync forward placeholder.

Related to RFC #4872 (https://github.com/vllm-project/vllm-omni/issues/4872):
``*_token_only`` placeholder builders expose a dual entry —
``build_forward_placeholder(source_outputs, ctx)`` for the sync forward path
and ``build_prewarm_placeholder(*, stage0_prompt, ctx, downstream_stage_id)`` for
the async-chunk prewarm path (no ``source_outputs`` yet).  Both must be fed by
the same ``_common`` length / packing helpers so async and sync placeholders stay
consistent, and the streaming fixup helper
(``adapter.compute_talker_prompt_ids_length``) must use the same length helper.

Length semantics
----------------
- forward (``mode="full"``) is the Qwen chat-template scan (golden-locked at 15
  for the canonical prompt ``test_common_helpers_golden.py``).
- prewarm (``mode="stage0_only"``) runs the **same** Qwen chat-template scan on
  the stage-0 input list (feeding it to both ``all``/``prompt`` roles), so the
  prewarm estimate equals the forward placeholder length and the adapter's
  streaming-fixup scan (15 == 15 for the golden prompt; 6 == 6 for a single user
  segment).  The connector fixup path replaces the estimate with the real
  length once the upstream chunk arrives.

These tests are CPU-only (no model loading).  They run both without a vllm
runtime (the import fallback is active) and with real vllm, where
``OmniTokensPrompt`` is built as a dict subclass; length capture uses a
``pack_placeholder_prompt`` spy so it works in both environments.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.model_executor.stage_input_processors import _common
from vllm_omni.model_executor.stage_input_processors import qwen3_omni as q3
from vllm_omni.model_executor.stage_input_processors._dispatch import OrchestratorInputContext
from vllm_omni.model_executor.stage_input_processors._registry import resolve_processor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

#: Canonical Qwen chat-template prompt used by the golden tests: im_start at 0/6,
#: one user segment of 6 tokens + one trailing assistant segment (+9) -> full 15.
GOLDEN_PROMPT = [151644, 872, 10, 11, 12, 13, 151644, 77091, 20, 21]

#: Single user segment (no trailing assistant marker): full == stage0_only == 6.
SINGLE_USER_PROMPT = [151644, 872, 10, 11, 12, 13]


def _shim_active() -> bool:
    """True when the test-support import fallback (no real vllm) is active."""
    try:
        import vllm_omni  # noqa: F401

        return bool(getattr(vllm_omni, "_SHIM_ACTIVE", False))
    except Exception:
        return False


def _source_outputs(prompt_token_ids, output_ids=()):
    return [
        SimpleNamespace(
            request_id="req-1",
            prompt_token_ids=list(prompt_token_ids),
            outputs=[SimpleNamespace(cumulative_token_ids=list(output_ids))],
        )
    ]


def _ctx(prompt=None):
    return OrchestratorInputContext(
        prompt=prompt,
        requires_multimodal_data=False,
        streaming_context=None,
    )


def _capture_pack(monkeypatch):
    """Spy on ``_common.pack_placeholder_prompt``; returns a recorder dict.

    ``pack_placeholder_prompt`` is the single packing chokepoint shared by both
    builders, so capturing ``prompt_len`` / ``voice_metadata`` verifies the
    builders' length logic without depending on how ``OmniTokensPrompt`` is
    constructed (import fallback vs real vllm).
    """

    recorded: dict[str, Any] = {}

    def _spy(*, prompt_len: int, voice_metadata: dict[str, Any] | None = None) -> Any:
        recorded["prompt_len"] = prompt_len
        recorded["voice_metadata"] = voice_metadata
        return "PLACEHOLDER"

    monkeypatch.setattr(_common, "pack_placeholder_prompt", _spy)
    return recorded


# ---------------------------------------------------------------------------
# Forward builder (sync path) locks the golden full-mode length.
# ---------------------------------------------------------------------------


def test_build_forward_placeholder_length_matches_golden(monkeypatch):
    recorded = _capture_pack(monkeypatch)
    out = q3.build_forward_placeholder(_source_outputs(GOLDEN_PROMPT), _ctx())
    assert len(out) == 1
    assert out[0] == "PLACEHOLDER"
    # Golden chat-template scan: sum(user segment) + 9 == 15.
    assert recorded["prompt_len"] == 15


def test_build_forward_placeholder_preserves_voice_metadata(monkeypatch):
    recorded = _capture_pack(monkeypatch)
    prompt = {"additional_information": {"speaker": ["ethan"], "language": ["English"]}}
    out = q3.build_forward_placeholder(_source_outputs([1, 2], [3]), _ctx(prompt=prompt))
    assert len(out) == 1
    assert recorded["voice_metadata"] == {"speaker": ["ethan"], "language": ["English"]}


# ---------------------------------------------------------------------------
# Prewarm builder (async path) is a stage0-only best-effort estimate.
# ---------------------------------------------------------------------------


def test_build_prewarm_placeholder_length_from_stage0(monkeypatch):
    recorded = _capture_pack(monkeypatch)
    out = q3.build_prewarm_placeholder(
        stage0_prompt=GOLDEN_PROMPT,
        ctx=_ctx(),
        downstream_stage_id=1,
    )
    assert len(out) == 1
    assert out[0] == "PLACEHOLDER"
    # stage0_only runs the chat-template scan on the stage-0 list -> 15
    # (the same number the forward builder and the adapter scan produce).
    assert recorded["prompt_len"] == 15
    # Voice metadata is intentionally not forwarded (matches pre-existing prewarm).
    assert recorded["voice_metadata"] is None


def test_build_prewarm_placeholder_accepts_request_like_stage0(monkeypatch):
    recorded = _capture_pack(monkeypatch)
    stage0_request = SimpleNamespace(prompt_token_ids=GOLDEN_PROMPT)
    q3.build_prewarm_placeholder(stage0_prompt=stage0_request, ctx=_ctx(), downstream_stage_id=1)
    assert recorded["prompt_len"] == 15


# ---------------------------------------------------------------------------
# Parity: forward and prewarm must agree where the prompt shape allows.
# ---------------------------------------------------------------------------


def test_forward_prewarm_parity_single_user_segment(monkeypatch):
    """Single user segment: forward (full) == prewarm (stage0_only) == 6."""
    recorded = _capture_pack(monkeypatch)
    q3.build_forward_placeholder(_source_outputs(SINGLE_USER_PROMPT), _ctx())
    forward_len = recorded["prompt_len"]
    recorded.clear()
    q3.build_prewarm_placeholder(stage0_prompt=SINGLE_USER_PROMPT, ctx=_ctx(), downstream_stage_id=1)
    assert forward_len == recorded["prompt_len"] == 6


def test_forward_prewarm_golden_relationship(monkeypatch):
    """Documented relationship for the chat-template golden prompt.

    forward (full) == 15 (golden-locked); prewarm (stage0_only) is now the
    **same chat-template scan** on the stage-0 list, so it also returns 15.
    The builder and the inline fallback
    (``adapter.compute_talker_prompt_ids_length``) therefore agree — the old
    ``len(stage0_prompt)`` == 10 estimate under-reserved KV slots and split a
    single request between 10 and 15 depending on whether resolution succeeded.
    """
    recorded = _capture_pack(monkeypatch)
    q3.build_forward_placeholder(_source_outputs(GOLDEN_PROMPT), _ctx())
    forward_len = recorded["prompt_len"]
    recorded.clear()
    q3.build_prewarm_placeholder(stage0_prompt=GOLDEN_PROMPT, ctx=_ctx(), downstream_stage_id=1)
    assert forward_len == 15
    assert recorded["prompt_len"] == 15
    assert forward_len == recorded["prompt_len"]  # full == stage0_only scan


# ---------------------------------------------------------------------------
# Packed placeholder structure (asserted with real vllm; field reads are
# skipped under the import fallback).
# ---------------------------------------------------------------------------


def test_pack_placeholder_prompt_structure():
    prompt = _common.pack_placeholder_prompt(prompt_len=4, voice_metadata={"speaker": ["ethan"]})
    empty_meta = _common.pack_placeholder_prompt(prompt_len=1)
    if _shim_active():
        # Under the import fallback, OmniTokensPrompt's real constructor is
        # bypassed, so field access returns fallback values. Field assertions
        # run where real vllm constructs the model properly.
        pytest.skip("OmniTokensPrompt field construction needs real vllm")
    assert prompt["prompt_token_ids"] == [0, 0, 0, 0]
    assert prompt["additional_information"] == {"speaker": ["ethan"]}
    assert empty_meta["prompt_token_ids"] == [0]
    assert empty_meta.get("additional_information") is None


# ---------------------------------------------------------------------------
# thinker2talker_token_only delegation keeps the public sync entry unchanged.
# ---------------------------------------------------------------------------


def test_thinker2talker_token_only_delegates_to_forward_placeholder(monkeypatch):
    recorded = _capture_pack(monkeypatch)
    prompt = {"additional_information": {"speaker": ["ethan"], "language": ["English"]}}
    out = q3.thinker2talker_token_only(
        _source_outputs([1, 2], [3]),
        prompt=prompt,
        requires_multimodal_data=False,
        streaming_context=None,
    )
    assert len(out) == 1
    assert out[0] == "PLACEHOLDER"
    assert recorded["voice_metadata"] == {"speaker": ["ethan"], "language": ["English"]}


# ---------------------------------------------------------------------------
# Streaming fixup helper shares the same length helper as the forward builder.
# ---------------------------------------------------------------------------


def test_adapter_compute_length_consistent_with_common():
    # Lazy import keeps this module import-light without a vllm runtime.
    from vllm_omni.distributed.omni_connectors import adapter

    assert adapter.compute_talker_prompt_ids_length(GOLDEN_PROMPT) == 15
    assert adapter.compute_talker_prompt_ids_length(GOLDEN_PROMPT) == _common.compute_placeholder_prompt_len(
        ids_or_prompt={"ids": {"all": GOLDEN_PROMPT, "prompt": GOLDEN_PROMPT}},
        mode="full",
    )
    # The prewarm stage0_only scan (what build_prewarm_placeholder and the
    # orchestrator inline fallback both use) must equal the adapter scan.
    assert adapter.compute_talker_prompt_ids_length(GOLDEN_PROMPT) == _common.compute_placeholder_prompt_len(
        ids_or_prompt=GOLDEN_PROMPT,
        mode="stage0_only",
    )
    assert adapter.compute_talker_prompt_ids_length(SINGLE_USER_PROMPT) == 6
    assert (
        _common.compute_placeholder_prompt_len(
            ids_or_prompt=SINGLE_USER_PROMPT,
            mode="stage0_only",
        )
        == 6
    )


# ---------------------------------------------------------------------------
# Registry surface: the resolved sync fn exposes build_prewarm_placeholder, so
# the orchestrator's prewarm resolution path resolves without a module scan.
# ---------------------------------------------------------------------------


def test_resolved_sync_fn_exposes_build_prewarm_placeholder():
    spec = resolve_processor(
        "vllm_omni.model_executor.stage_input_processors.qwen3_omni.thinker2talker_token_only",
        expected_kind=None,
    )
    assert callable(getattr(spec.fn, "build_prewarm_placeholder", None))
    assert callable(getattr(spec.fn, "build_forward_placeholder", None))
    # The attached builder is the same module-level object used by tests above.
    assert spec.fn.build_prewarm_placeholder is q3.build_prewarm_placeholder
