# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Alpamayo single-model class.

Exercises the pieces testable without materializing the 10B model on a GPU:
1. discrete trajectory-token logit masking
2. checkpoint partition into per-submodule buckets (vlm/expert/action_*)
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from vllm_omni.model_executor.models.alpamayo.alpamayo import (
    Alpamayo15ForConditionalGeneration as AlpamayoConditionalGeneration,
)

# Real-checkpoint path for the (skipped-unless-present) partition test. Override
# with ALPAMAYO_MODEL when the weights live elsewhere; the default is the
# canonical local mount. The skipif guard keeps this no-op in CI.
_MODEL_15 = os.environ.get("ALPAMAYO_MODEL", "/data/models/Alpamayo-1.5-10B")


# --------------------------------------------------------------------------- #
# logit masking
# --------------------------------------------------------------------------- #
def test_traj_logit_mask_basic():
    start, vocab_size = 151669, 4000
    end = start + vocab_size
    logits = torch.zeros(3, 156000)
    out = AlpamayoConditionalGeneration._apply_traj_logit_mask(logits, start, end)
    # masked range is -inf
    assert torch.isneginf(out[:, start:end]).all()
    # just outside the range is untouched
    assert out[:, start - 1].eq(0).all()
    assert out[:, end].eq(0).all()


def test_traj_logit_mask_leaves_future_start_token_samplable():
    """The <|traj_future_start|> trigger (155681) is above the masked range."""
    start, end = 151669, 151669 + 4000  # -> [151669, 155669)
    future_start = 155681
    logits = torch.zeros(1, 156000)
    out = AlpamayoConditionalGeneration._apply_traj_logit_mask(logits, start, end)
    assert not torch.isneginf(out[0, future_start])


def test_traj_logit_mask_handles_none_and_oversized_bounds():
    assert AlpamayoConditionalGeneration._apply_traj_logit_mask(None, 0, 10) is None
    logits = torch.zeros(2, 100)
    # bounds beyond vocab are clamped, no exception
    out = AlpamayoConditionalGeneration._apply_traj_logit_mask(logits, 90, 9999)
    assert torch.isneginf(out[:, 90:]).all()
    assert out[:, :90].eq(0).all()


# --------------------------------------------------------------------------- #
# robot_obs validation (request-supplied payload guard)
# --------------------------------------------------------------------------- #
def test_validate_robot_obs_accepts_well_formed():
    from vllm_omni.model_executor.models.alpamayo.alpamayo import _validate_robot_obs

    ok = {
        "ego_history_xyz": [[1.0, 2.0, 3.0]],
        "ego_history_rot": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
    }
    assert _validate_robot_obs(ok) is ok
    # tensors are also accepted
    tens = {"ego_history_xyz": torch.zeros(1, 4, 3), "ego_history_rot": torch.zeros(1, 4, 3, 3)}
    assert _validate_robot_obs(tens) is tens


def test_validate_robot_obs_rejects_malformed():
    from vllm_omni.model_executor.models.alpamayo.alpamayo import (
        _ROBOT_OBS_MAX_ELEMS,
        _validate_robot_obs,
    )

    assert _validate_robot_obs("not-a-dict") is None
    assert _validate_robot_obs({"ego_history_xyz": [[1, 2, 3]]}) is None  # missing rot
    assert _validate_robot_obs({"ego_history_xyz": 5, "ego_history_rot": 6}) is None  # wrong type
    oversized = {"ego_history_xyz": list(range(_ROBOT_OBS_MAX_ELEMS + 1)), "ego_history_rot": [1]}
    assert _validate_robot_obs(oversized) is None  # over element cap


# --------------------------------------------------------------------------- #
# Checkpoint partition into vlm / expert / action_* buckets
# --------------------------------------------------------------------------- #
def test_bucket_checkpoint_rules():
    items = [
        ("vlm.model.visual.blocks.0.attn.proj.bias", torch.zeros(1)),
        ("vlm.lm_head.weight", torch.zeros(1)),
        ("expert.layers.0.self_attn.q_proj.weight", torch.zeros(1)),
        ("expert.norm.weight", torch.zeros(1)),
        ("action_in_proj.encoder.trunk.0.weight", torch.zeros(1)),
        ("action_out_proj.weight", torch.zeros(1)),
        ("action_out_proj.bias", torch.zeros(1)),
        ("action_space.accel_mean", torch.zeros(1)),
        ("orphan.thing", torch.zeros(1)),  # falls through to vlm
    ]
    buckets = AlpamayoConditionalGeneration._bucket_checkpoint(items)
    names = {k: [n for n, _ in v] for k, v in buckets.items()}
    assert names["vlm"] == ["model.visual.blocks.0.attn.proj.bias", "lm_head.weight", "orphan.thing"]
    assert names["expert"] == ["layers.0.self_attn.q_proj.weight", "norm.weight"]
    assert names["action_in_proj"] == ["encoder.trunk.0.weight"]
    assert sorted(names["action_out_proj"]) == ["bias", "weight"]
    assert names["action_space"] == ["accel_mean"]


@pytest.mark.skipif(not os.path.isdir(_MODEL_15), reason="Alpamayo-1.5 weights not present locally")
def test_bucket_real_checkpoint_complete_partition():
    """Every tensor in the real Alpamayo-1.5 checkpoint lands in a known bucket
    with a stripped name that's consumable by the target submodule. Catches
    routing drift before the GPU load."""
    idx = json.load(open(os.path.join(_MODEL_15, "model.safetensors.index.json")))
    names = list(idx["weight_map"].keys())
    items = [(n, torch.empty(0)) for n in names]
    buckets = AlpamayoConditionalGeneration._bucket_checkpoint(items)
    counts = {k: len(v) for k, v in buckets.items()}

    # Expected from the released 1.5 checkpoint structure.
    assert counts["vlm"] == 750, f"vlm count: {counts['vlm']}"
    assert counts["expert"] == 397, f"expert count: {counts['expert']}"  # 36 layers * 11 + norm
    assert counts["action_in_proj"] == 10
    assert counts["action_out_proj"] == 2
    assert counts["action_space"] == 0  # released ckpt ships these as config-derived

    # VLM stripped names are consumable by the Qwen3-VL hf_to_vllm_mapper.
    consumable = ("model.visual.", "model.language_model.", "lm_head.")
    bad_vlm = [n for n, _ in buckets["vlm"] if not n.startswith(consumable)]
    assert not bad_vlm, f"vlm names not consumable by Qwen3-VL mapper: {bad_vlm[:5]}"
    # Expert stripped names are bare Qwen3Model keys.
    bad_exp = [n for n, _ in buckets["expert"] if not (n.startswith("layers.") or n.startswith("norm."))]
    assert not bad_exp, f"expert names not consumable by Qwen3Model: {bad_exp[:5]}"


# --------------------------------------------------------------------------- #
# Structural validation
# --------------------------------------------------------------------------- #
# Note: full meta-device construction needs a vLLM-validated ModelConfig which
# requires the architecture to be registered with vLLM's engine registry — that
# happens at vLLM LLM API integration. Structural correctness on CPU is already
# covered by test_bucket_real_checkpoint_complete_partition above (it verifies
# every real-checkpoint tensor lands in a bucket with a name that's consumable
# by the target submodule).


# --------------------------------------------------------------------------- #
# Pure helpers: trigger detection + action-token mRoPE positions
# --------------------------------------------------------------------------- #
def test_find_trigger_indices_future_start_only():
    """Without history_traj, the trigger condition is input == future_start."""
    ids = torch.tensor([100, 155681, 200, 155681, 151645])  # 2 triggers
    out = AlpamayoConditionalGeneration.find_trigger_indices(ids, 155681, 151645, has_history_traj=None)
    assert out.tolist() == [1, 3]


def test_find_trigger_indices_with_history_includes_im_end():
    """With history_traj set, im_end also triggers (the SGLang dual-trigger)."""
    ids = torch.tensor([100, 155681, 151645, 151645])
    hist = torch.tensor([True, False, True, False])
    # idx 1: future_start (always triggers). idx 2: im_end + hist (triggers).
    # idx 3: im_end but no hist (no trigger).
    out = AlpamayoConditionalGeneration.find_trigger_indices(ids, 155681, 151645, has_history_traj=hist)
    assert sorted(out.tolist()) == [1, 2]


def test_find_trigger_indices_empty_when_no_trigger():
    ids = torch.tensor([1, 2, 3])
    out = AlpamayoConditionalGeneration.find_trigger_indices(ids, 155681, 151645)
    assert out.numel() == 0


def test_action_token_positions_continues_after_current_mrope():
    """Each triggered request gets n_waypoints action positions starting at
    cur+1 (mirrors SGLang positions_list in _run_flow_matching)."""
    # 3D mRoPE (matching Qwen3-VL); batch of 3 requests with positions 10/20/30.
    cur = torch.tensor([[10, 20, 30], [10, 20, 30], [10, 20, 30]])  # (3, 3)
    active = torch.tensor([0, 2])  # triggered: request 0 and 2
    pos = AlpamayoConditionalGeneration.action_token_positions(cur, active, n_waypoints=4)
    assert pos.shape == (3, 8)  # 2 triggered * 4 waypoints
    # request 0: positions 11..14 (broadcast across 3 mrope dims since cur is equal here)
    assert pos[0, :4].tolist() == [11, 12, 13, 14]
    assert pos[0, 4:].tolist() == [31, 32, 33, 34]
    # All 3 mrope dims advance together when cur is equal across dims
    for d in range(3):
        assert pos[d, :4].tolist() == [11, 12, 13, 14]
        assert pos[d, 4:].tolist() == [31, 32, 33, 34]


def test_action_token_positions_empty_active():
    cur = torch.tensor([[10], [20], [30]])
    active = torch.empty(0, dtype=torch.long)
    pos = AlpamayoConditionalGeneration.action_token_positions(cur, active, n_waypoints=4)
    assert pos.shape == (3, 0)


def test_ar_class_declares_expert_and_action_heads_in_init_signature():
    """Sanity: AlpamayoConditionalGeneration.__init__ source contains the expert + action head wiring."""
    import inspect

    src = inspect.getsource(AlpamayoConditionalGeneration.__init__)
    for marker in ("self.expert", "self.action_in_proj", "self.action_out_proj", "self.action_space"):
        assert marker in src, f"missing {marker} in AlpamayoConditionalGeneration.__init__"
