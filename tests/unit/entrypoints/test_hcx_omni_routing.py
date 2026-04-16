# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HyperCLOVAX-SEED-Omni-8B routing and diffusion bridge.

These tests target the specific bugs fixed for HCX-Omni e2e inference:

  1. Fan-out routing (omni.py):
     HyperCLOVAX-SEED-Omni uses a fan-out topology where the thinker
     (stage 0) sends vision tokens to stage 1 AND audio tokens to stage 2
     independently.  The former linear `next_stage_id = stage_id + 1`
     assumption caused a RuntimeError when stage-1 tried to forward to
     stage-2 via a non-existent connector.  The fix computes downstream
     stage IDs from connector keys instead.

  2. additional_information → extra bridge (omni_diffusion.py):
     Stage input processors (e.g. thinker2vision_decoder) store model-
     specific data such as `vision_tokens` and `audio_tokens` in
     OmniTokensPrompt.additional_information.  The diffusion pipelines
     read these values from OmniDiffusionRequest.extra, so the generate()
     method must bridge the two.

  3. Empty batch guard (omni_stage.py / omni.py):
     When the thinker produces a text-only reply (no vision/audio tokens),
     the downstream diffusion stages must be skipped gracefully rather than
     raising "Cannot execute model with empty request list".
"""

import pytest


# ---------------------------------------------------------------------------
# 1. Fan-out routing — downstream_stage_ids computation
# ---------------------------------------------------------------------------

class TestFanoutRouting:
    """Verify connector-key-based downstream stage discovery.

    The core fix in omni.py:
        downstream_stage_ids = sorted([
            int(to) for (frm, to) in self.connectors.keys()
            if frm == str(stage_id)
        ])
    """

    def _compute_downstream(self, connectors: dict, stage_id: int) -> list[int]:
        """Replicate the fixed routing logic from omni.py."""
        return sorted([
            int(to) for (frm, to) in connectors.keys()
            if frm == str(stage_id)
        ])

    def test_fanout_topology_stage0(self):
        """Stage-0 (thinker) fans out to stage-1 AND stage-2 independently."""
        # HCX-Omni YAML: edges: [{from:0,to:1}, {from:0,to:2}]
        connectors = {
            ("0", "1"): object(),
            ("0", "2"): object(),
        }
        downstream = self._compute_downstream(connectors, stage_id=0)
        assert downstream == [1, 2], (
            "Thinker must forward to both vision decoder (1) and audio decoder (2)"
        )

    def test_fanout_topology_leaf_stages(self):
        """Stage-1 and stage-2 are leaf nodes with no outgoing edges."""
        connectors = {
            ("0", "1"): object(),
            ("0", "2"): object(),
        }
        assert self._compute_downstream(connectors, stage_id=1) == [], (
            "Vision decoder (stage-1) must not forward to any further stage"
        )
        assert self._compute_downstream(connectors, stage_id=2) == [], (
            "Audio decoder (stage-2) must not forward to any further stage"
        )

    def test_linear_topology_still_works(self):
        """Linear pipelines (0→1→2) continue to work correctly."""
        connectors = {
            ("0", "1"): object(),
            ("1", "2"): object(),
        }
        assert self._compute_downstream(connectors, stage_id=0) == [1]
        assert self._compute_downstream(connectors, stage_id=1) == [2]
        assert self._compute_downstream(connectors, stage_id=2) == []

    def test_no_connectors_from_stage(self):
        """Stage with no outgoing connectors returns empty list (terminal)."""
        connectors = {("0", "1"): object()}
        assert self._compute_downstream(connectors, stage_id=1) == []

    def test_downstream_ids_are_sorted(self):
        """Downstream stage IDs are returned in ascending order."""
        # Insert in reverse order to verify sorting
        connectors = {
            ("0", "3"): object(),
            ("0", "1"): object(),
            ("0", "2"): object(),
        }
        downstream = self._compute_downstream(connectors, stage_id=0)
        assert downstream == [1, 2, 3]

    def test_fanout_with_multiple_source_stages(self):
        """Connector map with multiple source stages: each sees only its own edges."""
        connectors = {
            ("0", "1"): object(),
            ("0", "2"): object(),
            ("1", "3"): object(),
            ("2", "3"): object(),
        }
        assert self._compute_downstream(connectors, stage_id=0) == [1, 2]
        assert self._compute_downstream(connectors, stage_id=1) == [3]
        assert self._compute_downstream(connectors, stage_id=2) == [3]
        assert self._compute_downstream(connectors, stage_id=3) == []


# ---------------------------------------------------------------------------
# 2. additional_information → extra bridge (omni_diffusion.py)
# ---------------------------------------------------------------------------

class TestAdditionalInfoBridge:
    """Verify that OmniTokensPrompt.additional_information is merged into
    OmniDiffusionRequest.extra.

    The fixed generate() in OmniDiffusion:
        extra: dict = {}
        for prompt in prompts:
            if isinstance(prompt, dict):
                ai = prompt.get("additional_information")
                if isinstance(ai, dict):
                    for k, v in ai.items():
                        if k not in extra:
                            extra[k] = v
        request = OmniDiffusionRequest(prompts, ..., extra=extra)
    """

    def _build_extra(self, prompts: list) -> dict:
        """Replicate the bridge logic from omni_diffusion.py."""
        extra: dict = {}
        for prompt in prompts:
            if isinstance(prompt, dict):
                ai = prompt.get("additional_information")
                if isinstance(ai, dict):
                    for k, v in ai.items():
                        if k not in extra:
                            extra[k] = v
        return extra

    def test_vision_tokens_propagated(self):
        """vision_tokens from additional_information must reach extra."""
        vision_tokens = list(range(729))
        prompts = [
            {
                "prompt_token_ids": [1, 2, 3],
                "additional_information": {"vision_tokens": vision_tokens},
            }
        ]
        extra = self._build_extra(prompts)
        assert "vision_tokens" in extra
        assert extra["vision_tokens"] == vision_tokens

    def test_audio_tokens_propagated(self):
        """audio_tokens and speakers from additional_information must reach extra."""
        audio_tokens = [[10, 20, 30]]
        prompts = [
            {
                "prompt_token_ids": [1, 2, 3],
                "additional_information": {
                    "audio_tokens": audio_tokens,
                    "speakers": ["fkms"],
                },
            }
        ]
        extra = self._build_extra(prompts)
        assert extra["audio_tokens"] == audio_tokens
        assert extra["speakers"] == ["fkms"]

    def test_no_additional_information_gives_empty_extra(self):
        """Prompts without additional_information produce an empty extra dict."""
        prompts = [{"prompt_token_ids": [1, 2, 3]}]
        extra = self._build_extra(prompts)
        assert extra == {}

    def test_string_prompt_ignored(self):
        """Plain string prompts are skipped (only dict prompts are inspected)."""
        prompts = ["hello world"]
        extra = self._build_extra(prompts)
        assert extra == {}

    def test_first_occurrence_wins_for_batched_prompts(self):
        """When multiple prompts carry the same key, the first value is used.

        This mirrors the single-request behaviour and prevents later prompts
        from silently overwriting earlier ones in a batch.
        """
        prompts = [
            {"additional_information": {"vision_tokens": [1, 2, 3]}},
            {"additional_information": {"vision_tokens": [4, 5, 6]}},  # must be ignored
        ]
        extra = self._build_extra(prompts)
        assert extra["vision_tokens"] == [1, 2, 3]

    def test_different_keys_merged_across_prompts(self):
        """Different keys from different prompts are all collected."""
        prompts = [
            {"additional_information": {"vision_tokens": [1, 2, 3]}},
            {"additional_information": {"audio_tokens": [[10, 20]]}},
        ]
        extra = self._build_extra(prompts)
        assert "vision_tokens" in extra
        assert "audio_tokens" in extra

    def test_non_dict_additional_information_ignored(self):
        """Non-dict additional_information values are skipped gracefully."""
        prompts = [
            {"additional_information": "invalid_value"},
            {"additional_information": None},
            {"additional_information": 42},
        ]
        extra = self._build_extra(prompts)
        assert extra == {}


# ---------------------------------------------------------------------------
# 3. Empty batch guard — text-only reply skips diffusion stages
# ---------------------------------------------------------------------------

class TestEmptyBatchGuard:
    """Verify that the empty batch guard in the routing loop works correctly.

    When the thinker produces a text-only reply (no image or audio tokens),
    thinker2vision_decoder / thinker2audio_decoder return an empty list.
    The routing loop must detect this and skip the diffusion stage (setting
    `any_forwarded = False` for that modality) rather than forwarding an
    empty request list to the diffusion engine.

    The property being tested: if ALL downstream stages return empty inputs,
    `any_forwarded` must remain False and the request is counted as complete.
    """

    def _simulate_routing(
        self,
        connectors: dict,
        stage_id: int,
        stage_inputs: dict[int, list],
    ) -> tuple[bool, list[int]]:
        """Simulate the routing loop body for one completed request.

        Returns:
            (any_forwarded, forwarded_stage_ids)
        """
        downstream_stage_ids = sorted([
            int(to) for (frm, to) in connectors.keys()
            if frm == str(stage_id)
        ])
        any_forwarded = False
        forwarded_to = []
        for next_stage_id in downstream_stage_ids:
            next_inputs = stage_inputs.get(next_stage_id, [])
            if not next_inputs:
                # Replicate: "No inputs for this modality, skip"
                continue
            # Would send here; record as forwarded
            any_forwarded = True
            forwarded_to.append(next_stage_id)
        return any_forwarded, forwarded_to

    def test_text_only_reply_skips_all_diffusion_stages(self):
        """Text-only thinker output → no forwarding to vision or audio decoders."""
        connectors = {("0", "1"): object(), ("0", "2"): object()}
        # Both downstream processors return empty lists (no tokens for either)
        stage_inputs = {1: [], 2: []}
        any_forwarded, forwarded_to = self._simulate_routing(connectors, 0, stage_inputs)
        assert not any_forwarded, (
            "Text-only reply must not forward to any diffusion stage"
        )
        assert forwarded_to == []

    def test_image_only_output_skips_audio_stage(self):
        """Vision tokens present but no audio → only vision decoder receives request."""
        connectors = {("0", "1"): object(), ("0", "2"): object()}
        stage_inputs = {
            1: [{"prompt_token_ids": list(range(729))}],  # vision tokens present
            2: [],  # no audio tokens
        }
        any_forwarded, forwarded_to = self._simulate_routing(connectors, 0, stage_inputs)
        assert any_forwarded
        assert forwarded_to == [1]
        assert 2 not in forwarded_to

    def test_audio_only_output_skips_vision_stage(self):
        """Audio tokens present but no vision → only audio decoder receives request."""
        connectors = {("0", "1"): object(), ("0", "2"): object()}
        stage_inputs = {
            1: [],  # no vision tokens
            2: [{"additional_information": {"audio_tokens": [[1, 2, 3]]}}],
        }
        any_forwarded, forwarded_to = self._simulate_routing(connectors, 0, stage_inputs)
        assert any_forwarded
        assert forwarded_to == [2]
        assert 1 not in forwarded_to

    def test_both_modalities_forward_to_both_stages(self):
        """Both vision and audio tokens → both decoders receive requests."""
        connectors = {("0", "1"): object(), ("0", "2"): object()}
        stage_inputs = {
            1: [{"prompt_token_ids": list(range(729))}],
            2: [{"additional_information": {"audio_tokens": [[1, 2, 3]]}}],
        }
        any_forwarded, forwarded_to = self._simulate_routing(connectors, 0, stage_inputs)
        assert any_forwarded
        assert sorted(forwarded_to) == [1, 2]

    def test_leaf_stage_never_forwards(self):
        """Leaf stages (no outgoing connectors) always result in any_forwarded=False."""
        connectors = {("0", "1"): object(), ("0", "2"): object()}
        # stage_id=1 (vision decoder) has no outgoing edges
        stage_inputs = {}  # irrelevant since no downstream stages exist
        any_forwarded, forwarded_to = self._simulate_routing(connectors, 1, stage_inputs)
        assert not any_forwarded
        assert forwarded_to == []
