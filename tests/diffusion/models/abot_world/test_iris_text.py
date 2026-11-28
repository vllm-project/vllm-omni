# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""ABot-World 0.5 Iris text conditioning: prompt switches, skills and action captions."""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.interaction.coordinator import InteractionCoordinator
from vllm_omni.diffusion.interaction.types import ChunkMediaSpec
from vllm_omni.diffusion.models.abot_world import iris_text
from vllm_omni.diffusion.models.abot_world.interaction import (
    ABotCameraInteractionHandler,
    ABotPromptInteractionHandler,
    ABotSkillInteractionHandler,
)
from vllm_omni.diffusion.models.abot_world.iris_text import IRIS_ACTION_CAPTIONS, IrisActionCaptions, IrisText
from vllm_omni.diffusion.models.abot_world.pipeline_abot_world import ABotWorldCausalPipeline, _ABotRequestInputs
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_D = 16


def _keys(*keys: str) -> frozenset[str]:
    return frozenset(keys)


# Per chunk: the key states the client reports in that window, and an
# optional skill request sent before the chunk's boundary. Expected values
# were recorded by replaying the same schedules through the upstream web
# worker loop (ExampleSkill + ActionConditions, amap-cvlab/ABot-World
# d4a4f2d7): whether each skill request was accepted, then the local text per
# chunk as a skill prompt or the action keys whose caption it holds.
_Chunk = tuple[list[frozenset[str]], str | None]
UPSTREAM_LOCAL_TEXT: dict[str, tuple[list[_Chunk], list[bool], list[str]]] = {
    "hold_crouch": (
        [([_keys("ctrl")], None), ([], None), ([_keys()], None), ([], None)],
        [],
        ["ctrl", "ctrl", "", ""],
    ),
    "tap_jump_pulses_one_chunk": ([([_keys("space"), _keys()], None), ([], None)], [], ["space", ""]),
    "held_space_is_a_single_pulse": ([([_keys("space")], None), ([], None), ([], None)], [], ["space", "", ""]),
    "roll_pulses_two_chunks": (
        [([_keys("alt"), _keys()], None), ([], None), ([], None)],
        [],
        ["alt", "alt", ""],
    ),
    "sprint_then_running_jump": (
        [([_keys("shift")], None), ([_keys("shift", "space"), _keys("shift")], None), ([], None), ([_keys()], None)],
        [],
        ["shift", "shift+space", "shift", ""],
    ),
    "aim_then_attack": (
        [
            ([_keys("mouseright")], None),
            ([_keys("mouseright", "mouseleft"), _keys("mouseright")], None),
            ([], None),
            ([], None),
        ],
        [],
        ["mouseright", "mouseright+mouseleft", "mouseright+mouseleft", "mouseright"],
    ),
    "new_tap_interrupts_roll": (
        [([_keys("alt"), _keys()], None), ([_keys("space"), _keys()], None), ([], None)],
        [],
        ["alt", "space", ""],
    ),
    "skill_lasts_four_chunks": (
        [([], "skill A"), ([], None), ([], None), ([], None), ([], None), ([], None)],
        [True],
        ["skill A", "skill A", "skill A", "skill A", "", ""],
    ),
    "skill_rejected_while_active": (
        [([], "skill A"), ([], "skill B"), ([], None), ([], None), ([], "skill C"), ([], None)],
        [True, False, True],
        ["skill A", "skill A", "skill A", "skill A", "skill C", "skill C"],
    ),
    "key_press_cancels_skill_requested_with_it": (
        [([_keys("ctrl")], "skill A"), ([], None), ([], None)],
        [True],
        ["ctrl", "ctrl", "ctrl"],
    ),
    "skill_hides_held_caption": (
        [([_keys("ctrl")], None), ([], "skill A"), ([], None), ([], None), ([], None), ([], None)],
        [True],
        ["ctrl", "skill A", "skill A", "skill A", "skill A", "ctrl"],
    ),
    "tap_cancels_skill": (
        [([], "skill A"), ([], None), ([_keys("space"), _keys()], None), ([], None), ([], "skill B")],
        [True, True],
        ["skill A", "skill A", "space", "", "skill B"],
    ),
    "tap_at_start_cancels_new_skill": (
        [([_keys("alt"), _keys()], "skill A"), ([], None), ([], None)],
        [True],
        ["alt", "alt", ""],
    ),
    "release_does_not_cancel_skill": (
        [([_keys("ctrl")], None), ([_keys()], "skill A"), ([], None)],
        [True],
        ["ctrl", "skill A", "skill A"],
    ),
}


def _expected_text(label: str) -> str:
    if not label or label.startswith("skill"):
        return label
    keys = tuple(label.split("+"))
    if len(keys) == 1:
        return IRIS_ACTION_CAPTIONS[keys[0]]
    return iris_text._COMBINATION_CAPTIONS[(keys[0], keys[1])]


class _IrisPipeline(ABotWorldCausalPipeline):
    """A real Iris-layout pipeline without checkpoints; records encoder calls."""

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.device = torch.device("cpu")
        self.vae_scale_factor_spatial = 16
        self._num_frame_per_block = 3
        self.vae_scale_factor_temporal = 4
        self.transformer = torch.nn.Module()
        self.transformer.config = type("Config", (), {"out_channels": 4, "use_relative_rope": True})()
        self.encoded: list[tuple[str, int]] = []
        self.cache_calls: list[bool] = []

    def _encode_text(self, prompt: str, *, max_sequence_length: int, dtype: torch.dtype) -> torch.Tensor:
        self.encoded.append((prompt, max_sequence_length))
        return torch.full((1, max_sequence_length, _D), float(len(self.encoded)), dtype=dtype)

    def _ar_text_caches(self, prompt_embeds: torch.Tensor, *, invalidate: bool) -> list:
        assert prompt_embeds.shape == (1, 512, _D)
        self.cache_calls.append(invalidate)
        return []

    def _ar_reference_caches(self) -> list:
        return []

    def _build_action_tensor(self, actions, *args) -> torch.Tensor:
        self.actions.append(actions)
        return torch.zeros(1)

    def peek_chunk_media(self, state: StepRequestState) -> ChunkMediaSpec:
        return ChunkMediaSpec(num_media_frames=12, fps=12.0, num_latent_frames=3)


def _iris_request(pipeline: _IrisPipeline, prompt: str = "A sunny canyon.") -> StepRequestState:
    inputs = _ABotRequestInputs(
        prompt=prompt,
        image=torch.zeros(3, 32, 32),
        reference_images=None,
        camera_action_script=None,
        height=32,
        width=32,
        num_frames=93,
        num_latent_frames=24,
        num_frame_per_block=3,
        output_type="latent",
        max_sequence_length=512,
        flow_shift=5.0,
        generator=torch.Generator().manual_seed(0),
    )
    state = StepRequestState(
        request_id="r",
        sampling=OmniDiffusionSamplingParams(),
        chunk_index=0,
        total_chunks=8,
        extra={
            "inputs": inputs,
            "dtype": torch.float32,
            "schedule": ((1000, 1.0),),
            "iris_text": IrisText(global_prompt=prompt),
        },
    )
    state.prompt_embeds = pipeline.encode_prompt(prompt, max_sequence_length=512, dtype=torch.float32)
    pipeline.actions = []
    return state


class _Rollout:
    """Drive the real coordinator and pipeline hooks chunk by chunk."""

    def __init__(self, prompt: str = "A sunny canyon.") -> None:
        self.pipeline = _IrisPipeline()
        self.coordinator = InteractionCoordinator.build(
            self.pipeline, OmniDiffusionConfig(model_class_name="ABotWorldCausalPipeline")
        )
        self.state = _iris_request(self.pipeline, prompt)
        self.clock = 0.0
        self.events = 0
        self.meta = self.coordinator.maybe_prepare_initial_session(self.state, self.pipeline)
        self.pipeline.prepare_next_chunk(self.state)
        self.pipeline._consume_iris_skill_chunk(self.state)

    def send(self, modality: str, payload: dict) -> bool:
        self.clock += 1.0
        self.events += 1
        try:
            self.coordinator.enqueue_parts(
                self.state,
                parts=[(modality, payload)],
                event_id=f"e{self.events}",
                received_at=self.clock,
                transition_chunks=None,
            )
        except ValueError:
            return False
        return True

    def next_chunk(self) -> str:
        """Apply the boundary, prepare the chunk, then finish it as post_decode does."""
        self.state.chunk_index += 1
        self.clock += 1.0
        self.meta = self.coordinator.apply_at_chunk_boundary(self.state, boundary_at=self.clock, num_latent_frames=3)
        self.pipeline.prepare_next_chunk(self.state)
        self.pipeline._consume_iris_skill_chunk(self.state)
        return self.state.extra["iris_text"].local_prompt


@pytest.mark.parametrize("name", sorted(UPSTREAM_LOCAL_TEXT))
def test_local_text_matches_the_upstream_web_worker(name: str) -> None:
    chunks, accepted, labels = UPSTREAM_LOCAL_TEXT[name]
    rollout = _Rollout()
    seen_accepted, seen_local = [], []
    for transitions, skill in chunks:
        if skill is not None:
            seen_accepted.append(rollout.send("skill", {"prompt": skill}))
        if transitions:
            rollout.send("camera", {"mode": "state", "transitions": [{"actions": sorted(t)} for t in transitions]})
        seen_local.append(rollout.next_chunk())

    assert seen_accepted == accepted
    assert seen_local == [_expected_text(label) for label in labels]
    changes = sum(1 for before, after in zip([""] + seen_local, seen_local) if before != after)
    local_encodes = [prompt for prompt, length in rollout.pipeline.encoded if length == 128]
    assert len(local_encodes) == 1 + changes
    assert rollout.pipeline.cache_calls.count(True) == changes


def test_request_prompt_fills_the_global_row_and_an_empty_local_row() -> None:
    pipeline = _IrisPipeline()

    embeds = pipeline.encode_prompt("  A sunny canyon. ", max_sequence_length=512, dtype=torch.float32)

    assert pipeline.encoded == [("  A sunny canyon. ", 384), ("", 128)]
    assert embeds.shape == (1, 512, _D)
    assert embeds[:, :384].eq(1).all() and embeds[:, 384:].eq(2).all()


def test_prompt_switch_reencodes_only_the_global_row() -> None:
    rollout = _Rollout()
    assert rollout.send("camera", {"mode": "state", "transitions": [{"actions": ["ctrl"]}]})
    rollout.next_chunk()
    local_row = rollout.state.prompt_embeds[:, 384:].clone()
    encoded = len(rollout.pipeline.encoded)

    assert rollout.send("prompt", {"prompt": " The canyon floods at dusk. "})
    local = rollout.next_chunk()

    assert local == IRIS_ACTION_CAPTIONS["ctrl"]
    assert rollout.pipeline.encoded[encoded:] == [("The canyon floods at dusk.", 384)]
    assert rollout.state.prompt_embeds[:, :384].eq(len(rollout.pipeline.encoded)).all()
    assert torch.equal(rollout.state.prompt_embeds[:, 384:], local_row)
    assert rollout.pipeline.cache_calls[-1] is True
    assert rollout.meta.completed_event_ids == [f"e{rollout.events}"]

    rollout.next_chunk()
    assert rollout.pipeline.cache_calls[-1] is False
    assert len(rollout.pipeline.encoded) == encoded + 1


def test_switching_back_to_the_request_prompt_is_a_change() -> None:
    rollout = _Rollout("A")
    rollout.send("prompt", {"prompt": "B"})
    rollout.next_chunk()
    rollout.send("prompt", {"prompt": "A"})
    rollout.next_chunk()

    assert [prompt for prompt, length in rollout.pipeline.encoded if length == 384] == ["A", "B", "A"]


def test_text_action_keys_never_reach_the_action_latent() -> None:
    rollout = _Rollout()
    rollout.send("camera", {"mode": "state", "transitions": [{"actions": ["W", "Shift"]}]})
    rollout.next_chunk()

    camera = rollout.state.interaction_sessions["camera"]
    assert rollout.pipeline.actions[-1] == (("w",), ("w",), ("w",))
    assert (camera.text_actions_held, camera.text_actions_tapped) == (_keys("shift"), _keys("shift"))
    assert rollout.state.extra["iris_text"].local_prompt == IRIS_ACTION_CAPTIONS["shift"]


def test_skill_acknowledgement_and_pending_rejection() -> None:
    rollout = _Rollout()
    assert rollout.send("skill", {"prompt": "  Draw the sword.  "})
    assert not rollout.send("skill", {"prompt": "Wave."})

    assert rollout.next_chunk() == "Draw the sword."
    assert rollout.meta.started_event_ids == rollout.meta.completed_event_ids == ["e1"]


def test_skill_and_prompt_updates_on_one_event_apply_together() -> None:
    rollout = _Rollout()
    rollout.clock += 1.0
    rollout.coordinator.enqueue_parts(
        rollout.state,
        parts=[("prompt", {"prompt": "Night falls."}), ("skill", {"prompt": "Light a torch."})],
        event_id="both",
        received_at=rollout.clock,
        transition_chunks=None,
    )

    assert rollout.next_chunk() == "Light a torch."
    assert rollout.state.extra["iris_text"].global_prompt == "Night falls."
    assert rollout.meta.completed_event_ids == ["both"]
    assert rollout.pipeline.cache_calls.count(True) == 1


@pytest.mark.parametrize(
    ("modality", "payload", "match"),
    [
        ("prompt", {"prompt": " "}, "nonempty prompt"),
        ("prompt", {"prompt": "p", "skill": "s"}, "nonempty prompt"),
        ("skill", {"skill": "s"}, "nonempty prompt"),
        ("skill", {"prompt": 3}, "nonempty prompt"),
    ],
)
def test_invalid_text_updates_are_rejected(modality, payload, match) -> None:
    handler = {"prompt": ABotPromptInteractionHandler, "skill": ABotSkillInteractionHandler}[modality](iris=True)
    state = StepRequestState(request_id="r", sampling=OmniDiffusionSamplingParams())
    with pytest.raises(ValueError, match=match):
        handler.enqueue(state, event_id="e", received_at=1.0, payload=payload, transition_chunks=None)
    assert modality not in state.interaction_sessions


@pytest.mark.parametrize(
    ("handler", "modality", "payload", "match"),
    [
        (ABotPromptInteractionHandler(), "prompt", {"prompt": "p"}, "Iris checkpoint"),
        (ABotSkillInteractionHandler(), "skill", {"prompt": "p"}, "Iris checkpoint"),
        (
            ABotCameraInteractionHandler(),
            "camera",
            {"mode": "state", "transitions": [{"actions": ["ctrl"]}]},
            "W/A/S/D/I/J/K/L action keys",
        ),
    ],
)
def test_abot_world_0_rejects_text_interaction(handler, modality, payload, match) -> None:
    state = StepRequestState(request_id="r", sampling=OmniDiffusionSamplingParams())
    with pytest.raises(ValueError, match=match):
        handler.validate_payload(state, event_id="e", payload=payload, transition_chunks=None)


def test_text_action_keys_are_state_mode_only() -> None:
    handler = ABotCameraInteractionHandler(iris=True)
    state = StepRequestState(request_id="r", sampling=OmniDiffusionSamplingParams())
    with pytest.raises(ValueError, match="W/A/S/D/I/J/K/L action keys"):
        handler.validate_payload(
            state, event_id="e", payload={"mode": "script", "frames": [["space"]]}, transition_chunks=None
        )


def test_handlers_bind_the_iris_layout_from_the_pipeline() -> None:
    coordinator = InteractionCoordinator.build(
        _IrisPipeline(), OmniDiffusionConfig(model_class_name="ABotWorldCausalPipeline")
    )
    state = StepRequestState(request_id="r", sampling=OmniDiffusionSamplingParams())
    for modality, payload in (
        ("prompt", {"prompt": "p"}),
        ("skill", {"prompt": "s"}),
        ("camera", {"mode": "state", "transitions": [{"actions": ["MouseLeft"]}]}),
    ):
        coordinator.get_handler(modality).validate_payload(state, event_id="e", payload=payload, transition_chunks=None)


def test_abot_world_0_never_reencodes_text() -> None:
    pipeline = _IrisPipeline()
    pipeline.transformer.config.use_relative_rope = False
    state = _iris_request(pipeline)
    del state.extra["iris_text"]
    state.prompt_embeds = torch.zeros(1, 512, _D)
    pipeline.encoded.clear()

    pipeline.prepare_next_chunk(state)

    assert pipeline.encoded == [] and pipeline.cache_calls == [False]


def test_action_captions_replay_a_held_and_tapped_schedule() -> None:
    captions = IrisActionCaptions()
    steps = [
        (_keys("shift"), _keys("shift")),
        (_keys("shift"), _keys("space")),
        (_keys("shift"), _keys()),
        (_keys(), _keys("alt")),
        (_keys(), _keys()),
        (_keys(), _keys()),
    ]

    actions = [captions.step(held, tapped)[0] for held, tapped in steps]

    assert actions == [("shift",), ("shift", "space"), ("shift",), ("alt",), ("alt",), ()]
