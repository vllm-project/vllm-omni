# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU unit tests for the π0.5 VLA model.

Deliberately lightweight: config plumbing and the checkpoint-boundary rule, the
state discretization path, relative actions, and the weight-load remap. Runs on
CPU with no weights and no lerobot. The full LeRobot parity check lives in
``tests/diffusion/models/pi05/test_pi05_parity.py``.

Most assertions here guard *silent* failure modes — cases where the wrong
behaviour still produces a well-shaped, finite action chunk:

  * normalization must run before state discretization;
  * quantile (``q01``/``q99``) norm stats must be recognized, since π0.5 defaults
    to them where π0 defaults to ``mean_std``;
  * relative-action checkpoints must be honoured, and an unresolvable
    ``relative_exclude_joints`` must raise rather than silently make every
    dimension relative;
  * MEM / RTC checkpoints must be rejected rather than served incorrectly.

    pytest tests/diffusion/models/pi05/test_pi05_units.py -v
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.pi05 import modeling_pi05
from vllm_omni.diffusion.models.pi05.config import (
    Pi05Config,
    UnsupportedCheckpointCapabilityError,
    load_lerobot_norm_stats,
    resolve_excluded_action_indices,
)
from vllm_omni.diffusion.models.pi05.modeling_pi05 import (
    OPENPI_ATTENTION_MASK_VALUE,
    GemmaVariantConfig,
    Pi05AdaRMSNorm,
    Pi05ForActionPrediction,
    _apply_norm,
    _build_norm_buffers,
    create_sinusoidal_pos_embedding,
    get_gemma_config,
    make_att_2d_masks,
    prepare_attention_masks_4d,
)
from vllm_omni.diffusion.models.pi05.processor_pi05 import (
    PI05_MAX_TOKEN_LEN,
    Pi05ImageProcessor,
    Pi05RelativeActions,
    build_model_inputs,
    build_pi05_prompt,
    discretize_state,
    normalize_state,
    prefix_token_budget,
    resize_with_pad,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ----------------------------------------------------------------------------
# Pi05Config dataclass + checkpoint resolver
# ----------------------------------------------------------------------------
_LEROBOT_CFG: dict[str, Any] = {
    "type": "pi05",
    "paligemma_variant": "gemma_2b",
    "action_expert_variant": "gemma_300m",
    "chunk_size": 50,
    "max_action_dim": 32,
    "max_state_dim": 32,
    "num_inference_steps": 10,
    "image_resolution": [224, 224],
    "tokenizer_max_length": 200,
    "dtype": "float32",
    "input_features": {
        "observation.images.base_0_rgb": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.images.left_wrist_0_rgb": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.images.right_wrist_0_rgb": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.state": {"type": "STATE", "shape": [32]},
    },
    "output_features": {"action": {"type": "ACTION", "shape": [32]}},
}

_EXPECTED_CAMERA_ORDER = [
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
]

_ACTION_NAMES = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]


def test_config_parses_lerobot_pi05_json():
    """``_LEROBOT_CFG`` mirrors the runtime keys of a real checkpoint."""
    c = Pi05Config.from_model_config(_LEROBOT_CFG)
    assert c.tokenizer_max_length == 200
    assert c.state_num_bins == 256
    assert c.image_feature_keys == _EXPECTED_CAMERA_ORDER
    assert c.image_resolution == (224, 224)


def test_config_tokenizer_length_differs_from_pi0():
    """π0 pads text to 48 tokens; π0.5 pads to 200 because the prompt also
    carries the serialized state."""
    assert Pi05Config().tokenizer_max_length == 200
    assert PI05_MAX_TOKEN_LEN == 200


def test_config_rejects_wrong_model_type():
    with pytest.raises(ValueError, match="type='pi05'"):
        Pi05Config.from_model_config(dict(_LEROBOT_CFG, type="pi0"))


def test_config_rejects_unsupported_partial_action_chunk():
    with pytest.raises(UnsupportedCheckpointCapabilityError, match="n_action_steps"):
        Pi05Config.from_model_config(dict(_LEROBOT_CFG, n_action_steps=10))


def test_config_derives_real_action_dim_from_output_features():
    config = Pi05Config.from_model_config(
        dict(_LEROBOT_CFG, output_features={"action": {"type": "ACTION", "shape": [7]}})
    )
    assert config.action_dim == 7
    assert config.max_action_dim == 32


def test_config_rejects_action_schema_wider_than_model():
    with pytest.raises(ValueError, match="exceeds max_action_dim"):
        Pi05Config.from_model_config(dict(_LEROBOT_CFG, output_features={"action": {"type": "ACTION", "shape": [33]}}))


def test_config_derives_real_state_dim_from_input_features():
    """The state width decides how many values are serialized into the prompt,
    so a 7-joint checkpoint must not be served a 32-value state prompt."""
    features: dict[str, Any] = dict(_LEROBOT_CFG["input_features"])
    features["observation.state"] = {"type": "STATE", "shape": [7]}
    config = Pi05Config.from_model_config(dict(_LEROBOT_CFG, input_features=features))
    assert config.state_dim == 7
    assert config.max_state_dim == 32


def test_config_state_dim_falls_back_to_max_state_dim():
    """LeRobot's ``validate_features`` fills in a ``max_state_dim``-wide state
    feature when the checkpoint declares none; match that."""
    features = {k: v for k, v in _LEROBOT_CFG["input_features"].items() if k != "observation.state"}
    assert Pi05Config.from_model_config(dict(_LEROBOT_CFG, input_features=features)).state_dim == 32


def test_config_rejects_state_schema_wider_than_model():
    features: dict[str, Any] = dict(_LEROBOT_CFG["input_features"])
    features["observation.state"] = {"type": "STATE", "shape": [33]}
    with pytest.raises(ValueError, match="exceeds max_state_dim"):
        Pi05Config.from_model_config(dict(_LEROBOT_CFG, input_features=features))


def test_config_rejects_inconsistent_openpi_metadata():
    with pytest.raises(ValueError, match="policy_server_config.action_dim"):
        Pi05Config.from_model_config(
            dict(
                _LEROBOT_CFG,
                output_features={"action": {"type": "ACTION", "shape": [7]}},
                policy_server_config={"action_dim": 32},
            )
        )


def test_config_non_square_resolution_raises():
    with pytest.raises(ValueError):
        Pi05Config(image_resolution=(224, 256))


# ----------------------------------------------------------------------------
# LeRobot normalization-stats sidecar
# ----------------------------------------------------------------------------
# LeRobot keeps stats out of config.json: policy_preprocessor.json holds the
# structure and one safetensors file per stateful step holds the numbers.
def _write_lerobot_checkpoint(
    tmp_path,
    *,
    norm_map: dict[str, str] | None = None,
    stats: dict[str, dict[str, list[float]]] | None = None,
    write_state_file: bool = True,
) -> str:
    """Write a minimal LeRobot-shaped checkpoint dir; return its path."""
    import json as _json

    import safetensors.torch as _st

    tmp_path.mkdir(parents=True, exist_ok=True)
    state_file = "policy_preprocessor_step_3_normalizer_processor.safetensors"
    step: dict = {
        "registry_name": "normalizer_processor",
        "config": {
            "eps": 1e-08,
            "features": {},
            "norm_map": norm_map if norm_map is not None else {"STATE": "QUANTILES", "ACTION": "QUANTILES"},
        },
    }
    if write_state_file:
        step["state_file"] = state_file
        flat = {
            f"{feature}.{stat}": torch.tensor(values, dtype=torch.float32)
            for feature, entry in (stats or {}).items()
            for stat, values in entry.items()
        }
        _st.save_file(flat, str(tmp_path / state_file))

    (tmp_path / "policy_preprocessor.json").write_text(
        _json.dumps({"name": "policy_preprocessor", "steps": [step]}), encoding="utf-8"
    )
    (tmp_path / "config.json").write_text(_json.dumps(_LEROBOT_CFG), encoding="utf-8")
    return str(tmp_path)


# A LeRobot state_dict carries every stat compute_stats emits, not just the two
# the norm_map selects.
_FULL_DATASET_STATS = {
    "observation.state": {
        "mean": [10.0, 20.0],
        "std": [1.0, 2.0],
        "min": [-5.0, -5.0],
        "max": [5.0, 5.0],
        "q01": [-1.0, -2.0],
        "q99": [1.0, 2.0],
    },
    "action": {
        "mean": [0.5, 0.5],
        "std": [0.1, 0.1],
        "min": [-1.0, -1.0],
        "max": [1.0, 1.0],
        "q01": [-0.25, -0.5],
        "q99": [0.25, 0.5],
    },
}


@pytest.mark.parametrize(
    "declared,expected",
    [
        ("QUANTILES", {"mode": "quantile", "q01": [-1.0, -2.0], "q99": [1.0, 2.0]}),
        ("MEAN_STD", {"mode": "mean_std", "mean": [10.0, 20.0], "std": [1.0, 2.0]}),
        ("IDENTITY", None),
    ],
)
def test_load_lerobot_norm_stats_takes_the_mode_from_norm_map(tmp_path, declared, expected):
    """The regression guard. ``_FULL_DATASET_STATS`` carries all six statistics,
    as a real state_dict does, so an implementation that sniffed the present keys
    would answer mean_std for every case here. Only norm_map distinguishes them.
    """
    path = _write_lerobot_checkpoint(
        tmp_path, norm_map={"STATE": declared, "ACTION": "QUANTILES"}, stats=_FULL_DATASET_STATS
    )
    stats = load_lerobot_norm_stats(path)

    if expected is None:
        assert "state" not in stats
    else:
        assert stats["state"] == expected
    assert stats["action"]["mode"] == "quantile"


@pytest.mark.parametrize("shape", ["no_sidecar", "no_state_file"])
def test_load_lerobot_norm_stats_absent_stats_are_none(tmp_path, shape):
    """``lerobot/pi05_base`` is the second shape: a normalizer step with no
    stats attached. Neither is an error — the client is then expected to send an
    already-normalized state."""
    if shape == "no_sidecar":
        path = str(tmp_path)
    else:
        path = _write_lerobot_checkpoint(tmp_path, write_state_file=False)

    assert load_lerobot_norm_stats(path) is None


def test_load_lerobot_norm_stats_missing_declared_state_file_raises(tmp_path):
    path = _write_lerobot_checkpoint(tmp_path, stats=_FULL_DATASET_STATS)
    (tmp_path / "policy_preprocessor_step_3_normalizer_processor.safetensors").unlink()

    with pytest.raises(FileNotFoundError, match="normalizer state"):
        load_lerobot_norm_stats(path)


def test_pipeline_rejects_missing_model_weights_before_initialization(tmp_path):
    from vllm_omni.diffusion.models.pi05.pipeline_pi05 import Pi05Pipeline

    pipeline = object.__new__(Pi05Pipeline)
    pipeline.model_dir = str(tmp_path)

    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        pipeline._initialize_model()


class _WideActionModel:
    """Stands in for the 3.6B model, which is the one collaborator here that is
    genuinely expensive to build. It returns the padded ``max_action_dim`` width
    the real model emits, so the assertion is about the pipeline's cropping."""

    def __init__(self, chunk_size: int, max_action_dim: int):
        self._shape = (1, chunk_size, max_action_dim)

    def sample_actions(self, **kwargs):
        del kwargs
        return torch.ones(self._shape)

    def _unnormalize_actions(self, actions):
        return actions


def test_pipeline_crops_actions_to_checkpoint_output_schema(monkeypatch):
    from vllm_omni.diffusion.models.pi05 import pipeline_pi05
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    config = Pi05Config.from_model_config(
        dict(_LEROBOT_CFG, output_features={"action": {"type": "ACTION", "shape": [7]}})
    )
    pipeline = object.__new__(pipeline_pi05.Pi05Pipeline)
    pipeline.config = config
    pipeline.tokenizer = object()
    pipeline._device = torch.device("cpu")
    pipeline.relative_actions = Pi05RelativeActions(
        enabled=False, exclude_joints=[], action_names=None, max_action_dim=config.max_action_dim
    )
    pipeline.model = _WideActionModel(config.chunk_size, config.max_action_dim)
    monkeypatch.setattr(
        pipeline_pi05,
        "build_model_inputs",
        lambda *args: ([torch.empty(0)], [torch.empty(0)], torch.empty(0), torch.empty(0)),
    )
    request = OmniDiffusionRequest(
        prompt="",
        sampling_params=OmniDiffusionSamplingParams(
            extra_args={"robot_obs": {"state": np.zeros(config.max_state_dim)}, "num_inference_steps": 2},
        ),
        request_id="crop-test",
    )

    result = pipeline.forward(request)

    assert result.output["actions"].shape == (config.chunk_size, 7)


def test_load_lerobot_norm_stats_unknown_mode_raises(tmp_path):
    """Fail loud: an unreproducible mode served anyway is silently wrong."""
    path = _write_lerobot_checkpoint(
        tmp_path, norm_map={"STATE": "SOMETHING_NEW", "ACTION": "QUANTILES"}, stats=_FULL_DATASET_STATS
    )
    with pytest.raises(ValueError, match="SOMETHING_NEW"):
        load_lerobot_norm_stats(path)


def test_load_lerobot_norm_stats_missing_declared_stat_raises(tmp_path):
    """norm_map promises QUANTILES but the state_dict has no q01/q99."""
    path = _write_lerobot_checkpoint(
        tmp_path, stats={"observation.state": {"mean": [0.0], "std": [1.0]}, "action": {"q01": [0.0], "q99": [1.0]}}
    )
    with pytest.raises(ValueError, match="q01"):
        load_lerobot_norm_stats(path)


def test_from_pretrained_backfills_sidecar_stats(tmp_path):
    """End to end: the stats reach state_norm_stats without any deploy yaml."""
    path = _write_lerobot_checkpoint(tmp_path, stats=_FULL_DATASET_STATS)
    config = Pi05Config.from_pretrained(path)

    assert config.norm_stats["state"]["mode"] == "quantile"
    assert config.state_norm_stats == config.norm_stats["state"]


def test_from_pretrained_does_not_override_explicit_norm_stats(tmp_path):
    """config.json wins over the sidecar, matching LeRobot's override rule."""
    import json as _json

    path = _write_lerobot_checkpoint(tmp_path, stats=_FULL_DATASET_STATS)
    explicit = {"state": {"mode": "min_max", "min": [0.0], "max": [1.0]}}
    (tmp_path / "config.json").write_text(_json.dumps({**_LEROBOT_CFG, "norm_stats": explicit}), encoding="utf-8")

    assert Pi05Config.from_pretrained(path).norm_stats == explicit


def test_sidecar_quantile_stats_reach_the_prompt(tmp_path):
    """The whole point: sidecar stats must change the discretized prompt bins."""
    path = _write_lerobot_checkpoint(
        tmp_path,
        stats={"observation.state": {"q01": [-1.0, -1.0], "q99": [1.0, 1.0]}, "action": _FULL_DATASET_STATS["action"]},
    )
    config = Pi05Config.from_pretrained(path)
    raw = np.array([1.0, -1.0], dtype=np.float32)

    with_stats = discretize_state(normalize_state(raw, state_dim=2, state_norm_stats=config.state_norm_stats))
    without = discretize_state(normalize_state(raw, state_dim=2, state_norm_stats=None))

    # q01=-1/q99=1 maps [-1, 1] onto itself, so this is identical to the
    # pass-through path: the top saturates at 255 and -1.0 sits exactly on the
    # first bin edge (underflow to -1 needs a value strictly below q01).
    assert with_stats.tolist() == without.tolist() == [255, 0]

    shifted = Pi05Config.from_pretrained(
        _write_lerobot_checkpoint(
            tmp_path / "shifted",
            stats={
                "observation.state": {"q01": [0.0, 0.0], "q99": [4.0, 4.0]},
                "action": _FULL_DATASET_STATS["action"],
            },
        )
    )
    moved = discretize_state(normalize_state(raw, state_dim=2, state_norm_stats=shifted.state_norm_stats))
    assert moved.tolist() != without.tolist(), "different quantiles must move the prompt bins"


# ----------------------------------------------------------------------------
# Checkpoint boundary: declared-but-unconsumed capabilities must raise
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "key,value",
    [
        ("use_visual_memory", True),
        ("use_proprioceptive_memory", True),
        ("rtc_config", {"mode": "trained", "inference_delay": 2}),
        ("n_obs_steps", 6),
    ],
)
def test_config_rejects_unsupported_capability(key, value):
    """MEM and RTC change what a correct action chunk looks like and are not
    visible in the weights. Serving such a checkpoint must fail loudly."""
    with pytest.raises(UnsupportedCheckpointCapabilityError, match=key):
        Pi05Config.from_model_config(dict(_LEROBOT_CFG, **{key: value}))


def test_config_rejects_empty_rtc_config():
    """``RTCConfig`` defaults ``enabled=True``, so ``rtc_config={}`` selects RTC
    with LeRobot's defaults rather than turning it off."""
    with pytest.raises(UnsupportedCheckpointCapabilityError, match="rtc_config"):
        Pi05Config.from_model_config(dict(_LEROBOT_CFG, rtc_config={}))


@pytest.mark.parametrize("key,value", [("rtc_config", None), ("n_obs_steps", 1), ("empty_cameras", 0)])
def test_config_accepts_capability_at_its_off_value(key, value):
    """The off value differs per capability: ``None`` for a config mapping, 1
    for an observation count, 0 for a camera count."""
    assert Pi05Config.from_model_config(dict(_LEROBOT_CFG, **{key: value})).tokenizer_max_length == 200


def test_config_accepts_rtc_training_max_delay():
    """``rtc_training_max_delay`` describes how the checkpoint was *trained*
    (clean action prefixes were sampled), not a request to run RTC at inference.
    Such a checkpoint is still correct to serve without RTC, so it must not be
    rejected — unlike a populated ``rtc_config``."""
    c = Pi05Config.from_model_config(dict(_LEROBOT_CFG, rtc_training_max_delay=10))
    assert c.tokenizer_max_length == 200


# ----------------------------------------------------------------------------
# Relative actions
# ----------------------------------------------------------------------------
def test_relative_actions_without_action_names_raises():
    """``action_feature_names`` is populated from dataset metadata at training
    time. There is no dataset at serving time, so a relative-action checkpoint
    that omits it cannot resolve ``relative_exclude_joints`` and must fail."""
    with pytest.raises(UnsupportedCheckpointCapabilityError, match="action_feature_names"):
        Pi05Config.from_model_config(dict(_LEROBOT_CFG, use_relative_actions=True, action_feature_names=None))


def test_relative_actions_with_action_names_accepted():
    c = Pi05Config.from_model_config(dict(_LEROBOT_CFG, use_relative_actions=True, action_feature_names=_ACTION_NAMES))
    assert c.use_relative_actions
    assert c.relative_exclude_joints == ["gripper"]


def test_relative_actions_empty_exclude_list_needs_no_action_names():
    c = Pi05Config.from_model_config(dict(_LEROBOT_CFG, use_relative_actions=True, relative_exclude_joints=[]))
    assert c.use_relative_actions


def test_resolve_excluded_action_indices_exact_match():
    assert resolve_excluded_action_indices(["gripper"], _ACTION_NAMES) == [6]


def test_resolve_excluded_action_indices_substring_match():
    """A checkpoint may spell the dimension ``gripper_position`` while the
    config just says ``gripper``."""
    assert resolve_excluded_action_indices(["gripper"], ["joint_0", "gripper_position"]) == [1]


def test_resolve_excluded_action_indices_unresolvable_raises():
    with pytest.raises(UnsupportedCheckpointCapabilityError):
        resolve_excluded_action_indices(["no_such_joint"], _ACTION_NAMES)


def _relative_step(enabled=True):
    return Pi05RelativeActions(
        enabled=enabled,
        exclude_joints=["gripper"],
        action_names=_ACTION_NAMES,
        max_action_dim=32,
    )


def test_relative_mask_excludes_gripper_and_padding():
    rel = _relative_step()
    assert rel.relative_mask[0]
    assert not rel.relative_mask[6], "gripper must stay absolute"
    assert not rel.relative_mask[len(_ACTION_NAMES) :].any(), "padded dims carry no signal"
    assert rel.num_relative_dims == 6


def test_relative_absolute_round_trip_is_exact():
    """The two directions are one object precisely so they cannot disagree."""
    rel = _relative_step()
    g = torch.Generator().manual_seed(0)
    actions = torch.randn(1, 50, 32, generator=g)
    state = torch.randn(32, generator=g)
    assert torch.allclose(rel.to_absolute(rel.to_relative(actions, state), state), actions, atol=1e-6)


def test_relative_transform_shifts_only_included_dims():
    rel = _relative_step()
    actions = torch.zeros(1, 4, 32)
    state = torch.arange(32, dtype=torch.float32)
    out = rel.to_absolute(actions, state)
    assert torch.allclose(out[..., 0], state[0].expand(1, 4))
    assert torch.allclose(out[..., 6], torch.zeros(1, 4)), "gripper must not be shifted"


def test_relative_disabled_is_identity():
    rel = _relative_step(enabled=False)
    actions = torch.randn(1, 4, 32)
    assert torch.equal(rel.to_absolute(actions, torch.randn(32)), actions)


def test_relative_rejects_mismatched_action_dim():
    rel = _relative_step()
    with pytest.raises(ValueError, match="max_action_dim"):
        rel.to_absolute(torch.zeros(1, 4, 8), torch.zeros(32))


# ----------------------------------------------------------------------------
# State discretization — π0.5's defining input path
# ----------------------------------------------------------------------------
def test_discretize_spans_the_full_bin_range():
    out = discretize_state(np.array([-1.0, 0.0, 1.0], dtype=np.float32), num_bins=256)
    assert out.tolist() == [0, 128, 255]


def test_discretize_underflows_to_minus_one_and_saturates_at_the_top():
    """Out-of-range input is asymmetric, and that asymmetry is LeRobot's.

    ``np.digitize`` returns 0 for anything below the first bin edge, so the
    ``- 1`` makes a below-range state land in bin ``-1``; above-range input
    saturates at ``num_bins - 1`` on its own. The checkpoint was trained with
    ``" -1"`` in the state prompt for under-range dims, so clipping that to 0
    would change the tokens the model sees — LeRobot parity catches it.
    """
    out = discretize_state(np.array([-9.0, 9.0], dtype=np.float32), num_bins=256)
    assert out.tolist() == [-1, 255]


@pytest.mark.parametrize(
    "stats",
    [
        {"mean": [5.0] * 32, "std": [1.0] * 32},
        {"min": [0.0] * 32, "max": [10.0] * 32},
        {"q01": [0.0] * 32, "q99": [10.0] * 32},
    ],
)
def test_normalize_state_supports_every_mode(stats):
    """π0.5 defaults STATE/ACTION to QUANTILES where π0 uses MEAN_STD, so the
    ``q01``/``q99`` schema must be recognized. An unrecognized entry would fall
    back to identity and wrongly bin every state dimension without raising."""
    out = normalize_state([5.0] * 32, state_dim=32, state_norm_stats=stats)
    assert np.allclose(out, 0.0, atol=1e-6)


def test_normalize_state_without_stats_passes_through():
    """LeRobot's NormalizeProcessor returns the tensor unchanged when stats are
    missing, so a client that normalizes its own state must not be re-scaled."""
    out = normalize_state([5.0, -5.0] + [0.0] * 30, state_dim=32, state_norm_stats=None)
    assert out[0] == 5.0 and out[1] == -5.0


def test_normalize_state_does_not_clip_out_of_range():
    """The parity-critical contract: normalization is affine and nothing else.

    Clamping to [-1, 1] here would hide an under-range dimension from
    ``discretize_state``, turning LeRobot's ``-1`` bin into ``0`` and changing
    the state tokens in the prompt. LeRobot applies
    ``2.0 * (x - q01) / (q99 - q01) - 1.0`` with no clip.
    """
    stats = {"q01": [0.0] * 32, "q99": [10.0] * 32}
    out = normalize_state([-10.0, 20.0] + [5.0] * 30, state_dim=32, state_norm_stats=stats)
    assert out[0] == pytest.approx(-3.0)  # would be -1.0 if clipped
    assert out[1] == pytest.approx(3.0)  # would be 1.0 if clipped
    # and the under-range dim must reach the -1 bin end to end
    assert discretize_state(out, num_bins=256)[0] == -1


def test_normalize_state_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_state([0.0] * 32, state_dim=32, state_norm_stats={"mode": "bogus"})


def test_normalization_must_precede_discretization():
    """The ordering constraint from LeRobot's pipeline, as an executable check.

    ``Pi05PrepareStateTokenizerProcessorStep`` bins over [-1, 1] and assumes the
    normalizer already ran. Reversed, every bin index is wrong and nothing
    raises — so assert the two orders actually differ.
    """
    stats = {"q01": [0.0] * 32, "q99": [10.0] * 32}
    raw = [5.0] * 32
    correct = discretize_state(normalize_state(raw, state_dim=32, state_norm_stats=stats))
    skipped = discretize_state(np.asarray(raw, dtype=np.float32))
    assert correct.tolist() != skipped.tolist()
    assert correct[0] == 128, "mid-range state should land mid-range"
    assert skipped[0] == 255, "unnormalized state saturates the top bin"


def test_prompt_serializes_the_declared_state_width():
    """LeRobot's tokenizer step discretizes the state at its real width and
    never pads to ``max_state_dim``."""
    prompt = build_pi05_prompt(task="x", state=[0.0] * 7, state_dim=7, state_norm_stats=None)
    bins = prompt.split("State: ")[1].split(";")[0].split()
    assert len(bins) == 7


def test_prompt_rejects_a_state_of_the_wrong_width():
    """Padding or truncating here would change the prompt tokens instead of
    reporting a misconfigured client."""
    with pytest.raises(ValueError, match="dimension"):
        build_pi05_prompt(task="x", state=[0.0, 0.0], state_dim=32, state_norm_stats=None)


def test_prompt_template_matches_lerobot():
    prompt = build_pi05_prompt(
        task="  pick_up the red\nblock ",
        state=[0.0] * 4,
        state_dim=4,
        state_norm_stats=None,
    )
    assert prompt.startswith("Task: pick up the red block, State: ")
    assert prompt.endswith(";\nAction: ")


# ----------------------------------------------------------------------------
# Input contract: 1..3 views x 256 tokens + a constant 200 text tokens
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("views,expected", [(1, 456), (2, 712), (3, 968)])
def test_prefix_token_budget(views, expected):
    cfg = Pi05Config(max_cameras=views, tokenizer_max_length=200)
    budget = prefix_token_budget(cfg, num_real_cameras=views)
    assert budget["image_tokens"] == 256 * views
    assert budget["text_tokens"] == 200
    assert budget["total_prefix_len"] == expected


# ----------------------------------------------------------------------------
# Shared math helpers (same contracts as π0)
# ----------------------------------------------------------------------------
def test_gemma_variant_dims():
    vlm = get_gemma_config("gemma_2b")
    assert (vlm.width, vlm.depth, vlm.mlp_dim, vlm.num_kv_heads, vlm.head_dim) == (2048, 18, 16384, 1, 256)
    expert = get_gemma_config("gemma_300m")
    assert (expert.width, expert.depth, expert.mlp_dim) == (1024, 18, 4096)


def test_gemma_unknown_variant_raises():
    with pytest.raises(ValueError):
        get_gemma_config("gemma_7b")


def test_sinusoidal_embedding_shape_and_oddity():
    out = create_sinusoidal_pos_embedding(torch.tensor([0.0, 1.0]), 64)
    assert out.shape == (2, 64)
    with pytest.raises(ValueError):
        create_sinusoidal_pos_embedding(torch.tensor([0.0]), 63)


def test_prepare_attention_masks_4d():
    mask = torch.tensor([[[True, False]]])
    out = prepare_attention_masks_4d(mask)
    assert out.shape == (1, 1, 1, 2)
    assert out[0, 0, 0, 0] == 0.0
    assert out[0, 0, 0, 1] == OPENPI_ATTENTION_MASK_VALUE


def test_make_att_2d_masks_respects_padding():
    pad = torch.tensor([[True, True, False]])
    att = torch.tensor([[0, 0, 0]])
    out = make_att_2d_masks(pad, att)
    assert not out[0, :, 2].any()


def test_build_norm_buffers_recognizes_quantile_stats():
    """A π0.5 checkpoint typically ships q01/q99. Returning None here would
    silently leave actions in normalized space."""
    stats = _build_norm_buffers({"action": {"q01": [0.0], "q99": [2.0]}}, "action")
    assert stats is not None
    assert torch.allclose(stats["min"], torch.tensor([0.0]))
    assert torch.allclose(stats["max"], torch.tensor([2.0]))


def test_apply_norm_quantile_round_trip_and_padded_tail():
    stats = _build_norm_buffers({"action": {"q01": [0.0, 0.0], "q99": [2.0, 4.0]}}, "action")
    x = torch.tensor([[0.5, -0.5, 7.0]])  # third entry is padding beyond the stats
    back = _apply_norm(_apply_norm(x, stats, inverse=False), stats, inverse=True)
    assert torch.allclose(back, x, atol=1e-5)
    assert back[0, 2] == 7.0, "padded tail must pass through untouched"


# ----------------------------------------------------------------------------
# Image processor
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("value,expected", [(0.0, -1.0), (0.25, -0.5), (1.0, 1.0)])
@pytest.mark.parametrize("container", ["numpy", "torch"])
def test_float_image_uses_lerobot_zero_one_domain(value, expected, container):
    if container == "numpy":
        image = np.full((4, 4, 3), value, dtype=np.float32)
    else:
        image = torch.full((3, 4, 4), value, dtype=torch.float32)

    out = Pi05ImageProcessor(image_size=4).preprocess_single(image)

    assert out.shape == (1, 3, 4, 4)
    assert torch.allclose(out, torch.full_like(out, expected))


@pytest.mark.parametrize("bad", [-0.01, 1.01, np.nan, np.inf])
@pytest.mark.parametrize("container", ["numpy", "torch"])
def test_float_image_rejects_values_outside_lerobot_domain(bad, container):
    if container == "numpy":
        image = np.full((4, 4, 3), bad, dtype=np.float32)
    else:
        image = torch.full((3, 4, 4), bad, dtype=torch.float32)

    with pytest.raises(ValueError):
        Pi05ImageProcessor(image_size=4).preprocess_single(image)


def test_uint8_image_uses_zero_to_255_domain():
    image = np.full((4, 4, 3), 64, dtype=np.uint8)
    out = Pi05ImageProcessor(image_size=4).preprocess_single(image)
    assert torch.allclose(out, torch.full_like(out, 64.0 / 255.0 * 2.0 - 1.0))


def test_resize_with_pad_pads_with_minus_one():
    out = resize_with_pad(torch.zeros(1, 3, 100, 200), 224, 224)
    assert out.shape == (1, 3, 224, 224)
    assert out[0, 0, 0, 0] == -1.0


class _FakeTokenizer:
    def __call__(self, text, **kwargs):
        del text
        length = kwargs["max_length"]
        return {"input_ids": [0] * length, "attention_mask": [1] * length}


def test_build_model_inputs_compacts_missing_middle_camera():
    config = Pi05Config(image_feature_keys=_EXPECTED_CAMERA_ORDER, max_cameras=3)
    observation = {
        _EXPECTED_CAMERA_ORDER[0]: np.zeros((4, 4, 3), dtype=np.uint8),
        _EXPECTED_CAMERA_ORDER[2]: np.full((4, 4, 3), 255, dtype=np.uint8),
        "state": np.zeros(32, dtype=np.float32),
    }

    images, masks, _, _ = build_model_inputs(observation, config, _FakeTokenizer(), torch.device("cpu"))

    assert [bool(mask.item()) for mask in masks] == [True, True, False]
    assert torch.all(images[0] == -1.0)
    assert torch.all(images[1] == 1.0), "right camera must compact into slot 1"
    assert torch.all(images[2] == -1.0), "empty graph padding belongs at the tail"


def test_build_model_inputs_requires_state():
    config = Pi05Config(image_feature_keys=_EXPECTED_CAMERA_ORDER)
    observation = {_EXPECTED_CAMERA_ORDER[0]: np.zeros((4, 4, 3), dtype=np.uint8)}
    with pytest.raises(ValueError, match="state"):
        build_model_inputs(observation, config, _FakeTokenizer(), torch.device("cpu"))


def test_build_model_inputs_requires_configured_camera():
    config = Pi05Config(image_feature_keys=_EXPECTED_CAMERA_ORDER)
    observation = {
        "unknown_camera": np.zeros((4, 4, 3), dtype=np.uint8),
        "state": np.zeros(32, dtype=np.float32),
    }
    with pytest.raises(ValueError, match="configured camera"):
        build_model_inputs(observation, config, _FakeTokenizer(), torch.device("cpu"))


# ----------------------------------------------------------------------------
# Model behaviour + weight-load remap (tiny action dims; full Gemma backbone)
# ----------------------------------------------------------------------------
# Module *structure* (no ``state_proj``, ``time_mlp_{in,out}`` widths, AdaRMS on
# the expert norms but not the prefix) is not asserted here: ``load_weights``
# audits every model parameter against the checkpoint and raises on a missing,
# unmatched or π0-shaped key, so a structural drift fails loudly at load time
# rather than silently. ``test_pi0_shaped_keys_are_not_silently_loaded`` guards
# that audit, and ``test_pi05_parity.py`` is the numerical oracle.
# Both Gemma towers keep their real widths but drop from 18 layers to 2 for
# every test in this section. Nothing here depends on depth: the weight remap
# matches by name, and the AR mask, prefix length and ODE determinism are
# shape-driven. Full-scale numerical correctness is ``test_pi05_parity.py``'s
# job, against the real checkpoint.
#
# Two dimensions must NOT be shrunk, both because something outside these
# configs is sized against them:
#
# * ``width`` — the vision tower's ``multi_modal_projector`` projects image
#   embeddings to the PaliGemma text width so the two can be concatenated.
#   Narrowing the text tower alone makes that concat fail.
# * the vision tower itself — So400m/14 on 224x224 is what yields exactly 256
#   tokens per view, which ``test_prefix_length_and_valid_len_for_each_view_count``
#   exists to verify.
#
# Depths are equal across the two towers on purpose: the action expert runs
# layer-by-layer alongside the VLM, so a mismatch would not be a valid model.
_SHRUNK_GEMMA = {
    "gemma_2b": GemmaVariantConfig(2048, 2, 1024, 8, 1, 256),
    "gemma_300m": GemmaVariantConfig(1024, 2, 512, 8, 1, 256),
}


@pytest.fixture(scope="module", autouse=True)
def _shrink_gemma_backbones():
    """Patch the module-level lookup ``Pi05ForActionPrediction.__init__`` calls."""
    patch = pytest.MonkeyPatch()
    patch.setattr(modeling_pi05, "get_gemma_config", lambda variant: _SHRUNK_GEMMA[variant])
    yield
    patch.undo()


def _tiny_pi05_model():
    """A fresh model. Use for tests that load weights or otherwise mutate it."""
    return Pi05ForActionPrediction(Pi05Config(max_action_dim=8, max_state_dim=8, chunk_size=4, n_action_steps=4))


@pytest.fixture(scope="module")
def tiny_model():
    """Shared read-only model, so the tests that only inspect it pay for one
    construction between them rather than one each."""
    return _tiny_pi05_model()


@pytest.fixture(autouse=True)
def _silence_pi05_loader():
    logging.getLogger("vllm_omni.diffusion.models.pi05.modeling_pi05").setLevel(logging.CRITICAL)
    yield


@pytest.mark.slow
def test_suffix_has_no_state_token(tiny_model):
    """π0's suffix is ``[state, actions...]`` with AR mask ``[1, 1, 0...]``;
    π0.5's is actions only with ``[1, 0...]``."""
    embs, pad, att, cond = tiny_model.embed_suffix(torch.randn(1, 4, 8), torch.tensor([1.0]))
    assert embs.shape[1] == 4
    assert att[0].tolist() == [1, 0, 0, 0]
    assert cond.shape == (1, tiny_model.expert_width)


# Not marked slow: constructs one small norm layer, no backbone.
def test_adarms_norm_returns_gate_only_when_conditioned():
    norm = Pi05AdaRMSNorm(8, cond_dim=4)
    out, gate = norm(torch.randn(1, 3, 8), torch.randn(1, 4))
    assert gate is not None and gate.shape == (1, 1, 8)

    plain = Pi05AdaRMSNorm(8, cond_dim=None)
    out, gate = plain(torch.randn(1, 3, 8))
    assert gate is None


@pytest.mark.slow
def test_lm_head_remap_to_embed_tokens():
    model = _tiny_pi05_model()
    target = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    payload = torch.full(dict(model.named_parameters())[target].shape, 0.5)
    model.load_weights([("model.paligemma_with_expert.paligemma.lm_head.weight", payload)], strict=False)
    assert torch.allclose(dict(model.named_parameters())[target], payload)


@pytest.mark.slow
def test_action_time_mlp_remapped_to_time_mlp():
    """Some exports carry the π0 parameter names for the timestep MLP."""
    model = _tiny_pi05_model()
    payload = torch.full(dict(model.named_parameters())["time_mlp_in.weight"].shape, 0.25)
    model.load_weights([("model.action_time_mlp_in.weight", payload)], strict=False)
    assert torch.allclose(dict(model.named_parameters())["time_mlp_in.weight"], payload)


@pytest.mark.slow
def test_pi0_shaped_keys_are_not_silently_loaded():
    """A π0 checkpoint pointed at the π0.5 class would leave the AdaRMS expert
    randomly initialized. ``state_proj`` is the tell."""
    model = _tiny_pi05_model()
    with pytest.raises(RuntimeError, match="π0-shaped"):
        model.load_weights([("model.state_proj.weight", torch.zeros(1024, 8))])


@pytest.mark.slow
@pytest.mark.parametrize("num_views", [1, 2, 3])
def test_prefix_length_and_valid_len_for_each_view_count(num_views, tiny_model):
    """The input contract: ``256 * views + 200`` total, with only the *valid*
    length varying per request."""
    live_text = 120
    images = [torch.zeros(1, 3, 224, 224) for _ in range(num_views)]
    masks = [torch.tensor([True]) for _ in range(num_views)]
    lang = torch.zeros(1, 200, dtype=torch.long)
    lang_mask = torch.zeros(1, 200, dtype=torch.bool)
    lang_mask[:, :live_text] = True

    embs, pad_masks, _ = tiny_model.embed_prefix(images, masks, lang, lang_mask)
    assert embs.shape[1] == 256 * num_views + 200
    assert int(pad_masks.sum()) == 256 * num_views + live_text


@pytest.mark.slow
def test_sample_actions_shape_and_determinism(tiny_model):
    """Flow matching is an ODE: fixed noise must give a bit-identical chunk."""
    model = tiny_model.eval()
    images = [torch.zeros(1, 3, 224, 224)]
    masks = [torch.tensor([True])]
    lang = torch.zeros(1, 200, dtype=torch.long)
    lang_mask = torch.ones(1, 200, dtype=torch.bool)
    noise = torch.randn(1, 4, 8, generator=torch.Generator().manual_seed(42))

    with torch.no_grad():
        a1 = model.sample_actions(
            images=images, image_masks=masks, lang_tokens=lang, lang_masks=lang_mask, noise=noise, num_steps=2
        )
        a2 = model.sample_actions(
            images=images, image_masks=masks, lang_tokens=lang, lang_masks=lang_mask, noise=noise, num_steps=2
        )
    assert a1.shape == (1, 4, 8)
    assert torch.isfinite(a1).all()
    assert torch.equal(a1, a2)


# ----------------------------------------------------------------------------
# Serving dtype
# ----------------------------------------------------------------------------
# The check that decides what runs lives on the pipeline, not on Pi05Config:
# Pi05Config.dtype records what the checkpoint declares and never reaches a
# cast, while _initialize_model casts using the top-level OmniDiffusionConfig.
@pytest.mark.parametrize(
    "declared,expected",
    [
        ("float32", torch.float32),
        ("bfloat16", torch.bfloat16),
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16),
    ],
)
def test_resolve_dtype_accepts_supported_serving_dtypes(declared, expected):
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.pi05.pipeline_pi05 import Pi05Pipeline

    assert Pi05Pipeline._resolve_dtype(OmniDiffusionConfig(dtype=declared)) is expected


@pytest.mark.parametrize("declared", ["float16", "fp16", torch.float16])
def test_resolve_dtype_rejects_unvalidated_dtypes(declared):
    """float16 resolves to a real torch dtype, so a permissive lookup would run
    the model in a precision nothing here has been checked in.

    Only dtypes that survive ``OmniDiffusionConfig`` normalization are worth
    asserting on: it maps a string it does not recognize onto bfloat16 with a
    warning, so e.g. ``"float64"`` never reaches this guard as float64.
    """
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.pi05.pipeline_pi05 import Pi05Pipeline

    with pytest.raises(ValueError, match="Unsupported π0.5 dtype"):
        Pi05Pipeline._resolve_dtype(OmniDiffusionConfig(dtype=declared))


@pytest.mark.slow
def test_bfloat16_runs_and_tracks_float32(tiny_model):
    """bfloat16 is a supported serving dtype, so it has to produce a usable
    chunk — and one that still resembles float32.

    The bound is deliberately loose. This is not a parity check (bfloat16
    deviates from float32 by ~1-2% of the action range on the real checkpoint,
    which is why recipes/lerobot/Pi05.md tells operators to validate it on their
    own task). What it catches is gross breakage: a cast that silently did not
    happen, activations overflowing to inf, or a dtype mismatch that would make
    the two outputs unrelated.
    """
    import copy

    images = [torch.zeros(1, 3, 224, 224)]
    masks = [torch.tensor([True])]
    lang = torch.zeros(1, 200, dtype=torch.long)
    lang_mask = torch.ones(1, 200, dtype=torch.bool)
    noise = torch.randn(1, 4, 8, generator=torch.Generator().manual_seed(7))

    def run(model, dtype):
        with torch.no_grad():
            return model.sample_actions(
                images=[img.to(dtype) for img in images],
                image_masks=masks,
                lang_tokens=lang,
                lang_masks=lang_mask,
                noise=noise.to(dtype),
                num_steps=2,
            ).float()

    reference = run(tiny_model.eval(), torch.float32)
    bf16_model = copy.deepcopy(tiny_model).to(torch.bfloat16).eval()

    # The cast actually happened — otherwise everything below passes trivially.
    assert next(bf16_model.parameters()).dtype is torch.bfloat16

    actual = run(bf16_model, torch.bfloat16)
    assert actual.shape == reference.shape
    assert torch.isfinite(actual).all()

    scale = float(reference.abs().max())
    assert float((actual - reference).abs().max()) < max(0.5 * scale, 1e-2)
