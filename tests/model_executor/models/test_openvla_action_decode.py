# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L1 tests for OpenVLA action de-tokenisation (CPU, no weights).

OpenVLA's policy output is seven ordinary generated tokens; turning them back
into a robot action is the half that exists in neither vLLM nor vllm-omni, and
it is pure arithmetic, so it is fully testable here.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.model_executor.models.openvla.action_decode import (
    OPENVLA_EMPTY_TOKEN_ID,
    OpenVLAActionDecoder,
    build_prompt,
    build_prompt_token_ids,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# openvla-7b's real shape: a padded 32064-entry embedding table whose last 64
# rows are padding, so the action bins start counting down from 32000.
_VOCAB_PADDED = 32064
_PAD_TO = 64
_UNPADDED = _VOCAB_PADDED - _PAD_TO
_N_BINS = 256

# Two embodiments so the "which one?" behaviour is exercised. The mask matches
# every embodiment in the real checkpoint: the last dimension (the gripper) is
# never un-normalised.
_STATS = {
    "bridge_orig": {
        "action": {
            "q01": [-1.0, -2.0, 0.0],
            "q99": [1.0, 2.0, 10.0],
            "mask": [True, True, False],
        }
    },
    "fractal": {
        "action": {
            "q01": [0.0, 0.0, 0.0],
            "q99": [4.0, 4.0, 4.0],
            "mask": [True, True, True],
        }
    },
}


def _hf_config(norm_stats=None, **overrides):
    config = SimpleNamespace(
        norm_stats=_STATS if norm_stats is None else norm_stats,
        n_action_bins=_N_BINS,
        pad_to_multiple_of=_PAD_TO,
        text_config=SimpleNamespace(vocab_size=_VOCAB_PADDED),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _decoder(**kwargs):
    return OpenVLAActionDecoder.from_hf_config(_hf_config(), **kwargs)


def _reference_decode(token_ids, stats, vocab_size=_UNPADDED, n_bins=_N_BINS):
    """The reference arithmetic, transcribed independently of the implementation."""
    bins = np.linspace(-1, 1, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    discretized = np.asarray(vocab_size) - np.asarray(token_ids)
    discretized = np.clip(discretized - 1, a_min=0, a_max=bin_centers.shape[0] - 1)
    normalized = bin_centers[discretized]
    q01 = np.asarray(stats["q01"])
    q99 = np.asarray(stats["q99"])
    mask = np.asarray(stats["mask"], dtype=bool)
    return np.where(mask, 0.5 * (normalized + 1) * (q99 - q01) + q01, normalized)


def test_decode_matches_the_reference_arithmetic():
    decoder = _decoder(default_unnorm_key="bridge_orig")
    # A high bin, the middle, and the very bottom of the action range.
    token_ids = [_UNPADDED - 1, _UNPADDED - 128, _UNPADDED - 255]
    got = decoder.decode(token_ids)
    want = _reference_decode(token_ids, _STATS["bridge_orig"]["action"])
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)


def test_bins_count_down_from_the_top_of_the_vocabulary():
    """The direction is the easy thing to get backwards.

    `bin = vocab_size - token_id`, so the *highest* action token id is the
    *lowest* bin and therefore the bottom of the action range.
    """
    decoder = _decoder(default_unnorm_key="bridge_orig")
    lowest_bin = decoder.decode([_UNPADDED - 1] * 3)
    highest_bin = decoder.decode([_UNPADDED - 255] * 3)
    assert lowest_bin[0] == pytest.approx(-1.0, abs=0.01)
    assert highest_bin[0] == pytest.approx(1.0, abs=0.01)


def test_masked_dimension_is_not_unnormalised():
    """The gripper dimension comes back as a raw bin centre, inside [-1, 1]."""
    decoder = _decoder(default_unnorm_key="bridge_orig")
    action = decoder.decode([_UNPADDED - 1, _UNPADDED - 1, _UNPADDED - 1])
    assert -1.0 <= action[2] <= 1.0
    # ...while an unmasked dimension of the same token spans its own q01..q99.
    assert action[1] == pytest.approx(-2.0, abs=0.02)


def test_out_of_range_tokens_clip_rather_than_raise():
    decoder = _decoder(default_unnorm_key="bridge_orig")
    # A token far below the action range (e.g. ordinary text) and one above it.
    low = decoder.decode([1, 1, 1])
    high = decoder.decode([_UNPADDED + 10] * 3)
    np.testing.assert_allclose(low, decoder.decode([_UNPADDED - 255] * 3))
    np.testing.assert_allclose(high, decoder.decode([_UNPADDED - 1] * 3))


def test_wrong_token_count_is_rejected():
    decoder = _decoder(default_unnorm_key="bridge_orig")
    with pytest.raises(ValueError, match="Expected 3 action tokens"):
        decoder.decode([_UNPADDED - 1, _UNPADDED - 2])


def test_unnorm_key_selects_the_embodiment():
    decoder = _decoder(default_unnorm_key="bridge_orig")
    token_ids = [_UNPADDED - 128] * 3
    bridge = decoder.decode(token_ids)
    fractal = decoder.decode(token_ids, "fractal")
    assert not np.allclose(bridge, fractal)
    np.testing.assert_allclose(fractal, _reference_decode(token_ids, _STATS["fractal"]["action"]), atol=1e-6)


def test_ambiguous_embodiment_names_the_options():
    decoder = _decoder()
    assert decoder.default_unnorm_key is None
    with pytest.raises(ValueError, match="bridge_orig"):
        decoder.decode([_UNPADDED - 1] * 3)


def test_single_embodiment_needs_no_key():
    single = {"only": _STATS["bridge_orig"]}
    decoder = OpenVLAActionDecoder.from_hf_config(_hf_config(norm_stats=single))
    assert decoder.default_unnorm_key == "only"
    assert decoder.decode([_UNPADDED - 1] * 3).shape == (3,)


def test_unknown_embodiment_is_rejected_at_construction():
    with pytest.raises(ValueError, match="Unknown unnorm_key"):
        _decoder(default_unnorm_key="nope")


def test_missing_padding_metadata_raises_instead_of_decoding_wrongly():
    """Guards a silent failure: a wrong vocab size clips every token to bin 0."""
    config = _hf_config()
    del config.pad_to_multiple_of
    with pytest.raises(ValueError, match="pad_to_multiple_of"):
        OpenVLAActionDecoder.from_hf_config(config)


def test_checkpoint_without_norm_stats_is_rejected():
    with pytest.raises(ValueError, match="norm_stats"):
        OpenVLAActionDecoder.from_hf_config(_hf_config(norm_stats={}))


def test_prompt_matches_the_checkpoints_training_format():
    assert build_prompt("Pick Up The Red Block") == (
        "In: What action should the robot take to pick up the red block?\nOut:"
    )


class _FakeTokenizer:
    def __init__(self, ids):
        self._ids = ids
        self.calls = []

    def encode(self, text, add_special_tokens=True):
        self.calls.append((text, add_special_tokens))
        return list(self._ids)


def test_prompt_token_ids_append_the_empty_token():
    tokenizer = _FakeTokenizer([1, 2, 3])
    ids = build_prompt_token_ids(tokenizer, "close the gripper")
    assert ids == [1, 2, 3, OPENVLA_EMPTY_TOKEN_ID]
    assert tokenizer.calls[0][1] is True


def test_prompt_token_ids_do_not_double_append():
    tokenizer = _FakeTokenizer([1, 2, OPENVLA_EMPTY_TOKEN_ID])
    assert build_prompt_token_ids(tokenizer, "close the gripper") == [
        1,
        2,
        OPENVLA_EMPTY_TOKEN_ID,
    ]


def test_policy_server_values_are_derived_from_the_checkpoint():
    values = _decoder(default_unnorm_key="bridge_orig").policy_server_values()
    assert values["action_dim"] == 3
    assert values["unnorm_key"] == "bridge_orig"
    assert values["supported_embodiments"] == ["bridge_orig", "fractal"]
    assert values["image_resolution"] == [224, 224]
