from types import SimpleNamespace

import pytest

from vllm_omni.worker import sampling_utils

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_fixed_seed_contract_separates_stage_sampler_from_talker_mtp_rng() -> None:
    assert hasattr(sampling_utils, "apply_fixed_seed_to_sampling_params")
    assert hasattr(sampling_utils, "get_tts_local_seed")

    stage_only = SimpleNamespace(seed=None, extra_args=None)
    sampling_utils.apply_fixed_seed_to_sampling_params(
        stage_only,
        17,
        seed_talker_mtp=False,
    )
    assert stage_only.seed == 17
    assert sampling_utils.get_tts_local_seed(stage_only) is None

    talker = SimpleNamespace(seed=None, extra_args={"preserved": True})
    sampling_utils.apply_fixed_seed_to_sampling_params(
        talker,
        23,
        seed_talker_mtp=True,
    )
    assert talker.seed == 23
    assert talker.extra_args == {"preserved": True, "tts_local_seed": 23}
    assert sampling_utils.get_tts_local_seed(talker) == 23

    outer_seed_only = SimpleNamespace(seed=31, extra_args={})
    assert sampling_utils.get_tts_local_seed(outer_seed_only) is None
