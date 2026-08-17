"""
Analogous to: https://github.com/vllm-project/vllm/blob/v0.23.0/tests/models/multimodal/generation/vlm_utils/case_filtering.py

This follows a similar pattern so that we can use `get_parametrized_options` to expand our test
configurations out, similar to the way in which vLLM expands supported input type tests. For now,
this mostly just means parametrizing over the test groups, which may be unique per model, but
doing so in this way lets us cleanly define test groups that may be heterogeneous + avoid having
excessive skip marks for features that may not be supported in every pipeline.
"""

# Engine initialization is currently quite expensive. For efficiency,
# when we run the tests, we should try to collapse compatible accelerations,
# because even with tiny models, the overhead of spinning up a new instance
# stacks up fast.
import itertools

import pytest

from tests.helpers.mark import hardware_marks
from tests.model_tests.diffusion.config_types import DiffusionAccs, DiffusionModelTestOpts, get_required_device_count
from vllm_omni.platforms import current_omni_platform

# These tests are intended to run on L4 GPUs since they test parallelism (i.e., need GPUs),
# but are intended to be smoke tests that we can properly validate without real model
# weights in the CI.
#
# As such, no DiffusionAccs may require more than 4 GPUs, since we would otherwise skip
# them in the CI.
MAX_CI_DEVICES = 4


def get_test_group_marks(model_name: str, test_group: list[DiffusionAccs] | None, model_marks: list | None) -> list:
    """Build the full set of pytest marks for a test group.

    For now, single device groups default to core model, and multi device groups
    default to advanced_model marks. We also validate and throw if a test group exceeds
    the max number of GPUs for the CI to prevent accidentally adding configurations that
    would be skipped in the CI."""
    marks = list(model_marks) if model_marks is not None else []

    required_devices = get_required_device_count(test_group)
    if required_devices > MAX_CI_DEVICES:
        raise ValueError(
            f"Test group {test_group} for {model_name} requires {required_devices} devices, "
            f"but the max CI device count is {MAX_CI_DEVICES}. "
        )

    marks.extend(hardware_marks(res={"cuda": "L4"}, num_cards=required_devices))
    # hardware_marks only adds skipif for num_cards > 1, so we currently handle the single-device case
    # directly here. This should probably be handled in a more common way later on.
    assert current_omni_platform is not None and current_omni_platform.device_count is not None
    device_count = current_omni_platform.device_count()
    if current_omni_platform.is_cuda() and device_count < required_devices:
        marks.append(pytest.mark.skip(reason=f"Need {required_devices} devices, got {device_count}"))
    if required_devices > 1:
        marks.append(pytest.mark.full_model)
    else:
        marks.append(pytest.mark.core_model)
    return marks


def get_model_parametrization(model_name: str, test_info: DiffusionModelTestOpts, online: bool):
    """Given a model & its corresponding test options, build the list of pytest params
    to be run for this model. The base case (no accelerations) is always included. Extra
    test groups are always appended for offline tests, but are only added to the online
    tests if online_base_only is overridden to False to avoid redundant testing.
    """
    test_groups: list[list[DiffusionAccs] | None] = [None]
    if not (online and test_info.online_base_only) and test_info.extra_test_groups:
        test_groups.extend(test_info.extra_test_groups)

    return [
        pytest.param(
            model_name,
            test_group,
            test_info.supported_tasks,
            test_info.check_multi_output and test_group is None,
            test_info.check_determinism and test_group is None,
            id=f"{model_name}[{'+'.join(test_group)}]" if test_group else model_name,
            marks=get_test_group_marks(model_name, test_group, test_info.marks),
        )
        for test_group in test_groups
    ]


def get_parametrized_options(
    test_settings: dict[str, DiffusionModelTestOpts],
    online: bool = False,
):
    """Converts all the DiffusionModelTestOpts into an expanded list of parameters
    based on which accelerations are available.

    When online=True and a model has online_base_only=True, only the base case
    (no accelerations) is included for that model's parametrization to minimize
    redundancy between offline / online tests.
    """
    # Get a list per model type, where each entry contains a tuple of all of
    # that model type's cases, then flatten them into the top level so that
    # we can consume them in one mark.parametrize call.
    parametrization = [
        get_model_parametrization(model_name, test_info, online=online)
        for model_name, test_info in test_settings.items()
    ]
    return list(itertools.chain(*parametrization))
