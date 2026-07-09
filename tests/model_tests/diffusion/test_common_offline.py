import pytest

from tests.model_tests.diffusion.case_filtering import get_parametrized_options
from tests.model_tests.diffusion.config_types import (
    DiffusionAccs,
    DiffusionTasks,
    build_omni_from_diff_accelerations,
)
from tests.model_tests.diffusion.model_settings import DIFFUSION_TEST_SETTINGS
from tests.model_tests.diffusion.task_runners import (
    run_and_validate_image_to_image_request,
    run_and_validate_text_to_image_determinism,
    run_and_validate_text_to_image_multi_output,
    run_and_validate_text_to_image_request,
)

# NOTE : Hardware marks are added dynamically based on test requirements
pytestmark = [pytest.mark.diffusion]


@pytest.mark.parametrize(
    "model_name,accelerations,supported_tasks,check_multioutput,check_determinism",
    get_parametrized_options(DIFFUSION_TEST_SETTINGS),
)
def test_pipeline_on_supported_tasks(
    model_name,
    accelerations: list[DiffusionAccs] | None,
    supported_tasks: list[DiffusionTasks],
    check_multioutput: bool,
    check_determinism: bool,
    tiny_model_paths: dict[str, str],
    subtests,
):
    """Run a smoke test on all of the pipelines supported tasks using a set of enabled accelerations."""
    assert len(supported_tasks) > 0
    # We initialize the Omni object before running the tasks, then run each task as a pytest subtask.
    # This lets us init the model once, but display separate failures in pytest, and avoid halting the
    # checks on other tasks if one fails.
    #
    # This allows us to have some degree of test isolation without the cost of redundant initialization,
    # since starting the server can take 10+ seconds, even for tiny models.
    #
    # NOTE: Be sure to install pytest-subtests if you're running on pytest < 9
    omni = build_omni_from_diff_accelerations(
        accelerations=accelerations,
        model=tiny_model_paths[model_name],
        enforce_eager=True,
    )
    try:
        for task_type in supported_tasks:
            with subtests.test(msg=task_type):
                if task_type == DiffusionTasks.TEXT_TO_IMAGE:
                    run_and_validate_text_to_image_request(omni)
                elif task_type == DiffusionTasks.IMAGE_TO_IMAGE:
                    run_and_validate_image_to_image_request(omni)
                else:
                    raise ValueError(f"Task type {task_type} is not yet supported")

        # NOTE: For now, we only check determinism + multi output for the base case,
        # since checking it on every extra acceleration configuration is redundant
        # (see case_filtering).
        if check_determinism:
            with subtests.test(msg="determinism"):
                run_and_validate_text_to_image_determinism(omni)
        if check_multioutput:
            with subtests.test(msg="multi_output"):
                run_and_validate_text_to_image_multi_output(omni)
    finally:
        omni.close()
