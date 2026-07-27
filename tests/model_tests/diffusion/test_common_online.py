"""
Analogous to test_common_offline, but for server tests. Validates the full
online serving stack (CLI arg parsing, subprocess, API routing, response
encoding) using tiny models.
"""

import pytest

from tests.helpers.runtime import OmniServer, OpenAIClientHandler
from tests.model_tests.diffusion.case_filtering import get_parametrized_options
from tests.model_tests.diffusion.config_types import (
    DiffusionAccs,
    DiffusionTasks,
    build_server_args_from_diff_accelerations,
)
from tests.model_tests.diffusion.model_settings import DIFFUSION_TEST_SETTINGS
from tests.model_tests.diffusion.task_runners import (
    run_and_validate_online_image_to_image_request,
    run_and_validate_online_text_to_image_determinism,
    run_and_validate_online_text_to_image_multi_output,
    run_and_validate_online_text_to_image_request,
    run_and_validate_online_text_to_video_request,
)

# NOTE : Hardware marks are added dynamically based on test requirements
pytestmark = [pytest.mark.diffusion]


@pytest.mark.parametrize(
    "model_name,accelerations,supported_tasks,check_multioutput,check_determinism",
    get_parametrized_options(DIFFUSION_TEST_SETTINGS, online=True),
)
def test_online_on_supported_tasks(
    model_name: str,
    accelerations: list[DiffusionAccs] | None,
    supported_tasks: list[DiffusionTasks],
    check_multioutput: bool,
    check_determinism: bool,
    tiny_model_paths: dict[str, str],
    run_level: str,
    subtests,
):
    """Smoke test: start a tiny model server and run each supported task via the API."""
    model_path = tiny_model_paths[model_name]
    server_args = build_server_args_from_diff_accelerations(accelerations)
    server_args.append("--enforce-eager")

    with OmniServer(model_path, server_args) as server:
        # TODO: We may want to revisit run_level validation here,
        # because checks for things like image size etc should not
        # depend on whether or not the weights are real or random
        client = OpenAIClientHandler(
            host=server.host,
            port=server.port,
            api_key="EMPTY",
            run_level=run_level,
            log_stats=server.log_stats,
        )
        for task_type in supported_tasks:
            with subtests.test(msg=task_type):
                if task_type == DiffusionTasks.TEXT_TO_IMAGE:
                    run_and_validate_online_text_to_image_request(server, client)
                elif task_type == DiffusionTasks.IMAGE_TO_IMAGE:
                    run_and_validate_online_image_to_image_request(server, client)
                elif task_type == DiffusionTasks.TEXT_TO_VIDEO:
                    run_and_validate_online_text_to_video_request(server, client)
                else:
                    raise ValueError(f"Task type {task_type} is not yet supported")

        # NOTE: For now, we only check determinism + multi output for the base case,
        # since checking it on every extra acceleration configuration is redundant
        # (see case_filtering).
        if check_determinism:
            with subtests.test(msg="determinism"):
                run_and_validate_online_text_to_image_determinism(server, client)
        if check_multioutput:
            with subtests.test(msg="multi_output"):
                run_and_validate_online_text_to_image_multi_output(server, client)
