"""
Analogous to test_common_offline, but for server tests. Validates the full
online serving stack (CLI arg parsing, subprocess, API routing, response
encoding) using tiny models.
"""

import pytest

from tests.helpers.runtime import OmniServer, OpenAIClientHandler
from tests.model_tests.case_filtering import get_parametrized_options
from tests.model_tests.config_types import DiffusionAccs, ModelTasks
from tests.model_tests.model_settings import MODEL_SETTINGS
from tests.model_tests.task_runners import TASKS_TO_RUNNER_MAP
from tests.model_tests.utils import build_server_args_from_diff_accelerations

# NOTE : Hardware marks are added dynamically based on test requirements
# Mark can also be determined by if it's a diffusion test opt or not
pytestmark = [pytest.mark.diffusion]


@pytest.mark.parametrize(
    "model_name,accelerations,supported_tasks,check_multioutput,check_determinism",
    get_parametrized_options(MODEL_SETTINGS, online=True),
)
def test_online_on_supported_tasks(
    model_name: str,
    accelerations: list[DiffusionAccs] | None,
    supported_tasks: list[ModelTasks],
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
                if task_type in TASKS_TO_RUNNER_MAP:
                    online_runner = TASKS_TO_RUNNER_MAP[task_type].online_validator
                    online_runner(server, client)
                else:
                    raise ValueError(f"Task type {task_type} is not yet supported")

        # NOTE: For now, we only check determinism + multi output for the base case,
        # since checking it on every extra acceleration configuration is redundant
        # (see case_filtering).
        # FIXME - Re-enable determinism / multi-out checks
        # if check_determinism:
        #     with subtests.test(msg="determinism"):
        #         run_and_validate_online_text_to_image_determinism(server, client)
        # if check_multioutput:
        #     with subtests.test(msg="multi_output"):
        #         run_and_validate_online_text_to_image_multi_output(server, client)
