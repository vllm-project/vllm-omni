"""Nightly-only e2e coverage for BAGEL diffusion multi-replica serving.

This test needs 4 H100 GPUs; keep it out of generic test_bagel_* jobs.
"""

import os

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams, dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "ByteDance-Seed/BAGEL-7B-MoT"
MULTI_REPLICA_DEPLOY = get_deploy_config_path("ci/bagel_multi_replicas_4gpu.yaml")
ROUTE_STRESS_REQUESTS = 6

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]

test_params = [
    OmniServerParams(
        model=MODEL,
        stage_config_path=MULTI_REPLICA_DEPLOY,
        server_args=["--disable-log-stats"],
        use_stage_cli=True,
    )
]


@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_multi_stage_diffusion_uses_multi_replica_dit_stage(omni_server, openai_client) -> None:
    request_config = {
        "model": omni_server.model,
        "messages": dummy_messages_from_mix_data(
            content_text="<|im_start|>A small red cube on a white table.<|im_end|>"
        ),
        "modalities": ["image"],
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "guidance_scale": 0.0,
            "seed": 42,
        },
    }

    responses = openai_client.send_diffusion_request(request_config, request_num=ROUTE_STRESS_REQUESTS)

    assert len(responses) == ROUTE_STRESS_REQUESTS
    assert all(response.success for response in responses)
    assert all(response.images is not None and response.images[0].size == (512, 512) for response in responses)
