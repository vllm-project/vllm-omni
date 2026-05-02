"""
Tests for Flux2 Klein; currently Dev is implemented separately,
but ideally these models will fold together in the future.

Coverage:
- FP8 + CacheDiT + Ulysses=2 + TP=2
- Layerwise CPU offload + Ulysses=2 + Ring=2
- Layerwise CPU offload + TP=2
- Layerwise CPU offload + HSDP
"""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

FOUR_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "L4"}, num_cards=4)
POSITIVE_PROMPT = "A cat sitting on a windowsill"
NEGATIVE_PROMPT = "blurry, low quality"


# For now, we only test a good configuration for this model to keep the CI
# light, but in the near future may explore using random models to check
# cross-feature compatibility more generally.
def _get_diffusion_feature_cases(model: str):
    return [
        # FP8 / Hybrid sequence parallelism
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--ulysses-degree",
                    "2",
                    "--quantization",
                    "fp8",
                    # NOTE: TP added for test coverage here since it doesn't
                    # slow things down too much, but fp8 + ulysses SP +
                    # cache_dit will generally get you the highest speedup here
                    "--tensor-parallel-size",
                    "2",
                ],
            ),
            marks=FOUR_CARD_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--enable-layerwise-offload",
                    "--ulysses-degree",
                    "2",
                    "--ring",
                    "2",
                ],
            ),
            id="layerwise_ulysses2_ring2",
            marks=FOUR_CARD_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--enable-layerwise-offload",
                    "--tensor-parallel-size",
                    "2",
                ],
            ),
            id="layerwise_tp2",
            marks=FOUR_CARD_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--enable-layerwise-offload",
                    "--use-hsdp",
                    "--hsdp-shard-size",
                    "2",
                ],
            ),
            id="layerwise_hsdp",
            marks=FOUR_CARD_FEATURE_MARKS,
        ),
    ]


@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(
        model="black-forest-labs/FLUX.2-klein-4B",
    ),
    indirect=True,
)
def test_flux2_klein(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    request_config = {
        "model": omni_server.model,
        "messages": [{"role": "user", "content": POSITIVE_PROMPT}],
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "negative_prompt": NEGATIVE_PROMPT,
            "true_cfg_scale": 4.0,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
