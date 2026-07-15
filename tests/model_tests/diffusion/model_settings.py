from tests.model_tests.diffusion import diff_model_builders
from tests.model_tests.diffusion.config_types import (
    DiffusionAccs,
    DiffusionModelTestOpts,
    DiffusionTasks,
)

# This object defines the (tiny) model configurations for common tests.
#
# TO ADD A NEW MODEL:
# Map the pipeline name (i.e., the name of the pipeline in _DIFFUSION_MODELS)
# to a DiffusionModelTestOpts and define the supported tasks. A base case with no
# accelerations is always run. Use extra_test_groups to define additional
# acceleration configurations to run a basic smoke test for (i.e., simple inference).
#
# The model will only be loaded once per test group, and will execute each of its
# supported tasks as a pytest subtest. The base case also runs additional checks
# that are independent of accelerations, e.g., for determinism and compatibility with
# multi-output, unless explicitly disabled in the test settings.
#
# TO RUN A SUBSET OF THE TESTS:
# These tests should be fast, but if you'd like to run a subset of the tests, you can do so
# with `-k`; the IDs of the tests are the name of the pipeline joined by +.
#
# Example:
# $ pytest test_common_offline.py -k test_pipeline_on_supported_tasks[Flux2KleinPipeline]
#   ^ Runs only the base (no accelerations) case for Flux2KleinPipeline
#
# $ pytest test_common_offline.py -k test_pipeline_on_supported_tasks[Flux2KleinPipeline
#   ^ Runs all test groups for Flux2KleinPipeline only
DIFFUSION_TEST_SETTINGS = {
    "Flux2KleinPipeline": DiffusionModelTestOpts(
        model="black-forest-labs/FLUX.2-klein-4B",
        builder=diff_model_builders.tiny_flux2_klein_builder,
        supported_tasks=[DiffusionTasks.TEXT_TO_IMAGE, DiffusionTasks.IMAGE_TO_IMAGE],
        extra_test_groups=[
            [DiffusionAccs.HSDP, DiffusionAccs.TEA_CACHE],
            [DiffusionAccs.SEQUENCE_PARALLEL, DiffusionAccs.CACHE_DIT, DiffusionAccs.LAYERWISE_OFFLOAD],
            [DiffusionAccs.CFG_PARALLEL, DiffusionAccs.TENSOR_PARALLEL, DiffusionAccs.CPU_OFFLOAD],
        ],
    )
}
