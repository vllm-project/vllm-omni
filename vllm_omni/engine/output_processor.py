import warnings

from vllm_omni.outputs.output_processor import *  # noqa: F401,F403

warnings.warn(
    "Importing from 'vllm_omni.engine.output_processor' is deprecated. "
    "Use 'vllm_omni.outputs.output_processor' instead.",
    DeprecationWarning,
    stacklevel=2,
)
