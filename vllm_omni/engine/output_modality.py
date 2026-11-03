import warnings

from vllm_omni.outputs.output_modality import *  # noqa: F401,F403

warnings.warn(
    "Importing from 'vllm_omni.engine.output_modality' is deprecated. Use 'vllm_omni.outputs.output_modality' instead.",
    DeprecationWarning,
    stacklevel=2,
)
