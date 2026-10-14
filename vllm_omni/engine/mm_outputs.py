import warnings

from vllm_omni.outputs.mm_outputs import *  # noqa: F401,F403

warnings.warn(
    "Importing from 'vllm_omni.engine.mm_outputs' is deprecated. Use 'vllm_omni.outputs.mm_outputs' instead.",
    DeprecationWarning,
    stacklevel=2,
)
