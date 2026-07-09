import warnings

warnings.warn(
    "Importing from 'vllm_omni.engine.output_processor' is deprecated. "
    "Use 'vllm_omni.outputs.output_processor' instead.",
    DeprecationWarning,
    stacklevel=2,
)
