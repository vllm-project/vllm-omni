from __future__ import annotations

from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.inputs.data import OmniSamplingParams


def validate_streaming_input_sampling_params(params: OmniSamplingParams) -> None:
    if (
        not isinstance(params, SamplingParams)
        or params.n > 1
        or params.output_kind == RequestOutputKind.FINAL_ONLY
        or params.stop
    ):
        raise ValueError(
            "Input streaming is currently supported only for SamplingParams "
            "with n == 1, output_kind != FINAL_ONLY, and without stop strings."
        )
