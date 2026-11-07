# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def propagate_target_size(prompt, sampling_params_list):
    if not isinstance(prompt, dict):
        return prompt
    dimensions = prompt.get("mm_processor_kwargs") or {}
    height = dimensions.get("target_h", dimensions.get("height"))
    width = dimensions.get("target_w", dimensions.get("width"))
    if height is not None and width is not None:
        params = sampling_params_list[0].clone()
        params.extra_args = {**(params.extra_args or {}), "target_h": height, "target_w": width}
        sampling_params_list[0] = params
    return prompt
