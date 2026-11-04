# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Selective online MXFP8 using the shared NPU provider, not an INT8 adapter."""


def online_mxfp8_config(mode, existing, *, num_layers: int):
    """Only attention/FFN parallel linears participate; conditioning stays intact."""
    if mode is None:
        return existing
    if mode != "mxfp8":
        raise ValueError("lingbot_npu_online_quant must be null or 'mxfp8'")
    if existing is not None:
        raise ValueError("Use either lingbot_npu_online_quant or quantization_config, not both")
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    ignored = ["transformer.c2ws_hidden_states_layer1", "transformer.c2ws_hidden_states_layer2"]
    for index in range(num_layers):
        ignored.extend(f"transformer.blocks.{index}.{name}" for name in ("cam_injector_layer1", "cam_injector_layer2"))
    # Plain nn.Linear time/gate/camera-scale/head modules never enter LinearBase
    # dispatch. Explicitly exclude the parallel camera linears as well.
    return DiffusionMXFP8Config(is_checkpoint_mxfp8_serialized=False, ignored_layers=ignored)
