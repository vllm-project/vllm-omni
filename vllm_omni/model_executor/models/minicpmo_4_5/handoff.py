# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-owned stage metadata and CPU tensor snapshots for request handoff."""

import torch

# Speech boundaries are independent of dialogue turn boundaries.
SPEECH_FLUSH_KEY = "speech_flush"
TEXT_WITHOUT_AUDIO_KEY = "text_without_audio"


def pack_handoff_tensor(value: torch.Tensor) -> bytes:
    """Keep legacy list-to-float32 values without creating Python scalar lists.

    EngineCore's ``model_intermediate_buffer`` has Any-valued fields, so raw
    tensors would lose their type on its msgpack wire. The existing Omni codec
    packs an owned byte snapshot and restores the dtype/shape at the consumer.
    """
    # NPU platform bootstrap imports the metadata constants before connector
    # initialization is safe. Load the codec only when transferring a tensor.
    from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer

    tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if tensor.numel() == 0:
        # Empty transposed tensors may report contiguous with a nonunit last
        # stride; the shared codec reinterprets that dimension as bytes.
        tensor = torch.empty(tensor.shape, dtype=torch.float32)
    return OmniSerializer.serialize(tensor)


def unpack_handoff_tensor(value: object) -> object:
    """Accept existing local tensors/lists as well as packed turn handoffs."""
    if not isinstance(value, bytes):
        return value
    from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer

    tensor = OmniSerializer.deserialize(value)
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32 or tensor.device.type != "cpu":
        raise ValueError("MiniCPM-o packed handoff must contain a CPU float32 tensor")
    return tensor
