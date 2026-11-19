# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Small CPU runtime builders shared by MiniCPM-o model and worker tests."""

from types import SimpleNamespace


def stage0_vision_runtime():
    import torch

    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
    )

    class _StageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(256, 2)

        def get_input_embeddings(self):
            return self.embed

        def get_audio_hidden_states(self, _data):
            return [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
    runtime.thinker = runtime.stage_model
    runtime.tokenizer = SimpleNamespace(
        unk_token_id=0,
        convert_tokens_to_ids=lambda token: {
            "<unit>": 1,
            "</unit>": 2,
            "<|listen|>": 3,
            "<|speak|>": 4,
            "<|tts_bos|>": 5,
            "<|tts_eos|>": 6,
            "<|tts_pad|>": 7,
            "<|chunk_eos|>": 8,
            "<|chunk_tts_eos|>": 9,
            "<|turn_eos|>": 10,
            "<|audio|>": 11,
            "<image>": 12,
            "</image>": 13,
            "<slice>": 14,
            "</slice>": 15,
        }.get(token, 0),
        encode=lambda text, add_special_tokens=False: [201, 5],
    )
    runtime.processor = SimpleNamespace(get_streaming_chunk_size=lambda: 4)
    runtime.device = "cpu"
    runtime._init_token_ids()
    # One 2-row block per frame keeps the expected token_ids readable; the real
    # resampler emits VISION_EMBEDS_PER_FRAME rows per slice.
    runtime._stage_vision_embeddings = lambda frames: [
        [torch.full((2, 2), float(index), dtype=torch.float32)] for index in range(len(frames))
    ]
    return runtime
