# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Matrix session lifecycle for the shared Wan tiny codec."""

import torch
from torch import nn

from vllm_omni.diffusion.models.wan2_2.taew2_2 import TAEW2Decoder, TAEW2Encoder, TAEW2EncodeState

from .graph import MatrixGraph


class MatrixCodec(nn.Module):
    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.encoder = TAEW2Encoder(checkpoint_path, torch.bfloat16)
        self.decoder = TAEW2Decoder(checkpoint_path, torch.bfloat16)
        self.encoder_state = TAEW2EncodeState()
        self.decoder_state = self.decoder.new_decode_state("matrix")
        self.use_graph = False
        self.graphs: dict[str, MatrixGraph] = {}

    def clear_cache(self) -> None:
        for graph in self.graphs.values():
            graph.close()
        self.graphs.clear()
        self.encoder_state = TAEW2EncodeState()
        self.decoder.release(self.decoder_state)
        self.decoder_state = self.decoder.new_decode_state("matrix")

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode_chunk(pixels, TAEW2EncodeState())

    def encode_stream(self, pixels: torch.Tensor) -> torch.Tensor:
        if self.use_graph and self.encoder_state.frames_encoded:
            if "encode" not in self.graphs:
                memory = self.encoder_state.memory

                def encode(value: torch.Tensor) -> torch.Tensor:
                    state = TAEW2EncodeState(memory=memory.copy(), frames_encoded=1)
                    output = self.encoder.encode_chunk(value, state)
                    for index, previous in memory.items():
                        previous.copy_(state.memory[index])
                    return output

                self.graphs["encode"] = MatrixGraph(encode, tuple(memory.values()))
            output = self.graphs["encode"](pixels)
            self.encoder_state.frames_encoded += pixels.shape[2]
            assert output is not None
            return output
        return self.encoder.encode_chunk(pixels, self.encoder_state)

    def decode_stream(self, latents: torch.Tensor) -> torch.Tensor:
        if self.use_graph and self.decoder_state.started:
            if "decode" not in self.graphs:
                memory = self.decoder_state.feat_map

                def decode(value: torch.Tensor) -> torch.Tensor:
                    state = self.decoder.new_decode_state("matrix-graph")
                    state.feat_map = memory.copy()
                    state.frames_decoded = 1
                    output = self.decoder.decode_chunk(value, state)
                    for index, previous in enumerate(memory):
                        if previous is not None:
                            previous.copy_(state.feat_map[index])
                    return output

                self.graphs["decode"] = MatrixGraph(decode, tuple(value for value in memory if value is not None))
            output = self.graphs["decode"](latents)
            self.decoder_state.frames_decoded += latents.shape[2]
            self.decoder_state.chunks_decoded += 1
            assert output is not None
            return output
        return self.decoder.decode_chunk(latents, self.decoder_state)
