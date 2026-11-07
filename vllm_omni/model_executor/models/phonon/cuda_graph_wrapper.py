# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fixed-address codec graphs with main-stream Mimi execution."""

import torch


class MimiDecodeRunner:
    def __init__(self, decoder, *, capture=True):
        self.decoder = decoder
        device = next(decoder.parameters()).device
        self.stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
        self.capture = capture and self.stream is not None
        self._forward = (
            torch.compile(
                decoder.forward,
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )
            if self.capture
            else decoder.forward
        )
        self.buckets = {}

    def _bucket(self, batch):
        if batch in self.buckets:
            return self.buckets[batch]
        parameter = next(self.decoder.parameters())
        latent = parameter.new_zeros(batch, self.decoder.config["quantizer_dimension"], 1)
        slots = torch.arange(batch, device=parameter.device)
        positions = torch.zeros(batch, dtype=torch.long, device=parameter.device)
        graph = None
        if self.capture:
            # Capture mutates streaming histories. Preserve every slot around warmup.
            state = {name: value.clone() for name, value in self.decoder.named_buffers()}
            for _ in range(3):
                self._forward(latent, slots, positions)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self._forward(latent, slots, positions)
            for name, value in self.decoder.named_buffers():
                value.copy_(state[name])
        else:
            output = None
        bucket = latent, slots, positions, graph, output
        self.buckets[batch] = bucket
        return bucket

    def decode(self, latent, slots, positions):
        if self.stream is None:
            return self._forward(latent, slots, positions), None
        main = torch.cuda.current_stream(latent.device)
        static_latent, static_slots, static_positions, graph, output = self._bucket(latent.shape[0])
        static_latent.copy_(latent)
        static_slots.copy_(slots)
        static_positions.copy_(positions)
        if graph is None:
            output = self._forward(static_latent, static_slots, static_positions)
        else:
            graph.replay()
        output = output.clone()
        finished = torch.cuda.Event()
        finished.record(main)
        return output, finished


class FlowSampleRunner:
    def __init__(self, model, *, capture=True, postprocess=None, state_buffers=()):
        self.model = model
        self.capture = capture and next(model.parameters()).is_cuda
        self.postprocess = postprocess
        self.state_buffers = state_buffers
        self.buckets = {}

    def _run(self, hidden, noise, slots):
        if self.postprocess is None:
            return self.model.sample(hidden, noise)
        return self.postprocess(hidden, noise, slots)

    def sample(self, hidden, noise, slots=None):
        if not self.capture:
            return self._run(hidden, noise, slots)
        batch = hidden.shape[0]
        if batch not in self.buckets:
            static_hidden = torch.zeros_like(hidden)
            static_noise = torch.zeros_like(noise)
            static_slots = torch.zeros_like(slots) if slots is not None else None
            saved_state = [buffer.clone() for buffer in self.state_buffers]
            stream = torch.cuda.Stream(device=hidden.device)
            stream.wait_stream(torch.cuda.current_stream(hidden.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._run(static_hidden, static_noise, static_slots)
            torch.cuda.current_stream(hidden.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self._run(static_hidden, static_noise, static_slots)
            for buffer, saved in zip(self.state_buffers, saved_state, strict=True):
                buffer.copy_(saved)
            self.buckets[batch] = static_hidden, static_noise, static_slots, graph, output
        static_hidden, static_noise, static_slots, graph, output = self.buckets[batch]
        static_hidden.copy_(hidden)
        static_noise.copy_(noise)
        if static_slots is not None:
            static_slots.copy_(slots)
        graph.replay()
        return output
