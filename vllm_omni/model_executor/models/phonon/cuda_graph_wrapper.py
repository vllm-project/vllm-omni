# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fixed-address codec graphs and single-in-flight side-stream handoff."""

import torch


class MimiDecodeRunner:
    def __init__(self, decoder, *, capture=True):
        self.decoder = decoder
        device = next(decoder.parameters()).device
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self.capture = capture and self.stream is not None
        self.buckets = {}
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream(device))

    def reset_slots(self, slot: int):
        if self.stream is None:
            self.decoder.reset_slots(slot)
            return
        # Decode, owned PCM snapshot and slot reset share the same stream order.
        with torch.cuda.stream(self.stream):
            self.decoder.reset_slots(slot)

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
                self.decoder(latent, slots, positions)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.stream):
                output = self.decoder(latent, slots, positions)
            for name, value in self.decoder.named_buffers():
                value.copy_(state[name])
        else:
            output = None
        bucket = latent, slots, positions, graph, output
        self.buckets[batch] = bucket
        return bucket

    def decode(self, latent, slots, positions):
        if self.stream is None:
            return self.decoder(latent, slots, positions), None
        main = torch.cuda.current_stream(latent.device)
        ready = torch.cuda.Event()
        ready.record(main)
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)
            static_latent, static_slots, static_positions, graph, output = self._bucket(latent.shape[0])
            static_latent.copy_(latent)
            static_slots.copy_(slots)
            static_positions.copy_(positions)
            copied = torch.cuda.Event()
            copied.record(self.stream)
            if graph is None:
                output = self.decoder(static_latent, static_slots, static_positions)
            else:
                graph.replay()
            # Own output storage before the next replay overwrites the graph output.
            output = output.clone()
            finished = torch.cuda.Event()
            finished.record(self.stream)
        # The producer may reuse Flow's output once the side-stream copy completes.
        main.wait_event(copied)
        return output, finished


class FlowSampleRunner:
    def __init__(self, model, *, capture=True):
        self.model = model
        self.capture = capture and next(model.parameters()).is_cuda
        self.buckets = {}

    def sample(self, hidden, noise):
        if not self.capture:
            return self.model.sample(hidden, noise)
        batch = hidden.shape[0]
        if batch not in self.buckets:
            static_hidden = torch.zeros_like(hidden)
            static_noise = torch.zeros_like(noise)
            stream = torch.cuda.Stream(device=hidden.device)
            stream.wait_stream(torch.cuda.current_stream(hidden.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.model.sample(static_hidden, static_noise)
            torch.cuda.current_stream(hidden.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self.model.sample(static_hidden, static_noise)
            self.buckets[batch] = static_hidden, static_noise, graph, output
        static_hidden, static_noise, graph, output = self.buckets[batch]
        static_hidden.copy_(hidden)
        static_noise.copy_(noise)
        graph.replay()
        return output
