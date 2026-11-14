# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SolarX Realization decoder, pinned to a7c61a6e02bef8539178a17b6e150a775b3a0ebb.

Upstream: https://github.com/SolarX-Team/Realization-Streaming-VAE-Decoder
The upstream runtime/tables are Apache-2.0; see LICENSE. Local changes expose
three spatial tile hooks. Temporal attention, refresh and compensation stay
unchanged. Graphs wrap only local layer passes, outside tile collectives.
"""

import torch

from .stream_decoder import StreamingDecoder


class TileStreamingDecoder(StreamingDecoder):
    def __init__(self, vae, *, group=None, use_graph=True):
        self.group = group
        self.graphs = {}
        self.layouts = {}
        self.capture_count = 0
        self.use_graph = use_graph
        super().__init__(vae, first_refresh=True, compensate=True, compile=False)

    def _make_plan(self, z):
        if self._plan is not None:
            self.layouts[self._plan["latent"]] = self._plan, self._k, self._v, self._rope_cache, self.graphs
        key = tuple(z.shape[-2:])
        if key in self.layouts:
            plan, self._k, self._v, self._rope_cache, self.graphs = self.layouts[key]
            return plan
        self.graphs = {}
        return super()._make_plan(z)

    def _local_tile_count(self, total):
        if self.group is None:
            return total
        return len(range(self.group.rank_in_group, total, self.group.world_size))

    def _owned_tiles(self, tiles):
        if self.group is None:
            return tiles
        return tiles[self.group.rank_in_group :: self.group.world_size]

    def _blend(self, tiles):
        if self.group is not None:
            tiles = torch.cat(self.vae._all_gather_tiled_results(list(tiles.split(1)), len(self._plan["tiles"])))
        return super()._blend(tiles)

    def _layers(self, span, z, fresh, slot, rope):
        if not self.use_graph:
            return super()._layers(span, z, fresh, slot, rope)
        key = (tuple(fresh), slot, span.start, span.stop, tuple(z.shape))
        if key not in self.graphs:
            stream = torch.cuda.Stream(device=z.device)
            stream.wait_stream(torch.cuda.current_stream(z.device))
            with torch.cuda.stream(stream):
                static = z.clone()
                for _ in range(2):
                    result = super()._layers(span, static, fresh, slot, rope)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    result = super()._layers(span, static, fresh, slot, rope)
            torch.cuda.current_stream(z.device).wait_stream(stream)
            self.graphs[key] = static, result, graph
            self.capture_count += 1
        static, result, graph = self.graphs[key]
        static.copy_(z)
        graph.replay()
        return result
