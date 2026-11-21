# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Complete ZAC decode graphs: RVQ → neural decoder → iSTFT/FIR → resampling.

Buckets are keyed by (num_seqs, total RVQ tokens), with one global slab tail.
Per-request lengths govern state commits and audio coverage; final remains a
device input. Eager and graph execution share the packed neural/synthesis core.
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

if TYPE_CHECKING:
    from .codec import ZACDecoderNetwork
    from .model_state import Zonos3RVQBuffer

logger = init_logger(__name__)


def _cuda_check(result: tuple, operation: str) -> tuple:
    if int(result[0]) != 0:
        raise RuntimeError(f"Codec SWITCH {operation} failed: {result[0]}")
    return result[1:]


class _CodecSwitch:
    """GPU integer dispatch, built with CUDA bindings (CUDA 12.8+).

    PyTorch's conditional capture API only exposes IF/WHILE. Compile the tiny
    handle setter before capture; retain its module for the graphs' lifetime.
    Each SWITCH body contains one already-captured packed core, so no Torch
    allocations or conditional capture contexts are needed inside the bodies.
    """

    def __init__(self, device: torch.device) -> None:
        from cuda.bindings import driver, nvrtc, runtime

        (version,) = _cuda_check(runtime.cudaRuntimeGetVersion(), "runtime version")
        if version < 12080:
            raise RuntimeError("Zonos3 Codec SWITCH graphs require CUDA 12.8 or newer")

        source = b"""
typedef unsigned long long cudaGraphConditionalHandle;
extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(cudaGraphConditionalHandle, unsigned int);
extern "C" __global__ void set_bucket(unsigned long long handle, const int *value) {
    cudaGraphSetConditional(handle, (unsigned int)*value);
}
"""
        major, minor = torch.cuda.get_device_capability(device)
        options = [f"--gpu-architecture=compute_{major}{minor}".encode(), b"--std=c++17"]
        (program,) = _cuda_check(nvrtc.nvrtcCreateProgram(source, b"zonos3_switch.cu", 0, [], []), "create setter")
        try:
            compiled = nvrtc.nvrtcCompileProgram(program, len(options), options)
            if int(compiled[0]) != 0:
                (size,) = _cuda_check(nvrtc.nvrtcGetProgramLogSize(program), "setter log size")
                log = bytearray(size)
                _cuda_check(nvrtc.nvrtcGetProgramLog(program, log), "setter log")
                raise RuntimeError(f"Codec SWITCH setter compilation failed: {log.decode()}")
            (size,) = _cuda_check(nvrtc.nvrtcGetPTXSize(program), "setter PTX size")
            ptx = bytearray(size)
            _cuda_check(nvrtc.nvrtcGetPTX(program, ptx), "setter PTX")
        finally:
            _cuda_check(nvrtc.nvrtcDestroyProgram(program), "destroy setter program")
        (self.module,) = _cuda_check(driver.cuModuleLoadData(bytes(ptx)), "load setter module")
        (self.function,) = _cuda_check(driver.cuModuleGetFunction(self.module, b"set_bucket"), "find setter")
        _cuda_check(driver.cuFuncLoad(self.function), "preload setter")

    def capture(self, index: torch.Tensor, cores: list[torch.cuda.CUDAGraph]) -> None:
        from cuda.bindings import driver, runtime

        stream = torch.cuda.current_stream().cuda_stream
        _, _, parent, _, _, _ = _cuda_check(runtime.cudaStreamGetCaptureInfo(stream), "inspect parent capture")
        (handle,) = _cuda_check(runtime.cudaGraphConditionalHandleCreate(parent, 0, 0), "create handle")
        handle_arg = ctypes.c_ulonglong(int(handle))
        index_arg = ctypes.c_void_p(index.data_ptr())
        arguments = (ctypes.c_void_p * 2)(ctypes.addressof(handle_arg), ctypes.addressof(index_arg))
        _cuda_check(
            driver.cuLaunchKernel(
                self.function, 1, 1, 1, 1, 1, 1, 0, driver.CUstream(stream), ctypes.addressof(arguments), 0
            ),
            "capture setter",
        )
        _, _, parent, dependencies, edges, count = _cuda_check(
            runtime.cudaStreamGetCaptureInfo(stream), "inspect setter dependencies"
        )
        params = runtime.cudaGraphNodeParams()
        params.type = runtime.cudaGraphNodeType.cudaGraphNodeTypeConditional
        params.conditional.handle = handle
        params.conditional.type = runtime.cudaGraphConditionalNodeType.cudaGraphCondTypeSwitch
        params.conditional.size = len(cores)
        (node,) = _cuda_check(runtime.cudaGraphAddNode(parent, dependencies, edges, count, params), "add node")
        for i, core in enumerate(cores):
            _cuda_check(
                runtime.cudaGraphAddChildGraphNode(
                    params.conditional.phGraph_out[i], [], 0, runtime.cudaGraph_t(core.raw_cuda_graph())
                ),
                f"attach bucket {i}",
            )
        _cuda_check(
            runtime.cudaStreamUpdateCaptureDependencies(
                stream,
                [node],
                None,
                1,
                runtime.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamSetCaptureDependencies,
            ),
            "join parent capture",
        )


class ZACCodecGraphWrapper:
    """GPU selection of the original (ready B, packed T) Codec buckets.

    A single replay owns readiness/compaction and mutually exclusive Codec
    branches. Neural/synthesis shapes match the two-stage buckets: scheduled
    but not-ready requests never inflate the executed Codec cohort. Private
    graph memory cannot alias concurrently running native AR graphs.
    """

    def __init__(
        self, network: ZACDecoderNetwork, assembly: Zonos3RVQBuffer, capacity: int, max_chunk: int, eager: bool
    ) -> None:
        from .codec import ZACPackedCodec

        self.assembly, self.capacity, self.max_chunk = assembly, capacity, max_chunk
        self.eager = eager
        self.codec = ZACPackedCodec(network, 2 * capacity, max_chunk, capacity * max_chunk)
        batches = sorted({min(capacity, 1 << i) for i in range((capacity - 1).bit_length() + 1)})
        self.capture_keys = [(b, b * t) for b in batches for t in sorted({1, 2, 4, 8, 15, max_chunk}) if t <= max_chunk]
        self.bucket = None
        self.pool = None
        self.cores = {}
        self.bridge = None

    def _execute(self, codes, slots, active, final, conditional=False):
        packed, destinations, lengths, flush, skip, order = self.assembly.prepare(
            codes, slots, active, final, slots.numel() * self.max_chunk
        )
        offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
        if not conditional:
            # Eager shares the exact packed core. Inactive lanes have unique
            # scratch slots; no GPU predicate is read on the host.
            audio, pcm_offsets, counts = self.codec(packed, destinations, offsets, flush)
            return audio, pcm_offsets, counts, skip, order

        bridge_codes, bridge_slots, bridge_offsets, bridge_final = self.bridge
        bridge_codes.copy_(packed)
        bridge_slots.copy_(destinations)
        bridge_offsets.copy_(offsets)
        bridge_final.copy_(flush)
        audio, pcm_offsets, counts = self.output
        pcm_offsets.zero_()
        counts.zero_()
        ready_count = (lengths.gt(0) | flush).long().sum()
        eligible = (ready_count > 0) & (ready_count <= self.bucket_batches) & (offsets[-1] <= self.bucket_tokens)
        # Ordered (B, T) keys select the smallest ready cohort, then token slab.
        # A zero-token final still runs; only no-ready uses the out-of-range case.
        self.case_index.copy_(torch.where(eligible, self.case_ids, len(self.capture_keys)).amin())
        self.switch.capture(self.case_index, [self.cores[key][0] for key in self.capture_keys])
        return audio, pcm_offsets, counts, skip, order

    def _audio_capacity(self, batch: int, tokens: int) -> int:
        synthesis = self.codec.synthesis
        wave = tokens * self.codec.scale * synthesis.ola.hop + batch * synthesis.ola.lookahead
        mixed = wave + batch * synthesis.half_fir
        return (mixed * synthesis.new + synthesis.orig - 1) // synthesis.orig + batch * synthesis.new

    @torch.inference_mode()
    def capture(self) -> None:
        device = self.codec.state.frames.device
        if self.eager or device.type != "cuda":
            return
        from vllm_omni.utils.cuda_stream import create_cuda_stream

        self.capture_stream = create_cuda_stream(device)
        self.capture_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(self.capture_stream):
            self._capture(device)
        self.capture_stream.synchronize()

    def _capture(self, device: torch.device) -> None:
        codes = torch.zeros(self.capacity, 10, dtype=torch.long, device=device)
        slots = torch.arange(self.capacity, 2 * self.capacity, device=device)
        active = torch.zeros(self.capacity, dtype=torch.bool, device=device)
        final = torch.zeros_like(active)
        bridge_codes = torch.zeros(self.capacity * self.max_chunk, 10, dtype=torch.long, device=device)
        bridge_slots = slots.clone()
        bridge_offsets = torch.zeros(self.capacity + 1, dtype=torch.long, device=device)
        bridge_final = final.clone()
        self.bridge = (bridge_codes, bridge_slots, bridge_offsets, bridge_final)
        # Stable parent outputs live outside the shared core workspace pool.
        self.output = (
            torch.empty(self._audio_capacity(self.capacity, bridge_codes.shape[0]), dtype=torch.float32, device=device),
            torch.zeros_like(bridge_offsets),
            torch.zeros(self.capacity, dtype=torch.long, device=device),
        )
        self.bucket_batches = torch.tensor([b for b, _ in self.capture_keys], dtype=torch.long, device=device)
        self.bucket_tokens = torch.tensor([t for _, t in self.capture_keys], dtype=torch.long, device=device)
        self.case_ids = torch.arange(len(self.capture_keys), dtype=torch.int32, device=device)
        self.case_index = torch.empty((), dtype=torch.int32, device=device)
        self.switch = _CodecSwitch(device)
        self.pool = torch.cuda.graph_pool_handle()
        capture_stream = torch.cuda.current_stream()
        # Separate captures let the allocator reuse intermediate workspaces
        # across mutually exclusive buckets, as in the two-stage wrapper.
        for batch, tokens in reversed(self.capture_keys):
            arguments = (bridge_codes[:tokens], bridge_slots[:batch], bridge_offsets[: batch + 1], bridge_final[:batch])
            for _ in range(2):
                self.codec(*arguments)
            torch.accelerator.synchronize()
            core = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(core, pool=self.pool, stream=capture_stream):
                result = self.codec(*arguments)
                values, boundaries, valid = result
                audio, pcm_offsets, counts = self.output
                audio[: values.numel()].copy_(values)
                pcm_offsets[: batch + 1].copy_(boundaries)
                pcm_offsets[batch + 1 :].copy_(boundaries[-1].expand(self.capacity - batch))
                counts[:batch].copy_(valid)
            self.cores[batch, tokens] = (core, result)
        # Parent metadata/output allocations must not alias a child's shared
        # intermediates. Only the serial Codec cores share a private pool.
        graph = torch.cuda.CUDAGraph()
        self._execute(codes, slots, active, final)
        torch.accelerator.synchronize()
        with torch.cuda.graph(graph, stream=capture_stream):
            output = self._execute(codes, slots, active, final, conditional=True)
        self.bucket = (graph, codes, slots, active, final, output)
        logger.info("Captured single-SWITCH Codec with %d (ready B, sigma_T) bodies", len(self.capture_keys))

    def __call__(self, codes, slots, active, final):
        if self.bucket is None:
            return self._execute(codes, slots, active, final)
        graph, static_codes, static_slots, static_active, static_final, output = self.bucket
        n = slots.numel()
        static_codes.zero_()
        static_codes[:n].copy_(codes)
        static_slots.copy_(torch.arange(self.capacity, 2 * self.capacity, device=slots.device))
        static_slots[:n].copy_(slots)
        static_active.zero_()
        static_active[:n].copy_(active)
        static_final.zero_()
        static_final[:n].copy_(final)
        graph.replay()
        audio, offsets, counts, skip, order = output
        # A small scheduled cohort cannot produce the global capacity's PCM.
        # Bound the owned snapshot by CPU-known input shape, without reading
        # readiness/lengths or copying the B=max capacity tail every AR step.
        return audio[: self._audio_capacity(n, n * self.max_chunk)], offsets[: n + 1], counts[:n], skip[:n], order[:n]
