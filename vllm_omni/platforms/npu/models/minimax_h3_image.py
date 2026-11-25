# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""NPU H3 image VAE graphs with scoped runtime settings and opt-in batching."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from types import MethodType
from typing import TYPE_CHECKING

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch


logger = init_logger(__name__)


@contextmanager
def _npu_runtime_settings(*, disable_internal_format: bool, disable_jit_compile: bool) -> Iterator[None]:
    """Restore effective process settings after forward, including setup failures.

    TorchNPU's internal-format property is write-only. Its native option getter
    returns an empty value for the default (enabled). JIT's public getter resolves
    the effective device-dependent default. Avoid redundant JIT writes, which
    synchronize the device even when the mode is unchanged.
    """
    with ExitStack() as restore:
        if disable_internal_format:
            import torch_npu

            previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") != b"disable"
            if previous:
                restore.callback(setattr, torch.npu.config, "allow_internal_format", previous)
                torch.npu.config.allow_internal_format = False
        if disable_jit_compile and not torch.npu.is_jit_compile_false():
            restore.callback(torch.npu.set_compile_mode, jit_compile=True)
            torch.npu.set_compile_mode(jit_compile=False)
        yield


def apply_minimax_h3_image_graphs(model: nn.Module, od_config: OmniDiffusionConfig) -> None:
    """Validate H3 graph admission and install forward-scoped runtime controls.

    Implicit graphs fall back to the native VAE when the runtime is incompatible;
    explicit graph enablement fails before model or process state is modified.
    Approximate precision and batching are opt-in. Other pipelines are unchanged.
    """
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3_image import MiniMaxH3ImagePipeline

    if not isinstance(model, MiniMaxH3ImagePipeline):
        return
    additional = od_config.additional_config or {}
    deterministic = additional.get("npu_deterministic", True)
    if type(deterministic) is not bool:
        raise ValueError("npu_deterministic must be a bool")
    enabled = additional.get("npu_vae_tile_graph", True)
    if type(enabled) is not bool:
        raise ValueError("npu_vae_tile_graph must be a bool")
    disable_internal_format = additional.get("npu_disable_internal_format", False)
    if type(disable_internal_format) is not bool:
        raise ValueError("npu_disable_internal_format must be a bool")
    disable_jit_compile = additional.get("npu_disable_jit_compile", False)
    if type(disable_jit_compile) is not bool:
        raise ValueError("npu_disable_jit_compile must be a bool")
    enabled = enabled and not od_config.enforce_eager and od_config.enable_cuda_graph_decode
    if enabled:
        reason = None
        if os.environ.get("TASK_QUEUE_ENABLE") == "2":
            reason = "TASK_QUEUE_ENABLE=2; set TASK_QUEUE_ENABLE=1 or 0 before starting the worker"
        elif os.environ.get("ASCEND_LAUNCH_BLOCKING") == "1":
            reason = "ASCEND_LAUNCH_BLOCKING=1; unset it or set it to 0 before starting the worker"
        elif not NPUExactGraphRunner.is_supported():
            reason = "missing NPUGraph support"
        if reason is not None:
            if additional.get("npu_vae_tile_graph") is True:
                raise ValueError(f"npu_vae_tile_graph cannot be enabled: {reason}; or set npu_vae_tile_graph=false")
            logger.warning("MiniMax H3 image VAE graphs disabled: %s. Using native VAE execution.", reason)
            enabled = False

    # Finish validation before installing any wrapper or changing runtime state.
    encoder_precision = additional.get("npu_vae_encoder_precision", "fp32")
    encoder_batch = additional.get("npu_vae_encoder_batch", 1)
    stream_count = additional.get("npu_vae_tile_streams", 1)
    batch_size = additional.get("npu_vae_tile_batch", 1)
    if enabled:
        if encoder_precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("npu_vae_encoder_precision must be fp32, fp16, or bf16")
        if type(encoder_batch) is not int or encoder_batch not in (1, 2, 4, 8, 16):
            raise ValueError("npu_vae_encoder_batch must be 1, 2, 4, 8, or 16")
        if encoder_batch > 1 and encoder_precision == "fp32":
            raise ValueError("Batched VAE encoder currently requires reduced precision")
        if type(stream_count) is not int or not 1 <= stream_count <= 8:
            raise ValueError("npu_vae_tile_streams must be an integer in [1, 8]")
        if type(batch_size) is not int or batch_size not in (1, 2, 4, 8, 16, 32):
            raise ValueError("npu_vae_tile_batch must be 1, 2, 4, 8, 16, or 32")
        if batch_size > 1 and stream_count > 1:
            raise ValueError("VAE tile batching and multiple streams must be evaluated independently")

    if deterministic or enabled or disable_internal_format or disable_jit_compile:
        original_forward = model.forward

        def runtime_forward(module: nn.Module, request: DiffusionRequestBatch) -> DiffusionOutput:
            # Inline engines share the caller's process. Setup never changes
            # global settings, and forward restores them even on exceptions.
            with ExitStack() as restore:
                if deterministic:
                    previous = torch.are_deterministic_algorithms_enabled()
                    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
                    restore.callback(torch.use_deterministic_algorithms, previous, warn_only=warn_only)
                    torch.use_deterministic_algorithms(True)
                with _npu_runtime_settings(
                    disable_internal_format=enabled or disable_internal_format,
                    disable_jit_compile=enabled or disable_jit_compile,
                ):
                    return original_forward(request)

        model.forward = MethodType(runtime_forward, model)
    if not enabled:
        return
    vae = model.vae
    if getattr(vae, "_h3_npu_tile_graph_installed", False):
        return
    if encoder_precision != "fp32":
        encoder_dtype = torch.float16 if encoder_precision == "fp16" else torch.bfloat16
        encoder_runner = None
        encoder_signature = None
        original_encode = vae._encode_moments

        def encode_moments(module, pixels):
            nonlocal encoder_runner, encoder_signature
            if pixels.device.type != "npu":
                return original_encode(pixels)
            signature = tuple(
                (value.data_ptr(), value.dtype, value.device)
                for component in (module.encoder, module.quant_conv)
                for value in (*component.parameters(), *component.buffers())
            )
            if signature != encoder_signature:
                encoder_runner = NPUExactGraphRunner(max_graphs=8, component_name="MiniMax H3 image VAE encoder")
                encoder_signature = signature
            assert encoder_runner is not None
            # Weight casts must be graph-owned, not borrowed from an autocast
            # cache whose lifetime ends when this call returns.
            with torch.autocast(device_type="npu", dtype=encoder_dtype, cache_enabled=False):
                return encoder_runner.run("encode", (pixels,), (), lambda value: (original_encode(value),))[0]

        def map_encode_tiles(tiles):
            if not tiles:
                return []
            if any(tile.shape[0] != 1 or tile.shape[1:] != tiles[0].shape[1:] for tile in tiles):
                return [encode_moments(vae, tile) for tile in tiles]
            outputs = []
            for start in range(0, len(tiles), encoder_batch):
                packed = torch.cat(tiles[start : start + encoder_batch], dim=0)
                outputs.extend(encode_moments(vae, packed).split(1, dim=0))
            return outputs

        if encoder_batch > 1:
            setattr(encode_moments, "map", map_encode_tiles)
        vae._encode_moments = MethodType(encode_moments, vae)
    original_tile = vae._decode_tile
    original_prepare = vae.prepare_decoder_weights
    runner = None
    streams_runner = None
    weight_signature = None

    def prepare(module, dtype=torch.bfloat16):
        nonlocal runner, streams_runner, weight_signature
        original_prepare(dtype)
        # Casting or moving weights invalidates the addresses captured by a graph.
        signature = tuple(
            (value.data_ptr(), value.dtype, value.device)
            for component in (module.post_quant_conv, module.decoder)
            for value in (*component.parameters(), *component.buffers())
        )
        if signature != weight_signature:
            runner = None
            streams_runner = None
            weight_signature = signature

    def decode_tile(module, z):
        nonlocal runner
        if z.device.type != "npu":
            return original_tile(z)
        if runner is None:
            runner = NPUExactGraphRunner(
                max_graphs=4,
                component_name="MiniMax H3 image VAE tile",
                disable_config_hint="set npu_vae_tile_graph=false",
            )
        return runner.run("decode", (z,), (), lambda value: (original_tile(value),))[0]

    def map_tiles(tiles):
        nonlocal streams_runner, runner
        if not tiles:
            return []
        if tiles[0].device.type != "npu":
            return [original_tile(tile) for tile in tiles]
        if batch_size > 1:
            if any(tile.shape[0] != 1 or tile.shape[1:] != tiles[0].shape[1:] for tile in tiles):
                return [original_tile(tile) for tile in tiles]
            if runner is None:
                runner = NPUExactGraphRunner(
                    max_graphs=8,
                    component_name="MiniMax H3 batched VAE tiles",
                    disable_config_hint="set npu_vae_tile_batch=1",
                )
            outputs = []
            for start in range(0, len(tiles), batch_size):
                packed = torch.cat(tiles[start : start + batch_size], dim=0)
                decoded = runner.run("decode_batch", (packed,), (), lambda value: (original_tile(value),))[0]
                outputs.extend(decoded.split(1, dim=0))
            return outputs
        if streams_runner is None:
            streams_runner = _NPUTileGraphs(original_tile, autocast_dtype=torch.bfloat16)
            streams_runner.replicas = stream_count
        return streams_runner.map(tiles)

    if stream_count > 1 or batch_size > 1:
        setattr(decode_tile, "map", map_tiles)
    vae.prepare_decoder_weights = MethodType(prepare, vae)
    vae._decode_tile = MethodType(decode_tile, vae)
    vae._h3_npu_tile_graph_installed = True


class _NPUTileGraphs:
    """NPU graphs of ``fn`` per tile input shape, replayed on parallel streams.

    Each graph replays ``fn``'s eager single-tile kernels. One tile leaves most
    of the NPU idle, so ``map`` spreads tiles round-robin over ``replicas``
    independent graphs (own buffers and stream); every tile still runs the
    same kernels, so the outputs are unchanged.
    """

    max_shapes = 4
    replicas = 4

    def __init__(self, fn, *, autocast_dtype: torch.dtype) -> None:
        self.failed = False
        self.fn = fn
        self.autocast_dtype = autocast_dtype
        self.entries: dict[tuple, list[tuple[torch.npu.NPUGraph, torch.Tensor, torch.Tensor, torch.npu.Stream]]] = {}

    def _capture(self, z: torch.Tensor):
        static_in = z.clone()
        stream = torch.npu.Stream(device=z.device)
        stream.wait_stream(torch.npu.current_stream(z.device))
        with torch.npu.stream(stream), torch.autocast(z.device.type, dtype=self.autocast_dtype):
            # Warm up outside the graph (lazy allocations, kernel selection).
            for _ in range(2):
                self.fn(static_in)
        torch.npu.current_stream(z.device).wait_stream(stream)
        graph = torch.npu.NPUGraph()
        # Each replica owns its capture stream, workspace and static buffers.
        with torch.npu.graph(graph, stream=stream), torch.autocast(z.device.type, dtype=self.autocast_dtype):
            static_out = self.fn(static_in)
        return graph, static_in, static_out, stream

    def _replicas(self, z: torch.Tensor, count: int):
        key = (tuple(z.shape), z.dtype, z.device)
        entries = self.entries.get(key)
        if entries is None:
            if len(self.entries) >= self.max_shapes:
                return None
            entries = self.entries[key] = []
        while len(entries) < count:
            try:
                entries.append(self._capture(z))
            except Exception:
                self.failed = True
                raise
        return entries

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return self.map([z])[0]

    def map(self, tiles: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.failed:
            raise RuntimeError("NPU tile capture failed; restart the stage process")
        if not tiles:
            return []
        if any(tile.shape != tiles[0].shape for tile in tiles):
            return [self.map([tile])[0] for tile in tiles]
        entries = self._replicas(tiles[0], min(self.replicas, len(tiles)))
        if entries is None:
            return [self.fn(tile) for tile in tiles]
        current = torch.npu.current_stream(tiles[0].device)
        for *_, stream in entries:
            stream.wait_stream(current)
        outputs = []
        for index, tile in enumerate(tiles):
            graph, static_in, static_out, stream = entries[index % len(entries)]
            with torch.npu.stream(stream):
                # Stage input and clone output before this replica is reused.
                static_in.copy_(tile)
                graph.replay()
                output = static_out.clone()
            # The consumer stream reads the output after the replica's stream.
            output.record_stream(current)
            outputs.append(output)
        for *_, stream in entries:
            current.wait_stream(stream)
        return outputs
