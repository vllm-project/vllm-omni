# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single-frame VoiceChat perception through vLLM's encoder graph manager."""

from collections import Counter
from collections.abc import Hashable
from copy import copy
from typing import cast

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.multimodal import MultiModalConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
    EncoderCudaGraphPathConfig,
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)

from vllm_omni.platforms import current_omni_platform

from .nemo_vendored.perception import AudioPerceptionModule

logger = init_logger(__name__)

Cache = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
Frame = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
Phase = tuple[int, int]
_INPUT_KEYS = ("mel", "channel", "time", "valid")
_OUTPUT_KEYS = ("length", "next_channel", "next_time", "next_valid")


def perception_step(perception: AudioPerceptionModule, mel: torch.Tensor, caches: Cache, drop: int) -> Frame:
    encoded, length, channel, time, valid = perception.encoder.cache_aware_stream_step(
        processed_signal=mel,
        processed_signal_length=torch.full((mel.shape[0],), mel.shape[-1], device=mel.device, dtype=torch.long),
        cache_last_channel=caches[0],
        cache_last_time=caches[1],
        cache_last_channel_len=caches[2],
        keep_all_outputs=True,
        drop_extra_pre_encoded=drop,
    )
    encoded, length = perception.modality_adapter(audio_signal=encoded, length=length)
    return perception.proj(encoded.transpose(1, 2)), length, channel, time, valid


def _copy_exact(target: torch.Tensor, source: torch.Tensor) -> None:
    if target.shape != source.shape:
        raise ValueError(f"VoiceChat perception buffer shape mismatch: {target.shape} != {source.shape}")
    target.copy_(source)


class _PerceptionEncoder(SupportsEncoderCudaGraph):
    """A ready audio frame is one item; caches remain explicit tensor inputs."""

    def __init__(self, perception: AudioPerceptionModule, phases: tuple[Phase, ...], caches: Cache) -> None:
        self.perception = perception
        self.phases = phases
        self.paths = dict(zip(("default", "steady"), phases))
        self.caches = caches
        self.outputs: dict[Phase, tuple[torch.Tensor, ...]] = {}

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        return EncoderCudaGraphConfig(
            modalities=["audio"],
            buffer_keys=list(_INPUT_KEYS),
            out_hidden_size=int(self.perception.cfg.output_dim),
            padding_logics=dict.fromkeys(_INPUT_KEYS, _copy_exact),
            paths={path: EncoderCudaGraphPathConfig(allow_zero_tokens=True) for path in self.paths},
        )

    def get_encoder_cudagraph_budget_range(self, vllm_config: VllmConfig) -> tuple[int, int]:
        return 1, 1

    def get_encoder_cudagraph_item_specs(self, mm_kwargs: dict[str, object]) -> list[EncoderItemSpec]:
        phase = self.phase(mm_kwargs)
        return [
            EncoderItemSpec(
                input_size=1,
                output_tokens=1,
                path_output_tokens={path: int(phase == geometry) for path, geometry in self.paths.items()},
            )
        ]

    def select_encoder_cudagraph_items(self, mm_kwargs: dict[str, object], indices: list[int]) -> dict[str, object]:
        assert indices == [0]
        return dict(mm_kwargs)

    @staticmethod
    def phase(inputs: dict[str, object]) -> Phase:
        mel = cast(torch.Tensor, inputs["mel"])
        return mel.shape[-1], cast(int, inputs["drop"])

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
        axis_keys: tuple[Hashable, ...] | None = (),
    ) -> EncoderCudaGraphCaptureInputs:
        phase = self.paths[path]
        values = {
            "mel": torch.zeros(1, self.perception.encoder._feat_in, phase[0], device=device, dtype=dtype),
            **{key: torch.zeros_like(cache) for key, cache in zip(_INPUT_KEYS[1:], self.caches)},
        }
        outputs = (torch.empty_like(values["valid"]), *(torch.empty_like(values[key]) for key in _INPUT_KEYS[1:]))
        # Capture replaces eager buffers; run() gives callers private copies of state.
        self.outputs[phase] = outputs
        values.update(zip(_OUTPUT_KEYS, outputs))
        return EncoderCudaGraphCaptureInputs(values=values)

    def prepare_encoder_cudagraph_replay_buffers(
        self, mm_kwargs: dict[str, object], max_batch_size: int, max_frames_per_batch: int, path: str = "default"
    ) -> EncoderCudaGraphReplayBuffers:
        return EncoderCudaGraphReplayBuffers({key: cast(torch.Tensor, mm_kwargs[key]) for key in _INPUT_KEYS})

    def encoder_cudagraph_forward(self, inputs: dict[str, torch.Tensor], path: str = "default") -> torch.Tensor:
        drop = self.paths[path][1]
        caches = inputs["channel"], inputs["time"], inputs["valid"]
        frame = perception_step(self.perception, inputs["mel"], caches, drop)
        for key, output in zip(_OUTPUT_KEYS, frame[1:]):
            _copy_exact(inputs[key], output)
        return frame[0]

    def encoder_eager_forward(self, mm_kwargs: dict[str, object], path: str = "default") -> torch.Tensor:
        phase = self.phase(mm_kwargs)
        outputs = self.outputs.get(phase)
        mel = cast(torch.Tensor, mm_kwargs["mel"])
        caches = cast(Cache, tuple(mm_kwargs[key] for key in _INPUT_KEYS[1:]))
        frame = perception_step(self.perception, mel, caches, cast(int, mm_kwargs["drop"]))
        if outputs is None:
            # An uncaptured phase needs its own buffers, independent of eager outputs.
            outputs = tuple(torch.empty_like(value) for value in frame[1:])
            self.outputs[phase] = outputs
        # Preserve the buffers that later graph replays write into.
        for target, output in zip(outputs, frame[1:]):
            _copy_exact(target, output)
        return frame[0]

    def postprocess_encoder_output(
        self,
        outputs: dict[str, torch.Tensor],
        indices: list[int],
        per_item_out_tokens: list[int],
        dest: dict[int, torch.Tensor] | list[torch.Tensor | None],
        clone: bool = False,
        batch_mm_kwargs: dict[str, object] | None = None,
    ) -> None:
        output = next(iter(outputs.values()))
        dest[0] = output.clone() if clone else output


class PerceptionCudaGraph:
    """Two fixed phase graphs; caller owns all returned state across replays."""

    def __init__(self, perception: AudioPerceptionModule, vllm_config: VllmConfig, phases: tuple[Phase, ...]) -> None:
        self.perception = perception
        parameter = next(perception.encoder.parameters())
        self.device, self.dtype = parameter.device, parameter.dtype
        caches = perception.encoder.get_initial_cache_state(batch_size=1, dtype=self.dtype, device=self.device)
        self.adapter = _PerceptionEncoder(perception, phases, caches)
        self._fallback_reasons: Counter[str] = Counter()
        # A frame and its caches are already placed locally and cannot be split.
        config = copy(vllm_config)
        config.compilation_config = copy(config.compilation_config)
        config.compilation_config.encoder_cudagraph_token_budgets = [1]
        config.compilation_config.encoder_cudagraph_max_vision_items_per_batch = 1
        config.compilation_config.encoder_cudagraph_max_frames_per_batch = 0
        config.model_config = copy(config.model_config)
        config.model_config.multimodal_config = MultiModalConfig(limit_per_prompt={"audio": 1, "video": 0})
        self.manager = EncoderCudaGraphManager(config, self.device, self.dtype, self.adapter)
        self.stream = torch.cuda.current_stream(self.device)
        # Capture must not change the stream used by the rest of the worker.
        capture_stream = torch.cuda.Stream(device=self.device)
        capture_stream.wait_stream(self.stream)
        with torch.cuda.stream(capture_stream):
            self.manager.capture(graph_pool=None)
        self.stream.wait_stream(capture_stream)

    def _ineligible_reason(self, mel: torch.Tensor, caches: Cache, drop: int) -> str | None:
        if torch.is_grad_enabled():
            return "grad_enabled"
        if torch.is_autocast_enabled("cuda"):
            return "autocast"
        if self.perception.training:
            return "training"
        # Both mel width and drop select the phase's capture geometry.
        if (mel.shape[-1], drop) not in self.adapter.phases:
            return "phase"
        if mel.shape[:2] != (1, self.perception.encoder._feat_in):
            return "shape"
        if mel.device != self.device:
            return "device"
        if mel.dtype != self.dtype:
            return "dtype"
        if any(
            (x.shape, x.dtype, x.device) != (y.shape, y.dtype, y.device) for x, y in zip(caches, self.adapter.caches)
        ):
            return "cache"
        if torch.cuda.current_stream(self.device) != self.stream:
            return "stream"
        return None

    def get_cumulative_stats(self) -> dict[str, object]:
        """Include eager fallbacks that bypass the encoder graph manager."""
        stats = self.manager.get_cumulative_stats()
        ineligible = sum(self._fallback_reasons.values())
        hits, misses = stats["graph_hits"], stats["graph_misses"] + ineligible
        return {
            **stats,
            "graph_misses": misses,
            "ineligible_misses": ineligible,
            "fallback_reasons": dict(self._fallback_reasons),
            "hit_rate": hits / (hits + misses) if hits + misses else 0.0,
        }

    def run(self, mel: torch.Tensor, caches: Cache, drop: int) -> Frame:
        reason = self._ineligible_reason(mel, caches, drop)
        if reason is not None:
            first_fallback = not self._fallback_reasons
            self._fallback_reasons[reason] += int(mel.shape[0])
            if first_fallback:
                logger.warning(
                    "VoiceChat perception CUDA Graph is enabled but a frame runs eagerly (%s); "
                    "get_cumulative_stats() includes fallback counts and reasons.",
                    reason,
                )
            return perception_step(self.perception, mel, caches, drop)
        phase = mel.shape[-1], drop
        inputs: dict[str, object] = {"mel": mel, "drop": drop, **dict(zip(_INPUT_KEYS[1:], caches))}
        encoded = self.manager.execute(inputs)[0]
        # Manager owns reusable buffers; private copies prevent cross-session overwrite.
        length, channel, time, valid = (value.clone() for value in self.adapter.outputs[phase])
        return encoded, length, channel, time, valid


def make_perception_graph(
    perception: AudioPerceptionModule, vllm_config: VllmConfig, phases: tuple[Phase, ...]
) -> PerceptionCudaGraph | None:
    compilation = vllm_config.compilation_config
    if (
        not compilation.cudagraph_mm_encoder
        or vllm_config.model_config.enforce_eager
        or compilation.cudagraph_mode == CUDAGraphMode.NONE
        or not current_omni_platform.is_cuda()
        or (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1)
    ):
        return None
    return PerceptionCudaGraph(perception, vllm_config, tuple(dict.fromkeys(phases)))
