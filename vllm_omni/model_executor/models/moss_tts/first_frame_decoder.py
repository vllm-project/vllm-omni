# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in MOSS Local first-frame decoding in the Talker process.

The default path resets a private streaming state pool after each call. The
experimental empty-history path specializes attention and owns no stream
state. Both retain every upsampling block; the regular codec stage
independently primes its persistent state with the same codes.
"""

from __future__ import annotations

import copy

import torch
from torch import nn


def first_audio_enabled(config) -> bool:
    connector = getattr(config.model_config, "stage_connector_config", {}) or {}
    extra = connector.get("extra", connector) if isinstance(connector, dict) else getattr(connector, "extra", {})
    enabled = (extra or {}).get("moss_talker_first_audio", False)
    if not isinstance(enabled, bool):
        raise ValueError("moss_talker_first_audio must be a boolean")
    if not enabled:
        return False
    model, parallel = config.model_config, config.parallel_config
    supported = (
        torch.device(config.device_config.device).type == "cuda"
        and bool(getattr(model, "use_v2_model_runner", False))
        and bool(getattr(model, "async_chunk", False))
        and parallel.tensor_parallel_size == parallel.pipeline_parallel_size == 1
        and parallel.distributed_executor_backend in (None, "uni")
        and not config.cache_config.enable_prefix_caching
        and getattr(config, "speculative_config", None) is None
    )
    if not supported:
        raise ValueError(
            "MOSS first audio requires CUDA MRV2 async chunks, in-process TP/PP=1, no prefix cache/speculation"
        )
    return True


class MossFirstFrameDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        from .modeling_moss_tts_codec import MossTTSCodecDecoder

        cfg = copy.copy(config)
        cfg.model_config = copy.copy(config.model_config)
        cfg.model_config.hf_config = copy.deepcopy(config.model_config.hf_config)
        cfg.model_config.hf_config.codec_async_output = True
        cfg.model_config.hf_config.codec_attention_backend = "triton_slot"
        cfg.model_config.hf_config.codec_private_graph_pool = True
        cfg.scheduler_config = copy.copy(config.scheduler_config)
        cfg.scheduler_config.max_num_seqs = 8
        cfg.compilation_config = copy.copy(config.compilation_config)
        cfg.compilation_config.cudagraph_capture_sizes = [1, 2, 4, 8]
        cfg.compilation_config.max_cudagraph_capture_size = 8
        self._empty_history = bool(getattr(cfg.model_config.hf_config, "moss_first_frame_empty_history", False))
        if self._empty_history:
            # Skip the general streaming graph setup; the private first-only
            # graph bank below owns compilation/capture and has no KV pool.
            cfg.model_config.enforce_eager = True
        self.decoder = MossTTSCodecDecoder(vllm_config=cfg)
        self.decoder._initial_stream_chunk_frames = 1
        self.decoder._stream_chunk_frames = 1
        self.decoder._stream_max_step_frames = 1
        self.decoder._streaming_graph_frame_sizes = [1]

    def load(self) -> set[str]:
        self.decoder.load_weights(())
        # Reference encoding belongs to the API processor, never this decoder.
        self.decoder._codec.encoder = None
        if self._empty_history:
            from .first_frame_special import StatelessFirstGraphs, specialize

            specialize(self.decoder._codec)
            self._special_graphs = StatelessFirstGraphs(self.decoder._codec)
        return set(dict(self.named_parameters()))

    @property
    def sample_rate(self) -> torch.Tensor:
        return self.decoder._sr_tensor

    def warmup(self) -> None:
        """Compatibility with the post-backbone graph hook; load captured us."""
        if self._empty_history:
            if not getattr(self, "_special_graphs", None):
                raise RuntimeError("First decoder must load before graph warmup")
        else:
            self.decoder._ensure_stream_session()

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """[B, NQ] codes -> owned float32 [B, channels, samples] PCM."""
        if self._empty_history:
            return self._special_graphs(codes)
        session = self.decoder._ensure_stream_session()
        parts = []
        for start in range(0, codes.shape[0], 8):
            chunk = codes[start : start + 8]
            slots = [session.acquire() for _ in range(len(chunk))]
            if any(slot is None for slot in slots):
                raise RuntimeError("First-frame decoder exhausted its private state pool")
            try:
                output = session.step(
                    {slot: chunk[row, :, None] for row, slot in enumerate(slots)},
                    terminal_slots=set(slots),
                )
                parts.append(torch.stack([output[slot] for slot in slots]))
            finally:
                for slot in slots:
                    session.release(slot)
        return torch.cat(parts, dim=0)
