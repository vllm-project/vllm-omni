# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage-1 Code2Wav model (48 kHz stereo SpectroStream) for Google Magenta RealTime 2."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import safetensors
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

from vllm_omni.model_executor.models.magenta_rt2.spectrostream import (
    SpectroStreamModel,
    SpectroStreamRowState,
    _unflatten_nested,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.transformers_utils.configs.magenta_rt2 import (
    MagentaRT2Config,
    magenta_rt2_allow_patterns,
    resolve_magenta_rt2_checkpoint,
)

logger = init_logger(__name__)


def _payload_codes(runtime_info: Mapping[str, Any] | None) -> torch.Tensor | list | tuple | None:
    if not isinstance(runtime_info, Mapping):
        return None
    codes = runtime_info.get("codes")
    if isinstance(codes, Mapping):
        audio = codes.get("audio")
        if isinstance(audio, torch.Tensor) and audio.numel() > 0:
            return audio
        if isinstance(audio, (list, tuple)) and audio:
            return audio
    audio = runtime_info.get("codes.audio")
    if isinstance(audio, torch.Tensor) and audio.numel() > 0:
        return audio
    if isinstance(audio, (list, tuple)) and audio:
        return audio
    return None


def _codec_ids_from_payload_or_input(
    input_ids: torch.Tensor,
    runtime_info: Mapping[str, Any] | None,
) -> torch.Tensor:
    audio = _payload_codes(runtime_info)
    if isinstance(audio, torch.Tensor):
        return audio.reshape(-1).to(device=input_ids.device, dtype=torch.long)
    if audio is not None:
        return torch.as_tensor(audio, device=input_ids.device, dtype=torch.long).reshape(-1)
    return input_ids.reshape(-1).to(dtype=torch.long)


class SpectroStreamDecodeGraphRunner:
    """Fixed-shape CUDA graph capturing 1-frame streaming ``SpectroStreamModel.step`` for all rows."""

    def __init__(
        self,
        codec: SpectroStreamModel,
        state: SpectroStreamRowState,
        num_rows: int,
        num_codebooks: int,
        device: torch.device,
    ) -> None:
        self.codec = codec
        self.state = state
        self.num_rows = int(num_rows)
        self.num_codebooks = int(num_codebooks)
        self.device = device
        with torch.inference_mode(False):
            self._codes_buf = torch.zeros((self.num_rows, 1, self.num_codebooks), dtype=torch.long, device=device)
            self._active_mask_buf = torch.ones((self.num_rows,), dtype=torch.bool, device=device)
        self._out_buf: torch.Tensor | None = None
        self._graph: torch.cuda.CUDAGraph | None = None

    def capture(self) -> None:
        if self.device.type != "cuda" or not torch.cuda.is_available() or self._graph is not None:
            return
        # Save state snapshot around capture warmup so scratch/session state starts clean.
        out = self.codec.step(self._codes_buf, self.state, active_mask=self._active_mask_buf)
        with torch.inference_mode(False):
            self._out_buf = torch.empty_like(out)
            self._out_buf.copy_(out)
        self.state.reset_rows(list(range(self.num_rows)))
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = self.codec.step(self._codes_buf, self.state, active_mask=self._active_mask_buf)
            self._out_buf.copy_(out)
        self.state.reset_rows(list(range(self.num_rows)))
        self._graph = graph

    def step_all_rows(self, codes_r1k: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        if self._graph is not None and self._out_buf is not None:
            self._codes_buf.copy_(codes_r1k)
            self._active_mask_buf.copy_(active_mask)
            self._graph.replay()
            return self._out_buf.clone()
        return self.codec.step(codes_r1k, self.state, active_mask=active_mask)


class SpectroStreamCode2Wav(nn.Module):
    """Stage-1 MRv2 ``LLM_GENERATION`` audio decoder for Magenta RealTime 2."""

    input_modalities = "audio"
    requires_request_ids = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self._async_chunk = bool(getattr(vllm_config.model_config, "async_chunk", False))
        hf_cfg = getattr(vllm_config.model_config, "hf_config", None)
        self.config = hf_cfg if isinstance(hf_cfg, MagentaRT2Config) else MagentaRT2Config()

        # Runner capability flags (matching OmniGenerationModelRunner contract).
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        self._num_codebooks = int(self.config.ss_active_quantizers)
        self._output_sample_rate = int(self.config.sample_rate)
        self._samples_per_frame = int(self.config.samples_per_frame)
        max_seqs = int(getattr(vllm_config.scheduler_config, "max_num_seqs", 8) or 8)
        duplex_max = int(getattr(vllm_config.model_config, "duplex_max_sessions", 1) or 1)
        self._max_codec_sessions = max(max_seqs, duplex_max)
        self._scratch_row = self._max_codec_sessions
        self._num_codec_rows = self._max_codec_sessions + 1

        self.spectrostream = SpectroStreamModel(self.config)
        self._codec_state: SpectroStreamRowState | None = None
        self._graph_runner: SpectroStreamDecodeGraphRunner | None = None
        self._request_rows: dict[str, int] = {}
        self._consumed_full_payload_requests: set[str] = set()
        self._use_cuda_graphs = bool(getattr(self.config, "spectrostream_cuda_graphs", True))
        self.allow_patterns_overrides = magenta_rt2_allow_patterns(self.model_path, self.config)

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        del hidden_states, sampling_metadata
        return None

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict[str, object]]:
        return [{"meta": {"magenta_rt2_dummy_profile": True}} for _ in range(num_reqs)]

    def _ensure_state(self, device: torch.device) -> SpectroStreamRowState:
        target_dtype = self.spectrostream.conv_dtype
        dtype_mismatch = (
            self._codec_state is not None
            and self._codec_state.out_conv_state is not None
            and self._codec_state.out_conv_state.dtype != target_dtype
        )
        device_mismatch = (
            self._codec_state is not None
            and self._codec_state.ola_buf is not None
            and self._codec_state.ola_buf.device != device
        )
        if self._codec_state is None or device_mismatch or dtype_mismatch:
            self._codec_state = self.spectrostream.init_state(
                self._num_codec_rows,
                device=device,
                dtype=target_dtype,
            )
            self._graph_runner = (
                SpectroStreamDecodeGraphRunner(
                    self.spectrostream,
                    self._codec_state,
                    self._num_codec_rows,
                    self._num_codebooks,
                    device,
                )
                if self._use_cuda_graphs and device.type == "cuda"
                else None
            )
        return self._codec_state

    def warm_up_codec_cache(self, *_: Any) -> None:
        device = self.spectrostream.quantizer_embedding.device
        self._ensure_state(device)
        if self._graph_runner is not None:
            self._graph_runner.capture()

    def _lease_row(self, request_id: str) -> int:
        if request_id in self._request_rows:
            return self._request_rows[request_id]
        used = set(self._request_rows.values())
        for row in range(self._max_codec_sessions):
            if row not in used:
                state = self._ensure_state(self.spectrostream.quantizer_embedding.device)
                state.reset_rows(row)
                self._request_rows[request_id] = row
                return row
        return self._scratch_row

    def release_request_rows(self, finished_req_ids: Iterable[str]) -> None:
        for req_id in finished_req_ids:
            self._consumed_full_payload_requests.discard(req_id)
            row = self._request_rows.pop(req_id, None)
            if row is not None and self._codec_state is not None:
                self._codec_state.reset_rows(row)

    def on_requests_finished(self, finished_req_ids: Iterable[str]) -> None:
        """MRv2 runner lifecycle hook invoked when requests finish or terminate."""
        self.release_request_rows(finished_req_ids)

    def _split_request_ids(
        self,
        ids: torch.Tensor,
        seq_token_counts: list[int] | None = None,
    ) -> list[torch.Tensor]:
        if seq_token_counts is not None and len(seq_token_counts) >= 1:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + int(count))
            n = ids.numel()
            return [ids[boundaries[i] : min(boundaries[i + 1], n)] for i in range(len(seq_token_counts))]
        if is_forward_context_available():
            slices = get_forward_context().ubatch_slices
            if slices is not None and len(slices) > 1 and not any(hasattr(s, "token_slice") for s in slices):
                boundaries = [0]
                for s in slices:
                    boundaries.append(boundaries[-1] + s)
                return [ids[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]
        return [ids]

    def _resolve_request_ids(
        self,
        num_req: int,
        request_ids: Any,
        runtime_additional_information: list[dict[str, Any]] | None,
    ) -> list[str | None]:
        out: list[str | None] = [None] * num_req
        if isinstance(request_ids, (list, tuple)):
            for i in range(min(num_req, len(request_ids))):
                if isinstance(request_ids[i], str) and request_ids[i]:
                    out[i] = request_ids[i]
        if runtime_additional_information:
            for i in range(min(num_req, len(runtime_additional_information))):
                if out[i] is not None:
                    continue
                info = runtime_additional_information[i]
                if isinstance(info, Mapping):
                    rid = info.get("request_id") or info.get("req_id")
                    if isinstance(rid, str) and rid:
                        out[i] = rid
        return out

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """Decode flat frame-major ``[F * 12]`` codec tokens into ``[F * 1920, 2]`` stereo 48 kHz PCM."""
        del positions, intermediate_tensors, inputs_embeds
        device = self.spectrostream.quantizer_embedding.device
        sr_tensor = torch.tensor(self._output_sample_rate, dtype=torch.int32)
        empty = torch.zeros((0, 2), dtype=torch.float32, device=device)

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        ids = input_ids.reshape(-1).to(device=device, dtype=torch.long)
        req_segments = self._split_request_ids(ids, kwargs.get("seq_token_counts"))
        num_req = len(req_segments)
        state_ids = self._resolve_request_ids(num_req, kwargs.get("request_ids"), runtime_additional_information)
        k = self._num_codebooks
        audios: list[torch.Tensor] = [empty] * num_req
        srs = [sr_tensor] * num_req

        # (req_idx, state_id, codes_fk [F, K])
        pending: list[tuple[int, str | None, torch.Tensor]] = []
        for i, seg in enumerate(req_segments):
            info = (
                runtime_additional_information[i]
                if runtime_additional_information is not None and i < len(runtime_additional_information)
                else None
            )
            meta = info.get("meta") if isinstance(info, Mapping) else None
            if isinstance(meta, Mapping) and meta.get("magenta_rt2_dummy_profile") is True:
                continue
            flat = _codec_ids_from_payload_or_input(seg, info)
            n = int(flat.numel())
            if n == 0 or n % k != 0:
                if n > 0:
                    logger.warning(
                        "SpectroStream Code2Wav token length %d not divisible by %d; skipping request %s.",
                        n,
                        k,
                        state_ids[i],
                    )
                continue
            frames = n // k
            codes_fk = flat.reshape(frames, k)
            sid = state_ids[i]
            if not self._async_chunk and sid is not None:
                if sid in self._consumed_full_payload_requests:
                    continue
                self._consumed_full_payload_requests.add(sid)
            pending.append((i, sid, codes_fk))

        if not pending:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": audios, "sr": srs},
            )

        # Non-streaming full-payload decode when async_chunk=False and no cross-chunk state is needed.
        if not self._async_chunk:
            for i, _sid, codes_fk in pending:
                wav = self.spectrostream.forward(codes_fk.unsqueeze(0))[0]
                audios[i] = wav
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": audios, "sr": srs},
            )

        state = self._ensure_state(device)
        rows = [self._lease_row(sid) if sid is not None else self._scratch_row for _, sid, _ in pending]
        max_frames = max(int(item[2].shape[0]) for item in pending)
        scratch_used = any(row == self._scratch_row for row in rows)

        step_codes = torch.zeros((self._num_codec_rows, 1, k), dtype=torch.long, device=device)
        step_active = torch.zeros((self._num_codec_rows,), dtype=torch.bool, device=device)
        row_outputs: dict[int, list[torch.Tensor]] = {item[0]: [] for item in pending}

        for frame_idx in range(max_frames):
            step_codes.zero_()
            step_active.zero_()
            active_entries: list[tuple[int, int]] = []
            for (req_i, _sid, codes_fk), row in zip(pending, rows, strict=True):
                if frame_idx < codes_fk.shape[0]:
                    step_codes[row, 0].copy_(codes_fk[frame_idx])
                    step_active[row] = True
                    active_entries.append((req_i, row))
            if self._graph_runner is not None:
                decoded = self._graph_runner.step_all_rows(step_codes, step_active)
            else:
                decoded = self.spectrostream.step(step_codes, state, active_mask=step_active)
            for req_i, row in active_entries:
                row_outputs[req_i].append(decoded[row])

        if scratch_used:
            state.reset_rows(self._scratch_row)

        for req_i, chunks in row_outputs.items():
            if chunks:
                audios[req_i] = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": srs},
        )

    def make_omni_output(
        self,
        model_output: Any,
        **_: Any,
    ) -> OmniOutput:
        if isinstance(model_output, OmniOutput):
            return model_output
        return OmniOutput(text_hidden_states=None, multimodal_outputs=model_output)

    def _find_flax_checkpoint(self) -> Path | None:
        return resolve_magenta_rt2_checkpoint(self.model_path, self.config)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        flat_flax: dict[str, torch.Tensor] = {}
        state = self.state_dict()
        loaded: set[str] = set()

        for name, tensor in weights:
            if name.startswith("params/soundstream/"):
                flat_flax[name] = tensor
                continue
            if name in state and state[name].shape == tensor.shape:
                state[name].data.copy_(tensor.to(dtype=state[name].dtype))
                loaded.add(name)

        if not flat_flax and not loaded:
            ckpt = self._find_flax_checkpoint()
            if ckpt is not None:
                with safetensors.safe_open(str(ckpt), framework="pt", device="cpu") as f:
                    for key in f.keys():  # noqa: SIM118
                        if key.startswith("params/soundstream/"):
                            flat_flax[key] = f.get_tensor(key)

        if flat_flax:
            nested = _unflatten_nested(flat_flax)
            if "params" in nested and "soundstream" in nested["params"]:
                self.spectrostream.load_flax_params(nested["params"]["soundstream"])
                loaded.update(self.state_dict().keys())
                logger.info("Loaded SpectroStream Flax weights (%d parameters)", len(loaded))
        return loaded


__all__ = [
    "SpectroStreamCode2Wav",
    "SpectroStreamDecodeGraphRunner",
]
