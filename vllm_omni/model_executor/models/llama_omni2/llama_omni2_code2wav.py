# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CosyVoice2 code-to-waveform stage for LLaMA-Omni 2."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.transformers_utils.repo_utils import hf_api
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.output_templates import OmniOutput

SAMPLE_RATE = 24000
_DECODER_YAML_CANDIDATES = ("cosyvoice.yaml", "flow.yaml")
_REQUIRED_ARTIFACTS = ("flow.pt", "hift.pt")
_DEFAULT_SPEAKER_EMBEDDING = Path(__file__).with_name("assets") / "default_english_speaker_embedding.npy"


@lru_cache(maxsize=1)
def _cached_default_speaker_embedding() -> torch.Tensor:
    embedding = torch.from_numpy(np.load(_DEFAULT_SPEAKER_EMBEDDING, allow_pickle=False)).to(torch.float32)
    if embedding.shape != (1, 192):
        raise ValueError(
            f"LLaMA-Omni 2 default speaker embedding must have shape (1, 192), got {tuple(embedding.shape)}"
        )
    if not torch.isfinite(embedding).all():
        raise ValueError("LLaMA-Omni 2 default speaker embedding contains non-finite values")
    if not torch.count_nonzero(embedding):
        raise ValueError("LLaMA-Omni 2 default speaker embedding is all zeros")
    return embedding


def load_default_speaker_embedding() -> torch.Tensor:
    return _cached_default_speaker_embedding().clone()


def _resolve_decoder_yaml(
    root: Path,
    yaml_name: str | None = None,
) -> Path:
    candidates = (yaml_name,) if yaml_name else _DECODER_YAML_CANDIDATES
    for candidate in candidates:
        if candidate and (root / candidate).is_file():
            return root / candidate
    expected = ", ".join(name for name in candidates if name)
    raise FileNotFoundError(f"LLaMA-Omni 2 CosyVoice2 decoder is missing a supported YAML under {root}: {expected}")


def validate_cosy2_decoder_dir(model_dir: str | os.PathLike[str]) -> Path:
    root = Path(model_dir)
    _resolve_decoder_yaml(root)
    for artifact in _REQUIRED_ARTIFACTS:
        path = root / artifact
        if not path.is_file():
            raise FileNotFoundError(f"LLaMA-Omni 2 CosyVoice2 decoder is missing {artifact}: {path}")
    return root


def _resolve_decoder_dir(model: str) -> Path:
    local = Path(model).expanduser()
    if local.is_dir():
        return validate_cosy2_decoder_dir(local)

    return validate_cosy2_decoder_dir(hf_api().snapshot_download(model))


def _load_cosy2_modules(
    model_dir: Path,
    device: torch.device,
    yaml_name: str | None = None,
) -> tuple[nn.Module, nn.Module]:
    try:
        from flashcosyvoice.modules.hifigan import HiFTGenerator
        from hyperpyyaml import load_hyperpyyaml
    except ImportError as exc:
        raise ImportError("LLaMA-Omni 2 Code2Wav requires step-audio2's flashcosyvoice runtime") from exc

    yaml_path = _resolve_decoder_yaml(model_dir, yaml_name)
    if yaml_path.name == "flow.yaml":
        with yaml_path.open(encoding="utf-8") as handle:
            config = load_hyperpyyaml(handle)
        flow = config["flow"]
    else:
        from flashcosyvoice.modules.flow import CausalMaskedDiffWithXvec

        flow = CausalMaskedDiffWithXvec()
    flow.load_state_dict(
        torch.load(
            model_dir / "flow.pt",
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    flow.to(device).eval()
    if hasattr(flow, "decoder") and hasattr(flow.decoder, "fp16"):
        flow.decoder.fp16 = False

    hift = HiFTGenerator()
    hift_state = {
        name.removeprefix("generator."): tensor
        for name, tensor in torch.load(
            model_dir / "hift.pt",
            map_location="cpu",
            weights_only=True,
        ).items()
    }
    hift.load_state_dict(hift_state, strict=True)
    hift.to(device).eval()

    def hift_inference(
        self: nn.Module,
        *,
        speech_feat: torch.Tensor,
        cache_source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(speech_feat, cache_source)

    hift.inference = MethodType(hift_inference, hift)
    return flow, hift


def _truthy(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.reshape(-1)[0].item()) if value.numel() else False
    if isinstance(value, (list, tuple)):
        return _truthy(value[0]) if value else False
    return bool(value)


def _optional_int(value: Any) -> int | None:
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item()) if value.numel() else None
    if isinstance(value, (list, tuple)):
        return _optional_int(value[0]) if value else None
    return int(value) if value is not None else None


def _request_id(info: Mapping[str, Any]) -> str | None:
    meta = info.get("meta")
    if not isinstance(meta, Mapping):
        return None
    value = meta.get("request_id")
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return str(value) if value is not None else None


def _codes(info: Mapping[str, Any]) -> list[int]:
    codes = info.get("codes")
    audio = codes.get("audio") if isinstance(codes, Mapping) else None
    if isinstance(audio, torch.Tensor):
        return [int(token) for token in audio.reshape(-1).tolist()]
    if isinstance(audio, (list, tuple)):
        return [int(token) for token in audio]
    return []


def _cross_fade(
    current: torch.Tensor,
    previous: torch.Tensor,
    window: torch.Tensor,
) -> torch.Tensor:
    overlap = min(
        int(window.numel() // 2),
        int(current.shape[-1]),
        int(previous.shape[-1]),
    )
    if overlap <= 0:
        return current
    result = current.clone()
    result[..., :overlap] = result[..., :overlap] * window[:overlap] + previous[..., -overlap:] * window[-overlap:]
    return result


@dataclass
class LlamaOmni2AudioChunk:
    request_id: str
    audio: torch.Tensor
    sample_rate: int
    sequence_index: int
    consumed_units: int
    finished: bool


@dataclass
class _Code2WavState:
    units: list[int] = field(default_factory=list)
    chunk_seq: int = -1
    token_offset: int = 0
    sequence_index: int = 0
    mel_cache: torch.Tensor | None = None
    source_cache: torch.Tensor | None = None
    speech_cache: torch.Tensor | None = None


@dataclass(frozen=True)
class _Code2WavWorkItem:
    output_index: int
    request_id: str
    finished: bool
    state: _Code2WavState


class LlamaOmni2Code2WavCore:
    def __init__(
        self,
        *,
        flow: nn.Module,
        hift: nn.Module,
        device: str | torch.device,
        mel_cache_len: int = 8,
        source_cache_len: int = 8 * 480,
        speaker_embedding: torch.Tensor | None = None,
    ) -> None:
        self.flow = flow
        self.hift = hift
        self.device = torch.device(device)
        self.mel_cache_len = int(mel_cache_len)
        self.source_cache_len = int(source_cache_len)
        if speaker_embedding is None:
            speaker_embedding = load_default_speaker_embedding()
        if speaker_embedding.shape != (1, 192):
            raise ValueError(
                f"LLaMA-Omni 2 speaker embedding must have shape (1, 192), got {tuple(speaker_embedding.shape)}"
            )
        self.speaker_embedding = speaker_embedding.to(
            device=self.device,
            dtype=torch.float32,
        )
        self._states: dict[str, _Code2WavState] = {}
        self._finished_request_ids: set[str] = set()
        self._profile_batch_ranges = os.getenv(
            "VLLM_OMNI_LLAMA_OMNI2_PROFILE_BATCHES",
            "",
        ).lower() in {"1", "true", "yes", "on"}
        self._window = torch.from_numpy(np.hamming(2 * self.source_cache_len).astype(np.float32)).to(self.device)

    def __contains__(self, request_id: object) -> bool:
        return request_id in self._states

    def cancel(self, request_id: str) -> None:
        self._states.pop(request_id, None)
        self._finished_request_ids.discard(request_id)

    @staticmethod
    def _clone_state(state: _Code2WavState) -> _Code2WavState:
        return _Code2WavState(
            units=list(state.units),
            chunk_seq=state.chunk_seq,
            token_offset=state.token_offset,
            sequence_index=state.sequence_index,
            mel_cache=state.mel_cache,
            source_cache=state.source_cache,
            speech_cache=state.speech_cache,
        )

    @staticmethod
    def _tensor_shape(value: torch.Tensor | None) -> tuple[int, ...] | None:
        return tuple(value.shape) if value is not None else None

    def _work_signature(self, item: _Code2WavWorkItem) -> tuple[Any, ...]:
        state = item.state
        return (
            len(state.units),
            state.token_offset,
            item.finished,
            self._tensor_shape(state.mel_cache),
            self._tensor_shape(state.source_cache),
            self._tensor_shape(state.speech_cache),
        )

    def _batch_range(self, component: str, batch_size: int):
        if not self._profile_batch_ranges:
            return nullcontext()
        return torch.profiler.record_function(f"llama_omni2.code2wav.{component}[batch={batch_size}]")

    def _run_flow(
        self,
        *,
        token: torch.Tensor,
        token_len: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_token_len: torch.Tensor,
        prompt_feat: torch.Tensor,
        prompt_feat_len: torch.Tensor,
        embedding: torch.Tensor,
        finished: bool,
    ) -> tuple[torch.Tensor, Any]:
        inference = getattr(self.flow, "inference", None)
        if callable(inference):
            return inference(
                token=token,
                token_len=token_len,
                prompt_token=prompt_token,
                prompt_token_len=prompt_token_len,
                prompt_feat=prompt_feat,
                prompt_feat_len=prompt_feat_len,
                embedding=embedding,
                finalize=finished,
            )
        return self.flow(
            token=token,
            token_len=token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=embedding,
            streaming=not finished,
            finalize=finished,
        )

    def _process_group(
        self,
        items: list[_Code2WavWorkItem],
    ) -> list[LlamaOmni2AudioChunk]:
        batch_size = len(items)
        states = [item.state for item in items]
        finished = items[0].finished
        token = torch.tensor(
            [state.units for state in states],
            dtype=torch.long,
            device=self.device,
        )
        empty_prompt_token = torch.zeros(
            (batch_size, 0),
            dtype=torch.int32,
            device=self.device,
        )
        empty_prompt_feat = torch.zeros(
            (batch_size, 0, 80),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            with self._batch_range("flow", batch_size):
                mel, _ = self._run_flow(
                    token=token,
                    token_len=torch.full(
                        (batch_size,),
                        token.shape[1],
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    prompt_token=empty_prompt_token,
                    prompt_token_len=torch.zeros(
                        batch_size,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    prompt_feat=empty_prompt_feat,
                    prompt_feat_len=torch.zeros(
                        batch_size,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    embedding=self.speaker_embedding.expand(batch_size, -1),
                    finished=finished,
                )

            ratio = int(getattr(self.flow, "token_mel_ratio", 1))
            mel = mel[..., states[0].token_offset * ratio :]
            if states[0].mel_cache is not None:
                mel = torch.cat(
                    (
                        torch.cat([state.mel_cache for state in states], dim=0),
                        mel,
                    ),
                    dim=-1,
                )
            if states[0].source_cache is None:
                cache_source = torch.zeros(
                    (batch_size, 1, 0),
                    dtype=mel.dtype,
                    device=mel.device,
                )
            else:
                cache_source = torch.cat(
                    [state.source_cache for state in states],
                    dim=0,
                )
            with self._batch_range("hift", batch_size):
                speech, source = self.hift.inference(
                    speech_feat=mel,
                    cache_source=cache_source,
                )
            if states[0].speech_cache is not None:
                speech = _cross_fade(
                    speech,
                    torch.cat([state.speech_cache for state in states], dim=0),
                    self._window,
                )

        if not finished and speech.shape[-1] >= self.source_cache_len:
            emitted = speech[..., : -self.source_cache_len]
        else:
            emitted = speech
        with self._batch_range("d2h", batch_size):
            emitted_cpu = emitted.to(torch.float32).detach().cpu().contiguous()
        lookahead = int(getattr(self.flow, "pre_lookahead_len", 0))
        chunks: list[LlamaOmni2AudioChunk] = []
        for row, item in enumerate(items):
            state = item.state
            state.mel_cache = (
                mel[
                    row : row + 1,
                    ...,
                    -self.mel_cache_len :,
                ]
                .detach()
                .clone()
            )
            state.source_cache = (
                source[
                    row : row + 1,
                    ...,
                    -self.source_cache_len :,
                ]
                .detach()
                .clone()
            )
            state.speech_cache = (
                speech[
                    row : row + 1,
                    ...,
                    -self.source_cache_len :,
                ]
                .detach()
                .clone()
            )
            state.token_offset = len(state.units) if finished else max(0, len(state.units) - lookahead)
            chunks.append(
                LlamaOmni2AudioChunk(
                    request_id=item.request_id,
                    audio=emitted_cpu[row].reshape(-1),
                    sample_rate=SAMPLE_RATE,
                    sequence_index=state.sequence_index,
                    consumed_units=len(state.units),
                    finished=finished,
                )
            )
            state.sequence_index += 1
        return chunks

    def process_batch(
        self,
        requests: list[tuple[str, list[int], bool]],
        *,
        chunk_seqs: list[int | None] | None = None,
    ) -> list[LlamaOmni2AudioChunk | None]:
        request_ids = [request_id for request_id, _, _ in requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("duplicate LLaMA-Omni 2 Code2Wav request ids in one batch")
        if chunk_seqs is None:
            chunk_seqs = [None] * len(requests)
        if len(chunk_seqs) != len(requests):
            raise ValueError("LLaMA-Omni 2 Code2Wav chunk_seqs must align with requests")

        tentative: dict[str, _Code2WavState] = {}
        outputs: list[LlamaOmni2AudioChunk | None] = [None] * len(requests)
        work: list[_Code2WavWorkItem] = []
        terminal_empty: set[str] = set()
        for output_index, ((request_id, new_units, finished), chunk_seq) in enumerate(
            zip(requests, chunk_seqs, strict=True)
        ):
            if request_id in self._finished_request_ids:
                raise ValueError(f"LLaMA-Omni 2 Code2Wav request {request_id!r} already finished")
            previous = self._states.get(request_id)
            expected_chunk_seq = 0 if previous is None else previous.chunk_seq + 1
            if chunk_seq is not None and chunk_seq != expected_chunk_seq:
                raise ValueError(
                    "LLaMA-Omni 2 Code2Wav chunk_seq must be monotonic: "
                    f"request_id={request_id!r}, expected={expected_chunk_seq}, "
                    f"actual={chunk_seq}"
                )
            state = self._clone_state(previous or _Code2WavState())
            state.chunk_seq = expected_chunk_seq if chunk_seq is None else chunk_seq
            state.units.extend(int(unit) for unit in new_units)
            if not state.units and not finished:
                raise ValueError("nonterminal Code2Wav chunks require codec units")
            tentative[request_id] = state
            lookahead = int(getattr(self.flow, "pre_lookahead_len", 0))
            if not finished and len(state.units) <= lookahead:
                continue
            if not state.units:
                outputs[output_index] = LlamaOmni2AudioChunk(
                    request_id=request_id,
                    audio=torch.empty(0, dtype=torch.float32),
                    sample_rate=SAMPLE_RATE,
                    sequence_index=state.sequence_index,
                    consumed_units=0,
                    finished=True,
                )
                terminal_empty.add(request_id)
                continue
            work.append(
                _Code2WavWorkItem(
                    output_index=output_index,
                    request_id=request_id,
                    finished=finished,
                    state=state,
                )
            )

        groups: dict[tuple[Any, ...], list[_Code2WavWorkItem]] = {}
        for item in work:
            groups.setdefault(self._work_signature(item), []).append(item)
        for items in groups.values():
            chunks = self._process_group(items)
            for item, chunk in zip(items, chunks, strict=True):
                outputs[item.output_index] = chunk

        finished_ids = terminal_empty | {item.request_id for item in work if item.finished}
        for request_id, state in tentative.items():
            if request_id in finished_ids:
                self._states.pop(request_id, None)
                self._finished_request_ids.add(request_id)
            else:
                self._states[request_id] = state
        return outputs

    def process(
        self,
        request_id: str,
        new_units: list[int],
        *,
        finished: bool,
    ) -> LlamaOmni2AudioChunk | None:
        return self.process_batch([(request_id, new_units, finished)])[0]


class LlamaOmni2Code2Wav(nn.Module):
    input_modalities = "audio"

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        core: LlamaOmni2Code2WavCore | None = None,
    ) -> None:
        super().__init__()
        del prefix
        self.vllm_config = vllm_config
        self.have_multimodal_outputs = True
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True
        self.make_empty_intermediate_tensors = lambda: None

        if core is None:
            model_dir = _resolve_decoder_dir(vllm_config.model_config.model)
            device = torch.device(vllm_config.device_config.device)
            flow, hift = _load_cosy2_modules(model_dir, device)
            core = LlamaOmni2Code2WavCore(
                flow=flow,
                hift=hift,
                device=device,
            )
        self.core = core

    @property
    def sampler(self) -> Sampler:
        return Sampler()

    def get_language_model(self) -> nn.Module:
        return self

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        hidden_size = self.vllm_config.model_config.get_hidden_size()
        return torch.zeros(
            (input_ids.numel(), hidden_size),
            dtype=torch.float32,
            device=input_ids.device,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> OmniOutput:
        del positions
        infos = runtime_additional_information or []
        if not infos and input_ids is not None and input_ids.numel():
            infos = [
                {
                    "codes": {"audio": input_ids.reshape(-1)},
                    "meta": {"request_id": "sync-0", "finished": True},
                }
            ]

        requests: list[tuple[str, list[int], bool]] = []
        chunk_seqs: list[int | None] = []
        codec_units: list[torch.Tensor] = []
        for info in infos:
            info = info if isinstance(info, Mapping) else {}
            meta = info.get("meta")
            meta = meta if isinstance(meta, Mapping) else {}
            request_id = _request_id(info)
            codes = _codes(info)
            codec_units.append(torch.tensor(codes, dtype=torch.long))
            finished = _truthy(meta.get("stream_finished")) or _truthy(meta.get("finished"))
            if request_id is None:
                if codes:
                    raise ValueError("LLaMA-Omni 2 Code2Wav payload is missing meta.request_id")
                continue
            requests.append((request_id, codes, finished))
            chunk_seqs.append(_optional_int(meta.get("chunk_seq")))
        chunks = self.core.process_batch(
            requests,
            chunk_seqs=chunk_seqs,
        )

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": [chunk.audio if chunk is not None else None for chunk in chunks],
                "sr": [
                    (torch.tensor(chunk.sample_rate, dtype=torch.int32) if chunk is not None else None)
                    for chunk in chunks
                ],
                "finished": [
                    (torch.tensor(chunk.finished, dtype=torch.bool) if chunk is not None else None) for chunk in chunks
                ],
                "sequence_index": [
                    (torch.tensor(chunk.sequence_index, dtype=torch.int64) if chunk is not None else None)
                    for chunk in chunks
                ],
                "consumed_units": [
                    (torch.tensor(chunk.consumed_units, dtype=torch.int64) if chunk is not None else None)
                    for chunk in chunks
                ],
                "codec_units": codec_units,
            },
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Any = None,
    ) -> None:
        del hidden_states, sampling_metadata
        return None

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        del weights
        return {
            *(f"core.flow.{name}" for name, _ in self.core.flow.named_parameters()),
            *(f"core.hift.{name}" for name, _ in self.core.hift.named_parameters()),
        }

    def on_requests_finished(self, finished_req_ids: Iterable[str]) -> None:
        for request_id in finished_req_ids:
            self.core.cancel(request_id)
