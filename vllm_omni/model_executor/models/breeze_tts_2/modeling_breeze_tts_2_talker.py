"""Breeze-TTS-2 stage-0 talker for vLLM-Omni.

The scheduler samples Breeze codebook-0 tokens one at a time. The other 15
codebooks are kept in the request-local Omni buffer: after each backbone step,
the eager depth decoder completes the frame and the next backbone step
consumes the summed 16-codebook embedding. This preserves the upstream Breeze
generation order without importing HuggingFace GenerationMixin.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.sequence import IntermediateTensors
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.model_executor.models.breeze_tts_2.configuration_breeze_tts_2 import (
    BreezeTTS2Config,
)
from vllm_omni.model_executor.models.breeze_tts_2.modeling_breeze_tts_2_depth import (
    BreezeDepthDecoderForCausalLM,
)
from vllm_omni.model_executor.models.breeze_tts_2.modeling_breeze_tts_2_text import (
    BreezeTTS2TextEncoder,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput


def _prefixed(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _normalize_request_infos(infos: Any) -> list[dict[str, Any]] | None:
    """Normalize runner metadata to one mutable dict per request."""
    if not isinstance(infos, Sequence) or isinstance(infos, (str, bytes, bytearray)):
        return None
    if not all(isinstance(info, dict) for info in infos):
        return None
    return infos


def _iter_request_rows(
    states: Sequence[dict[str, Any]],
    spans: Sequence[tuple[int, int]] | None,
    num_rows: int,
) -> Iterable[tuple[dict[str, Any], int, int]]:
    """Map request state to logits rows for full or sampled hidden layouts."""
    if num_rows <= 0 or not states:
        return
    if spans is not None and len(spans) == len(states):
        if all(0 <= int(start) <= int(end) <= num_rows for start, end in spans):
            for state, (start, end) in zip(states, spans, strict=True):
                if int(start) < int(end):
                    yield state, int(start), int(end)
            return
        # The runner passes full-sequence spans to make_omni_output but only
        # one sampled row per request to compute_logits.
        if num_rows == len(states):
            for index, state in enumerate(states):
                yield state, index, index + 1
            return
    if num_rows == len(states):
        for index, state in enumerate(states):
            yield state, index, index + 1
        return
    raise RuntimeError(
        "Breeze talker cannot align request state with logits rows: "
        f"states={len(states)}, rows={num_rows}, spans={spans}"
    )


def _check_weight_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
    """Fail early for Breeze-owned, non-sharded checkpoint tensors."""
    actual = tuple(int(dim) for dim in tensor.shape)
    if actual != expected:
        raise ValueError(f"Breeze weight {name!r} has shape {actual}, expected {expected}")


class BreezeTTS2TalkerForGeneration(nn.Module):
    """Stage-0 Breeze talker: text/audio conditioning -> codec frames."""

    have_multimodal_outputs = True
    has_preprocess = True
    has_postprocess = True
    # Stage 0 emits a tiny per-frame codec payload. Let the AR runner snapshot
    # it through the async-output path instead of the final-only client path.
    use_async_omni_output = True
    eager_omni_postprocess_before_async_output = True
    omni_pooler_payload_include_hidden = False
    input_modalities = "text"
    gpu_resident_buffer_keys: set[tuple[str, str]] = {
        ("hidden_states", "last"),
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config: BreezeTTS2Config = vllm_config.model_config.hf_config
        self._async_chunk = bool(vllm_config.model_config.async_chunk)
        self._golden_dump_dir = os.environ.get("VLLM_OMNI_BREEZE_GOLDEN_DIR")

        backbone_config = self.config.backbone_config
        backbone_vllm_config = vllm_config.with_hf_config(
            backbone_config,
            architectures=["Qwen3ForCausalLM"],
        )
        backbone_vllm_config.model_config.hf_text_config = backbone_config
        self.model = Qwen3Model(
            vllm_config=backbone_vllm_config,
            prefix=_prefixed(prefix, "model"),
        )
        # Upstream's Qwen3 adapter keeps only decoder layers/norm/RoPE and
        # replaces the standard text embedding with Breeze's summed audio/text
        # tables.  Breeze always supplies ``inputs_embeds`` through preprocess,
        # so remove the unused Qwen embedding instead of allocating and then
        # reporting an uninitialized 151936x2048 parameter at load time.
        self.model.embed_tokens = nn.Identity()

        self.hidden_size = int(backbone_config.hidden_size)
        self.num_codebooks = int(self.config.num_codebooks)
        self.codebook_vocab_size = int(self.config.vocab_size)
        codec_config = self.config.codec_config
        self.codebook_size = (
            int(codec_config.get("codebook_size", self.codebook_vocab_size))
            if isinstance(codec_config, Mapping)
            else self.codebook_vocab_size
        )
        self.codebook_eos_token_id = int(self.config.codebook_eos_token_id)
        self.audio_token_id = int(self.config.audio_token_id)
        self.audio_eos_token_id = int(self.config.audio_eos_token_id)
        self.text_pad_token_id = int(self.config.pad_token_id)

        self.lm_head = ParallelLMHead(
            self.codebook_vocab_size + 1,
            self.hidden_size,
            bias=False,
            prefix=_prefixed(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.codebook_vocab_size + 1)

        # Upstream Breeze uses embedding(code_i + i * vocab_size) and sums
        # over codebooks before entering the Qwen3 backbone.
        self.embed_audio_tokens = nn.Embedding(
            self.num_codebooks * self.codebook_vocab_size,
            self.hidden_size,
        )
        # Keep Breeze's standalone text table even though the normal path
        # uses T5Gemma2 outputs. It is part of the checkpoint contract and is
        # also the correct fallback when a checkpoint omits text_encoder.
        self.embed_text_tokens = nn.Embedding(int(self.config.text_vocab_size), self.hidden_size)
        self.register_buffer(
            "audio_token_offsets",
            torch.arange(self.num_codebooks, dtype=torch.long) * self.codebook_vocab_size,
            persistent=False,
        )

        self.text_encoder = BreezeTTS2TextEncoder(self.config.text_encoder_config)
        self.text_encoder_proj = nn.Linear(
            int(self.config.text_encoder_config.hidden_size),
            self.hidden_size,
            bias=False,
        )
        self.depth_decoder = BreezeDepthDecoderForCausalLM(self.config.depth_decoder_config)
        if bool(self.config.tie_codebooks_embeddings):
            # The upstream HF model ties these tables. Sharing the Parameter
            # also prevents a tied checkpoint from leaving the depth input
            # embedding randomly initialized when only one key is serialized.
            self.depth_decoder.model.embed_tokens.weight = self.embed_audio_tokens.weight

        self._codec_disallowed_mask: torch.Tensor | None = None
        self._batch_state: list[dict[str, Any]] | None = None
        self._batch_state_spans: Sequence[tuple[int, int]] | None = None

    # ------------------------------------------------------------------
    # Embedding and backbone hooks
    # ------------------------------------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        """Embed complete generated codec frames; scalar ids are invalid."""
        if input_ids.ndim < 1 or input_ids.shape[-1] != self.num_codebooks:
            raise ValueError(
                "Breeze embed_input_ids requires complete codec frames with shape "
                f"(..., {self.num_codebooks}), got {tuple(input_ids.shape)}"
            )
        codes = input_ids.to(dtype=torch.long)
        offsets = self.audio_token_offsets.to(device=codes.device)
        ids = codes.clamp(0, self.codebook_vocab_size - 1) + offsets
        return self.embed_audio_tokens(ids).sum(dim=-2)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    # ------------------------------------------------------------------
    # Prompt/text conditioning
    # ------------------------------------------------------------------

    def _encode_text_segments(
        self,
        input_ids: torch.Tensor,
        text_mask: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> torch.Tensor:
        text_ids = input_ids[text_mask.to(device=input_ids.device, dtype=torch.bool)].to(torch.long)
        lengths = [int(x) for x in text_lengths.reshape(-1).tolist() if int(x) > 0]
        if int(text_ids.numel()) != sum(lengths):
            raise ValueError(f"Breeze text mask/length mismatch: {text_ids.numel()} != {sum(lengths)}")
        if not lengths:
            return input_ids.new_empty((0, self.hidden_size), dtype=self.text_encoder.dtype)
        segments = list(torch.split(text_ids, lengths))
        padded = pad_sequence(segments, batch_first=True, padding_value=0)
        attn = torch.zeros_like(padded, dtype=torch.bool)
        for row, length in enumerate(lengths):
            attn[row, :length] = True
        hidden = self.text_encoder(padded, attention_mask=attn)
        projected = self.text_encoder_proj(hidden)
        return torch.cat([projected[row, :length] for row, length in enumerate(lengths)], dim=0)

    def _build_prompt_embeddings(self, prompt_ids: torch.Tensor, info: Mapping[str, Any]) -> torch.Tensor:
        device = next(self.parameters()).device
        prompt_ids = prompt_ids.reshape(-1).to(device=device, dtype=torch.long)
        text_mask = torch.as_tensor(info.get("text_ids_mask"), device=device, dtype=torch.bool).reshape(-1)
        text_lengths = torch.as_tensor(info.get("text_ids_len"), device=device, dtype=torch.long)
        if text_mask.numel() != prompt_ids.numel():
            raise ValueError("Breeze text_ids_mask length does not match prompt length")
        output = torch.zeros(
            (prompt_ids.numel(), self.hidden_size),
            device=device,
            dtype=self.embed_text_tokens.weight.dtype,
        )
        text_embeds = self._encode_text_segments(prompt_ids, text_mask, text_lengths)
        output[text_mask] = text_embeds.to(dtype=output.dtype)

        audio_mask = prompt_ids == self.audio_token_id
        input_values = info.get("input_values")
        if bool(audio_mask.any()):
            if input_values is None:
                raise ValueError("Breeze audio placeholders require input_values")
            codes = torch.as_tensor(input_values, device=device, dtype=torch.long)
            if codes.ndim != 2 or codes.shape[-1] != self.num_codebooks:
                raise ValueError(f"Breeze input_values must be (T, {self.num_codebooks})")
            if int(codes.shape[0]) != int(audio_mask.sum()):
                raise ValueError("Breeze reference code count does not match audio placeholders")
            output[audio_mask] = self.embed_input_ids(codes).to(dtype=output.dtype)

        eos_mask = prompt_ids == self.audio_eos_token_id
        if bool(eos_mask.any()):
            eos_codes = torch.full(
                (1, self.num_codebooks),
                self.codebook_eos_token_id,
                dtype=torch.long,
                device=device,
            )
            output[eos_mask] = self.embed_input_ids(eos_codes).expand(int(eos_mask.sum()), -1)
        return output

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build prefill embeddings and consume one completed frame per decode."""
        del input_embeds
        info = dict(info_dict)
        span_len = int(input_ids.shape[0])
        is_prefill = bool(info.get("_omni_is_prefill", span_len > 1))
        if is_prefill:
            full_ids = info.get("prompt_ids")
            if not isinstance(full_ids, torch.Tensor):
                raise RuntimeError("Breeze prefill is missing prompt_ids")
            full_embeds = info.get("breeze_prompt_embeds")
            if not isinstance(full_embeds, torch.Tensor):
                full_embeds = self._build_prompt_embeddings(full_ids, info).detach().cpu().contiguous()
            offset = int(info.get("breeze_prefill_offset", info.get("_omni_num_computed_tokens", 0)) or 0)
            take = full_embeds[offset : offset + span_len].to(device=input_ids.device)
            if take.shape[0] != span_len:
                raise RuntimeError("Breeze prompt embedding slice is shorter than scheduled span")
            return (
                torch.full_like(input_ids, self.text_pad_token_id),
                take,
                {
                    "breeze_prompt_embeds": full_embeds,
                    "breeze_prefill_offset": offset + span_len,
                    "breeze_generated_frames": int(info.get("breeze_generated_frames", 0) or 0),
                    "breeze_max_new_frames": int(info.get("breeze_max_new_frames", -1)),
                },
            )

        frame = info.get("breeze_current_frame")
        if isinstance(frame, torch.Tensor) and frame.numel() == self.num_codebooks:
            frame = frame.to(device=input_ids.device, dtype=torch.long).reshape(1, 1, -1)
            embeds = self.embed_input_ids(frame).reshape(1, -1)
            safe_id = frame[..., 0].reshape_as(input_ids).to(dtype=input_ids.dtype)
            return safe_id, embeds, {}
        has_breeze_state = any(
            key in info
            for key in (
                "breeze_current_frame",
                "breeze_audio_codes",
                "breeze_generated_frames",
                "breeze_prompt_embeds",
            )
        )
        if has_breeze_state:
            # A real decode must consume the full frame produced from the
            # preceding backbone hidden state. A scalar fallback would silently
            # fill codebooks 1..15 with EOS and corrupt AR state.
            raise RuntimeError(
                "Breeze decode is missing breeze_current_frame; refusing scalar fallback "
                "because it would fill codebooks 1..15 with EOS"
            )
        # Runner-only dummy execution has no Breeze codec state. It must not
        # invent a semantically invalid audio frame; a stable zero embedding is
        # sufficient for shape/profiling warmup.
        dummy_embeds = torch.zeros(
            (*input_ids.shape, self.hidden_size),
            device=input_ids.device,
            dtype=self.embed_audio_tokens.weight.dtype,
        )
        return input_ids, dummy_embeds, {}

    def postprocess(self, hidden_states: torch.Tensor, **_: Any) -> dict[str, Any]:
        if hidden_states.numel() == 0:
            return {}
        return {"hidden_states": {"last": hidden_states[-1].detach()}}

    # ------------------------------------------------------------------
    # Main head, depth decoder and Omni output
    # ------------------------------------------------------------------

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor | None:
        del sampling_metadata
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        logits = self._main_head_logits(hidden_states)

        # ``make_omni_output`` runs immediately before this hook. It records
        # requests whose previous hidden state selected EOS or reached the
        # frame budget, so the scheduler receives a real stop token instead
        # of depending on an arbitrary deploy-level max_tokens timeout.
        states = self._batch_state or []
        if states:
            for state, row_start, row_end in _iter_request_rows(
                states,
                getattr(self, "_batch_state_spans", None),
                int(logits.shape[0]),
            ):
                if state.get("breeze_force_eos", False):
                    row = logits[row_start:row_end]
                    row.fill_(float("-inf"))
                    row[:, self.codebook_vocab_size] = 0.0
        return logits

    def _main_head_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is None:
            raise RuntimeError("Breeze main head returned no logits")
        if self._codec_disallowed_mask is None or self._codec_disallowed_mask.device != logits.device:
            mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
            if self.codebook_size < self.codebook_vocab_size:
                mask[self.codebook_size : self.codebook_vocab_size] = True
            self._codec_disallowed_mask = mask
        return logits.masked_fill(self._codec_disallowed_mask, float("-inf"))

    @torch.no_grad()
    def _generate_depth_codes(self, backbone_hidden: torch.Tensor, code0: int) -> torch.Tensor:
        sequence = torch.tensor([[0, code0]], dtype=torch.long, device=backbone_hidden.device)
        generated = [int(code0)]
        for _ in range(self.num_codebooks - 1):
            depth_out = self.depth_decoder(
                input_ids=sequence,
                backbone_last_hidden_state=backbone_hidden.reshape(1, -1),
                use_cache=False,
            )
            logits = depth_out.logits[:, -1, :].clone()
            if self.codebook_size < logits.shape[-1]:
                logits[:, self.codebook_size :] = float("-inf")
            next_code = int(torch.argmax(logits, dim=-1).item())
            generated.append(next_code)
            sequence = torch.cat([sequence, torch.tensor([[next_code]], device=sequence.device)], dim=1)
        return torch.tensor(generated, dtype=torch.long, device=backbone_hidden.device).unsqueeze(0)

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            self._batch_state = None
            self._batch_state_spans = None
            return model_outputs
        hidden = model_outputs
        infos = kwargs.get("model_intermediate_buffer", [])
        spans = kwargs.get("request_token_spans")
        info_items = _normalize_request_infos(infos)
        if info_items is None:
            self._batch_state = None
            self._batch_state_spans = None
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})
        # The runner invokes compute_logits after this method, so retain the
        # same request ordering and spans for per-request EOS masking.
        self._batch_state = info_items
        self._batch_state_spans = spans
        per_request_codes: list[torch.Tensor] = []

        for index, info in enumerate(info_items):
            if spans is not None and index < len(spans):
                row_start = max(0, int(spans[index][0]))
                row_end = min(int(spans[index][1]), int(hidden.shape[0]))
            elif len(info_items) == 1:
                row_start, row_end = 0, int(hidden.shape[0])
            else:
                raise RuntimeError("Breeze make_omni_output requires request_token_spans for batched decode")
            if row_end <= row_start:
                per_request_codes.append(hidden.new_empty((0, self.num_codebooks), dtype=torch.long))
                continue
            row_hidden = hidden[row_end - 1]
            if self._golden_dump_dir and bool(info.get("_omni_is_prefill", False)):
                self._dump_golden_prefill(info, row_hidden)
            main_logits = self._main_head_logits(row_hidden.reshape(1, -1))
            code0 = int(torch.argmax(main_logits, dim=-1).item())
            generated_frames = int(info.get("breeze_generated_frames", 0) or 0)
            max_new_frames = int(info.get("breeze_max_new_frames", -1))
            terminal = code0 == self.codebook_vocab_size or (
                max_new_frames > 0 and generated_frames >= max_new_frames
            )
            if terminal:
                info["breeze_force_eos"] = True
                existing = info.get("breeze_audio_codes")
                per_request_codes.append(
                    existing
                    if isinstance(existing, torch.Tensor) and existing.numel() and not self._async_chunk
                    else hidden.new_empty((0, self.num_codebooks), dtype=torch.long)
                )
                continue
            if code0 < 0 or code0 >= self.codebook_size:
                raise RuntimeError(f"Breeze main head produced invalid codec id {code0}")

            frame = self._generate_depth_codes(row_hidden, code0)
            accumulated = info.get("breeze_audio_codes")
            accumulated = (
                torch.cat([accumulated.to(frame.device), frame], dim=0)
                if isinstance(accumulated, torch.Tensor) and accumulated.numel()
                else frame
            )
            info["breeze_audio_codes"] = accumulated
            info["breeze_current_frame"] = frame[0].detach()
            info["breeze_generated_frames"] = generated_frames + 1
            info["breeze_force_eos"] = False
            per_request_codes.append(frame if self._async_chunk else accumulated)
        if self._golden_dump_dir:
            for info in info_items:
                self._dump_golden_frames(info)
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs={"codes": {"audio": per_request_codes}})

    def _dump_golden_frames(self, info: dict[str, Any]) -> None:
        """Write terminal codec frames for opt-in golden-parity diagnostics."""
        codes = info.get("breeze_audio_codes")
        if not info.get("breeze_force_eos", False) or not isinstance(codes, torch.Tensor) or codes.numel() == 0:
            return
        prompt_ids = info.get("prompt_ids")
        if not isinstance(prompt_ids, torch.Tensor):
            raise RuntimeError("Breeze golden dump is missing prompt_ids")
        digest = hashlib.sha256(prompt_ids.detach().cpu().numpy().tobytes()).hexdigest()[:16]
        output_dir = Path(self._golden_dump_dir)  # type: ignore[arg-type]
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "template": info.get("template"),
                "prompt_ids": prompt_ids.cpu(),
                "codes": codes.detach().cpu().to(torch.long).contiguous(),
                "generated_frames": int(info.get("breeze_generated_frames", 0) or 0),
                "finished": True,
            },
            output_dir / f"breeze_{digest}.pt",
        )


    def _dump_golden_prefill(self, info: dict[str, Any], hidden: torch.Tensor) -> None:
        """Write the terminal-prompt hidden row for opt-in parity diagnostics."""
        prompt_ids = info.get("prompt_ids")
        if not isinstance(prompt_ids, torch.Tensor):
            raise RuntimeError("Breeze golden dump is missing prompt_ids")
        digest = hashlib.sha256(prompt_ids.detach().cpu().numpy().tobytes()).hexdigest()[:16]
        output_dir = Path(self._golden_dump_dir)  # type: ignore[arg-type]
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "template": info.get("template"),
                "prompt_ids": prompt_ids.cpu(),
                "hidden": hidden.detach().reshape(-1).cpu().to(torch.float32).contiguous(),
            },
            output_dir / f"prefill_{digest}.pt",
        )

    def on_requests_finished(self, finished_req_ids: Iterable[str]) -> None:
        """Drop cross-step references retained for the finished AR requests."""
        del finished_req_ids
        # Codec/frame state lives in request-local additional_information and is
        # freed with the Request. Only the short logits-to-state mapping is
        # model-owned and must not leak into the next micro-batch.
        self._batch_state = None
        self._batch_state_spans = None

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        params = dict(self.named_parameters())
        backbone_weights: list[tuple[str, torch.Tensor]] = []
        text_weights: list[tuple[str, torch.Tensor]] = []
        depth_weights: list[tuple[str, torch.Tensor]] = []
        for name, tensor in weights:
            # This table belongs to Breeze's custom audio embedding module,
            # although its checkpoint key is nested under backbone_model.
            # Handle it before forwarding the remaining backbone weights to
            # Qwen3Model.
            if name == "backbone_model.embed_tokens.embed_audio_tokens.weight":
                target = "embed_audio_tokens.weight"
                param = params.get(target)
                if param is None:
                    raise ValueError(f"Missing Breeze parameter: {target}")
                _check_weight_shape(name, tensor, tuple(param.shape))
                default_weight_loader(param, tensor)
                loaded.add(target)
                continue
            if name.startswith("backbone_model."):
                # Qwen3Model.load_weights() is called on the nested module and
                # therefore expects names relative to that module (the same
                # convention used by MOSS-TTS), not the outer ``model.``
                # namespace used by ``named_parameters()``.
                backbone_weights.append((name[len("backbone_model.") :], tensor))
                continue
            if name.startswith("text_encoder."):
                text_name = name[len("text_encoder.") :]
                text_weights.append((text_name, tensor))
                continue
            if name.startswith("depth_decoder."):
                depth_weights.append((name[len("depth_decoder.") :], tensor))
                continue
            if name in {"lm_head.weight", "embed_text_tokens.weight"} or name.startswith("text_encoder_proj."):
                target = name
            else:
                continue
            param = params.get(target)
            if param is None:
                continue
            # ParallelLMHead stores a TP-local shard while checkpoint tensors
            # are global; validate against the global Breeze head shape.
            expected_shape = (
                (self.codebook_vocab_size + 1, self.hidden_size)
                if name == "lm_head.weight"
                else tuple(param.shape)
            )
            _check_weight_shape(name, tensor, expected_shape)
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, tensor)
            loaded.add(target)

        if backbone_weights:
            loaded.update("model." + key for key in self.model.load_weights(backbone_weights))
        if text_weights:
            loaded.update("text_encoder." + key for key in self.text_encoder.load_weights(text_weights))
        if depth_weights:
            loaded.update("depth_decoder." + key for key in self.depth_decoder.load_weights(depth_weights))
            if bool(getattr(self.config, "tie_codebooks_embeddings", False)):
                # The checkpoint serializes only one side of the tied table.
                # ``named_parameters(remove_duplicate=True)`` exposes it as the
                # talker-owned ``embed_audio_tokens.weight``, so report both
                # aliases to the strict weight tracker.
                if "depth_decoder.model.embed_tokens.weight" in loaded:
                    loaded.add("embed_audio_tokens.weight")
                if "embed_audio_tokens.weight" in loaded:
                    loaded.add("depth_decoder.model.embed_tokens.weight")
        return loaded

__all__ = ["BreezeTTS2TalkerForGeneration"]
