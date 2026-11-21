# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sesame CSM stage 0 core: Llama backbone, first-codebook head and depth decoder.

One engine token is one audio frame. The backbone hidden state gives the first
codebook through ``lm_head``, the depth decoder completes the frame, and the sum
of the frame's codebook embeddings is the backbone's next input.

Generation ends on a frame whose codebooks, except the last, are all the EOS id.
That cannot be seen from the first codebook alone, so the step after such a
frame is forced to emit ``stop_token_id``, an engine-only id one past the
codebook vocabulary.

A prompt may carry one reference recording. Its waveform arrives with the
request, is encoded with the Mimi codec at prefill, and the resulting frame
embeddings replace the prompt's audio placeholders.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from itertools import islice
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MimiModel
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaModel
from vllm.model_executor.models.utils import AutoWeightsLoader, make_layers, maybe_prefix

from vllm_omni.model_executor.models.csm.csm_depth_decoder import CsmDepthDecoder, draw_frame_uniforms
from vllm_omni.model_executor.models.csm.csm_mimi import rename_codec_weights
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker.sampling_utils import get_tts_local_seed

# Depth-decoder sampling of sesame/csm-1b's generation_config.json (depth_decoder_*).
_DEFAULT_DEPTH_SAMPLING = {"do_sample": True, "temperature": 0.9, "top_k": 50, "top_p": 1.0}
# Request field holding the reference recording: a mono waveform at the codec's sample rate.
REFERENCE_AUDIO_KEY = "csm_ref_audio"


def stop_frame_mask(codes: torch.Tensor, codebook_eos_token_id: int = 0) -> torch.Tensor:
    """Frames that end generation: every codebook except the last is the EOS id."""
    return (codes[..., :-1] == codebook_eos_token_id).all(dim=-1)


def empty_frame_mask(codes: torch.Tensor, codebook_eos_token_id: int = 0) -> torch.Tensor:
    """Frames where audio is cut: every codebook, including the last, is the EOS id."""
    return (codes == codebook_eos_token_id).all(dim=-1)


class CsmAudioEmbeddings(nn.Module):
    """Embeds an audio frame as the sum of its codebook embeddings."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.embed_audio_tokens = nn.Embedding(config.num_codebooks * config.vocab_size, config.hidden_size)
        self.register_buffer(
            "audio_tokens_offsets", torch.arange(config.num_codebooks) * config.vocab_size, persistent=False
        )

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Embed ``codes`` of shape ``[..., num_codebooks]``."""
        return self.embed_audio_tokens(codes + self.audio_tokens_offsets).sum(dim=-2)


class CsmBackbone(nn.Module):
    """Llama decoder layers and final norm of the CSM backbone."""

    hf_to_vllm_mapper = LlamaModel.hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, config: Any, prefix: str = "") -> None:
        super().__init__()
        self.embed_tokens = CsmAudioEmbeddings(config)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: LlamaDecoderLayer(vllm_config=vllm_config, prefix=prefix, config=config),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Return the hidden states after the final norm."""
        hidden_states = inputs_embeds
        residual = None
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load backbone weights, stacking q/k/v and gate/up projections."""
        return AutoWeightsLoader(self).load_weights(weights, mapper=self.hf_to_vllm_mapper)


class CsmTalkerForConditionalGeneration(nn.Module):
    """CSM backbone, first-codebook head and depth decoder for ``sesame/csm-1b``."""

    # The repository also ships the same weights in the original format (model.safetensors, ckpt.pt).
    allow_patterns_overrides = ["transformers-*.safetensors"]
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        if not getattr(config, "tie_codebooks_embeddings", True):
            raise ValueError("CSM checkpoints with untied codebook embeddings are not supported")
        self.config = config
        self.num_codebooks = int(config.num_codebooks)
        self.codebook_vocab_size = int(config.vocab_size)
        self.codebook_eos_token_id = int(config.codebook_eos_token_id)

        self.embed_text_tokens = VocabParallelEmbedding(
            config.text_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.text_vocab_size,
            prefix=maybe_prefix(prefix, "embed_text_tokens"),
        )
        self.backbone_model = CsmBackbone(
            vllm_config=vllm_config, config=config, prefix=maybe_prefix(prefix, "backbone_model")
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.depth_decoder = CsmDepthDecoder(config.depth_decoder_config)
        # Tied in the checkpoint (tie_codebooks_embeddings).
        self.depth_decoder.model.embed_tokens.weight = self.backbone_model.embed_tokens.embed_audio_tokens.weight
        # Encodes reference audio; runs in float32 whatever dtype the stage is configured with.
        self.codec_model = MimiModel(config.codec_config).to(torch.float32)
        self.audio_token_id = int(config.audio_token_id)
        self.audio_eos_token_id = int(config.audio_eos_token_id)

        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = False
        self.has_talker_mtp = True
        self.mtp_output_key = ("codes", "audio")
        self.mtp_accepts_per_row_generators = True
        # talker_mtp records the stop frame per request id.
        self.mtp_accepts_req_infos = True
        # Enables mrv2_sampling_context, which gives compute_logits the request order.
        self.logitsprocs_need_output_token_ids = True
        self.omni_pooler_payload_include_hidden = False
        self.stop_token_id = self.codebook_vocab_size
        self.logits_vocab_size = self.codebook_vocab_size + 1
        sampling = getattr(vllm_config.model_config, "subtalker_sampling_params", None)
        self.mtp_sampling_params = {**_DEFAULT_DEPTH_SAMPLING, **(sampling if isinstance(sampling, Mapping) else {})}

        stop_logits = torch.full((self.logits_vocab_size,), float("-inf"))
        stop_logits[self.stop_token_id] = 0.0
        self.register_buffer("_stop_logits", stop_logits, persistent=False)
        self.register_buffer("_no_stop", torch.zeros((), dtype=torch.bool), persistent=False)
        self.register_buffer("_zero_text_step", torch.zeros((1, config.hidden_size)), persistent=False)
        self._stop_flags: dict[str, torch.Tensor] = {}
        self._forced_stop_mask: torch.Tensor | None = None
        # Per request: reference frame embeddings followed by the audio-EOS embedding, and the prompt
        # position of the first placeholder (None until a prefill span reaches it).
        self._references: dict[str, tuple[torch.Tensor, int | None]] = {}

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        """Embed prompt text tokens."""
        return self.embed_text_tokens(input_ids)

    def embed_audio_frames(self, codes: torch.Tensor) -> torch.Tensor:
        """Embed audio frames of shape ``[..., num_codebooks]`` as backbone inputs."""
        return self.backbone_model.embed_tokens(codes)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        """Run the backbone and return its hidden states."""
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        return self.backbone_model(positions, inputs_embeds)

    def compute_logits(
        self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None
    ) -> torch.Tensor | None:
        """Return first-codebook logits plus the stop column, ``[N, logits_vocab_size]``."""
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        logits = F.pad(self.lm_head(hidden_states), (0, 1), value=float("-inf"))
        mask = self._forced_stop_mask
        # The mask is set only while sampling; prompt-logprob calls have other row counts.
        if mask is not None and mask.shape[0] == logits.shape[0]:
            logits = torch.where(mask, self._stop_logits.to(logits.dtype), logits)
        return logits

    @contextmanager
    def mrv2_sampling_context(self, *, req_ids: Sequence[str], num_output_tokens: torch.Tensor) -> Iterator[None]:
        """Force the stop token for requests whose previous frame was a stop frame."""
        flags = self._stop_flags
        if flags:
            self._forced_stop_mask = torch.stack([flags.get(req_id, self._no_stop) for req_id in req_ids])[:, None]
        try:
            yield
        finally:
            self._forced_stop_mask = None

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        """Allocate this step's token-major frame outputs; the runner fills them after sampling."""
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        spans = kwargs.get("request_token_spans")
        num_tokens = spans[-1][1] if spans else model_outputs.shape[0]
        device = model_outputs.device
        return OmniOutput(
            text_hidden_states=model_outputs,
            multimodal_outputs={
                "codes": {"audio": torch.zeros((num_tokens, self.num_codebooks), dtype=torch.long, device=device)},
                "meta": {"codec_frame_valid": torch.zeros((num_tokens,), dtype=torch.int8, device=device)},
            },
        )

    def preprocess(
        self, input_ids: torch.Tensor, input_embeds: torch.Tensor | None, **info: Any
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Per-request input hook.

        Prompt rows keep their text embeddings, except the audio placeholders of
        a reference recording; decode rows take the stored frame embedding.
        """
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        is_prefill = info.get("_omni_is_prefill")
        if is_prefill is None:
            is_prefill = input_ids.shape[0] != 1
        if is_prefill:
            waveform = info.get(REFERENCE_AUDIO_KEY)
            if waveform is not None:
                input_embeds = self._embed_reference_audio(input_ids, input_embeds, waveform, info)
            return input_ids, input_embeds, {}
        # CSM adds nothing to the frame embedding on decode steps.
        return input_ids, input_embeds, {"mtp_inputs": (self._zero_text_step, self._zero_text_step)}

    def _embed_reference_audio(
        self, input_ids: torch.Tensor, input_embeds: torch.Tensor, waveform: Any, info: Mapping[str, Any]
    ) -> torch.Tensor:
        """Put the reference recording's frame embeddings on this prompt span's audio placeholders.

        The prompt holds one placeholder per frame followed by the audio-EOS
        token, which takes the embedding of an all-EOS frame. A prompt that
        does not match the recording's frame count is rejected.
        """
        req_id = info.get("req_id")
        frames, start = self._references.get(req_id, (None, None))
        if frames is None:
            samples = torch.as_tensor(waveform, dtype=torch.float32).to(input_embeds.device).reshape(1, 1, -1)
            codes = self.codec_model.encode(samples).audio_codes[0].T
            audio_eos = codes.new_full((1, self.num_codebooks), self.codebook_eos_token_id)
            frames = self.embed_audio_frames(torch.cat((codes, audio_eos)))
        num_frames = frames.shape[0] - 1

        is_audio = input_ids == self.audio_token_id
        is_audio_eos = input_ids == self.audio_eos_token_id
        computed = int(info.get("_omni_num_computed_tokens") or 0)
        if start is None and bool(is_audio.any()):
            start = computed + int(is_audio.int().argmax())
        if start is None:
            expected_audio = expected_audio_eos = torch.zeros_like(is_audio)
            index = None
        else:
            index = computed + torch.arange(input_ids.shape[0], device=input_ids.device) - start
            expected_audio = (index >= 0) & (index < num_frames)
            expected_audio_eos = index == num_frames
        prompt_len = info.get("_omni_prompt_len")
        prompt_done = prompt_len is not None and computed + input_ids.shape[0] >= int(prompt_len)
        matches = torch.equal(is_audio, expected_audio) and torch.equal(is_audio_eos, expected_audio_eos)
        # By the end of the prompt every frame and the audio-EOS token must have had a position.
        if prompt_done and (start is None or start + num_frames >= int(prompt_len)):
            matches = False
        if not matches:
            self._references.pop(req_id, None)
            raise ValueError(
                f"The CSM prompt of request {req_id} does not match its reference audio: the recording has "
                f"{num_frames} frames, which need as many audio placeholders followed by one audio-EOS token"
            )
        if index is not None:
            replaced = (is_audio | is_audio_eos)[:, None]
            reference = frames.index_select(0, index.clamp(0, num_frames)).to(input_embeds.dtype)
            input_embeds = torch.where(replaced, reference, input_embeds)

        if req_id is not None:
            if prompt_done:
                # The whole prompt is embedded; the frames are not needed again.
                self._references.pop(req_id, None)
            else:
                self._references[req_id] = (frames, start)
        return input_embeds

    def mtp_frame_valid(self, layer0: torch.Tensor) -> torch.Tensor:
        """Rows whose first codebook is a code rather than the stop token."""
        return (layer0 >= 0) & (layer0 < self.codebook_vocab_size)

    def talker_mtp(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
        do_sample: bool | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
        generators: Sequence[torch.Generator | None] | None = None,
        sample_uniforms: torch.Tensor | None = None,
        req_ids: Sequence[str] | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Complete the frames of the just-sampled first codebooks.

        Returns the next backbone input ``[B, hidden]`` and the codes ``[B, num_codebooks]``.
        """
        first_codes = input_ids.reshape(-1).to(torch.long)
        bsz = first_codes.shape[0]
        # A forced stop token is not a code; its row is computed on code 0 and discarded.
        is_code = self.mtp_frame_valid(first_codes)
        first_codes = torch.where(is_code, first_codes, torch.zeros_like(first_codes))

        defaults = self.mtp_sampling_params
        do_sample = bool(defaults["do_sample"] if do_sample is None else do_sample)
        if do_sample and sample_uniforms is None:
            if generators is None and generator is not None:
                generators = [generator] * bsz
            sample_uniforms = draw_frame_uniforms(
                bsz,
                self.depth_decoder.num_steps,
                self.codebook_vocab_size,
                device=first_codes.device,
                generators=generators,
            )
        codes, frame_embeds = self.generate_frame(
            last_talker_hidden.reshape(bsz, -1),
            first_codes,
            do_sample=do_sample,
            temperature=float(defaults["temperature"] if temperature is None else temperature),
            top_k=int(defaults["top_k"] if top_k is None else top_k),
            top_p=float(defaults["top_p"] if top_p is None else top_p),
            uniforms=sample_uniforms,
        )
        if req_ids is not None:
            # A request that already received the stop token stays stopped.
            stop = stop_frame_mask(codes, self.codebook_eos_token_id) | ~is_code
            for row, req_id in enumerate(req_ids):
                self._stop_flags[req_id] = stop[row]
        codes = torch.where(is_code[:, None], codes, torch.zeros_like(codes))
        return frame_embeds.to(input_embeds.dtype), codes

    mtp = talker_mtp
    get_mtp_seed = staticmethod(get_tts_local_seed)

    def on_requests_finished(self, req_ids: Iterable[str]) -> None:
        """Drop the per-request state of finished or aborted requests."""
        for req_id in req_ids:
            self._stop_flags.pop(req_id, None)
            self._references.pop(req_id, None)

    def generate_frame(
        self,
        backbone_hidden: torch.Tensor,
        first_codes: torch.Tensor,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        uniforms: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Complete the frames started by ``first_codes`` with the depth decoder."""
        return self.depth_decoder.generate_frame(
            backbone_hidden,
            first_codes,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            uniforms=uniforms,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load a transformers-format CSM checkpoint, whichever Mimi layout it and transformers use."""
        return AutoWeightsLoader(self).load_weights(rename_codec_weights(weights, self.codec_model))
