# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage 1 of Chatterbox: S3 speech tokens to audio.

Every step, the stage is handed the chunk of every request that has one
ready. All of them go through one flow call, one row per request, so
requests are the batch dimension here as they are in stage 0. The vocoder
batches the rows whose mels are equally long and never pads one, so the
audio a request plays does not depend on what shares its step.

Upstream ships S3Gen with the streaming hooks of the CosyVoice 2 decoder it
is derived from but not the loop that uses them, and its flow's own
streaming branch sizes the decoder mask from the uncropped encoder length.
``flow_mels`` is ``S3Token2Wav.flow_inference`` plus
``CausalMaskedDiffWithXvec.inference`` (0.1.7) for a batch, with each row's
mask built from its own cropped length.

Seams follow CosyVoice 2's ``token2wav``. A chunk is decoded from the last
``LEFT_CONTEXT_TOKENS`` of context plus its own tokens, the frames before the
offset are dropped, the previous chunk's last eight mel frames are vocoded
again, and the overlap is cross-faded with a Hamming window. Bounding the
context keeps a chunk's cost constant instead of growing with the sentence.

The flow attends over its whole input, so a frame's mel depends on the
tokens decoded with it and on the noise the flow starts from. Upstream draws
that noise once per utterance. Drawing it per chunk would make every chunk a
different sample of the utterance, and the context a chunk decodes again
would disagree with the audio already played. The decoder instead takes the
noise by position from one fixed buffer, so a mel frame starts from the same
noise every time it is decoded, whichever chunk it falls in. The flow is
therefore deterministic for a given token sequence and voice, where upstream
gives a new sample on every call; the vocoder's source module still draws
its own phase and noise per call. Chunks still differ from a one-chunk
decode of the same tokens, because a chunk is decoded before the tokens
that follow it exist and from a bounded context.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.chatterbox.s3gen_core.configs import CFM_PARAMS
from vllm_omni.model_executor.models.chatterbox.s3gen_core.decoder import ConditionalDecoder
from vllm_omni.model_executor.models.chatterbox.s3gen_core.flow import CausalMaskedDiffWithXvec
from vllm_omni.model_executor.models.chatterbox.s3gen_core.flow_matching import CausalConditionalCFM
from vllm_omni.model_executor.models.chatterbox.s3gen_core.mask import make_pad_mask
from vllm_omni.model_executor.models.chatterbox.s3gen_core.upsample_encoder import UpsampleConformerEncoder
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import HiFTGenerator
from vllm_omni.model_executor.models.glm_tts.vocoder import ConvRNNF0Predictor
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

# CosyVoice 2's caches: eight mel frames, and HiFT's 480 samples per frame
# (upsample rates 8 * 5 * 3 times an iSTFT hop of 4).
MEL_CACHE_FRAMES = 8
SAMPLES_PER_FRAME = 480
SOURCE_CACHE_SAMPLES = MEL_CACHE_FRAMES * SAMPLES_PER_FRAME
# Tokens of already-played context a chunk is decoded with (two seconds).
LEFT_CONTEXT_TOKENS = 50
# Seed of the flow-noise buffer: fixed, so every process draws the same one.
NOISE_SEED = 0
# The checkpoint also holds the reference encoders, which run in the API
# process (conditioning.VoiceConditioner), not in this stage.
REFERENCE_ENCODER_PREFIXES = ("tokenizer.", "speaker_encoder.")


@dataclass
class Reference:
    """One request's voice, as stage 1 needs it.

    Attributes:
        prompt_token: Shape (1, P), S3 tokens of the reference clip.
        prompt_feat: Shape (1, 2P, 80), its mel at the output sample rate.
        embedding: Shape (1, 192), its x-vector.
    """

    prompt_token: torch.Tensor
    prompt_feat: torch.Tensor
    embedding: torch.Tensor


@dataclass
class StreamState:
    """What one request carries from one chunk to the next.

    Attributes:
        reference: The request's voice, which arrives with its first chunk
            and with no later one.
        mel: Shape (1, 80, 8), the last frames decoded, vocoded again next time.
        source: Shape (1, 1, 3840), HiFT's source signal for those frames.
        speech: Shape (1, 3840), the samples held back for the cross-fade.
    """

    reference: Reference
    mel: torch.Tensor
    source: torch.Tensor
    speech: torch.Tensor


@dataclass
class Chunk:
    """One request's share of a decode step.

    Attributes:
        tokens: Shape (N,), every valid token of the utterance so far.
        token_offset: Tokens already played; their frames are dropped.
        reference: The request's voice.
        state: The previous chunk's state, or None for a first chunk.
        finalize: Whether this is the utterance's last chunk.
    """

    tokens: torch.Tensor
    token_offset: int
    reference: Reference
    state: StreamState | None
    finalize: bool


def window(tokens: torch.Tensor, token_offset: int) -> tuple[torch.Tensor, int]:
    """Keep the last ``LEFT_CONTEXT_TOKENS`` played tokens and what follows.

    Args:
        tokens: Shape (N,), every valid token so far.
        token_offset: Tokens already played.

    Returns:
        The kept tokens and the offset of the first new token within them.
    """
    start = max(0, token_offset - LEFT_CONTEXT_TOKENS)
    return tokens[start:], token_offset - start


def fade_in_out(fade_in: torch.Tensor, fade_out: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """CosyVoice's overlap-add, in place on ``fade_in``.

    Args:
        fade_in: Shape (B, T), the new audio.
        fade_out: Shape (B, W / 2), the held-back tail of the previous chunk.
        window: Shape (W,), rising half applied to the new audio, falling
            half to the tail.

    Returns:
        ``fade_in`` with its first W / 2 samples blended.
    """
    overlap = window.shape[0] // 2
    fade_in[..., :overlap] = fade_in[..., :overlap] * window[:overlap] + fade_out[..., -overlap:] * window[overlap:]
    return fade_in


def encode(encoder: UpsampleConformerEncoder, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """``UpsampleConformerEncoder.forward`` with the padding zeroed.

    Upstream masks the token embeddings but not the encoder's own input
    projection, whose bias and LayerNorm turn a padded row's zeros into a
    constant. The pre-lookahead layer then convolves three positions ahead
    of every frame, so a short row's last tokens would mix that constant in
    and the row would not decode as it does alone. Zeroing after ``embed``
    changes nothing when there is nothing to pad.

    Nothing else of upstream's ``forward`` applies: the encoder is built
    without dynamic chunking, CNN module or CMVN, so its chunk masks are the
    padding masks.

    Args:
        encoder: ``flow.encoder``.
        tokens: Shape (B, T, 512), the masked token embeddings.
        lengths: Shape (B,), each row's token count.

    Returns:
        Shape (B, 2T, 512).
    """
    masks = ~make_pad_mask(lengths, tokens.shape[1]).unsqueeze(1)
    hidden, pos_emb, masks = encoder.embed(tokens, masks)
    hidden = encoder.pre_lookahead_layer(hidden * masks.transpose(1, 2).to(hidden))
    hidden = encoder.forward_layers(hidden, masks, pos_emb, masks)
    hidden, lengths = encoder.up_layer(hidden.transpose(1, 2).contiguous(), lengths)
    hidden = hidden.transpose(1, 2).contiguous()
    masks = ~make_pad_mask(lengths, hidden.shape[1]).unsqueeze(1)
    hidden, pos_emb, masks = encoder.up_embed(hidden, masks)
    hidden = encoder.forward_up_layers(hidden, masks, pos_emb, masks)
    return encoder.after_norm(hidden)


def flow_mels(
    flow: CausalMaskedDiffWithXvec,
    tokens: list[torch.Tensor],
    references: list[Reference],
    finalize: list[bool],
    n_timesteps: int,
    meanflow: bool,
    noise: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Mels for a batch of token rows in one flow call.

    Row ``i`` is laid out ``[prompt_i | tokens_i | padding]``. Prompts differ
    in length between requests (a reference clip shorter than the
    conditioning window gives a shorter prompt), so every length here is
    per row.

    Args:
        flow: The loaded flow.
        tokens: B rows of shape (N_i,), valid speech tokens.
        references: B voices.
        finalize: Per row, whether the lookahead frames are kept.
        n_timesteps: Flow steps.
        meanflow: Whether the checkpoint is a distilled meanflow model (plain
            Euler steps) or a standard one (Euler steps with the flow's own
            classifier-free guidance, on a cosine schedule).
        noise: Shape (B, 80, F) covering the last F mel frames of the flow's
            input; the flow draws the frames before them. None lets it draw
            them all, which is the distribution upstream's ``flow_inference``
            samples from. ``S3GenDecoder`` passes the whole input, taken by
            position from its buffer.

    Returns:
        B mels of shape (1, 80, F_i) with ``F_i = 2 * N_i``, minus the
        lookahead's six frames when not final: the generated frames only.
    """
    device = flow.input_embedding.weight.device
    ratio = flow.token_mel_ratio
    prompt_lens = [ref.prompt_token.shape[1] for ref in references]
    row_lens = [prompt + row.numel() for prompt, row in zip(prompt_lens, tokens, strict=True)]
    valid = [
        length * ratio - (0 if final else flow.pre_lookahead_len * ratio)
        for length, final in zip(row_lens, finalize, strict=True)
    ]

    token = nn.utils.rnn.pad_sequence(
        [torch.cat([ref.prompt_token[0], row]) for ref, row in zip(references, tokens, strict=True)],
        batch_first=True,
    )
    token_len = torch.tensor(row_lens, device=device)
    embedding = flow.spk_embed_affine_layer(
        nn.functional.normalize(torch.cat([ref.embedding for ref in references]), dim=1)
    )
    token_mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
    h = flow.encoder_proj(encode(flow.encoder, flow.input_embedding(token) * token_mask, token_len))

    frames = torch.arange(h.shape[1], device=device)
    mask = (frames[None, :] < torch.tensor(valid, device=device)[:, None]).unsqueeze(1).to(h)
    conds = torch.zeros_like(h)
    for i, ref in enumerate(references):
        conds[i, : prompt_lens[i] * ratio] = ref.prompt_feat[0]
    feat, _ = flow.decoder(
        mu=h.transpose(1, 2).contiguous(),
        mask=mask,
        spks=embedding,
        cond=conds.transpose(1, 2),
        n_timesteps=n_timesteps,
        noised_mels=noise,
        meanflow=meanflow,
    )
    return [feat[i : i + 1, :, prompt_lens[i] * ratio : valid[i]] for i in range(len(tokens))]


class S3GenDecoder(nn.Module):
    """S3Gen's flow and vocoder with a batched, streaming decode.

    Parameter names match the checkpoint (``flow.*``, ``mel2wav.*``).

    The flow never draws noise here: every mel frame starts from the noise
    at its own position in ``flow_noise``, so a frame decoded again in a
    later chunk, or in another batch, starts from the same noise. The same
    tokens and voice always give the same mel.

    Args:
        config: The checkpoint's constants; the flow's variant, its step
            count and the silence tokens all come from it.
        max_utterance_tokens: The longest utterance the decoder will be
            handed, in speech tokens: the stage's context length. The noise
            buffer covers it, so no utterance runs past the buffer.
    """

    def __init__(self, config: ChatterboxConfig, max_utterance_tokens: int) -> None:
        super().__init__()
        self.config = config
        # The architecture ``S3Token2Mel`` and ``S3Token2Wav`` build (0.1.7).
        encoder = UpsampleConformerEncoder(
            output_size=512,
            attention_heads=8,
            linear_units=2048,
            num_blocks=6,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalize_before=True,
            input_layer="linear",
            pos_enc_layer_type="rel_pos_espnet",
            selfattention_layer_type="rel_selfattn",
            input_size=512,
            use_cnn_module=False,
            macaron_style=False,
        )
        estimator = ConditionalDecoder(
            in_channels=320,
            out_channels=80,
            causal=True,
            channels=[256],
            dropout=0.0,
            attention_head_dim=64,
            n_blocks=4,
            num_mid_blocks=12,
            num_heads=8,
            act_fn="gelu",
            meanflow=config.meanflow,
        )
        self.flow = CausalMaskedDiffWithXvec(
            encoder=encoder,
            decoder=CausalConditionalCFM(spk_emb_dim=80, cfm_params=CFM_PARAMS, estimator=estimator),
            token_mel_ratio=config.token_mel_ratio,
            pre_lookahead_len=config.pre_lookahead_len,
        )
        self.mel2wav = HiFTGenerator(
            sampling_rate=config.sample_rate,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            f0_predictor=ConvRNNF0Predictor(),
        )

        # Upstream silences the first 20 ms and fades the next 20 ms in, to
        # hide spillover from the reference clip.
        n_trim = config.sample_rate // 50
        trim_fade = torch.zeros(2 * n_trim)
        trim_fade[n_trim:] = (torch.cos(torch.linspace(torch.pi, 0, n_trim)) + 1) / 2
        self.register_buffer("trim_fade", trim_fade, persistent=False)
        self.register_buffer(
            "fade_window", torch.hamming_window(2 * SOURCE_CACHE_SAMPLES, periodic=False), persistent=False
        )
        # Upstream appends these to the last tokens so speech does not end clipped.
        self.register_buffer(
            "silence", torch.full((config.n_silence_tokens,), config.silence_token, dtype=torch.long), persistent=False
        )

        # The flow's noise by position: the reference prompt's frames, then
        # the utterance's. The prompt region holds the longest reference the
        # conditioning can produce (``dec_cond_seconds`` of tokens). The
        # utterance region is drawn in two parts from one generator: first
        # the frames of the default output cap and the silence, then the
        # frames up to ``max_utterance_tokens``. The first part is the whole
        # buffer of a decoder sized for the default cap, so every position an
        # utterance within the cap uses has the same noise whatever the
        # context is.
        # Drawn on the CPU and then moved: vLLM builds the stage inside a
        # default-device context, where a CPU generator cannot fill the
        # tensor and a device generator would draw different noise.
        ratio = config.token_mel_ratio
        self.prompt_noise_frames = config.dec_cond_seconds * config.token_rate * ratio
        capped_frames = self.prompt_noise_frames + (config.max_new_tokens + config.n_silence_tokens) * ratio
        generator = torch.Generator().manual_seed(NOISE_SEED)
        self.register_buffer(
            "flow_noise",
            torch.cat(
                [
                    torch.randn(1, config.mel["num_mels"], frames, generator=generator, device="cpu")
                    for frames in (capped_frames, max(0, max_utterance_tokens - config.max_new_tokens) * ratio)
                ],
                dim=2,
            ).to(trim_fade.device),
            persistent=False,
        )

        # One entry per request mid-stream, keyed by the scheduler's request
        # id. No lock: forward and on_requests_finished run on one thread.
        self.streams: dict[str, StreamState] = {}

    def vocode(self, mels: list[torch.Tensor], cache: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """HiFT over B mels, one call per mel length.

        Rows are never padded to share a call. HiFT and its F0 predictor are
        not causal: padding reaches the last twelve frames of the row before
        it, whether the padding is zeros or the row's last frame repeated,
        and a non-final chunk holds back only eight. Rows of equal length
        share a call, which leaves each as it is alone.

        Args:
            mels: B mels of shape (1, 80, F_i).
            cache: Shape (B, 1, S), source samples written over each row's
                first S samples; S is 0 for first chunks.

        Returns:
            B waveforms of shape (1, F_i * 480) and B sources (1, 1, F_i * 480).
        """
        frames = [mel.shape[2] for mel in mels]
        speech: dict[int, torch.Tensor] = {}
        source: dict[int, torch.Tensor] = {}
        for length in dict.fromkeys(frames):
            rows = [i for i, n in enumerate(frames) if n == length]
            wavs, sources = self.mel2wav.inference(
                speech_feat=torch.cat([mels[i] for i in rows]), cache_source=cache[rows]
            )
            for k, i in enumerate(rows):
                speech[i], source[i] = wavs[k : k + 1], sources[k : k + 1]
        return [speech[i] for i in range(len(mels))], [source[i] for i in range(len(mels))]

    @torch.inference_mode()
    def chunk_mels(self, chunks: list[Chunk]) -> list[torch.Tensor]:
        """Every chunk's new mel frames, from one flow call.

        Row ``i`` of the flow's noise is ``[prompt region, its first 2 * P_i
        frames | utterance region, the frames of the row's tokens by their
        position in the utterance | zero padding]``. The appended silence
        tokens continue the positions.

        Args:
            chunks: The step's chunks, one per request.

        Returns:
            Per chunk, in order: shape (1, 80, F), the frames after the
            chunk's token offset.

        Raises:
            RuntimeError: If a non-final chunk brings fewer new frames than
                the mel cache holds.
        """
        ratio = self.config.token_mel_ratio
        rows: list[torch.Tensor] = []
        offsets: list[int] = []
        noise: list[torch.Tensor] = []
        for chunk in chunks:
            kept, offset = window(chunk.tokens, chunk.token_offset)
            new_frames = (kept.numel() - offset - self.config.pre_lookahead_len) * ratio
            if not chunk.finalize and new_frames < MEL_CACHE_FRAMES:
                raise RuntimeError(
                    f"chatterbox_s3gen got a non-final chunk of {new_frames} new mel frames; the cross-fade "
                    f"needs {MEL_CACHE_FRAMES}: codec_chunk_frames must be at least "
                    f"{math.ceil(MEL_CACHE_FRAMES / ratio)}"
                )
            row = torch.cat([kept, self.silence]) if chunk.finalize else kept
            prompt_frames = chunk.reference.prompt_token.shape[1] * ratio
            # ``kept`` is the tail of the utterance so far.
            start = self.prompt_noise_frames + (chunk.tokens.numel() - kept.numel()) * ratio
            end = start + row.numel() * ratio
            rows.append(row)
            offsets.append(offset)
            # (F_i, 80), the layout pad_sequence pads.
            noise.append(torch.cat([self.flow_noise[0, :, :prompt_frames], self.flow_noise[0, :, start:end]], dim=1).T)
        mels = flow_mels(
            self.flow,
            rows,
            [chunk.reference for chunk in chunks],
            [chunk.finalize for chunk in chunks],
            self.config.n_cfm_timesteps,
            self.config.meanflow,
            nn.utils.rnn.pad_sequence(noise, batch_first=True).transpose(1, 2),
        )
        return [mel[:, :, offset * ratio :] for mel, offset in zip(mels, offsets, strict=True)]

    @torch.inference_mode()
    def chunked_decode_streaming(self, chunks: list[Chunk]) -> list[tuple[torch.Tensor, StreamState | None]]:
        """Decode every chunk's new audio in one pass.

        Rows carrying a source cache and first chunks are vocoded separately,
        because HiFT writes the cache over the first samples of every row it
        is given; within each, see ``vocode``.

        Args:
            chunks: The step's chunks, one per request.

        Returns:
            Per chunk, in order: the new audio of shape (1, n) and the state
            for the request's next chunk (None after the last).
        """
        mels = self.chunk_mels(chunks)

        speech: dict[int, torch.Tensor] = {}
        source: dict[int, torch.Tensor] = {}
        continuing = [(i, chunk.state) for i, chunk in enumerate(chunks) if chunk.state is not None]
        first = [i for i, chunk in enumerate(chunks) if chunk.state is None]
        if continuing:
            for i, state in continuing:
                mels[i] = torch.cat([state.mel, mels[i]], dim=2)
            wavs, sources = self.vocode(
                [mels[i] for i, _ in continuing], torch.cat([state.source for _, state in continuing])
            )
            blended = fade_in_out(
                torch.cat([wav[:, :SOURCE_CACHE_SAMPLES] for wav in wavs]),
                torch.cat([state.speech for _, state in continuing]),
                self.fade_window,
            )
            for k, (i, _) in enumerate(continuing):
                wavs[k][:, :SOURCE_CACHE_SAMPLES] = blended[k : k + 1]
                speech[i], source[i] = wavs[k], sources[k]
        if first:
            wavs, sources = self.vocode([mels[i] for i in first], self.fade_window.new_zeros(len(first), 1, 0))
            for k, i in enumerate(first):
                wavs[k][:, : self.trim_fade.shape[0]] *= self.trim_fade
                speech[i], source[i] = wavs[k], sources[k]

        results: list[tuple[torch.Tensor, StreamState | None]] = []
        for i, chunk in enumerate(chunks):
            if chunk.finalize:
                results.append((speech[i], None))
            else:
                results.append(
                    (
                        speech[i][:, :-SOURCE_CACHE_SAMPLES],
                        StreamState(
                            reference=chunk.reference,
                            mel=mels[i][:, :, -MEL_CACHE_FRAMES:],
                            source=source[i][:, :, -SOURCE_CACHE_SAMPLES:],
                            speech=speech[i][:, -SOURCE_CACHE_SAMPLES:],
                        ),
                    )
                )
        return results

    def decode_step(
        self,
        input_ids: torch.Tensor,
        counts: list[int],
        payloads: list[dict] | None,
        request_ids: list[str] | None,
    ) -> list[torch.Tensor]:
        """One engine step: every scheduled request's tokens to its new audio.

        Args:
            input_ids: The step's flat token ids; per request, every valid
                speech token of the utterance so far.
            counts: Tokens per request, in batch order.
            payloads: Per request, the inter-stage payload of its chunk;
                empty for a request the runner holds none for. Only a
                request's first chunk has to carry its voice.
            request_ids: Per request, the scheduler's id. None on the
                engine's profiling run, and only then.

        Returns:
            Per request, 1-D float32 audio: only the samples new in this
            step, empty when the request had nothing to decode.

        Raises:
            RuntimeError: If a request has tokens but no stream metadata:
                decoding such a chunk as a whole utterance would play wrong
                audio with no error.
        """
        device, dtype = self.trim_fade.device, self.trim_fade.dtype
        audios = [torch.zeros(0, device=input_ids.device)] * len(counts)
        if request_ids is None:
            return audios

        flat = input_ids.reshape(-1)
        chunks: list[Chunk] = []
        owners: list[tuple[int, str]] = []
        start = 0
        for idx, (count, raw, request_id) in enumerate(
            zip(counts, payloads or [{}] * len(counts), request_ids, strict=True)
        ):
            tokens = flat[start : start + count].long()
            start += count
            # The async processor's terminal payload carries no tokens.
            if count == 0:
                continue
            # Read by section: beside the chunk's payload the runner keeps
            # keys of its own in the same dict (the request id, the sampling
            # parameters), which are not payload fields.
            meta, embed = raw.get("meta") or {}, raw.get("embed") or {}
            offset, final = meta.get("left_context_size"), meta.get("stream_finished")
            if offset is None or final is None:
                raise RuntimeError(
                    f"chatterbox_s3gen got {count} tokens for request {request_id} without stream metadata"
                )
            # A stream has state from the end of its first chunk to its
            # last, so a chunk without state is a first chunk and brings the
            # voice. A later chunk decodes in the voice the stream began
            # with, whatever its own payload holds.
            state = self.streams.get(request_id)
            reference = (
                state.reference
                if state is not None
                else Reference(
                    prompt_token=embed["speech_token"].to(device),
                    prompt_feat=embed["speech_feat"].to(device=device, dtype=dtype),
                    embedding=embed["embedding"].to(device=device, dtype=dtype),
                )
            )
            chunks.append(
                Chunk(
                    tokens=tokens.to(device),
                    token_offset=offset,
                    reference=reference,
                    state=state,
                    finalize=bool(final),
                )
            )
            owners.append((idx, request_id))

        if not chunks:
            return audios
        for (idx, request_id), (audio, state) in zip(owners, self.chunked_decode_streaming(chunks), strict=True):
            if state is None:
                self.streams.pop(request_id, None)
            else:
                self.streams[request_id] = state
            audios[idx] = audio.reshape(-1).float()
        return audios

    def on_requests_finished(self, finished_req_ids: Iterable[str]) -> None:
        """Free finished requests' stream state.

        The runner calls this every step with the scheduler's finished ids,
        aborts included. An aborted stream never sends a final chunk, so
        this is the only place its state is freed.

        Args:
            finished_req_ids: The scheduler's ids of the requests that ended.
        """
        for request_id in finished_req_ids:
            self.streams.pop(request_id, None)


class ChatterboxS3Gen(S3GenDecoder):
    """``S3GenDecoder`` as a vLLM-Omni generation stage.

    Args:
        vllm_config: The engine's config; its ``hf_config`` is the
            checkpoint's ``ChatterboxConfig``.
        prefix: vLLM's module prefix, unused: the stage is the root module.

    Raises:
        ValueError: If one step's token budget cannot hold every slot's
            longest utterance.
    """

    # Without this the runner discards OmniOutput.multimodal_outputs.
    have_multimodal_outputs = True
    # Has the default runner keep each request's inter-stage payload and
    # merge every chunk into it. Model Runner V2 does not read this: it
    # builds the payload anew for every chunk, from the request's own
    # additional_information and that chunk alone, which is why the voice
    # is kept in the stream state.
    enable_update_additional_information = True
    # Has the runner pass the scheduler's request ids, the ids that
    # on_requests_finished is later given. The payload's own id is the
    # upstream stage's external id and may differ.
    requires_request_ids = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config: ChatterboxConfig = vllm_config.model_config.hf_config
        # The scheduler gives a new request what is left of the step's token
        # budget, and the decoder cannot tell part of an utterance from one.
        # That budget is the field vLLM's scheduler reads: the scheduled-token
        # limit whenever one is set, 0 included. No utterance is longer than
        # stage 0's context, whatever max_tokens a request asks for, and the
        # deploy file keeps this stage's context at least that long.
        scheduler = vllm_config.scheduler_config
        field = "max_num_batched_tokens" if scheduler.max_num_scheduled_tokens is None else "max_num_scheduled_tokens"
        budget = getattr(scheduler, field)
        max_model_len = vllm_config.model_config.max_model_len
        needed = scheduler.max_num_seqs * max_model_len
        if budget < needed:
            raise ValueError(
                f"chatterbox_s3gen needs {field} >= max_num_seqs ({scheduler.max_num_seqs}) * "
                f"max_model_len ({max_model_len}) = {needed}, got {budget}: an utterance scheduled in part "
                "would be decoded as a whole one"
            )
        super().__init__(config, max_model_len)
        # The repo holds other checkpoints too; this stage loads its own.
        self.allow_patterns_overrides = [config.s3gen_weights]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        *,
        seq_token_counts: list[int],
        model_intermediate_buffer: list[dict] | None = None,
        request_ids: list[str] | None = None,
        **runner_kwargs: object,
    ) -> OmniOutput:
        """Decode the step's tokens. Emits delta audio, see ``decode_step``.

        Args:
            input_ids: The step's flat token ids.
            positions: Unused; the stage has no positional state.
            intermediate_tensors: Unused; the stage is not pipeline-parallel.
            inputs_embeds: Unused; the tokens are decoded as ids.
            seq_token_counts: Tokens per request, in batch order.
            model_intermediate_buffer: Per request, the inter-stage payload
                of its chunk.
            request_ids: Per request, the scheduler's id. Absent only on the
                runner's profiling run.
            **runner_kwargs: Whatever else the runner passes to every model.

        Returns:
            ``multimodal_outputs`` with, per request, ``audio`` (1-D float32,
            the samples new in this step) and ``sr`` (int32 scalar).
        """
        audios = self.decode_step(input_ids, seq_token_counts, model_intermediate_buffer, request_ids)
        rate = torch.tensor(self.config.sample_rate, dtype=torch.int32)
        return OmniOutput(text_hidden_states=None, multimodal_outputs={"audio": audios, "sr": [rate] * len(audios)})

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the S3Gen checkpoint the config names.

        Strict: a key this stage does not own, or a parameter the file does
        not fill, fails the load instead of leaving random weights behind.

        Args:
            weights: The checkpoint's tensors by name, the reference
                encoders' included.

        Returns:
            The names of every parameter filled, as vLLM requires.
        """
        state = {name: tensor for name, tensor in weights if not name.startswith(REFERENCE_ENCODER_PREFIXES)}
        self.load_state_dict(state, strict=True)
        return set(state)
