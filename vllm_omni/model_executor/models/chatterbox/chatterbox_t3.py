# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage 0 of Chatterbox Turbo: T3 on vLLM's GPT-2 backbone.

vLLM runs the backbone (``tfmr.*`` in the checkpoint) with paged attention and
continuous batching. Everything T3 adds around it is here: the text and speech
embeddings, the speech head, the speaker projection, the prompt layout
``[speaker | prompt tokens | text | start-of-speech]`` and the start-token
mask on the speech logits vLLM samples from.

Mirrors ``chatterbox/models/t3/t3.py`` (0.1.7): ``T3.__init__`` for the
modules, ``prepare_input_embeds`` and ``inference_turbo`` for the layout.
"""

from collections.abc import Iterable

import torch
from torch import nn
from transformers import GPT2Config
from vllm.config import VllmConfig
from vllm.model_executor.models.gpt2 import GPT2Model
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

# Checkpoint key -> parameter name on T3Heads.
HEAD_KEYS = {
    "text_emb.weight": "text_emb.weight",
    "speech_emb.weight": "speech_emb.weight",
    "speech_head.weight": "speech_head.weight",
    "speech_head.bias": "speech_head.bias",
    "cond_enc.spkr_enc.weight": "spkr_enc.weight",
    "cond_enc.spkr_enc.bias": "spkr_enc.bias",
}
BACKBONE_PREFIX = "tfmr."
# Training-only. Upstream also deletes tfmr.wte after loading, but vLLM's
# GPT2Model owns a wte parameter and its loader fails on any parameter the
# checkpoint did not fill, so wte is loaded and left idle.
UNUSED_KEYS = frozenset({"text_head.weight"})


class T3Heads(nn.Module):
    """The four modules T3 wraps around its backbone."""

    def __init__(self, config: ChatterboxConfig) -> None:
        super().__init__()
        self.text_emb = nn.Embedding(config.text_vocab_size, config.hidden_size)
        self.speech_emb = nn.Embedding(config.speech_vocab_size, config.hidden_size)
        self.speech_head = nn.Linear(config.hidden_size, config.speech_vocab_size, bias=True)
        self.spkr_enc = nn.Linear(config.speaker_embed_size, config.hidden_size)


def prefill_embeds(
    heads: T3Heads,
    text_ids: torch.Tensor,
    cond_tokens: torch.Tensor,
    speaker_emb: torch.Tensor,
    start_speech_token: int,
) -> torch.Tensor:
    """Embed one request's prompt the way ``T3.prepare_input_embeds`` does.

    Turbo has no perceiver and none of T3's separate text and speech position
    embeddings (the GPT-2 backbone applies its own ``wpe``), so the
    conditioning is the projected speaker embedding followed by the prompt
    tokens through ``speech_emb``; the text follows, then the start-of-speech
    token ``inference_turbo`` appends.

    Args:
        heads: The loaded heads.
        text_ids: Shape (T,), GPT-2 ids of the normalized text.
        cond_tokens: Shape (C,), S3 tokens of the reference clip.
        speaker_emb: Shape (1, 256), the voice encoder's embedding.
        start_speech_token: The start-of-speech id.

    Returns:
        Shape (1 + C + T + 1, H), in the heads' dtype.
    """
    start = heads.speech_emb.weight[start_speech_token : start_speech_token + 1]
    return torch.cat(
        [heads.spkr_enc(speaker_emb), heads.speech_emb(cond_tokens), heads.text_emb(text_ids), start], dim=0
    )


def prefill_slice(
    heads: T3Heads,
    config: ChatterboxConfig,
    text_ids: torch.Tensor,
    cond_tokens: torch.Tensor,
    speaker_emb: torch.Tensor,
    num_computed: int,
    span: int,
) -> torch.Tensor:
    """The rows of the prompt embedding a prefill step was scheduled.

    Slicing by computed progress is what the preprocess phase contract
    prescribes; it is correct however the scheduler splits the prompt,
    including a one-token tail. A span that runs past the end of the prompt
    (a preempted request recomputed from zero, whose span also covers the
    tokens it had generated) gets only the prompt's rows; the caller embeds
    the rest.

    Args:
        heads: The loaded heads.
        config: The model config.
        text_ids: Shape (T,), GPT-2 ids of the normalized text.
        cond_tokens: Shape (C,), S3 tokens of the reference clip.
        speaker_emb: Shape (1, 256), the voice encoder's embedding.
        num_computed: Tokens computed before this step, ``_omni_num_computed_tokens``.
        span: Tokens scheduled this step.

    Returns:
        Shape (n, H) with ``n <= span``: ``span`` rows, or the remaining
        prompt rows when the span runs past the prompt.
    """
    embeds = prefill_embeds(heads, text_ids, cond_tokens, speaker_emb, config.start_speech_token)
    return embeds[num_computed : num_computed + span]


def speech_logits(heads: T3Heads, hidden: torch.Tensor, config: ChatterboxConfig) -> torch.Tensor:
    """Map hidden states to the speech logits vLLM samples from.

    The start token is ``-inf``: upstream never masks it but never wants it
    sampled either, and every placeholder prompt id relies on it being
    unreachable.

    Args:
        heads: The loaded heads.
        hidden: Shape (N, H).
        config: The model config.

    Returns:
        Shape (N, ``config.speech_vocab_size``).
    """
    logits = heads.speech_head(hidden)
    logits[:, config.start_speech_token] = float("-inf")
    return logits


def split_t3_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> tuple[list[tuple[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """Route the T3 checkpoint's keys to the backbone and the heads.

    Args:
        weights: ``(name, tensor)`` pairs as vLLM's loader yields them.

    Returns:
        The backbone weights with the ``tfmr.`` prefix stripped, for
        ``GPT2Model.load_weights``, and a state dict for ``T3Heads``.

    Raises:
        KeyError: On a key that belongs to neither, so a checkpoint with a
            different layout fails to load instead of loading partially.
    """
    backbone: list[tuple[str, torch.Tensor]] = []
    heads: dict[str, torch.Tensor] = {}
    for name, tensor in weights:
        if name.startswith(BACKBONE_PREFIX):
            backbone.append((name.removeprefix(BACKBONE_PREFIX), tensor))
        elif name in HEAD_KEYS:
            heads[HEAD_KEYS[name]] = tensor
        elif name not in UNUSED_KEYS:
            raise KeyError(f"unexpected key {name!r} in the T3 checkpoint")
    return backbone, heads


class ChatterboxT3ForConditionalGeneration(nn.Module, SupportsPP):
    """T3: text and reference conditioning in, S3 speech tokens out."""

    # The runner replaces the placeholder prompt's embeddings through preprocess.
    has_preprocess = True
    # Has the runner call make_omni_output and keep what it returns.
    have_multimodal_outputs = True
    # Stage 1 reads the sampled ids, never the hidden states. Read by the
    # default runner only; Model Runner V2 copies no hidden state when
    # streaming and always copies it otherwise.
    omni_pooler_payload_include_hidden = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError(
                "Chatterbox T3 requires enable_prefix_caching=False: its placeholder prompts would share "
                "cache entries across voices and texts"
            )
        config: ChatterboxConfig = vllm_config.model_config.hf_config
        self.config = config
        backbone = GPT2Config(
            vocab_size=config.text_vocab_size,
            n_positions=config.max_position_embeddings,
            n_embd=config.hidden_size,
            n_layer=config.num_hidden_layers,
            n_head=config.num_attention_heads,
            activation_function=config.activation_function,
            layer_norm_epsilon=config.layer_norm_epsilon,
        )
        self.tfmr = GPT2Model(
            vllm_config=vllm_config.with_hf_config(backbone, architectures=["ChatterboxT3ForConditionalGeneration"]),
            prefix=maybe_prefix(prefix, "tfmr"),
        )
        self.heads = T3Heads(config)
        self.make_empty_intermediate_tensors = self.tfmr.make_empty_intermediate_tensors
        # The repo also holds the 10-step S3Gen weights; vLLM's loader would
        # otherwise read every safetensors file in the snapshot.
        self.allow_patterns_overrides = [config.t3_weights]

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed sampled speech ids. Part of vLLM's text-generation contract."""
        return self.heads.speech_emb(input_ids)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        *,
        _omni_is_prefill: bool,
        _omni_prompt_len: int,
        _omni_num_computed_tokens: int,
        ids: dict[str, list[int]],
        embed: dict[str, torch.Tensor],
        **runner_metadata: object,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Embed one request's scheduled tokens.

        A prefill span is replaced by its rows of the real prompt embedding.
        When vLLM recomputes a preempted request the span also covers the
        speech ids already generated; those are embedded as a decode step
        would. A decode span is one sampled speech id.

        Args:
            input_ids: The scheduled ids for this request.
            input_embeds: The runner's embeddings for the span, unused.
            _omni_is_prefill: Whether the span is part of the prompt.
            _omni_prompt_len: The request's prompt length.
            _omni_num_computed_tokens: Tokens computed before this step.
            ids: ``additional_information["ids"]``: ``prompt`` and ``speech_token``.
            embed: ``additional_information["embed"]``: ``voice`` and the stage 1 reference.
            **runner_metadata: Other per-request fields the runner passes.

        Returns:
            The ids, their embeddings, and no payload update.

        Raises:
            TypeError: If the request carries no ``ids`` or ``embed``, that
                is, its prompt was not built with ``conditioning.build_prompt``.
        """
        if not _omni_is_prefill:
            return input_ids, self.embed_input_ids(input_ids), {}
        device, dtype = input_ids.device, self.heads.text_emb.weight.dtype
        in_prompt = min(input_ids.shape[0], _omni_prompt_len - _omni_num_computed_tokens)
        prompt_rows = prefill_slice(
            self.heads,
            self.config,
            torch.tensor(ids["prompt"], dtype=torch.long, device=device),
            torch.tensor(ids["speech_token"], dtype=torch.long, device=device),
            embed["voice"].to(device=device, dtype=dtype),
            _omni_num_computed_tokens,
            in_prompt,
        )
        embeds = torch.cat([prompt_rows, self.embed_input_ids(input_ids[in_prompt:])])
        return input_ids, embeds, {}

    def preprocess_decode_batch(
        self, *, input_ids: torch.Tensor, req_infos: list[dict]
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
        """Embed a step's decode rows in one lookup.

        The default runner calls this in place of ``preprocess`` for the
        rows that are one decode token each.

        Args:
            input_ids: Shape (N,), one sampled speech id per request.
            req_infos: Per request, the fields ``preprocess`` is given.

        Returns:
            The ids, their embeddings of shape (N, H), and no payload update
            for any request.
        """
        return input_ids, self.embed_input_ids(input_ids), [{} for _ in req_infos]

    def preprocess_decode_batch_mrv2(
        self, *, input_ids: torch.Tensor, input_embeds: torch.Tensor, req_infos: list[dict]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
        """A step's decode rows on Model Runner V2, which has already embedded them.

        V2 fills ``input_embeds`` with ``embed_input_ids`` of every row
        before any hook, and that is all a decode row needs. It also unpacks
        five values from this hook: the two in the middle feed a
        residual-codebook predictor, which this model does not have, so they
        are empty.

        Args:
            input_ids: Shape (N,), one sampled speech id per request.
            input_embeds: Shape (N, H), their embeddings.
            req_infos: Per request, the fields ``preprocess`` is given.

        Returns:
            The ids and embeddings as given, two tensors of shape (N, 0), and
            no payload update for any request.
        """
        empty = input_embeds.new_empty((len(req_infos), 0))
        return input_ids, input_embeds, empty, empty, [{} for _ in req_infos]

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Speech logits with the start token masked."""
        return speech_logits(self.heads, hidden_states, self.config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **runner_kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run the backbone.

        Returns:
            The hidden states, shape (N, H). A tensor and nothing wrapped
            around it: Model Runner V2 captures this call in a CUDA graph and
            sizes the graph's output from it.
        """
        return self.tfmr(input_ids, positions, intermediate_tensors, inputs_embeds)

    def make_omni_output(self, model_outputs: torch.Tensor, **runner_kwargs: object) -> OmniOutput:
        """Wrap a step's hidden states for the runner. Emits no audio and no tokens.

        Stage 1 gets the sampled ids, which the chunk processor reads from
        the request itself. The one entry here exists because the
        runner-side transport calls the chunk processor for a request only
        on a step that has a payload for it, and the processor must see
        every sampled token. It is one flag per token row, which the runner
        slices to each request's rows, and every flag is false: the payload
        holds no codec frame.

        Args:
            model_outputs: ``forward``'s hidden states, shape (N, H).
            **runner_kwargs: Whatever else the runner passes.

        Returns:
            The hidden states and the per-row flags.
        """
        return OmniOutput(
            text_hidden_states=model_outputs,
            multimodal_outputs={
                "meta": {"codec_frame_valid": torch.zeros(model_outputs.shape[0], dtype=torch.bool, device="cpu")}
            },
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load ``t3_turbo_v1.safetensors``.

        Returns:
            The names of every parameter filled, as vLLM requires.
        """
        backbone, heads = split_t3_weights(weights)
        loaded = self.tfmr.load_weights(backbone)
        self.heads.load_state_dict(heads, strict=True)
        return {f"tfmr.{name}" for name in loaded} | {f"heads.{name}" for name in heads}
