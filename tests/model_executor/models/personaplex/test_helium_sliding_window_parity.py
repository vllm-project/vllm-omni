# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Helium past its 3000-frame context matches Moshi's streaming semantics (RFC #7389 W6).

The temporal transformer runs on vLLM's paged KV with a per-layer sliding
window; a duplex call past the window then reads a block table whose pages
behind the window were freed, and rotates queries and keys at positions past
3000. Moshi's streaming reference keeps a ring of ``context`` (3000) KV cells
whose oldest cell is labelled as the next write, so a query attends to the keys
with ``0 <= q - k < 2999`` (``test_helium_window.py`` pins this against the
repo's port of the ring), and applies RoPE at absolute positions. A small random
Helium (real head size, real context) runs through vLLM on the default compiled
and CUDA-graph path, and every token's logprob is compared with a dense fp32
reference of those semantics, before and after the window.
"""

from __future__ import annotations

import math

import pytest
import torch
from safetensors.torch import save_file
from transformers import LlamaConfig

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model]

CONTEXT = 3000
# Keys a query attends to once Moshi's ring has wrapped.
MOSHI_KEYS = CONTEXT - 1
HIDDEN, LAYERS, HEADS, HEAD_DIM, INTER, VOCAB = 512, 2, 4, 128, 1024, 1024
PROMPT, DECODE = 3500, 1500
ARCH = "HeliumSlidingWindowParityLM"


def _random_moshi_weights(seed: int = 0) -> dict[str, torch.Tensor]:
    """Moshi-layout checkpoint. Queries and keys are scaled up so attention is
    sharp enough for a wrong window or position to show in the logits."""
    g = torch.Generator().manual_seed(seed)

    def normal(*shape: int, std: float) -> torch.Tensor:
        return torch.randn(*shape, generator=g) * std

    weights = {"embed.weight": normal(VOCAB, HIDDEN, std=1.0)}
    for i in range(LAYERS):
        p = f"transformer.layers.{i}."
        q = normal(HEADS * HEAD_DIM, HIDDEN, std=2 / math.sqrt(HIDDEN))
        k = normal(HEADS * HEAD_DIM, HIDDEN, std=2 / math.sqrt(HIDDEN))
        v = normal(HEADS * HEAD_DIM, HIDDEN, std=1 / math.sqrt(HIDDEN))
        weights[p + "self_attn.in_proj_weight"] = torch.cat([q, k, v])
        weights[p + "self_attn.out_proj.weight"] = normal(HIDDEN, HEADS * HEAD_DIM, std=1 / math.sqrt(HEADS * HEAD_DIM))
        weights[p + "gating.linear_in.weight"] = normal(2 * INTER, HIDDEN, std=1 / math.sqrt(HIDDEN))
        weights[p + "gating.linear_out.weight"] = normal(HIDDEN, INTER, std=1 / math.sqrt(INTER))
        weights[p + "norm1.alpha"] = torch.ones(1, 1, HIDDEN)
        weights[p + "norm2.alpha"] = torch.ones(1, 1, HIDDEN)
    weights["out_norm.alpha"] = torch.ones(1, 1, HIDDEN)
    weights["text_linear.weight"] = normal(VOCAB, HIDDEN, std=3 / math.sqrt(HIDDEN))
    return {name: w.to(torch.bfloat16).contiguous() for name, w in weights.items()}


def _register_parity_model() -> None:
    from vllm import ModelRegistry
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding

    from vllm_omni.model_executor.models.personaplex.modeling_helium import HeliumModel
    from vllm_omni.model_executor.models.personaplex.personaplex_talker import (
        PersonaPlexTalkerForConditionalGeneration as Talker,
    )

    class HeliumSlidingWindowParityLM(torch.nn.Module):
        """The talker's Helium backbone with a token embedding and the text head,
        so vLLM can drive it from token ids. The Moshi-format weights load
        through the talker's own mapping."""

        _load_temporal = Talker._load_temporal
        _load_direct = staticmethod(Talker._load_direct)
        _load_shard = staticmethod(Talker._load_shard)

        def __init__(self, *, vllm_config, prefix: str = "") -> None:
            super().__init__()
            config = vllm_config.model_config.hf_config
            self.model = HeliumModel(vllm_config=vllm_config, prefix="model")
            self.embed = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, prefix="lm_head")
            self.logits_processor = LogitsProcessor(config.vocab_size)
            self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
            # Piecewise CUDA graphs replay from the addresses seen at capture,
            # so the backbone reads its embeddings from one persistent buffer.
            self.register_buffer(
                "_embeds",
                torch.zeros(vllm_config.scheduler_config.max_num_batched_tokens, config.hidden_size),
                persistent=False,
            )

        def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.embed(input_ids)

        def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **_):
            if inputs_embeds is None:
                inputs_embeds = self._embeds[: input_ids.shape[0]]
                inputs_embeds.copy_(self.embed(input_ids))
            return self.model(
                input_ids=None,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )

        def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
            return self.logits_processor(self.lm_head, hidden_states)

        def load_weights(self, weights) -> set[str]:
            params = dict(self.named_parameters(remove_duplicate=False))
            loaded: set[str] = set()
            for name, weight in weights:
                if name == "embed.weight":
                    self.embed.weight.data.copy_(weight)
                    loaded.add(name)
                elif name == "text_linear.weight":
                    self._load_direct("lm_head.weight", weight, params, loaded)
                else:
                    loaded |= self._load_temporal(name, weight, params)
            return loaded

    ModelRegistry.register_model(ARCH, HeliumSlidingWindowParityLM)


def _rms_norm(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-8) * alpha.reshape(-1)


def _moshi_rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Moshi ``apply_rope``: interleaved pairs, angles in fp32 at absolute positions."""
    ds = torch.arange(HEAD_DIM // 2, device=x.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(10000.0) * 2 / HEAD_DIM))
    angles = positions.float().view(-1, 1, 1) * freqs.view(1, 1, -1)
    cos, sin = torch.cos(angles), torch.sin(angles)
    real, imag = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = real * cos - imag * sin
    out[..., 1::2] = real * sin + imag * cos
    return out


@torch.no_grad()
def _reference_logprobs(
    weights: dict[str, torch.Tensor],
    token_ids: list[int],
    *,
    window: int | None = MOSHI_KEYS,
    rope_period: int | None = None,
) -> torch.Tensor:
    """Dense fp32 Helium with Moshi's streaming context: keys with ``0 <= q - k < window``.

    ``window=None`` attends to every earlier key, and ``rope_period`` wraps the
    rotary positions; both are wrong past 3000 and must be told apart."""
    w = {name: t.float().cuda() for name, t in weights.items()}
    ids = torch.tensor(token_ids, device="cuda")
    positions = torch.arange(ids.numel(), device="cuda")
    rope_positions = positions if rope_period is None else positions % rope_period
    delta = positions.view(-1, 1) - positions.view(1, -1)
    visible = delta >= 0
    if window is not None:
        visible &= delta < window
    x = w["embed.weight"][ids]
    for i in range(LAYERS):
        p = f"transformer.layers.{i}."
        q, k, v = (_rms_norm(x, w[p + "norm1.alpha"]) @ w[p + "self_attn.in_proj_weight"].T).split(HIDDEN, dim=-1)
        q, k, v = (t.view(-1, HEADS, HEAD_DIM) for t in (q, k, v))
        q, k = _moshi_rope(q, rope_positions), _moshi_rope(k, rope_positions)
        heads = []
        for h in range(HEADS):
            scores = (q[:, h] @ k[:, h].T) / math.sqrt(HEAD_DIM)
            heads.append(torch.softmax(scores.masked_fill(~visible, float("-inf")), dim=-1) @ v[:, h])
        x = x + torch.cat(heads, dim=-1) @ w[p + "self_attn.out_proj.weight"].T
        gate, up = (_rms_norm(x, w[p + "norm2.alpha"]) @ w[p + "gating.linear_in.weight"].T).chunk(2, dim=-1)
        x = x + (torch.nn.functional.silu(gate) * up) @ w[p + "gating.linear_out.weight"].T
    logits = _rms_norm(x, w["out_norm.alpha"]) @ w["text_linear.weight"].T
    return torch.log_softmax(logits, dim=-1).cpu()


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_helium_matches_moshi_past_the_sliding_window(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    # The fp32 reference must not use TF32; monkeypatch restores the global flag.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    from vllm_omni.model_executor.models.personaplex.configuration_helium import HeliumConfig

    _register_parity_model()
    dims = dict(
        hidden_size=HIDDEN,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        head_dim=HEAD_DIM,
        num_key_value_heads=HEADS,
        intermediate_size=INTER,
        vocab_size=VOCAB,
        max_position_embeddings=8192,
    )
    # config.json only has to load; the engine runs on the HeliumConfig below.
    LlamaConfig(architectures=[ARCH], **dims).save_pretrained(tmp_path)

    def helium_config(_loaded):
        config = HeliumConfig(sliding_window=CONTEXT, rms_norm_eps=1e-8, **dims)
        config.architectures = [ARCH]
        return config

    weights = _random_moshi_weights()
    save_file(weights, str(tmp_path / "model.safetensors"))
    prompt = torch.randint(0, VOCAB, (PROMPT,), generator=torch.Generator().manual_seed(1)).tolist()

    llm = LLM(
        model=str(tmp_path),
        hf_overrides=helium_config,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=6144,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.25,
        enable_prefix_caching=False,
        seed=0,
    )
    try:
        scheduler = llm.llm_engine.engine_core.engine_core.scheduler
        (group,) = scheduler.kv_cache_config.kv_cache_groups
        assert type(group.kv_cache_spec).__name__ == "SlidingWindowSpec"
        assert group.kv_cache_spec.sliding_window == MOSHI_KEYS
        (out,) = llm.generate(
            [{"prompt_token_ids": prompt}],
            SamplingParams(
                max_tokens=DECODE,
                temperature=0.0,
                ignore_eos=True,
                logprobs=0,
                prompt_logprobs=0,
                detokenize=False,
            ),
        )
    finally:
        # The engine runs in this process: release its distributed state and
        # memory so later tests in the same pytest process start clean.
        del llm
        cleanup_dist_env_and_memory()

    generated = list(out.outputs[0].token_ids)
    tokens = prompt + generated
    # vLLM's logprob of every token given its prefix: chunked prefill for the
    # prompt, then one decode step per generated token.
    vllm_logprobs = torch.tensor(
        [out.prompt_logprobs[t][prompt[t]].logprob for t in range(1, PROMPT)]
        + [out.outputs[0].logprobs[j][generated[j]].logprob for j in range(DECODE)]
    )
    targets = torch.tensor(tokens[1:])

    def errors(reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token |logprob error| and whether greedy decode picked the reference argmax."""
        error = (vllm_logprobs - reference[:-1].gather(1, targets.view(-1, 1)).squeeze(1)).abs()
        top2 = reference[:-1].topk(2, dim=-1).values
        clear = (top2[:, 0] - top2[:, 1]) > 0.1
        decode = torch.arange(len(targets)) >= PROMPT - 1
        agree = reference[:-1].argmax(-1) == targets
        return error, agree[decode & clear]

    error, agree = errors(_reference_logprobs(weights, tokens))
    # error[r] scores the query at position r; the ring first drops a key at r = MOSHI_KEYS.
    inside, past = error[:MOSHI_KEYS], error[MOSHI_KEYS:]
    # bf16 vLLM against the fp32 reference: as close past the window as inside it.
    assert inside.mean() < 0.05 and inside.max() < 0.3
    assert past.mean() < 0.05 and past.max() < 0.3
    assert past.mean() < 1.5 * inside.mean()
    assert agree.float().mean() > 0.99

    # The comparison can tell the failures W6 guards against apart: attending
    # past the window, or rotating at positions that wrap at the window.
    for wrong in (dict(window=None), dict(rope_period=CONTEXT)):
        wrong_error, wrong_agree = errors(_reference_logprobs(weights, tokens, **wrong))
        assert wrong_error[MOSHI_KEYS:].mean() > 0.3, wrong
        assert wrong_agree.float().mean() < 0.9, wrong
