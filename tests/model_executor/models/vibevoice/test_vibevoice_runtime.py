# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for VibeVoice per-step row/state handling (no weights, no GPU)."""

from types import SimpleNamespace

import pytest
import torch
from transformers.models.vibevoice_acoustic_tokenizer.modeling_vibevoice_acoustic_tokenizer import (
    VibeVoiceAcousticTokenizerCausalConv1d,
    VibeVoiceAcousticTokenizerConv1dPaddingCache,
)

from vllm_omni.model_executor.models.vibevoice.negative_kv import VibeVoiceNegativeKV
from vllm_omni.model_executor.models.vibevoice.stateful import VibeVoiceStatefulInference
from vllm_omni.model_executor.models.vibevoice.vibevoice import (
    VibeVoiceAudioTokenDecodeOutput,
    VibeVoiceModel,
    _split_padding_cache,
    _stack_padding_caches,
)
from vllm_omni.transformers_utils.configs.vibevoice import VibeVoiceConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

BOS, AUDIO, EOS_AUDIO = 1, 3, 2
HIDDEN, LATENT = 4, 2


@pytest.fixture
def inference(mocker):
    model = mocker.Mock(spec=VibeVoiceModel)
    model.config = VibeVoiceConfig(
        audio_config={"hidden_size": LATENT},
        text_config={"hidden_size": HIDDEN, "num_attention_heads": 1, "num_key_value_heads": 1},
    )
    model.sample_audio_latent.side_effect = lambda pos, neg, noise: torch.zeros(pos.shape[0], 1, LATENT)
    model.decode_audio_tokens.side_effect = lambda latents, acoustic, semantic: VibeVoiceAudioTokenDecodeOutput(
        audio=torch.full((len(acoustic), 1, 3200), 0.5),
        next_embeddings=torch.full((len(acoustic), 1, HIDDEN), 9.0),
        acoustic_caches=[VibeVoiceAcousticTokenizerConv1dPaddingCache() for _ in acoustic],
        semantic_caches=[VibeVoiceAcousticTokenizerConv1dPaddingCache() for _ in semantic],
    )
    negative_kv = mocker.Mock(spec=VibeVoiceNegativeKV)
    negative_kv.forward_step.side_effect = lambda ids, embeds: [torch.zeros(1, HIDDEN) for _ in ids]
    return VibeVoiceStatefulInference(
        model=model, negative_kv=negative_kv, audio_bos_token_id=BOS, audio_token_id=AUDIO
    )


def _step(inference, rows, sampled):
    """rows: (request_id, num_computed, num_scheduled, prompt_len[, seed]); returns (fed embeds, payload)."""
    inference.begin_step()
    fed = [
        inference.add_row(
            rid,
            num_computed=c,
            num_scheduled=n,
            prompt_len=p,
            input_embeds=torch.ones(n, HIDDEN),
            seed=seed[0] if seed else None,
        )
        for rid, c, n, p, *seed in rows
    ]
    inference.record_hidden(torch.ones(len(rows), HIDDEN))
    inference.record_sampled([[token] for token in sampled])
    return fed, inference.finish_step([row[0] for row in rows])


def test_chunked_prefill_one_token_tail_and_feedback(inference):
    # Non-final chunk: its sampled token is discarded even if it is audio_token.
    _, payload = _step(inference, [("a", 0, 4, 5)], [AUDIO])
    assert payload["model_outputs"] == []
    # One-token prompt tail is still prefill; its sample is valid.
    _, payload = _step(inference, [("a", 4, 1, 5)], [BOS])
    inference._negative_kv.reset_segment.assert_called_once_with("a")
    _, payload = _step(inference, [("a", 5, 1, 5)], [AUDIO])
    assert payload["meta"] == {"req_id": ["a"], "sparse_audio": ["1"]}
    (chunk,) = payload["model_outputs"]
    assert chunk.dtype == torch.float32 and chunk.shape == (3200,) and chunk.device.type == "cpu"
    # The decoded feedback replaces the next decode input.
    fed, _ = _step(inference, [("a", 6, 1, 5)], [EOS_AUDIO])
    assert torch.equal(fed[0], torch.full((1, HIDDEN), 9.0))


def test_audio_subset_shares_one_noise_draw(inference):
    for rid in ("a", "b", "c"):
        _step(inference, [(rid, 0, 2, 2)], [BOS])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        expected = torch.randn(2 * 2, LATENT)
        torch.manual_seed(7)
        _, payload = _step(inference, [("c", 2, 1, 2), ("a", 2, 1, 2), ("b", 2, 1, 2)], [AUDIO, EOS_AUDIO, AUDIO])
    assert payload["meta"]["req_id"] == ["c", "b"]
    inference._negative_kv.forward_step.assert_called_once()
    assert inference._negative_kv.forward_step.call_args.args[0] == ["c", "b"]
    torch.testing.assert_close(inference._model.sample_audio_latent.call_args.args[2], expected, atol=0, rtol=0)


def _noise(inference):
    return inference._model.sample_audio_latent.call_args.args[2]


def test_seeded_noise_is_private_to_the_request(inference):
    """Same seed -> same stream, whatever the batch mates, history or global RNG; globals untouched."""
    for rid, seed in ("a", 5), ("b", 5), ("c", 6), ("d", None):
        _step(inference, [(rid, 0, 2, 2, seed)], [BOS])
    torch.manual_seed(1)
    before = torch.get_rng_state()
    _step(inference, [("a", 2, 1, 2), ("c", 2, 1, 2)], [AUDIO, AUDIO])
    first = _noise(inference)
    _step(inference, [("b", 2, 1, 2)], [AUDIO])  # alone, reordered history, other request id
    assert torch.equal(_noise(inference)[:1], first[:1])  # a and b share seed 5 -> same first frame
    assert not torch.equal(first[0], first[1])  # seed 6 differs
    assert torch.equal(torch.get_rng_state(), before)  # seeded rows never consume the global stream
    _step(inference, [("a", 3, 1, 2), ("c", 3, 1, 2)], [AUDIO, AUDIO])
    assert not torch.equal(_noise(inference), first)  # the stream advances per frame
    with torch.random.fork_rng(devices=[]):  # replay: a request's frames are a pure function of its seed
        generator = torch.Generator().manual_seed(5)
        frames = [torch.randn(LATENT, generator=generator) for _ in range(2)]
    assert torch.equal(first[0], frames[0]) and torch.equal(_noise(inference)[0], frames[1])


def test_unseeded_rows_keep_the_shared_draw_beside_seeded_rows(inference):
    for rid, seed in (("s", 3), ("u", None), ("v", None)):
        _step(inference, [(rid, 0, 2, 2, seed)], [BOS])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9)
        expected = torch.randn(2 * 2, LATENT)
        torch.manual_seed(9)
        _step(inference, [("u", 2, 1, 2), ("s", 2, 1, 2), ("v", 2, 1, 2)], [AUDIO] * 3)
    noise = _noise(inference)
    assert noise.shape == (6, LATENT)
    torch.testing.assert_close(noise[[0, 2]], expected[:2], atol=0, rtol=0)  # first halves of u, v
    torch.testing.assert_close(noise[[3, 5]], expected[2:], atol=0, rtol=0)
    assert torch.equal(noise[1], noise[4])  # the seeded row repeats its draw in the unused half


class _NoisyTower(torch.nn.Module):
    """Stands in for the sampled VAE: latents = waveform + randn noise, drawn like the real encoder."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def encode(self, values, sample=True):
        return SimpleNamespace(latents=(values + torch.randn_like(values)).transpose(1, 2))  # (B, tokens, 1)


@pytest.fixture
def encoder():
    stub = SimpleNamespace(
        samples_per_token=2,
        audio_tower=_NoisyTower(),
        latent_bias_factor=0.0,
        latent_scaling_factor=1.0,
        multi_modal_projector=lambda x: x,
    )
    stub._encode_references = lambda rows, counts: VibeVoiceModel._encode_references(stub, rows, counts)
    return lambda *args: VibeVoiceModel.reference_embeddings(stub, *args)


def _refs(*lengths):
    rows = [torch.zeros(1, n) for n in lengths]
    return rows, [torch.tensor(n // 2) for n in lengths]


def _seed(*seeds):
    return [torch.tensor([s is not None, s or 0]) for s in seeds]


def test_seeded_reference_noise_depends_only_on_audio_and_seed(encoder):
    rows, counts = _refs(8, 4, 6)
    torch.manual_seed(1)
    before = torch.get_rng_state()
    together = encoder(rows, counts, _seed(5, 5, 6))
    assert torch.equal(torch.get_rng_state(), before)  # the global stream is untouched
    torch.manual_seed(99)
    alone = encoder(rows[1:2], counts[1:2], _seed(5))  # different batch mates, padding and global RNG
    assert torch.equal(together[1], alone[0])
    other = encoder(rows[:1], counts[:1], _seed(6))
    assert not torch.equal(together[0], other[0])
    assert torch.equal(together[0], encoder(rows[:1], counts[:1], _seed(5))[0])


def test_unseeded_reference_keeps_batched_global_draw(encoder):
    rows, counts = _refs(8, 4)
    for seeds in (None, _seed(None, None)):
        torch.manual_seed(3)
        reference = encoder(rows, counts, seeds)
        torch.manual_seed(3)
        padded = torch.stack([torch.nn.functional.pad(r, (0, 8 - r.shape[-1])) for r in rows])
        expected = (padded + torch.randn_like(padded)).transpose(1, 2)
        assert torch.equal(reference[0], expected[0, :4]) and torch.equal(reference[1], expected[1, :2])
    # A seeded item neither changes nor consumes the unseeded item's global draw.
    torch.manual_seed(3)
    mixed = encoder(rows, counts, _seed(None, 4))
    torch.manual_seed(3)
    assert torch.equal(mixed[0], (rows[0][None] + torch.randn_like(rows[0][None])).transpose(1, 2)[0, :4])
    with pytest.raises(ValueError, match="aligned"):
        encoder(rows, counts, _seed(1))


def test_release_and_unsupported_recompute(inference):
    _step(inference, [("a", 0, 2, 2)], [BOS])
    with pytest.raises(RuntimeError, match="preemption"):
        _step(inference, [("a", 0, 2, 2)], [BOS])
    inference.release({"a"})
    inference._negative_kv.release.assert_called_with("a")
    # A released id can start again from scratch.
    _step(inference, [("a", 0, 2, 2)], [BOS])


def test_decode_caches_follow_request_order_and_release(inference):
    _step(inference, [("a", 0, 2, 2), ("b", 0, 2, 2)], [AUDIO, AUDIO])
    states = inference._states
    acoustic = [states[rid].acoustic_cache for rid in ("a", "b")]
    semantic = [states[rid].semantic_cache for rid in ("a", "b")]
    assert all(isinstance(cache, VibeVoiceAcousticTokenizerConv1dPaddingCache) for cache in acoustic + semantic)
    assert len({id(cache) for cache in acoustic + semantic}) == 4

    # Reordered rows must feed each request's own previous decode caches.
    _step(inference, [("b", 2, 1, 2), ("a", 2, 1, 2)], [AUDIO, AUDIO])
    _, fed_acoustic, fed_semantic = inference._model.decode_audio_tokens.call_args.args
    assert all(actual is expected for actual, expected in zip(fed_acoustic, reversed(acoustic), strict=True))
    assert all(actual is expected for actual, expected in zip(fed_semantic, reversed(semantic), strict=True))

    continuing_acoustic = states["b"].acoustic_cache
    continuing_semantic = states["b"].semantic_cache
    inference.release({"a"})
    _step(inference, [("a", 0, 2, 2), ("b", 3, 1, 2)], [AUDIO, AUDIO])
    _, fed_acoustic, fed_semantic = inference._model.decode_audio_tokens.call_args.args
    assert fed_acoustic[0] is None and fed_semantic[0] is None
    assert fed_acoustic[1] is continuing_acoustic and fed_semantic[1] is continuing_semantic


def test_stacked_conv_caches_match_per_request_streaming():
    # Batched decode stacks per-request causal-conv caches; a joining request starts from zeros.
    torch.manual_seed(0)
    conv = VibeVoiceAcousticTokenizerCausalConv1d(2, 2, kernel_size=3, cache_key="c")
    chunks = torch.randn(3, 2, 1, 2, 1)  # step, request, (1, channels, time)
    single = [VibeVoiceAcousticTokenizerConv1dPaddingCache() for _ in range(2)]
    batched = [None, None]
    with torch.inference_mode():
        for step, rows in enumerate(([0], [0, 1], [0, 1])):
            expected = torch.cat([conv(chunks[step, r], padding_cache=single[r]) for r in rows])
            cache = _stack_padding_caches([batched[r] for r in rows])
            if cache is None:
                cache = VibeVoiceAcousticTokenizerConv1dPaddingCache()
            actual = conv(torch.cat([chunks[step, r] for r in rows]), padding_cache=cache)
            for r, row_cache in zip(rows, _split_padding_cache(cache, len(rows)), strict=True):
                batched[r] = row_cache
            torch.testing.assert_close(actual, expected)
