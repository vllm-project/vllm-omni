# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Check captured cache ownership and padded text masking against eager models."""

import pytest
import torch
from transformers import LlamaConfig, T5Gemma2TextConfig
from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2TextEncoder
from vllm.platforms import current_platform

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.breeze_tts_2.depth_decoder import BreezeDepthDecoder, sample_logits
from vllm_omni.model_executor.models.breeze_tts_2.modeling_breeze import BreezeForConditionalGeneration
from vllm_omni.model_executor.models.breeze_tts_2.text_encoder_graph import (
    BreezeTextEncoderCompiled,
    BreezeTextEncoderGraph,
)

pytestmark = [pytest.mark.core_model]
CAPTURE_RNG_SUPPORTED = current_platform.is_cuda() and hasattr(torch.cuda.CUDAGraph, "register_generator_state")


@pytest.fixture
def full_precision_matmul():
    # Different padded shapes can select different TF32 kernels. Compare
    # masking and replay semantics independently of that reduced precision.
    original_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(original_precision)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.inference_mode()
def test_text_graph_masks_padding_and_local_attention_across_replays(full_precision_matmul) -> None:
    torch.manual_seed(42)
    config = T5Gemma2TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention"],
        dropout_rate=0.0,
    )
    config._attn_implementation = "sdpa"
    encoder = T5Gemma2TextEncoder(config).to("cuda").eval()
    encoder.embed_tokens.eoi_token_index = 17
    encoder.embed_tokens.eoi_embedding.fill_(0.4)
    projection = torch.nn.Linear(32, 48, bias=False).to("cuda").eval()
    graph = BreezeTextEncoderGraph(encoder, projection, 64)
    for length in (31, 51, 17, 31):
        prompt = torch.randint(0, 64, (1, length), device="cuda")
        prompt[:, 3] = 17
        expected = projection(encoder(input_ids=prompt).last_hidden_state)[0]
        actual = graph.run(prompt)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    batched = BreezeTextEncoderGraph(encoder, projection, 64, batch_size=3)
    prompts = [torch.randint(0, 64, (length,), device="cuda") for length in (31, 51, 17)]
    outputs = batched.run_batch(prompts)
    for prompt, actual in zip(prompts, outputs, strict=True):
        expected = projection(encoder(input_ids=prompt[None]).last_hidden_state)[0]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    compiled = BreezeTextEncoderCompiled(encoder, projection)
    long_prompts = [torch.randint(0, 64, (length,)) for length in (211, 173, 129)]
    for batch in ([long_prompts[0]], long_prompts, [long_prompts[1]]):
        outputs = compiled.run_batch(batch)
        for prompt, actual in zip(batch, outputs, strict=True):
            expected = projection(encoder(input_ids=prompt[None].to("cuda")).last_hidden_state)[0]
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def _make_depth_decoder(vocab_size: int = 35) -> BreezeDepthDecoder:
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=vocab_size,
        max_position_embeddings=16,
        rope_theta=500000.0,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 32.0,
            "low_freq_factor": 0.001953125,
            "high_freq_factor": 0.0078125,
            "original_max_position_embeddings": 16,
        },
    )
    config.num_codebooks = 16
    config.audio_embed_size = 48
    depth = BreezeDepthDecoder(config).to("cuda").eval()
    torch.nn.init.normal_(depth.codebooks_head, std=0.1)
    return depth


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.inference_mode()
def test_depth_graph_survives_interleaved_batches_and_workspace_allocations(full_precision_matmul, monkeypatch) -> None:
    torch.manual_seed(42)
    depth = _make_depth_decoder()
    hidden = torch.randn(2, 48, device="cuda")
    first = torch.tensor([7, 11], device="cuda")
    generators = [torch.Generator(device="cuda").manual_seed(seed) for seed in (42, 99)]
    expected = torch.cat(
        [
            depth.generate_frame(hidden[i : i + 1], first[i : i + 1], temperature=0, top_k=0, top_p=1, generator=g)
            for i, g in enumerate(generators)
        ]
    )
    for batch in (1, 2, 1, 2):
        # Unowned allocations made before graph capture would be recycled here.
        pressure = [torch.full((batch, 2, depth.num_codebooks, 8), float("nan"), device="cuda") for _ in range(32)]
        actual = depth.generate_frames(
            hidden[:batch], first[:batch], temperature=0, top_k=0, top_p=1, generators=generators[:batch]
        )
        torch.testing.assert_close(actual, expected[:batch], atol=0, rtol=0)
        del pressure

    def sample(batch: int) -> torch.Tensor:
        return depth.generate_frames(
            hidden[:batch],
            first[:batch],
            temperature=0.9,
            top_k=10,
            top_p=0.8,
            generators=[torch.Generator(device="cuda").manual_seed(seed) for seed in (42, 99)[:batch]],
        )

    sampled = sample(1)
    sampled_graph = depth._graphs[(1, 1, False)]
    greedy_graph = depth._graphs[(1, 1, True)]
    assert sampled_graph is not greedy_graph
    sample(2)
    torch.testing.assert_close(sample(1), sampled, atol=0, rtol=0)
    for temperature, top_k, top_p in ((0.0, 0, 1.0), (0.7, 20, 0.95), (0.0, 1, 0.2), (1.2, 0, 0.8)):
        rng_before = generators[0].get_state()
        actual = depth.generate_frames(
            hidden[:1], first[:1], temperature=temperature, top_k=top_k, top_p=top_p, generators=generators[:1]
        )
        if temperature == 0:
            assert depth._graphs[(1, 1, True)] is greedy_graph
            torch.testing.assert_close(actual, expected[:1], atol=0, rtol=0)
            torch.testing.assert_close(generators[0].get_state(), rng_before, atol=0, rtol=0)
        else:
            assert depth._graphs[(1, 1, False)] is sampled_graph
    torch.testing.assert_close(sample(1), sampled, atol=0, rtol=0)

    graph = None
    for scale in (4.0, 0.5, 4.0):
        expected_cfg = depth._generate_frame(
            hidden, first[:1], depth._allocate_cache(hidden), 0, 0, 1, generators[0], guidance_scale=scale
        )
        actual_cfg = depth.generate_frames(
            hidden, first[:1], temperature=0, top_k=0, top_p=1, generators=generators[:1], guidance_scale=scale
        )
        torch.testing.assert_close(actual_cfg, expected_cfg, atol=0, rtol=0)
        if graph is not None:
            assert depth._graphs[(2, 2, True)] is graph
        graph = depth._graphs[(2, 2, True)]
        sample(2)

    # Use a still-cold sampled CFG entry to exercise platforms without
    # captured generator support, while reusing the existing compiled model.
    fallback_key = (2, 2, False)
    assert fallback_key not in depth._graphs
    reference_generator = torch.Generator(device="cuda").set_state(generators[0].get_state())
    inactive_rng = generators[1].get_state()
    exponential = torch.Tensor.exponential_
    observed_generators = []

    def record_eager_noise(tensor, *args, **kwargs):
        observed_generators.append(kwargs.get("generator"))
        return exponential(tensor, *args, **kwargs)

    fallback_graph = None
    retained = []
    with monkeypatch.context() as platform_guard:
        platform_guard.setattr(current_platform, "is_cuda", lambda: False)
        for scale in (4.0, 0.5, 4.0):
            expected = depth._generate_frame(
                hidden,
                first[:1],
                depth._allocate_cache(hidden),
                0.9,
                10,
                0.8,
                reference_generator,
                guidance_scale=scale,
            )
            observed_generators.clear()
            with monkeypatch.context() as noise_spy:
                noise_spy.setattr(torch.Tensor, "exponential_", record_eager_noise)
                actual = depth.generate_frames(
                    hidden,
                    first[:1],
                    temperature=0.9,
                    top_k=10,
                    top_p=0.8,
                    generators=generators[:1],
                    guidance_scale=scale,
                )
            entry = depth._graphs[fallback_key]
            assert entry.noise_generators is None
            assert len(observed_generators) == depth.num_codebooks - 1
            assert all(generator is generators[0] for generator in observed_generators)
            if fallback_graph is not None:
                assert entry is fallback_graph
            fallback_graph = entry
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(generators[0].get_state(), reference_generator.get_state(), atol=0, rtol=0)
            torch.testing.assert_close(generators[1].get_state(), inactive_rng, atol=0, rtol=0)
            retained.append((actual, expected.clone()))
    for actual, expected in retained:
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@torch.inference_mode()
def test_depth_graph_buckets_bound_shapes_and_preserve_requests(full_precision_matmul, monkeypatch) -> None:
    def reject_eager_noise(*args, **kwargs):
        raise AssertionError("Sampled depth replay must not issue eager RNG draws")

    exponential = torch.Tensor.exponential_
    observed_generators = []

    def record_eager_noise(tensor, *args, **kwargs):
        observed_generators.append(kwargs.get("generator"))
        return exponential(tensor, *args, **kwargs)

    torch.manual_seed(17)
    # Match the checkpoint's multinomial draw size, including reserved IDs.
    depth = _make_depth_decoder(vocab_size=2051)
    hidden = torch.randn(32, 48, device="cuda")
    uncond = torch.randn_like(hidden)
    first = torch.arange(32, device="cuda")
    first_logits = torch.randn(32, depth.vocab_size, device="cuda")
    first_logits[:, -3:] = -torch.inf
    seeds = [42 + index for index in range(32)]
    seeds[0], seeds[-1] = 2**63 + 17, 2**64 - 1
    generators = [torch.Generator(device="cuda").manual_seed(seed) for seed in seeds]
    reference_generators = [torch.Generator(device="cuda").manual_seed(seed) for seed in seeds]

    expected_greedy = torch.cat(
        [
            depth.generate_frame(
                hidden[index : index + 1],
                first[index : index + 1],
                temperature=0,
                top_k=0,
                top_p=1,
                generator=reference_generators[index],
            )
            for index in range(32)
        ]
    )
    retained: list[tuple[torch.Tensor, torch.Tensor]] = []
    for batch in (*range(1, 33), 17, 11, 3, 1):
        rows = list(reversed(range(batch)))
        actual = depth.generate_frames(
            hidden[rows], first[rows], temperature=0, top_k=0, top_p=1, generators=[generators[row] for row in rows]
        )
        expected = expected_greedy[rows]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        retained.append((actual, expected.clone()))
    assert set(depth._graphs) == {(batch, 1, True) for batch in (1, 2, 4, 8, 16, 32)}
    for generator, reference in zip(generators, reference_generators, strict=True):
        torch.testing.assert_close(generator.get_state(), reference.get_state(), atol=0, rtol=0)

    retired_generators: list[tuple[torch.Generator, torch.Tensor]] = []
    for temperature, guidance_scale in ((0.9, 1.0), (0.0, 4.0), (0.9, 4.0)):
        batches: tuple[int, ...] = (3, 5, 11, 7, 3, 1)
        if guidance_scale == 1:
            batches = (16, 17, 31, 32, 17, 31, 16, *batches)
        for step, batch in enumerate(batches):
            if guidance_scale == 1 and step == 4:
                # New requests reuse rows in the same 32-request graph. Their
                # seeds must replace the previous owners, not just offsets.
                for row, seed in ((0, 2**64 - 1), (16, 2**63 + 321)):
                    retired_generators.append((generators[row], generators[row].get_state()))
                    generators[row] = torch.Generator(device="cuda").manual_seed(seed)
                    reference_generators[row] = torch.Generator(device="cuda").manual_seed(seed)
            rows = list(reversed(range(batch)))
            batch_hidden = hidden[rows]
            if guidance_scale != 1:
                batch_hidden = torch.cat((batch_hidden, uncond[rows]))
            expected, first_codes = [], []
            for row in rows:
                # The talker and depth decoder share each request's generator.
                # Exercise a codebook-0 sample immediately before graph replay.
                row_first = sample_logits(first_logits[row : row + 1], temperature, 10, 0.8, generators[row])
                reference_first = sample_logits(
                    first_logits[row : row + 1], temperature, 10, 0.8, reference_generators[row]
                )
                torch.testing.assert_close(row_first, reference_first, atol=0, rtol=0)
                first_codes.append(row_first)
                row_hidden = hidden[row : row + 1]
                if guidance_scale != 1:
                    row_hidden = torch.cat((row_hidden, uncond[row : row + 1]))
                expected.append(
                    depth._generate_frame(
                        row_hidden,
                        reference_first,
                        depth._allocate_cache(row_hidden),
                        temperature,
                        10,
                        0.8,
                        reference_generators[row],
                        guidance_scale=guidance_scale if guidance_scale != 1 else None,
                    )
                )
            request_bucket = 1 << (batch - 1).bit_length()
            branches = 2 if guidance_scale != 1 else 1
            graph_key = (request_bucket * branches, branches, temperature == 0)
            observed_generators.clear()
            with monkeypatch.context() as rng_guard:
                if CAPTURE_RNG_SUPPORTED and temperature > 0 and graph_key in depth._graphs:
                    # A warmed replay must generate noise within the CUDA
                    # graph, without dispatching new RNG kernels from Python.
                    rng_guard.setattr(torch.Tensor, "exponential_", reject_eager_noise)
                elif not CAPTURE_RNG_SUPPORTED:
                    rng_guard.setattr(torch.Tensor, "exponential_", record_eager_noise)
                actual = depth.generate_frames(
                    batch_hidden,
                    torch.cat(first_codes),
                    temperature=temperature,
                    top_k=10,
                    top_p=0.8,
                    generators=[generators[row] for row in rows],
                    guidance_scale=guidance_scale,
                )
            if not CAPTURE_RNG_SUPPORTED:
                # ROCm fills noise eagerly for live requests only, in the same
                # per-codebook order as independent scalar sampling.
                expected_draws = (
                    [generators[row] for row in rows for _ in range(depth.num_codebooks - 1)] if temperature > 0 else []
                )
                assert observed_generators == expected_draws
            expected_frames = torch.cat(expected)
            torch.testing.assert_close(actual, expected_frames, atol=0, rtol=0)
            retained.append((actual, expected_frames.clone()))
            # Check the next codebook-0 sample as well as the depth result: a
            # replay must return the advanced RNG stream to its live request.
            for row in rows:
                actual_next = sample_logits(first_logits[row : row + 1], temperature, 10, 0.8, generators[row])
                expected_next = sample_logits(
                    first_logits[row : row + 1], temperature, 10, 0.8, reference_generators[row]
                )
                torch.testing.assert_close(actual_next, expected_next, atol=0, rtol=0)
            # Padding must not draw RNG, and inactive request streams must not advance.
            for generator, reference in zip(generators, reference_generators, strict=True):
                torch.testing.assert_close(generator.get_state(), reference.get_state(), atol=0, rtol=0)
            for generator, state in retired_generators:
                torch.testing.assert_close(generator.get_state(), state, atol=0, rtol=0)

    expected_keys = {(batch, 1, True) for batch in (1, 2, 4, 8, 16, 32)}
    expected_keys.update((batch, 1, False) for batch in (1, 4, 8, 16, 32))
    expected_keys.update((batch, 2, greedy) for batch in (2, 8, 16, 32) for greedy in (False, True))
    assert set(depth._graphs) == expected_keys
    for (batch_bucket, branches, greedy), entry in depth._graphs.items():
        if greedy or not CAPTURE_RNG_SUPPORTED:
            assert entry.noise_generators is None
        else:
            assert entry.noise_generators is not None
            assert len(entry.noise_generators) == batch_bucket // branches
            assert {id(generator) for generator in entry.noise_generators}.isdisjoint(
                id(generator) for generator in generators
            )
    for actual, expected in retained:
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.inference_mode()
def test_batched_decode_embedding_matches_scalar_bfloat16() -> None:
    # Use the checkpoint's RVQ table dimensions without loading its backbone.
    model = BreezeForConditionalGeneration.__new__(BreezeForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.num_codebooks, model.hidden_size = 16, 2048
    model.register_buffer("offsets", torch.arange(16, device="cuda") * 2051)
    model.depth_decoder = torch.nn.Module()
    torch.manual_seed(17)
    model.depth_decoder.embed_tokens = torch.nn.Embedding(16 * 2051, 2048, device="cuda", dtype=torch.bfloat16)
    for seed in (17, 42, 99):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        frames = torch.randint(0, 2051, (32, 16), device="cuda", generator=generator)
        frames[1] = frames[0]
        for batch in (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 7, 1):
            # Reverse physical rows and revisit smaller batches after expansion.
            current_frames = [row[None] for row in frames[:batch].flip(0)]
            if batch > 1:
                # CFG branches share their generated audio frame.
                current_frames[-2] = current_frames[-1]
            infos = [
                {
                    "_omni_is_prefill": False,
                    "breeze_state": {"current": current},
                    "global_request_id": [str(index)],
                    "breeze_prompt": {},
                    "breeze_sampling": {},
                }
                for index, current in enumerate(current_frames)
            ]
            ids = torch.zeros(batch, device="cuda", dtype=torch.long)
            expected = torch.cat(
                [
                    model.preprocess(input_ids=ids[index : index + 1], input_embeds=None, **info)[1]
                    for index, info in enumerate(infos)
                ]
            )
            output_ids, actual, updates = model.preprocess_decode_batch(input_ids=ids, req_infos=infos)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert output_ids is ids
            assert updates == [{} for _ in infos]
