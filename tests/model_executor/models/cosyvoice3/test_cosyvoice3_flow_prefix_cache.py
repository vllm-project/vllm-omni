# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

STEPS, DIM, DEPTH, PROMPT, LOOKAHEAD = 3, 32, 2, 30, 3


def _tiny_streaming_flow(monkeypatch):
    import vllm_omni.model_executor.models.cosyvoice3.code2wav_core.packed_dit as packed_dit
    import vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav as module
    from tests.model_executor.models.cosyvoice3.test_cosyvoice3_components import _force_torch_sdpa
    from tests.model_executor.models.cosyvoice3.test_cosyvoice3_full_batch import MelOutput
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.flow_prefix_cache import paged_attention_reference
    from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

    monkeypatch.setattr(module, "CausalHiFTGenerator", MelOutput)
    monkeypatch.setattr(module, "cosyvoice3_packed_streaming_enabled", lambda: True)
    monkeypatch.setattr(module, "cosyvoice3_packed_inference_enabled", lambda: True)
    # The full-recompute baseline runs the packed path eagerly on CPU, with a
    # page-size-1 reference attention in place of FA3.
    monkeypatch.setattr(packed_dit, "native_mish", torch.nn.functional.mish)
    monkeypatch.setattr(
        packed_dit,
        "packed_fa3",
        lambda q, k, v, lens, table, cu, _: paged_attention_reference(q, k, v, lens, table, cu),
    )
    config = CosyVoice3Config()
    config.flow["pre_lookahead_layer"]["channels"] = 32
    config.flow["decoder"]["estimator"].update(dim=DIM, depth=DEPTH, heads=4, dim_head=8)
    torch.manual_seed(0)
    with _force_torch_sdpa():
        model = module.CosyVoice3Code2Wav(config).eval()
    estimator = packed_dit.PackedDiT(model.flow_model.decoder.estimator)
    estimator.compiled_full_forward = estimator.forward_full
    model._packed_estimator = estimator
    model._stream_hift_from_feat = lambda mel, cache_state=None, finalize=False: (mel, None if finalize else {})
    return model


def _hops(hop_tokens):
    """Codec chunks as talker2code2wav_async_chunk emits them: (prefix length, offset, finalize)."""
    emitted = 0
    for hop in hop_tokens:
        yield emitted + hop + LOOKAHEAD, emitted, False
        emitted += hop
    yield emitted + LOOKAHEAD + 5, emitted, True


def _stream(model, request, hop_tokens):
    outputs, state = [], None
    for prefix, offset, finalize in _hops(hop_tokens):
        item = dict(request, token=request["token"][:, :prefix], token_offset_tokens=offset, finalize=finalize)
        [(mel, state)] = model.forward_streaming_batch([dict(item, cache_state=state)], n_timesteps=STEPS)
        outputs.append(mel)
    return outputs


@pytest.mark.parametrize(
    "hop_tokens",
    [
        [45, 50, 100],  # the connector pads the first hop to a 25-token attention chunk
        [17, 20, 33],  # unaligned hops leave a partial chunk to recompute
    ],
)
@pytest.mark.parametrize("pages", [64, 20])  # 20 pages run out on the last non-final hop
@torch.inference_mode()
def test_prefix_cache_matches_full_recompute(monkeypatch, hop_tokens, pages):
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.flow_prefix_cache import PAGE_FRAMES

    model = _tiny_streaming_flow(monkeypatch)
    request = dict(
        token=torch.randint(0, 64, (1, sum(hop_tokens) + LOOKAHEAD + 5)),
        prompt_token=torch.randint(0, 64, (1, PROMPT)),
        prompt_feat=torch.randn(1, 2 * PROMPT, 80),
        embedding=torch.randn(1, 192),
    )
    expected = _stream(model, request, hop_tokens)

    page_bytes = 2 * STEPS * PAGE_FRAMES * (2 * DEPTH * DIM + DIM) * 4
    monkeypatch.setenv("COSYVOICE3_FLOW_PREFIX_CACHE_GIB", str((pages + 0.5) * page_bytes / 2**30))
    actual = _stream(model, request, hop_tokens)

    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference, rtol=1e-4, atol=1e-4)
    assert model._flow_prefix_cache.num_pages == pages
    assert model._flow_prefix_cache.used_pages == 0
