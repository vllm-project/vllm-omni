# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Falcon Perception split-RoPE parity against the reference implementation.

The reference (``tiiuae/Falcon-Perception``, ``falcon_perception/rope.py``)
operates on ``(batch, seq, heads, dim)``; the port operates on vLLM's flat
``(num_tokens, heads, dim)``. The reference math is transcribed inline here so
the test is self-contained — it was verified bit-exact (``max|diff| == 0.0``)
against the actual reference module when the port was written.

Getting any of this subtly wrong produces a model that loads and runs but emits
garbage, so these are exact-equality assertions, not tolerances.
"""

import einops as E
import pytest
import torch

from vllm_omni.model_executor.models.falcon_perception.rope import (
    apply_3d_rotary_emb,
    apply_golden_freqs_cis_to_visual_pos,
    compute_image_spatial_positions,
    compute_pos_hw,
    pack_image_grid_positions,
    precompute_freqs_cis,
    unpack_image_grid_positions,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HEAD_DIM = 128
N_HEADS = 16
ROPE_DIM = HEAD_DIM // 2  # temporal half
N_FREQS = ROPE_DIM // 2  # complex pairs / golden freq rows


# --------------------------------------------------------------------------
# Reference math, transcribed from falcon_perception/rope.py @ origin/main.
# --------------------------------------------------------------------------
def _ref_precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def _ref_apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = E.rearrange(freqs_cis, "b s d -> b s 1 d")
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def _ref_golden_freqs(freqs_hfp, pos_bsp):
    theta = torch.einsum("bsp,hfp->bshf", pos_bsp.float(), freqs_hfp.float())
    return torch.polar(torch.ones_like(theta), theta)


def _ref_golden_rotary(x, freqs_cis):
    xf = x.float()
    x_even, x_odd = xf[..., 0::2], xf[..., 1::2]
    cos, sin = freqs_cis.real, freqs_cis.imag
    out = torch.empty_like(xf)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out.type_as(x)


def _ref_apply_3d(xq, xk, freqs_cis, freqs_cis_2d):
    xq_t, xq_hw = xq.chunk(2, dim=-1)
    xk_t, xk_hw = xk.chunk(2, dim=-1)
    xq_t, xk_t = _ref_apply_rotary_emb(xq_t, xk_t, freqs_cis)
    if freqs_cis_2d is not None:
        xq_hw = _ref_golden_rotary(xq_hw, freqs_cis_2d)
        xk_hw = _ref_golden_rotary(xk_hw, freqs_cis_2d)
    return (
        torch.cat([xq_t, xq_hw], dim=-1).type_as(xq),
        torch.cat([xk_t, xk_hw], dim=-1).type_as(xk),
    )


# --------------------------------------------------------------------------


def _fixture(seq_len: int = 50, grid_h: int = 5, grid_w: int = 7, img_start: int = 3):
    torch.manual_seed(0)
    q = torch.randn(1, seq_len, N_HEADS, HEAD_DIM)
    k = torch.randn(1, seq_len, N_HEADS, HEAD_DIM)
    golden = torch.randn(N_HEADS, N_FREQS, 2)

    pos_t = torch.arange(seq_len).clamp(max=20)
    freqs_cis = _ref_precompute_freqs_cis(ROPE_DIM, 4096)[pos_t].unsqueeze(0)

    pos_hw = torch.zeros(seq_len, 2)
    n_img = grid_h * grid_w
    pos_hw[img_start : img_start + n_img] = compute_image_spatial_positions(grid_h, grid_w)
    return q, k, golden, freqs_cis, pos_hw


def test_precompute_freqs_cis_matches_reference():
    ours = precompute_freqs_cis(ROPE_DIM, 4096, 10000.0)
    ref = _ref_precompute_freqs_cis(ROPE_DIM, 4096, 10000.0)
    assert ours.shape == (4096, N_FREQS)
    assert torch.equal(ours, ref)


def test_golden_freqs_match_reference_and_are_identity_for_text():
    _, _, golden, _, pos_hw = _fixture()
    ours = apply_golden_freqs_cis_to_visual_pos(golden, pos_hw)
    ref = _ref_golden_freqs(golden, pos_hw.unsqueeze(0))[0]
    assert torch.equal(ours, ref)

    # Text tokens sit at (0, 0) -> theta 0 -> exactly 1+0j, so the spatial half
    # is a genuine no-op outside images (and can be skipped during decode).
    text_row = ours[0]
    assert torch.equal(text_row.real, torch.ones_like(text_row.real))
    assert torch.equal(text_row.imag, torch.zeros_like(text_row.imag))


def test_apply_3d_rotary_matches_reference_with_and_without_image():
    q, k, golden, freqs_cis, pos_hw = _fixture()
    golden_cis = apply_golden_freqs_cis_to_visual_pos(golden, pos_hw)

    ours_q, ours_k = apply_3d_rotary_emb(q[0], k[0], freqs_cis[0], golden_cis)
    ref_q, ref_k = _ref_apply_3d(q, k, freqs_cis, golden_cis.unsqueeze(0))
    assert torch.equal(ours_q, ref_q[0])
    assert torch.equal(ours_k, ref_k[0])

    # Decode step: no image in the batch, spatial half must pass through.
    dec_q, dec_k = apply_3d_rotary_emb(q[0], k[0], freqs_cis[0], None)
    ref_dq, ref_dk = _ref_apply_3d(q, k, freqs_cis, None)
    assert torch.equal(dec_q, ref_dq[0])
    assert torch.equal(dec_k, ref_dk[0])

    # The compiled backbone always supplies a spatial-frequency tensor. Its
    # tensor gate must nevertheless preserve the reference's decode path
    # exactly, rather than relying on an identity float32 rotation.
    gated_q, gated_k = apply_3d_rotary_emb(q[0], k[0], freqs_cis[0], golden_cis, torch.tensor([False]))
    assert torch.equal(gated_q, ref_dq[0])
    assert torch.equal(gated_k, ref_dk[0])


def test_spatial_grid_is_aspect_normalised_and_h_major():
    """The grid must be h-major to match the patchify order in embed_multimodal."""
    grid = compute_image_spatial_positions(2, 3)
    assert grid.shape == (6, 2)
    # h (row) index changes slowest.
    assert torch.equal(grid[0:3, 0], grid[0].expand(3, 2)[:, 0])
    assert grid[0, 0] < grid[3, 0]
    # Aspect normalisation: wider than tall -> w spans a wider range than h.
    assert grid[:, 1].max() > grid[:, 0].max()


# --------------------------------------------------------------------------
# The grid rides in rows 1/2 of the M-RoPE positions buffer.
#
# It cannot ride in state set by ``embed_multimodal``: on a multimodal encoder
# cache hit (same image, different query) vLLM serves the stored embedding and
# never makes that call, so a stashed grid is None, ``pos_hw`` is None, and the
# golden 2-D RoPE is silently dropped for every image token. The model still
# runs — it just emits garbage (measured: 18 masks instead of 121).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("grid", [(5, 7), (7, 5), (62, 62), (26, 46), (1, 9), (9, 1), (1, 1)])
def test_packed_grid_round_trips_exactly(grid):
    """Reconstruction must be bit-exact, not close — it feeds a rotation."""
    grid_h, grid_w = grid
    got = unpack_image_grid_positions(pack_image_grid_positions(grid_h, grid_w))
    assert torch.equal(got, compute_image_spatial_positions(grid_h, grid_w))


def _positions_with_image(seq_len: int, spans: list[tuple[int, tuple[int, int]]]) -> torch.Tensor:
    """A ``[3, seq_len]`` buffer with packed grids at the given start offsets."""
    positions = torch.zeros((3, seq_len), dtype=torch.long)
    positions[0] = torch.arange(seq_len)
    for start, (grid_h, grid_w) in spans:
        packed = pack_image_grid_positions(grid_h, grid_w)
        positions[1:3, start : start + packed.shape[0]] = packed.t()
    return positions


def test_compute_pos_hw_scatters_only_onto_image_tokens():
    grid_h, grid_w, img_start = 5, 7, 3
    n_img = grid_h * grid_w
    input_ids = torch.full((50,), 999, dtype=torch.long)
    input_ids[img_start : img_start + n_img] = 227

    got = compute_pos_hw(input_ids, _positions_with_image(50, [(img_start, (grid_h, grid_w))]), image_token_id=227)
    expected = torch.zeros(50, 2)
    expected[img_start : img_start + n_img] = compute_image_spatial_positions(grid_h, grid_w)
    assert torch.equal(got, expected)


def test_compute_pos_hw_keeps_two_batched_images_of_different_sizes_apart():
    """A batch concatenates requests, so one shared ``max()`` would be wrong."""
    first, second = (5, 7), (3, 11)
    n_first, n_second = first[0] * first[1], second[0] * second[1]
    start_a, start_b = 2, 2 + n_first + 6

    input_ids = torch.full((start_b + n_second + 4,), 999, dtype=torch.long)
    input_ids[start_a : start_a + n_first] = 227
    input_ids[start_b : start_b + n_second] = 227
    positions = _positions_with_image(input_ids.shape[0], [(start_a, first), (start_b, second)])

    got = compute_pos_hw(input_ids, positions, image_token_id=227)
    assert torch.equal(got[start_a : start_a + n_first], compute_image_spatial_positions(*first))
    assert torch.equal(got[start_b : start_b + n_second], compute_image_spatial_positions(*second))


def test_compute_pos_hw_survives_a_window_that_starts_mid_image():
    """Prefix caching resumes inside the image span; the runner slices these
    rows by ``num_computed_tokens``, so only a tail of the grid is present.

    Carrying the grid *dimensions* on every token — not just the indices — is
    what makes that tail reconstruct to the same values as the full pass.
    """
    grid_h, grid_w, img_start = 5, 7, 3
    n_img = grid_h * grid_w
    input_ids = torch.full((50,), 999, dtype=torch.long)
    input_ids[img_start : img_start + n_img] = 227
    positions = _positions_with_image(50, [(img_start, (grid_h, grid_w))])

    full = compute_pos_hw(input_ids, positions, image_token_id=227)
    # Resume 20 tokens in, i.e. 17 patches into the image.
    resumed = compute_pos_hw(input_ids[20:], positions[:, 20:], image_token_id=227)
    assert torch.equal(resumed, full[20:])


def test_compute_pos_hw_returns_none_when_there_is_nothing_to_rotate():
    # Pure decode step: no image tokens in the batch.
    assert compute_pos_hw(torch.full((4,), 999), _positions_with_image(4, []), image_token_id=227) is None
    # Profile run: image tokens but a zeroed positions buffer, so no packed grid.
    assert compute_pos_hw(torch.full((4,), 227), torch.zeros((3, 4), dtype=torch.long), image_token_id=227) is None


def test_mrope_positions_carry_pos_t_and_the_grid_together():
    """End-to-end: what ``get_mrope_input_positions`` writes is what
    ``compute_pos_hw`` reads back, and row 0 still carries ``pos_t`` alone."""
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.falcon_perception.configuration_falcon_perception import (
        FalconPerceptionConfig,
    )
    from vllm_omni.model_executor.models.falcon_perception.falcon_perception_thinker import (
        FalconPerceptionThinker,
    )

    config = FalconPerceptionConfig()
    grid_h, grid_w = 3, 5
    tokens = (
        [7, 8]
        + config.image_structural_token_ids
        + [config.img_id] * (grid_h * grid_w)
        + [config.img_end_id]
        + [9, 10, 11]
    )
    mm_features = [
        SimpleNamespace(data=SimpleNamespace(get_data=lambda: {"image_grid_hw": torch.tensor([grid_h, grid_w])}))
    ]
    stub = SimpleNamespace(config=config, _end_of_image_token_id=config.img_end_id)

    positions, delta = FalconPerceptionThinker.get_mrope_input_positions(stub, tokens, mm_features=mm_features)

    # Row 0: the whole image block collapses onto the <image_cls> position.
    img_block = slice(2, 2 + 5 + grid_h * grid_w + 1)
    assert torch.equal(positions[0][img_block], torch.full((5 + grid_h * grid_w + 1,), 2))
    assert positions[0].tolist()[-3:] == [3, 4, 5]
    assert delta == int(positions[0].max()) + 1 - len(tokens)

    # Rows 1/2: the grid, recovered exactly.
    pos_hw = compute_pos_hw(torch.tensor(tokens), positions, image_token_id=config.img_id)
    patches = slice(2 + 5, 2 + 5 + grid_h * grid_w)
    assert torch.equal(pos_hw[patches], compute_image_spatial_positions(grid_h, grid_w))
    assert torch.equal(pos_hw[:2], torch.zeros(2, 2))
    assert torch.equal(pos_hw[patches.stop :], torch.zeros(len(tokens) - patches.stop, 2))


def test_mrope_positions_reject_a_grid_that_disagrees_with_the_placeholder_run():
    """A prompt update that leaves the literal ``<|image|>`` in place gives one
    patch token too many; that must fail loudly, not rotate by a part-patch."""
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.falcon_perception.configuration_falcon_perception import (
        FalconPerceptionConfig,
    )
    from vllm_omni.model_executor.models.falcon_perception.falcon_perception_thinker import (
        FalconPerceptionThinker,
    )

    config = FalconPerceptionConfig()
    tokens = [*config.image_structural_token_ids, *([config.img_id] * 16), config.img_end_id]
    mm_features = [SimpleNamespace(data=SimpleNamespace(get_data=lambda: {"image_grid_hw": torch.tensor([3, 5])}))]
    stub = SimpleNamespace(config=config, _end_of_image_token_id=config.img_end_id)

    with pytest.raises(ValueError, match="must agree exactly"):
        FalconPerceptionThinker.get_mrope_input_positions(stub, tokens, mm_features=mm_features)


# --------------------------------------------------------------------------
# freqs_cis_golden is a BUFFER, not a parameter. vLLM's missing-weight guard
# only inspects named_parameters()
# (model_loader/default_loader.py: weights_to_load = {name for name, _ in
# model.named_parameters()}), so a checkpoint without this key would leave
# torch.empty memory serving as the learned 2-D rotation — no exception, and
# masks that look plausible but are wrong. load_weights must catch it.
# --------------------------------------------------------------------------


def _thinker_stub():
    """Minimal stand-in exposing only what ``load_weights`` reads before the guard."""
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.falcon_perception.configuration_falcon_perception import (
        FalconPerceptionConfig,
    )

    config = FalconPerceptionConfig()
    golden = torch.zeros((config.num_attention_heads, config.head_dim // 4, 2))
    return SimpleNamespace(
        named_parameters=dict,
        config=config,
        model=SimpleNamespace(freqs_cis_golden=golden),
        _map_weight_name=lambda _name: None,
    )


@pytest.fixture
def _no_tp(monkeypatch):
    """``load_weights`` reads the TP rank/size; no distributed group in a CPU test."""
    import vllm_omni.model_executor.models.falcon_perception.falcon_perception_thinker as mod

    monkeypatch.setattr(mod, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: 1)


def test_load_weights_raises_when_the_golden_freqs_buffer_is_absent(_no_tp):
    from vllm_omni.model_executor.models.falcon_perception.falcon_perception_thinker import (
        FalconPerceptionThinker,
    )

    stub = _thinker_stub()
    # A checkpoint carrying everything *except* freqs_cis_golden.
    weights = iter([("tok_embeddings.weight", torch.zeros(4, 4))])
    with pytest.raises(ValueError, match="freqs_cis_golden"):
        FalconPerceptionThinker.load_weights(stub, weights)


def test_load_weights_accepts_the_checkpoint_when_the_buffer_is_present(_no_tp):
    from vllm_omni.model_executor.models.falcon_perception.falcon_perception_thinker import (
        FalconPerceptionThinker,
    )

    stub = _thinker_stub()
    n_heads = stub.config.num_attention_heads
    golden = torch.randn(n_heads, stub.config.head_dim // 4, 2)
    loaded = FalconPerceptionThinker.load_weights(stub, iter([("freqs_cis_golden", golden)]))
    assert "model.freqs_cis_golden" in loaded
    assert torch.equal(stub.model.freqs_cis_golden, golden)
