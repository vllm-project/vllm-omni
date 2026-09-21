# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Only the request whose own rows hold the <|fim_middle|> placeholder is img2img: its VAE patches go
through the generation expert, while the text, img2text and decode requests batched with it -- or
running after it -- keep the understanding expert whatever their length or order. The rows of each
request come from prepare_runner_inputs; a chunked prefill keeps its block start per request, and a
request whose image the encoder cache served (a CFG companion, a resumed request) is matched to the
same size seen before."""

import pytest
import torch
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange

from vllm_omni.model_executor.models.bagel.bagel import OmniBagelForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

FIM, PAD, TEXT = 7, 5, 1  # <|fim_middle|>, <|image_pad|>, a text token


def make_model():
    model = OmniBagelForConditionalGeneration.__new__(OmniBagelForConditionalGeneration)
    model._img2img_token_id = FIM
    model._ropes_pending = []
    model._ropes_metadata = {}
    model._reset_img2img_state()
    return model


def mrope(ids, features):
    return OmniBagelForConditionalGeneration.get_mrope_input_positions(None, ids, features)[0][0].tolist()


def feature(modality, offset, length, is_embed=None):
    return MultiModalFeatureSpec(
        data=None,
        modality=modality,
        identifier="x",
        mm_position=PlaceholderRange(offset=offset, length=length, is_embed=is_embed),
    )


class Req:
    """One request's prompt: token ids, RoPE positions, the prompt rows of its VAE patches and the
    (num_vae, num_vit, H, W) its encoder run reports."""

    def __init__(self, req_id, ids, pos, vae_rows=(), info=None):
        self.req_id, self.ids, self.pos, self.vae_rows, self.info = req_id, ids, pos, set(vae_rows), info


def text_req(req_id, n):
    return Req(req_id, [TEXT] * n, list(range(n)))


def img2text_req(req_id, num_patches, pre=1, post=1):
    ids = [TEXT] * pre + [PAD] * (num_patches + 2) + [TEXT] * post
    return Req(req_id, ids, mrope(ids, [feature("image", pre, num_patches + 2)]))


def img2img_req(req_id, num_vae=8, num_vit=6, size=(64, 48), pre=0, post=1):
    """text, [<VS> VAE patches <VE> | separator | <VS> ViT patches <VE>], text: num_vae / num_vit count
    the markers, as _process_img2img_input reports them."""
    block = num_vae + 1 + num_vit
    ids = [TEXT] * pre + [FIM] * block + [TEXT] * post
    is_embed = torch.tensor([True] * num_vae + [False] + [True] * num_vit)
    pos = mrope(ids, [feature("img2img", pre, block, is_embed)])
    return Req(req_id, ids, pos, range(pre + 1, pre + num_vae - 1), (num_vae, num_vit, *size))


def run_step(model, chunks, pending=(), padding=0):
    """One model step over ``chunks`` = [(req, tokens computed before, tokens scheduled now)], rows in
    that order, plus ``padding`` stale <|fim_middle|> rows past the batch (CUDA-graph padding).
    Returns (MoT used, rows marked as VAE patches, queued rope metadata per request)."""
    ids, pos = [], []
    for req, computed, n in chunks:
        ids += req.ids[computed : computed + n]
        pos += req.pos[computed : computed + n]
    ids += [FIM] * padding
    pos += [0] * padding
    input_ids, positions = torch.tensor(ids), torch.tensor(pos)
    for info in pending:
        model._register_img2img_info(info)  # what _process_img2img_input does per image
    restored, _ = model.prepare_runner_inputs(
        None,
        positions,
        torch.zeros(len(ids), 1),
        req_ids=[c[0].req_id for c in chunks],
        num_computed_tokens=[c[1] for c in chunks],
        num_scheduled_tokens=[c[2] for c in chunks],
        input_ids_buffer=input_ids,
    )
    use_mot = model._route_img2img(restored, positions)
    mask = model._vae_token_mask
    rows = set(mask.nonzero().flatten().tolist()) if mask is not None else set()
    metas, model._ropes_pending = model._ropes_pending, []
    return use_mot, rows, metas


def expected_rows(chunks):
    """Batch rows of the VAE patches every chunk brings in."""
    rows, row0 = set(), 0
    for req, computed, n in chunks:
        rows |= {row0 + r - computed for r in req.vae_rows if computed <= r < computed + n}
        row0 += n
    return rows


def full(req):
    return req, 0, len(req.ids)


@pytest.mark.parametrize("neighbour", ["text", "img2text"])
@pytest.mark.parametrize("img2img_first", [True, False])
def test_long_neighbour_in_the_same_batch_keeps_the_understanding_expert(neighbour, img2img_first):
    img = img2img_req("img")
    other = text_req("other", 60) if neighbour == "text" else img2text_req("other", 40)
    assert len(other.ids) > len(img.ids)  # long enough for the old length-based gate to misroute it
    chunks = [full(img), full(other)] if img2img_first else [full(other), full(img)]

    model = make_model()
    use_mot, rows, metas = run_step(model, chunks, pending=[img.info])

    assert use_mot and rows == expected_rows(chunks)
    assert model._pending_img2img_info == []
    img_meta, other_meta = (metas[0], metas[1]) if img2img_first else (metas[1], metas[0])
    assert img_meta["image_shape"] == [64, 48] and img_meta["prefill_position_count"] == len(img.ids)
    assert img_meta["ropes"] == [img.pos[-1] + 1]
    assert other_meta == {"ropes": [other.pos[-1] + 1]}, "a neighbour must not inherit the img2img metadata"


def test_requests_after_an_img2img_one_keep_the_understanding_expert():
    model = make_model()
    img = img2img_req("img")
    assert run_step(model, [full(img)], pending=[img.info])[0]

    for req in (text_req("t", 80), img2text_req("i", 60)):
        use_mot, rows, metas = run_step(model, [full(req)])
        assert not use_mot and rows == set() and metas == []


def test_decode_neighbours_get_their_own_metadata():
    a, b, img = text_req("a", 501), text_req("b", 601), img2img_req("img")
    chunks = [(a, 500, 1), (b, 600, 1), full(img)]

    use_mot, rows, metas = run_step(make_model(), chunks, pending=[img.info])

    assert use_mot and rows == expected_rows(chunks)
    assert metas[0] == {"ropes": [501]} and metas[1] == {"ropes": [601]}
    assert metas[2]["image_shape"] == [64, 48]


@pytest.mark.parametrize(
    "cuts", [(3,), (7,), (12,), (13,), (14,), (17,), (20,), (5,), (3, 9), (7, 15), (13, 14), (6, 21)]
)
def test_chunked_prefill_marks_the_same_rows_as_one_chunk(cuts):
    # rows: text 0-4, <VS> 5, patches 6-11, <VE> 12, separator 13, ViT 14-19, text 20-23
    img = img2img_req("img", num_vae=8, num_vit=6, pre=5, post=4)
    bounds = (0, *cuts, len(img.ids))
    model, seen = make_model(), set()
    for computed, end in zip(bounds, bounds[1:]):
        n = end - computed
        neighbour = text_req(f"t{computed}", 30)
        chunk = (img, computed, n)
        # the encoder runs in the step whose chunk brings in the block's first row
        pending = [img.info] if computed <= 5 < end else []
        use_mot, rows, metas = run_step(model, [full(neighbour), chunk], pending=pending)

        assert rows == expected_rows([full(neighbour), chunk])
        assert use_mot == bool(rows)
        seen |= {r - 30 + computed for r in rows}
        if FIM not in img.ids[computed:end]:
            assert metas == [], "a chunk without block rows is an ordinary forward"
            continue
        assert metas[0] == {"ropes": [30]}
        assert metas[1]["image_shape"] == [64, 48] and metas[1]["prefill_position_count"] == end
        assert metas[1]["ropes"] == [img.pos[end - 1] + 1]
    assert seen == img.vae_rows


def test_companion_served_by_the_encoder_cache_finds_its_size():
    model = make_model()
    parent = img2img_req("p", pre=3)
    assert run_step(model, [full(parent)], pending=[parent.info])[0]

    # same image, no encoder run: cfg_text keeps the image and drops the text
    companion = img2img_req("p__cfg_text", pre=0, post=2)
    use_mot, rows, metas = run_step(model, [full(companion), full(text_req("t", 50))])
    assert use_mot and rows == expected_rows([full(companion), full(text_req("t", 50))])
    assert metas[0]["image_shape"] == [64, 48] and metas[1] == {"ropes": [50]}


def test_encoder_outputs_match_by_size_not_order():
    small = img2img_req("small", num_vae=8, num_vit=6, size=(64, 48))
    large = img2img_req("large", num_vae=12, num_vit=10, size=(96, 64))
    chunks = [full(small), full(large)]

    use_mot, rows, metas = run_step(make_model(), chunks, pending=[large.info, small.info])

    assert use_mot and rows == expected_rows(chunks)
    assert metas[0]["image_shape"] == [64, 48] and metas[1]["image_shape"] == [96, 64]


def test_stale_encoder_output_does_not_leak_onto_a_text_batch():
    model = make_model()
    img = img2img_req("img")
    use_mot, rows, metas = run_step(model, [full(text_req("t", 40))], pending=[img.info])
    assert not use_mot and rows == set() and metas == []
    assert model._pending_img2img_info == []


def test_padding_rows_past_the_batch_are_ignored():
    model = make_model()
    img = img2img_req("img")
    assert run_step(model, [full(img)], pending=[img.info])[0]

    use_mot, rows, _ = run_step(model, [full(text_req("t", 40))], padding=6)
    assert not use_mot and rows == set()

    chunks = [full(img2img_req("img2"))]
    use_mot, rows, _ = run_step(model, chunks, padding=6)
    assert use_mot and rows == expected_rows(chunks)


def test_a_request_prefilled_again_starts_over():
    model = make_model()
    img = img2img_req("img", pre=2)
    use_mot, rows, _ = run_step(model, [(img, 0, 6)], pending=[img.info])  # cut inside the VAE block
    assert use_mot and rows == expected_rows([(img, 0, 6)])

    # preempted and resumed from scratch; the encoder cache still holds the image
    use_mot, rows, metas = run_step(model, [full(img)])
    assert use_mot and rows == expected_rows([full(img)])
    assert metas[0]["prefill_position_count"] == len(img.ids)


def test_dummy_runs_without_a_layout_neither_route_nor_touch_the_inputs():
    """Profiling and CUDA-graph capture call forward without prepare_runner_inputs; reading input_ids
    or positions there would sync the device inside a graph capture."""
    model = make_model()
    img = img2img_req("img")
    model._register_img2img_info(img.info)  # the profiling run embeds dummy img2img data

    class Untouchable:
        def __getitem__(self, _):
            raise AssertionError("input_ids must not be read on a dummy run")

    assert not model._route_img2img(Untouchable(), Untouchable())
    assert model._pending_img2img_info == [] and model._vae_token_mask is None and model._ropes_pending == []
