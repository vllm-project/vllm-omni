# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chunked prefix-KV execution against independent masked attention references."""

from dataclasses import dataclass

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vllm_omni.diffusion.attention.backends.abstract import AttentionBackend
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.forward_context import ForwardContext, MixFusionStep, override_forward_context
from vllm_omni.diffusion.hooks.mixfusion import apply_mixfusion, plan_foreign_hooks
from vllm_omni.diffusion.mixfusion.plan import MixFusionRoot, MixFusionSequenceRequest, mixfusion_compatible_hook
from vllm_omni.diffusion.mixfusion.prefix_kv import (
    PackedDocument,
    PackedDocumentGeometry,
    build_document_plan,
    pack_document_queries,
    run_prefix_attention,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _DenseBackend(AttentionBackend):
    @classmethod
    def supports_dense_4d_mask(cls, attention_spec=None):
        return True


class _MaskFreeBackend(AttentionBackend):
    pass


class _Attention:
    forward_packed = Attention.forward_packed
    _forward_packed_dense_buckets = Attention._forward_packed_dense_buckets

    def __init__(self, masked=True):
        self.backend = _DenseBackend if masked else _MaskFreeBackend
        self.calls = []

    def effective_attention(self):
        return None, self.backend, None

    def _effective_kv_cache_dtype(self):
        return None

    def forward(self, q, k, v, metadata):
        self.calls.append((q.shape, k.shape, metadata.attn_mask))
        return F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=metadata.attn_mask,
            enable_gqa=True,
        ).transpose(1, 2)


@pytest.mark.parametrize("masked", [True, False])
def test_prefix_segments_match_dense_reference_with_mixed_phases(masked):
    torch.manual_seed(123)
    # A later step, a first step, and its other CFG branch. Prefix and image
    # sizes differ, and the first step contains a causal tail after the image.
    geometries = (
        PackedDocumentGeometry("old", 0, 5, 5, 3, ((4, 8),), (4, 8)),
        PackedDocumentGeometry("new", 0, 10, 10, 0, ((4, 9),), (4, 9), capture_length=3),
        PackedDocumentGeometry("new", 1, 8, 10, 0, ((2, 7),), (2, 7), capture_length=1),
    )
    docs, qs, ks, vs, expected = [], [], [], [], []
    for geo in geometries:
        n, p = geo.query_length, geo.prefix_length
        q, k, v = torch.randn(n, 4, 8), torch.randn(n, 2, 8), torch.randn(n, 2, 8)
        pk, pv = torch.randn(p, 2, 8), torch.randn(p, 2, 8)
        all_k, all_v = torch.cat((pk, k)), torch.cat((pv, v))
        mask = torch.ones(p + n, p + n, dtype=torch.bool).tril()
        for start, end in geo.full_spans:
            mask[start:end, start:end] = True
        ref = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            all_k.transpose(0, 1)[None],
            all_v.transpose(0, 1)[None],
            attn_mask=mask[None, None, p:],
            enable_gqa=True,
        )[0].transpose(0, 1)
        expected.append(ref)
        docs.append(PackedDocument(geo, q, q, q, q, prefix_kv=((pk, pv),)))
        qs.append(q)
        ks.append(k)
        vs.append(v)
    plan = build_document_plan(geometries, torch.device("cpu"))
    attn = _Attention(masked)

    def packed_attention():
        q, k, v = (pack_document_queries(rows, plan) for rows in (qs, ks, vs))
        result = run_prefix_attention(attn, q, k, v, plan, docs, 0)
        return [result.index_select(0, idx) for idx in plan.document_indices]

    actual = packed_attention()
    for result, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, ref)
    assert docs[0].captured_kv == []
    torch.testing.assert_close(docs[1].captured_kv[0][0], ks[1][:3])
    assert docs[1].captured_kv[0][0].untyped_storage().data_ptr() != ks[1].untyped_storage().data_ptr()
    # Mutation of the other CFG branch must not change the old request.
    vs[-1].fill_(100)
    torch.testing.assert_close(packed_attention()[0], actual[0])


def test_equal_prefix_shapes_share_dense_attention_bucket():
    attn = _Attention()
    q, k, v = torch.randn(6, 2, 4), torch.randn(14, 1, 4), torch.randn(14, 1, 4)
    attn.forward_packed(
        q,
        k,
        v,
        cu_seqlens=torch.tensor([0, 3, 6]),
        cu_seqlens_k=torch.tensor([0, 7, 14]),
        max_seqlen=3,
        max_seqlen_k=7,
        causal=True,
        sequence_lengths=((3, 3), (7, 7)),
    )
    assert len(attn.calls) == 1
    assert attn.calls[0][0][0] == 2
    assert attn.calls[0][2][0, 0, 0].tolist() == [True] * 5 + [False] * 2


@dataclass
class _Output:
    value: torch.Tensor

    def __getitem__(self, index):
        assert index == 0
        return self.value


class _Decoder(nn.Module):
    _mixfusion_plan = {"": MixFusionRoot(request=MixFusionSequenceRequest())}

    def forward(
        self,
        inputs_embeds=None,
        position_ids=None,
        custom_pos_emb=None,
        query_lens=None,
        seq_lens=None,
        mixfusion_documents=None,
    ):
        return _Output(inputs_embeds * 2)


def test_prefix_kv_root_restores_order_and_recomputes_chunks_after_departure():
    model = _Decoder()
    apply_mixfusion(model, model._mixfusion_plan)
    docs = []
    for idx, length in enumerate([5, 9]):
        x = torch.arange(length * 8, dtype=torch.float32).reshape(1, length, 8) + idx
        docs.append(
            PackedDocument(
                PackedDocumentGeometry(str(idx), 0, length, length, 0, ((1, length),), (1, length)),
                x,
                torch.arange(length)[None],
                x,
                x,
            )
        )
    for active in [docs, docs[::-1], docs[:1]]:
        step = MixFusionStep()
        with override_forward_context(ForwardContext(mixfusion_step=step)):
            outputs = model(mixfusion_documents=active)
        assert step.plan.chunk_size == (4 if len(active) == 2 else active[0].geometry.query_length - 1)
        assert step.plan.num_chunks == (3 if len(active) == 2 else 1)
        assert len(outputs) == len(active)
        for out, doc in zip(outputs, active, strict=True):
            torch.testing.assert_close(out, doc.hidden_states * 2)
        assert tuple(doc.request_id for doc in step.plan.documents) == tuple(d.geometry.request_id for d in active)


def test_prefix_kv_root_checks_descendant_hooks():
    model = _Decoder()
    model.layers = nn.ModuleList([nn.Linear(8, 8)])

    @mixfusion_compatible_hook
    def token_count(module, args):
        module.token_count = args[0].shape[0]

    model.layers[0].register_forward_pre_hook(token_count)
    assert plan_foreign_hooks(model, model._mixfusion_plan) == []
    unknown = model.layers[0].register_forward_hook(lambda module, args, output: output)
    assert plan_foreign_hooks(model, model._mixfusion_plan) == ["layers.0:torch-native"]
    unknown.remove()
    assert plan_foreign_hooks(model, model._mixfusion_plan) == []
