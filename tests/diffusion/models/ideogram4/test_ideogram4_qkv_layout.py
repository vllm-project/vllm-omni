import torch


def _remap_qkv_for_parallel(weight: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    w = weight.view(3, num_heads, head_dim, -1)
    q = w[0].reshape(num_heads * head_dim, -1)
    k = w[1].reshape(num_heads * head_dim, -1)
    v = w[2].reshape(num_heads * head_dim, -1)
    return torch.cat([q, k, v], dim=0)


def test_qkv_weight_layout_matches_reference_view():
    num_heads = 2
    head_dim = 3
    in_features = 4
    batch = 2
    seq = 5
    hidden = num_heads * head_dim

    x = torch.randn(batch, seq, in_features)
    packed_weight = torch.randn(3 * hidden, in_features)

    ref_out = x @ packed_weight.T
    ref_qkv = ref_out.view(batch, seq, 3, num_heads, head_dim)
    ref_q, ref_k, ref_v = ref_qkv.unbind(dim=2)

    remapped = _remap_qkv_for_parallel(packed_weight, num_heads, head_dim)
    out = x @ remapped.T
    q, k, v = out.split([hidden, hidden, hidden], dim=-1)

    assert torch.allclose(q.view(batch, seq, num_heads, head_dim), ref_q)
    assert torch.allclose(k.view(batch, seq, num_heads, head_dim), ref_k)
    assert torch.allclose(v.view(batch, seq, num_heads, head_dim), ref_v)
