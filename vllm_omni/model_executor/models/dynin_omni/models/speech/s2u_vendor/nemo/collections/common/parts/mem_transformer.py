import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nemo.collections.common.parts.normalization import LayerVarNorm

float32_min = np.finfo(np.float32).min
float16_min = np.finfo(np.float16).min


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]


def get_norm(norm_type, d_model, ln_eps):
    if norm_type == 'ln':
        norm = nn.LayerNorm(d_model, eps=ln_eps)
    else:
        assert norm_type == 'var_ln'
        norm = LayerVarNorm(d_model, eps=ln_eps)
    return norm


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, *, dropout, pre_lnorm=False, ln_eps=1e-5, norm_type='ln'):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = get_norm(norm_type, d_model, ln_eps)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = self.CoreNet(inp)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class AdaptiveFFN(nn.Module):
    def __init__(self, d_model, d_inner, *, dropout, ln_eps=1e-5, gate_mix_prob=0.5, gate_sample_prob=0.25,
                 identity_loss_weight=1.0, identity_threshold=0.9, init_identiy_bias=2.0,
                 ffn_residual=True, norm_in_ffn=True):
        super(AdaptiveFFN, self).__init__()

        self.ffn_net = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_residual = ffn_residual

        self.layer_norm = nn.LayerNorm(d_model, eps=ln_eps)
        self.norm_in_ffn = norm_in_ffn

        self.gate_net = GateNet(d_model, gate_num=2, init_identiy_bias=init_identiy_bias)
        self.gate_mix_prob = gate_mix_prob
        self.gate_sample_prob = gate_sample_prob
        assert 0 <= self.gate_mix_prob <= 1.0
        assert 0 <= self.gate_sample_prob <= 1.0
        assert 0 <= self.gate_mix_prob + self.gate_sample_prob <= 1.0
        self.identity_threshold = identity_threshold

    def forward(self, inputs, *, pad_mask):
        # [T, B, D] => [T, B, gate_num]
        gate_prob = self.gate_net(inputs)
        # paddings may leads to NAN after softmax
        # gate_prob = gate_prob.masked_fill(pad_mask.unsqueeze(2), 0.)
        identity_prob, ffn_prob = torch.chunk(gate_prob, 2, dim=-1)

        ffn_output = self._ffn_forward(inputs)

        if self.training:
            r = random.random()
            if r < self.gate_mix_prob:
                # learn gate
                output = inputs * identity_prob + ffn_output * ffn_prob
                adaptive_prob = identity_prob
            elif r < self.gate_mix_prob + self.gate_sample_prob:
                # exploit
                identity_mask = torch.bernoulli(identity_prob)
                output = inputs * identity_mask + ffn_output * (1 - identity_mask)
                adaptive_prob = None
            else:
                # explore, by uniform sample branches
                mask_size = (inputs.shape[0], inputs.shape[1], 1)
                identity_mask = torch.bernoulli(torch.full(mask_size, 0.5, device=inputs.device)).type(inputs.dtype)
                output = inputs * identity_mask + ffn_output * (1 - identity_mask)
                adaptive_prob = None
        else:
            identity_mask = identity_prob > self.identity_threshold
            output = inputs * identity_mask + ffn_output * ~identity_mask
            adaptive_prob = identity_prob

        if not self.norm_in_ffn:
            output = self.layer_norm(output)

        return output, adaptive_prob

    def _ffn_forward(self, inp):
        output = inp

        output = self.ffn_net(output)
        if self.ffn_residual:
            output = output + inp

        if self.norm_in_ffn:
            output = self.layer_norm(output)

        return output


class GateNet(nn.Module):
    def __init__(self, in_features, gate_num, init_identiy_bias):
        super(GateNet, self).__init__()

        assert gate_num == 2
        self.weight = nn.Parameter(torch.Tensor(gate_num, in_features))
        self.bias = nn.Parameter(torch.tensor([init_identiy_bias] + [0.] * (gate_num - 1)))

    def forward(self, inputs):
        logits = F.linear(inputs, self.weight, self.bias)
        prob = F.softmax(logits, dim=-1)
        return prob


def _rel_shift_uni(x):
    # x: qlen x rlen x bsz x n_head
    zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                           device=x.device, dtype=x.dtype)
    x_padded = torch.cat([zero_pad, x], dim=1)

    x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

    x = x_padded[1:].view_as(x)

    # if zero_triu:
    #     ones = torch.ones((x.size(0), x.size(1)))
    #     x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

    return x


def _rel_shift_bi(x, klen):
    # x: qlen x rlen x bsz x n_head
    """perform relative shift to form the relative attention score."""
    x_size = x.size()
    assert klen * 2 == x_size[1]

    x = x.reshape(x_size[1], x_size[0], x_size[2], x_size[3])
    x = torch.narrow(x, dim=0, start=1, length=x.size()[0]-1)
    x = x.reshape(x_size[0], x_size[1] - 1, x_size[2], x_size[3])
    x = torch.narrow(x, dim=1, start=0, length=klen)

    return x


def check_rel_shift_bi():
    """in x, 14 means query 1, rel emb at position 4, -32 mean query 3, rel emb at position -2"""
    x = torch.tensor([[14, 13, 12, 11, 10, -11, -12, -13],
                      [24, 23, 22, 21, 20, -21, -22, -23],
                      [34, 33, 32, 31, 30, -31, -32, -33],
                      [44, 43, 42, 41, 40, -41, -42, -43]], dtype=torch.float32)
    x = x.unsqueeze(-1).unsqueeze(-1)
    shifted_x = _rel_shift_bi(x, klen=4)
    shifted_x = shifted_x.squeeze(-1).squeeze(-1)
    assert torch.equal(shifted_x,
                       torch.tensor([[10., -11., -12., -13.],
                                     [21.,  20., -21., -22.],
                                     [32.,  31.,  30., -31.],
                                     [43.,  42.,  41.,  40.]]))
    return shifted_x


def create_pad_mask(lens, max_len):
    mask = torch.arange(max_len).to(lens.device) >= lens.unsqueeze(-1)
    return mask


class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, *, dropout, dropatt, pre_lnorm=False, ln_eps=1e-5, uni_attn=True,
                 norm_type='ln', pos_enc='xl'):
        super(RelPartialLearnableMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = get_norm(norm_type, d_model, ln_eps)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

        self.uni_attn = uni_attn

        self.pos_enc = pos_enc
        if self.pos_enc == 'xl':
            self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)
        else:
            assert self.pos_enc is None

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, bsz = w.size(0), w.size(1)

        if self.pre_lnorm:
            w_norm = self.layer_norm(w)
        else:
            w_norm = w
        w_heads = self.qkv_net(w_norm)

        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

        if mems is not None:
            assert self.uni_attn
            k_mems, v_mems, real_mlen = mems
            new_mems = w_head_k, w_head_v

            w_head_k = torch.cat([k_mems, w_head_k], 0)
            w_head_v = torch.cat([v_mems, w_head_v], 0)
        else:
            new_mems = None

        #### compute attention score
        if self.pos_enc == 'xl':
            rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head
        else:
            rw_head_q = w_head_q
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head

        if self.pos_enc == 'xl':
            rr_head_q = w_head_q + r_r_bias
            r_head_k = self.r_net(r)
            r_head_k = r_head_k.view(r.size(0), self.n_head, self.d_head)                # qlen x n_head x d_head
            BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
            BD = self._rel_shift(BD, attn_score.size(1))
            # [qlen x klen x bsz x n_head]
            attn_score = attn_score + BD

        attn_score.mul_(self.scale)

        neg_min = float16_min if attn_score.dtype == torch.float16 else float32_min
        #### compute attention probability
        if self.uni_attn:
            # attn_mask: [qlen, klen, 1] -> [qlen, klen, 1, 1]
            attn_score = attn_score.float().masked_fill(
                attn_mask.unsqueeze(-1), neg_min).type_as(attn_score)
        else:
            # attn_mask: [klen, bsz] -> [1, klen, bsz, 1]
            attn_score = attn_score.masked_fill(attn_mask.unsqueeze(0).unsqueeze(-1), neg_min)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output, new_mems

    def _rel_shift(self, x, klen):
        if self.uni_attn:
            return _rel_shift_uni(x)
        else:
            return _rel_shift_bi(x, klen)


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, *, dropout, dropatt, pre_lnorm, ln_eps, uni_attn, norm_type='ln',
                 pos_enc='xl', adaptive_ffn=None):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model, d_head, dropout=dropout, dropatt=dropatt,
                                                         pre_lnorm=pre_lnorm, ln_eps=ln_eps, uni_attn=uni_attn,
                                                         norm_type=norm_type, pos_enc=pos_enc)
        self.use_adaptive_ffn = adaptive_ffn is not None
        if not self.use_adaptive_ffn:
            self.pos_ff = PositionwiseFF(d_model, d_inner, dropout=dropout, pre_lnorm=pre_lnorm, ln_eps=ln_eps, norm_type=norm_type)
        else:
            assert not pre_lnorm and norm_type == 'ln'
            self.pos_ff = AdaptiveFFN(d_model, d_inner, dropout=dropout, ln_eps=ln_eps, **adaptive_ffn)

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, *, dec_attn_mask=None, pad_mask=None, mems=None):

        output, new_mems = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        if not self.use_adaptive_ffn:
            output = self.pos_ff(output)
            ada_prob = None
        else:
            output, ada_prob = self.pos_ff(output, pad_mask=pad_mask)

        return output, new_mems, (ada_prob,)


class RelTransformerBlock(nn.Module):
    def __init__(self, n_layer, d_model, n_head, d_head, d_inner, dropout, dropout_att,
                 pre_lnorm=False, norm_output=False, ln_eps=1e-5, uni_attn=True, norm_type='ln', pos_enc='xl',
                 layer_drop=0.0, adaptive_ffn=None):
        super(RelTransformerBlock, self).__init__()

        self.n_layer = n_layer
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.drop = nn.Dropout(dropout)

        self.uni_attn = uni_attn
        self.att_trunc_len = -1

        self.clamp_len = 0

        self.layer_drop = layer_drop

        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(
                RelPartialLearnableDecoderLayer(
                    n_head, d_model, d_head, d_inner, dropout=dropout,
                    dropatt=dropout_att, pre_lnorm=pre_lnorm, ln_eps=ln_eps, uni_attn=uni_attn,  norm_type=norm_type,
                    pos_enc=pos_enc, adaptive_ffn=adaptive_ffn)
            )

        if norm_output:
            self.output_norm = nn.LayerNorm(d_model, eps=ln_eps)
        else:
            self.output_norm = identity

        self.use_adaptive_ffn = adaptive_ffn is not None
        if self.use_adaptive_ffn:
            self.identity_loss_weight = adaptive_ffn.identity_loss_weight

        self.pos_enc = pos_enc
        self._create_params()

    def _create_params(self):
        if self.pos_enc == 'xl':
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        else:
            assert self.pos_enc is None
            self.pos_emb = None
            self.r_w_bias = None
            self.r_r_bias = None

    def _forward(self, dec_inp, lens=None, mems=None):
        is_decoding = mems is not None
        qlen, bsz, _ = dec_inp.size()

        mlen = mems[0][0].size(0) if mems is not None else 0
        klen = mlen + qlen

        if not self.uni_attn or self.use_adaptive_ffn:
            assert lens is not None
            pad_mask = create_pad_mask(lens, max_len=qlen)
            # [B, L] -> [L, B]
            pad_mask = pad_mask.transpose(0, 1).contiguous()
        else:
            pad_mask = None

        if self.uni_attn:
            dec_attn_mask = torch.triu(
                dec_inp.new_ones(qlen, klen), diagonal=1+mlen).bool()[:,:,None]
        else:
            assert pad_mask is not None
            dec_attn_mask = pad_mask

        hids = []
        new_kv_mems = []
        core_out = dec_inp

        if self.pos_enc == 'xl':
            if self.uni_attn:
                pos_s, pos_e = klen-1, -1
            else:
                pos_s, pos_e = klen, -qlen
            pos_seq = torch.arange(pos_s, pos_e, -1.0, device=dec_inp.device,
                                   dtype=dec_inp.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            pos_emb = self.drop(pos_emb)
        else:
            pos_emb = None

        if self.use_adaptive_ffn:
            adaptive_prob = []
        else:
            adaptive_prob = None

        # hids.append(core_out)
        for i, layer in enumerate(self.layers):
            if self.layer_drop > 0 and self.training and np.random.random() < self.layer_drop:
                continue

            mems_i = None if not is_decoding else mems[i]
            core_out, kv_mems, extra = layer(core_out, pos_emb, self.r_w_bias,
                    self.r_r_bias, dec_attn_mask=dec_attn_mask, pad_mask=pad_mask, mems=mems_i)
            # hids.append(core_out)
            new_kv_mems.append(kv_mems)

            if self.use_adaptive_ffn:
                if extra[0] is not None:
                    adaptive_prob.append(extra[0])

        core_out = self.output_norm(core_out)

        if self.use_adaptive_ffn:
            if len(adaptive_prob) == 0:
                assert self.training
                adaptive_ffn_loss = torch.tensor(0., dtype=dec_inp.dtype, device=dec_inp.device)
            else:
                adaptive_prob_t = torch.stack(adaptive_prob)
                zero_guard_eps = 1e-12
                adaptive_logp = torch.log(adaptive_prob_t + zero_guard_eps)
                # pad_mask: [L, B] => [1, L, B, 1]
                adaptive_logp = adaptive_logp.masked_fill(pad_mask.unsqueeze(-1).unsqueeze(0), 0.)
                avg_adaptive_logp = adaptive_logp.sum() / (len(adaptive_prob) * lens.sum())
                adaptive_ffn_loss = -avg_adaptive_logp * self.identity_loss_weight
        else:
            adaptive_ffn_loss = None

        new_mems = []
        if is_decoding:
            new_mems = self._update_decode_mems(new_kv_mems, mems, [self.att_trunc_len] * len(self.layers))

        return core_out, new_mems, (adaptive_ffn_loss,)

    def forward(self, data, *, lens=None, mems=None):
        # data: [T, B, D]
        output, new_mems, extra = self._forward(data, lens=lens, mems=mems)

        if mems is None:
            assert len(new_mems) == 0

        return output, tuple(new_mems), extra

    def _update_decode_mems(self, step_mem, prev_mems, mem_len):
        assert prev_mems is not None and len(step_mem) == len(prev_mems)

        with torch.no_grad():
            new_mems = []
            for i in range(len(step_mem)):
                mem_len_i = mem_len[i]
                k_mem, v_mem = step_mem[i]
                k_prev_mem, v_prev_mem, real_mlen = prev_mems[i]
                new_k_mem = torch.cat([k_prev_mem, k_mem], 0)
                new_v_mem = torch.cat([v_prev_mem, v_mem], 0)
                real_mlen = real_mlen + k_mem.size(0)
                if mem_len_i > 0:
                    new_k_mem = self.neg_slice(new_k_mem, mem_len, True)
                    new_v_mem = self.neg_slice(new_v_mem, mem_len, True)
                    real_mlen = torch.min(real_mlen, mem_len)
                new_mems.append((new_k_mem, new_v_mem, real_mlen))
        return new_mems


def identity(x):
    return x
