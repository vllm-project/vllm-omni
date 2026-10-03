# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import MimiConfig
from transformers.models.mimi.modeling_mimi import MimiSplitResidualVectorQuantizer

from vllm_omni.model_executor.models.personaplex.duplex.stage0 import (
    PersonaPlexStage0DuplexRuntime,
)
from vllm_omni.model_executor.models.personaplex.personaplex_mimi import (
    FRAME_SIZE,
    PersonaPlexMimiCodec,
    _MimiStreamingTransformer,
    _StreamConv1d,
    _StreamConvTr1d,
)

pytestmark = pytest.mark.core_model

SEED = 4321
CUDA_DEVICE = torch.device("cuda")
DIM = 16
CARD = 32
ACTIVE_SCHEDULE = [
    (True, True, True),
    (True, False, True),
    (False, False, False),
    (False, True, True),
    (True, True, False),
    (True, True, True),
]


def _mask(rows: tuple[bool, ...], device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.bool, device=device)


class _ArgmaxQuantizer(MimiSplitResidualVectorQuantizer):
    """Mimi's split RVQ at test size, over random codebooks."""

    def __init__(self, dim: int, card: int = CARD, codebooks: int = 8) -> None:
        config = MimiConfig(
            hidden_size=dim,
            codebook_size=card,
            codebook_dim=8,
            vector_quantization_hidden_dimension=8,
            num_quantizers=codebooks,
        )
        super().__init__(config)
        self.proj = nn.Parameter(torch.randn(codebooks, card, dim))
        for rvq in (self.semantic_residual_vector_quantizer, self.acoustic_residual_vector_quantizer):
            for layer in rvq.layers:
                layer.codebook.embed_sum.normal_()
                layer.codebook.cluster_usage.uniform_(0.5, 2.0)

    def encode(self, x: torch.Tensor, num_quantizers: int | None = None) -> torch.Tensor:
        codes = torch.einsum("qcd,bdt->qbtc", self.proj, x).argmax(dim=-1)
        return codes if num_quantizers is None else codes[:num_quantizers]


def _make_small_codec(device: torch.device, batch_size: int, **halves: bool) -> PersonaPlexMimiCodec:
    torch.manual_seed(SEED)
    codec = PersonaPlexMimiCodec.__new__(PersonaPlexMimiCodec)
    nn.Module.__init__(codec)
    codec.device = device
    codec.dtype = torch.float32

    def conv(cin: int, cout: int, kernel: int, stride: int = 1, pad_mode: str = "constant") -> _StreamConv1d:
        return _StreamConv1d(nn.Conv1d(cin, cout, kernel, stride=stride, device=device), pad_mode=pad_mode)

    def convtr(cin: int, cout: int, stride: int) -> _StreamConvTr1d:
        return _StreamConvTr1d(nn.ConvTranspose1d(cin, cout, 2 * stride, stride=stride, device=device))

    codec._enc_stages = [
        ("conv", conv(1, 4, 7)),
        ("act", nn.ELU()),
        ("res", (nn.ELU(), conv(4, 2, 3), nn.ELU(), conv(2, 4, 1))),
        ("conv", conv(4, 8, 8, stride=4)),
        ("conv", conv(8, 8, 10, stride=5)),
        ("conv", conv(8, DIM, 12, stride=6)),
        ("conv", conv(DIM, DIM, 16, stride=8)),
    ]
    codec._downsample = conv(DIM, DIM, 4, stride=2, pad_mode="replicate")
    codec._upsample = _StreamConvTr1d(nn.ConvTranspose1d(DIM, DIM, 4, stride=2, device=device))
    codec._dec_stages = [
        ("conv", conv(DIM, 8, 7)),
        ("act", nn.ELU()),
        ("convtr", convtr(8, 8, stride=8)),
        ("res", (nn.ELU(), conv(8, 4, 3), nn.ELU(), conv(4, 8, 1))),
        ("convtr", convtr(8, 4, stride=6)),
        ("convtr", convtr(4, 4, stride=5)),
        ("convtr", convtr(4, 4, stride=4)),
        ("act", nn.ELU()),
        ("conv", conv(4, 1, 3)),
    ]
    codec.encoder_transformer = _MimiStreamingTransformer(num_layers=2, dim=DIM, num_heads=2, context=8).to(device)
    codec.decoder_transformer = _MimiStreamingTransformer(num_layers=1, dim=DIM, num_heads=2, context=8).to(device)
    for parameter in (*codec.encoder_transformer.parameters(), *codec.decoder_transformer.parameters()):
        nn.init.normal_(parameter, std=0.1)
    codec.model = nn.Module()
    codec.model.quantizer = _ArgmaxQuantizer(DIM).to(device)
    codec.streaming_init(batch_size, **halves)
    return codec


def _assert_same_streaming_state(a: PersonaPlexMimiCodec, b: PersonaPlexMimiCodec) -> None:
    for state_a, state_b in zip(a._conv_states(), b._conv_states(), strict=True):
        carry_a = state_a.partial if isinstance(state_a, _StreamConvTr1d) else state_a.prev
        carry_b = state_b.partial if isinstance(state_b, _StreamConvTr1d) else state_b.prev
        assert torch.equal(carry_a, carry_b)
        assert torch.equal(state_a._fresh, state_b._fresh)
    for transformer_a, transformer_b in (
        (a.encoder_transformer, b.encoder_transformer),
        (a.decoder_transformer, b.decoder_transformer),
    ):
        for kv_a, kv_b in zip(transformer_a._kv, transformer_b._kv, strict=True):
            assert torch.equal(kv_a.end_offset, kv_b.end_offset)
            assert torch.equal(kv_a.start_offset, kv_b.start_offset)
        assert torch.equal(transformer_a._offset, transformer_b._offset)


def _half_tensors(codec: PersonaPlexMimiCodec, half: str) -> list[torch.Tensor]:
    convs, transformer = codec._half_state(half)
    holders = [*convs, transformer, *(transformer._kv or ())]
    return [value for holder in holders for value in vars(holder).values() if isinstance(value, torch.Tensor)]


@pytest.mark.cpu
def test_dequantize_matches_transformers_decode() -> None:
    codec = _make_small_codec(torch.device("cpu"), batch_size=3)
    codes = torch.randint(0, CARD, (3, 8, 2), generator=torch.Generator().manual_seed(SEED))
    with torch.no_grad():
        assert torch.equal(codec._dequantize(codes), codec.model.quantizer.decode(codes))


@pytest.mark.cpu
@pytest.mark.parametrize("half", ["encode", "decode"])
def test_one_sided_codec_matches_full_and_rejects_the_other_half(half: str) -> None:
    batch_size = 3
    other = "decode" if half == "encode" else "encode"
    full = _make_small_codec(torch.device("cpu"), batch_size)
    codec = _make_small_codec(torch.device("cpu"), batch_size, **{other: False})
    generator = torch.Generator().manual_seed(SEED)

    assert [t.shape for t in _half_tensors(codec, half)] == [t.shape for t in _half_tensors(full, half)]
    assert _half_tensors(codec, other) == []

    for step, rows in enumerate(ACTIVE_SCHEDULE * 2):
        if step == 5:
            full.reset_slot(1)
            codec.reset_slot(1)
        elif step == 9:
            full.reset_streaming()
            codec.reset_streaming()
        if half == "encode":
            x = torch.randn(batch_size, FRAME_SIZE, generator=generator)
        else:
            x = torch.randint(0, CARD, (batch_size, 8), generator=generator)
        assert torch.equal(
            getattr(codec, f"{half}_frame")(x, _mask(rows)), getattr(full, f"{half}_frame")(x, _mask(rows))
        )

    if other == "encode":
        with pytest.raises(RuntimeError, match="no encode streaming state"):
            codec.encode_frame(torch.zeros(batch_size, FRAME_SIZE))
    else:
        with pytest.raises(RuntimeError, match="no decode streaming state"):
            codec.decode_frame(torch.zeros(batch_size, 8, dtype=torch.long))
    with pytest.raises(ValueError, match="streaming_init needs"):
        codec.streaming_init(batch_size, encode=False, decode=False)


class _GraphCodec:
    def __init__(self) -> None:
        self.inits: list[tuple[int, bool]] = []
        self.captures = 0

    def streaming_init(self, batch_size: int, *, decode: bool = True) -> None:
        self.inits.append((batch_size, decode))

    def capture_encode_graph(self) -> bool:
        self.captures += 1
        return True


@pytest.mark.cpu
@pytest.mark.parametrize("cuda_graph", [False, True])
def test_load_encoder_builds_the_shared_encoder_once(cuda_graph: bool) -> None:
    codecs: list[_GraphCodec] = []

    def factory() -> _GraphCodec:
        codecs.append(_GraphCodec())
        return codecs[-1]

    runtime = PersonaPlexStage0DuplexRuntime(
        SimpleNamespace(),
        model_path="/unused",
        device="cpu",
        codec_factory=factory,
        max_sessions=4,
    )
    runtime.load_encoder(cuda_graph=cuda_graph)

    assert len(codecs) == 1
    assert codecs[0].inits == [(4, False)]
    assert codecs[0].captures == int(cuda_graph)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["encode", "decode"])
def test_graph_replay_matches_eager(kind: str) -> None:
    batch_size = 3
    graphed = _make_small_codec(CUDA_DEVICE, batch_size)
    eager = _make_small_codec(CUDA_DEVICE, batch_size)
    assert getattr(graphed, f"capture_{kind}_graph")()
    _assert_same_streaming_state(graphed, eager)

    generator = torch.Generator().manual_seed(SEED)
    mask_device = "cpu" if kind == "encode" else CUDA_DEVICE
    run_graphed, run_eager = (getattr(codec, f"{kind}_frame") for codec in (graphed, eager))
    for rows in ACTIVE_SCHEDULE * 2:
        x = (
            torch.randn(batch_size, FRAME_SIZE, generator=generator)
            if kind == "encode"
            else torch.randint(0, CARD, (batch_size, 8), generator=generator).to(CUDA_DEVICE)
        )
        active = _mask(rows, mask_device)
        assert torch.equal(run_graphed(x, active), run_eager(x, active))
        _assert_same_streaming_state(graphed, eager)

    graphed.reset_slot(1)
    eager.reset_slot(1)
    x = (
        torch.randn(batch_size, FRAME_SIZE, generator=generator)
        if kind == "encode"
        else torch.randint(0, CARD, (batch_size, 8), generator=generator).to(CUDA_DEVICE)
    )
    assert torch.equal(run_graphed(x, _mask((True, True, True), mask_device)), run_eager(x, None))
    _assert_same_streaming_state(graphed, eager)
