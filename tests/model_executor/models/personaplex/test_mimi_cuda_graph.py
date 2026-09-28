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


def _reference_conv_step(state: SimpleNamespace, x: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """The pre-refactor ``_StreamConv1d.__call__``, which reassigned its carry."""
    if state.pad_mode == "replicate":
        pad = state.prev.shape[-1]
        edge = x[..., 0:1].expand(-1, -1, pad)
        fresh = (state.fresh & active).view(-1, 1, 1)
        state.prev = torch.where(fresh, edge.to(state.prev.dtype), state.prev)
    state.fresh[active] = False
    x = torch.cat([state.prev, x], dim=-1)
    t = x.shape[-1]
    num_frames = max(0, (t - state.kernel) // state.stride + 1)
    prev = x[..., num_frames * state.stride :]
    state.prev = torch.where(active.view(-1, 1, 1), prev, state.prev)
    if num_frames == 0:
        return x.new_zeros(x.shape[0], state.conv.out_channels, 0)
    return state.conv(x[..., : (num_frames - 1) * state.stride + state.kernel])


def _reference_convtr_step(state: SimpleNamespace, x: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """The pre-refactor ``_StreamConvTr1d.__call__``, which reassigned its carry."""
    out = state.conv(x)
    length = out.shape[-1]
    tail = state.kernel - state.stride
    pt = state.partial.shape[-1]
    merge = state.partial.clone()
    if state.conv.bias is not None:
        merge = merge - state.conv.bias[:, None]
        merge[state.fresh & active] = 0.0
        state.fresh[active] = False
    out[..., :pt] += merge
    partial = out[..., length - tail :].clone()
    state.partial = torch.where(active.view(-1, 1, 1), partial, state.partial)
    return out[..., : length - tail]


def _conv(kind: str, bias: bool) -> nn.Module:
    torch.manual_seed(SEED)
    if kind == "convtr":
        return nn.ConvTranspose1d(2, 3, kernel_size=4, stride=2, bias=bias)
    return nn.Conv1d(2, 3, kernel_size=4, stride=2, bias=bias)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("kind", "pad_mode", "bias"),
    [
        ("conv", "constant", True),
        ("conv", "replicate", True),
        ("convtr", None, True),
        ("convtr", None, False),
    ],
)
def test_stream_conv_in_place_carry_matches_reassigning_reference(kind: str, pad_mode: str | None, bias: bool) -> None:
    conv = _conv(kind, bias)
    batch_size, samples = 3, 4
    if kind == "convtr":
        stream = _StreamConvTr1d(conv)
    else:
        stream = _StreamConv1d(conv, pad_mode=pad_mode)
    stream.reset(batch_size, torch.device("cpu"), torch.float32)
    carry = stream.partial if kind == "convtr" else stream.prev
    reference = SimpleNamespace(
        conv=conv,
        kernel=stream.kernel,
        stride=stream.stride,
        pad_mode=pad_mode,
        fresh=stream._fresh.clone(),
    )
    if kind == "convtr":
        reference.partial = stream.partial.clone()
    else:
        reference.prev = stream.prev.clone()
    carry_ptr, fresh_ptr = carry.data_ptr(), stream._fresh.data_ptr()

    generator = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for rows in ACTIVE_SCHEDULE:
            active = _mask(rows)
            x = torch.randn(batch_size, 2, samples, generator=generator)
            out = stream(x, active)
            if kind == "convtr":
                expected = _reference_convtr_step(reference, x.clone(), active)
                expected_carry = reference.partial
            else:
                expected = _reference_conv_step(reference, x.clone(), active)
                expected_carry = reference.prev
            torch.testing.assert_close(out, expected, rtol=0.0, atol=0.0)
            torch.testing.assert_close(carry, expected_carry, rtol=0.0, atol=0.0)
            assert torch.equal(stream._fresh, reference.fresh)

    # The state never moved to new storage, which is what a CUDA graph replay needs.
    assert (stream.partial if kind == "convtr" else stream.prev).data_ptr() == carry_ptr
    assert stream._fresh.data_ptr() == fresh_ptr


class _ArgmaxQuantizer(MimiSplitResidualVectorQuantizer):
    """Mimi's split RVQ at test size, over random codebooks.

    Decode is the real Transformers codebook lookup; encode is a deterministic
    stand-in, an argmax over a projection per codebook.
    """

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
        codes = torch.einsum("qcd,bdt->qbtc", self.proj, x).argmax(dim=-1)  # [Q, B, T]
        return codes if num_quantizers is None else codes[:num_quantizers]


def _make_small_codec(device: torch.device, batch_size: int) -> PersonaPlexMimiCodec:
    """A PersonaPlexMimiCodec over the real streaming stages, small enough for a unit test.

    The encoder maps a 1920-sample frame to one code frame with the same stride
    structure as Mimi (4 * 5 * 6 * 8 then a stride-2 resampler); the decoder
    maps it back (a stride-2 resampler then 8 * 6 * 5 * 4).
    """
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
    codec.streaming_init(batch_size)
    return codec


def _assert_same_streaming_state(a: PersonaPlexMimiCodec, b: PersonaPlexMimiCodec) -> None:
    for state_a, state_b in zip(a._conv_states(), b._conv_states(), strict=True):
        carry_a = state_a.partial if isinstance(state_a, _StreamConvTr1d) else state_a.prev
        carry_b = state_b.partial if isinstance(state_b, _StreamConvTr1d) else state_b.prev
        assert torch.equal(carry_a, carry_b)
        assert torch.equal(state_a._fresh, state_b._fresh)
    transformers = [
        (a.encoder_transformer, b.encoder_transformer),
        (a.decoder_transformer, b.decoder_transformer),
    ]
    for transformer_a, transformer_b in transformers:
        for kv_a, kv_b in zip(transformer_a._kv, transformer_b._kv, strict=True):
            assert torch.equal(kv_a.end_offset, kv_b.end_offset)
            assert torch.equal(kv_a.start_offset, kv_b.start_offset)
        assert torch.equal(transformer_a._offset, transformer_b._offset)


@pytest.mark.cpu
def test_dequantize_matches_transformers_decode() -> None:
    codec = _make_small_codec(torch.device("cpu"), batch_size=3)
    generator = torch.Generator().manual_seed(SEED)
    codes = torch.randint(0, CARD, (3, 8, 2), generator=generator)

    with torch.no_grad():
        want = codec.model.quantizer.decode(codes)
        got = codec._dequantize(codes)

    assert want.abs().sum() > 0
    assert torch.equal(got, want)


class _GraphCodec:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.captures = 0

    def streaming_init(self, batch_size: int) -> None:
        self.batch_sizes.append(batch_size)

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
    assert codecs[0].batch_sizes == [4]
    assert codecs[0].captures == int(cuda_graph)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["encode", "decode"])
def test_graph_replay_matches_eager(kind: str) -> None:
    batch_size = 3
    graphed = _make_small_codec(CUDA_DEVICE, batch_size)
    eager = _make_small_codec(CUDA_DEVICE, batch_size)

    assert getattr(graphed, f"capture_{kind}_graph")()
    # Warmup advanced every row; capture leaves the codec at a fresh stream.
    _assert_same_streaming_state(graphed, eager)

    generator = torch.Generator().manual_seed(SEED)
    # Stage 0 hands the encoder host PCM and a host mask; Code2Wav hands the
    # decoder device codes and a device mask.
    mask_device = "cpu" if kind == "encode" else CUDA_DEVICE

    def frame() -> torch.Tensor:
        if kind == "encode":
            return torch.randn(batch_size, FRAME_SIZE, generator=generator)
        return torch.randint(0, CARD, (batch_size, 8), generator=generator).to(CUDA_DEVICE)

    run_graphed, run_eager = (getattr(codec, f"{kind}_frame") for codec in (graphed, eager))
    for rows in ACTIVE_SCHEDULE * 2:
        x = frame()
        active = _mask(rows, mask_device)
        assert torch.equal(run_graphed(x, active), run_eager(x, active))
        _assert_same_streaming_state(graphed, eager)

    # A recycled row restarts like a fresh stream under replay as well.
    graphed.reset_slot(1)
    eager.reset_slot(1)
    x = frame()
    assert torch.equal(run_graphed(x, _mask((True, True, True), mask_device)), run_eager(x, None))
    _assert_same_streaming_state(graphed, eager)
