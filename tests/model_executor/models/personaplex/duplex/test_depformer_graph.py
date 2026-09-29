# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from tests.model_executor.models.personaplex.duplex.test_stage0_runtime import _duplex_info
from vllm_omni.model_executor.models.personaplex.configuration_personaplex import (
    PersonaPlexDepformerConfig,
)
from vllm_omni.model_executor.models.personaplex.duplex.stage0 import PersonaPlexStage0DuplexRuntime
from vllm_omni.model_executor.models.personaplex.personaplex_depformer import PersonaPlexDepformer
from vllm_omni.model_executor.models.personaplex.personaplex_depformer_graph import (
    PersonaPlexDepformerGraphs,
)
from vllm_omni.model_executor.models.personaplex.personaplex_talker import (
    PersonaPlexTalkerForConditionalGeneration,
)

pytestmark = pytest.mark.core_model

HIDDEN = 24
NUM_STEPS = 8


class _FakeCodec:
    """Shared streaming encoder whose code for a row is that row's frame count."""

    def __init__(self) -> None:
        self.frames: list[int] = []

    def streaming_init(self, batch_size: int) -> None:
        self.frames = [0] * batch_size

    def encode_frame(self, pcm: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        codes = torch.zeros((len(self.frames), 8), dtype=torch.long)
        for row, is_active in enumerate(active.tolist()):
            if is_active:
                self.frames[row] += 1
                codes[row] = 100 + 7 * self.frames[row] + row
        # Like the real codec, the codes come back on the device of its input.
        return codes.to(pcm.device)

    def reset_slot(self, row: int) -> None:
        self.frames[row] = 0


class _FakeTalker:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.dtype = torch.float32

    def _build_prefill_embed(self, tokens, offset, span, device, silence=None, user_sine=None):
        del offset, silence, user_sine
        return tokens[:span].to(device=device, dtype=torch.float32)[:, None].expand(-1, 4).contiguous()

    def _build_frame_embeds(self, text_tokens, last_agent, *, user_d0, user_d1):
        del text_tokens, last_agent, user_d1
        return user_d0[:, :1].to(torch.float32).expand(-1, 4).contiguous()


def _runtime(device: torch.device, max_sessions: int) -> PersonaPlexStage0DuplexRuntime:
    voice_embeddings = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    runtime = PersonaPlexStage0DuplexRuntime(
        _FakeTalker(device),
        model_path="/unused",
        device=str(device),
        codec=_FakeCodec(),
        max_sessions=max_sessions,
        tokenizer=lambda _text: [7, 8, 9],
        voice_loader=lambda _voice: {"embeddings": voice_embeddings},
    )
    runtime.load_encoder()
    return runtime


def _depformer(device: torch.device, dtype: torch.dtype = torch.float32) -> PersonaPlexDepformer:
    torch.manual_seed(0)
    config = PersonaPlexDepformerConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=8,
        num_key_value_heads=4,
        intermediate_size=48,
        dep_q=NUM_STEPS,
        num_active_codebooks=NUM_STEPS,
        card=2048,
    )
    model = PersonaPlexDepformer(config, temporal_hidden_size=HIDDEN, text_card=100)
    for param in model.parameters():
        torch.nn.init.normal_(param, std=0.3)
    return model.to(device=device, dtype=dtype).eval()


def _graphs(depformer, runtime, buckets, *, dtype=torch.float32) -> PersonaPlexDepformerGraphs:
    return PersonaPlexDepformerGraphs(
        depformer,
        runtime,
        buckets=buckets,
        num_steps=NUM_STEPS,
        hidden_size=HIDDEN,
        dtype=dtype,
        device=runtime.device,
    )


def _prepare(runtime: PersonaPlexStage0DuplexRuntime, sessions: list[str], seq: int) -> None:
    runtime.encode_appends([_duplex_info(seq=seq, session_id=session) for session in sessions])
    for session in sessions:
        runtime.prepare_append(_duplex_info(seq=seq, session_id=session), prompt_len=17 + seq, request_id=session)


def _reference_step(runtime, depformer, request_ids, text, hidden) -> torch.Tensor:
    """The unpadded post-sample path: gather, depformer, record_samples."""
    tokens, provided = runtime.depformer_teacher_forcing(request_ids)
    codes = depformer(text, hidden, audio_tokens=tokens, audio_provided=provided, num_steps=NUM_STEPS)
    runtime.record_samples(request_ids=request_ids, text_tokens=text, agent_codes=codes)
    return codes.cpu()


def _step_inputs(rows: int, device, dtype=torch.float32, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    text = torch.randint(0, 100, (rows,), generator=generator)
    hidden = torch.randn(rows, 1, HIDDEN, generator=generator)
    return text.to(device), hidden.to(device=device, dtype=dtype)


def _frame_state(runtime) -> tuple[torch.Tensor, ...]:
    live = runtime.max_sessions
    return (
        runtime._last_text[:live].cpu(),
        runtime._last_agent[:live].cpu(),
        runtime._teacher_tokens.cpu(),
        runtime._teacher_provided.cpu(),
    )


def _assert_same_state(left, right) -> None:
    for a, b in zip(_frame_state(left), _frame_state(right), strict=True):
        assert torch.equal(a, b)
    for key, state in left.sessions.items():
        assert right.sessions[key].sampled_identity == state.sampled_identity


@pytest.mark.cpu
def test_step_commits_exactly_like_record_samples() -> None:
    device = torch.device("cpu")
    depformer = _depformer(device)
    graph_runtime, reference = _runtime(device, 4), _runtime(device, 4)
    graphs = _graphs(depformer, graph_runtime, [1, 2, 4])
    sessions = ["a", "b", "c"]
    # A session that reappears in the batch commits once, from its first row.
    request_ids = [*sessions, "a"]

    for seq in (1, 2, 3):
        _prepare(graph_runtime, sessions, seq)
        _prepare(reference, sessions, seq)
        text, hidden = _step_inputs(4, device, seed=seq)
        codes = graphs.run(request_ids, text, hidden)
        expected = _reference_step(reference, depformer, request_ids, text, hidden)
        assert codes.device.type == "cpu"
        assert torch.equal(codes, expected)
        _assert_same_state(graph_runtime, reference)

    # A second post-sample step of an already committed frame changes nothing.
    text, hidden = _step_inputs(4, device, seed=9)
    before = _frame_state(graph_runtime)
    graphs.run(request_ids, text, hidden)
    for a, b in zip(before, _frame_state(graph_runtime), strict=True):
        assert torch.equal(a, b)


@pytest.mark.cpu
def test_a_superseded_epoch_row_reads_neutral_and_commits_to_scratch() -> None:
    device = torch.device("cpu")
    depformer = _depformer(device)
    graph_runtime, reference = _runtime(device, 1), _runtime(device, 1)
    graphs = _graphs(depformer, graph_runtime, [1, 2])
    old, new = _duplex_info(seq=1, session_id="s", epoch=0), _duplex_info(seq=1, session_id="s", epoch=1)
    for runtime in (graph_runtime, reference):
        runtime.encode_appends([old, new])
        with pytest.raises(Exception, match="superseded"):
            runtime.prepare_append(old, prompt_len=18, request_id="old")
        runtime.prepare_append(new, prompt_len=18, request_id="new")

    text, hidden = _step_inputs(2, device)
    codes = graphs.run(["old", "new"], text, hidden)

    assert torch.equal(codes, _reference_step(reference, depformer, ["old", "new"], text, hidden))
    _assert_same_state(graph_runtime, reference)
    assert graph_runtime.sessions[("s", 1)].sampled_identity == (1, 1)


def _bare_talker(depformer, runtime) -> PersonaPlexTalkerForConditionalGeneration:
    talker = PersonaPlexTalkerForConditionalGeneration.__new__(PersonaPlexTalkerForConditionalGeneration)
    torch.nn.Module.__init__(talker)
    talker.depformer = depformer
    talker._personaplex_duplex_stage0_runtime = runtime
    talker._dtype = torch.float32
    talker.mtp_hidden_size = HIDDEN
    talker.num_active_codebooks = NUM_STEPS
    return talker


@pytest.mark.cpu
def test_talker_builds_graphs_at_vllm_capture_sizes_and_runs_through_them() -> None:
    device = torch.device("cpu")
    depformer = _depformer(device)
    runtime, reference = _runtime(device, 2), _runtime(device, 2)
    talker = _bare_talker(depformer, runtime)
    talker.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 2, 4, 8, 16, 24]),
    )

    talker._depformer_graphs = talker._build_depformer_graphs(runtime)
    assert talker._depformer_graphs.buckets == [1, 2, 4, 8]

    _prepare(runtime, ["a", "b"], 1)
    _prepare(reference, ["a", "b"], 1)
    text, hidden = _step_inputs(2, device)
    codes = talker.post_sample_talker_mtp(
        input_ids=text,
        hidden_states=hidden.reshape(2, HIDDEN),
        req_ids=["a", "b"],
        req_infos=[{}, {}],
    )
    assert torch.equal(codes, _reference_step(reference, depformer, ["a", "b"], text, hidden))
    _assert_same_state(runtime, reference)


_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.cuda
@_CUDA
def test_graph_replay_matches_padded_eager() -> None:
    rows = 3
    device = torch.device("cuda")
    dtype = torch.bfloat16
    depformer = _depformer(device, dtype)
    buckets = [1, 2, 4, 8]
    graph_runtime, eager_runtime = _runtime(device, 8), _runtime(device, 8)
    graphs = _graphs(depformer, graph_runtime, buckets, dtype=dtype)
    eager = _graphs(depformer, eager_runtime, buckets, dtype=dtype)
    assert graphs.capture() is True
    assert sorted(graphs._graphs) == buckets
    sessions = [f"s{i}" for i in range(rows)]

    # Four steps wrap the ring of pinned slot-upload buffers.
    for seq in (1, 2, 3, 4):
        _prepare(graph_runtime, sessions, seq)
        _prepare(eager_runtime, sessions, seq)
        # Padding rows hold garbage: NaN hidden and arbitrary token ids.
        graphs._hidden.fill_(float("nan"))
        graphs._text.fill_(97)
        text, hidden = _step_inputs(rows, device, dtype, seed=seq)
        codes = graphs.run(sessions, text, hidden)
        assert codes.device.type == "cuda"
        assert torch.equal(codes, eager.run(sessions, text, hidden))
        _assert_same_state(graph_runtime, eager_runtime)


@pytest.mark.cuda
@_CUDA
def test_graph_rows_above_the_largest_bucket_match_the_unpadded_path() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    depformer = _depformer(device, dtype)
    graph_runtime, reference = _runtime(device, 3), _runtime(device, 3)
    graphs = _graphs(depformer, graph_runtime, [1, 2], dtype=dtype)
    assert graphs.capture() is True
    sessions = ["a", "b", "c"]
    for seq in (1, 2):
        _prepare(graph_runtime, sessions, seq)
        _prepare(reference, sessions, seq)
        text, hidden = _step_inputs(3, device, dtype, seed=seq)
        codes = graphs.run(sessions, text, hidden)
        # Rows above the largest bucket run eagerly, and like a replay they
        # hand back device codes; the reference is already on the host.
        assert codes.device.type == "cuda"
        assert torch.equal(codes.cpu(), _reference_step(reference, depformer, sessions, text, hidden))
        _assert_same_state(graph_runtime, reference)
