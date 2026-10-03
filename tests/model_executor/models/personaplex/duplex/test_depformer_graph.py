# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from tests.model_executor.models.personaplex.duplex.test_stage0_runtime import (
    _duplex_info,
    _FakeCodec,
)
from tests.model_executor.models.personaplex.duplex.test_stage0_runtime import (
    _runtime as _make_runtime,
)
from vllm_omni.model_executor.models.personaplex.configuration_personaplex import PersonaPlexDepformerConfig
from vllm_omni.model_executor.models.personaplex.duplex.stage0 import PersonaPlexStage0DuplexRuntime
from vllm_omni.model_executor.models.personaplex.personaplex_depformer import PersonaPlexDepformer
from vllm_omni.model_executor.models.personaplex.personaplex_depformer_graph import PersonaPlexDepformerGraphs
from vllm_omni.model_executor.models.personaplex.personaplex_talker import PersonaPlexTalkerForConditionalGeneration

pytestmark = pytest.mark.core_model

HIDDEN = 24
NUM_STEPS = 8


def _runtime(device: torch.device, max_sessions: int) -> PersonaPlexStage0DuplexRuntime:
    rt = _make_runtime(_FakeCodec(), max_sessions, str(device))
    rt.load_encoder()
    return rt


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
        depformer, runtime, buckets=buckets, num_steps=NUM_STEPS, hidden_size=HIDDEN, dtype=dtype, device=runtime.device
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


def _assert_same_state(left, right) -> None:
    live = left.max_sessions
    for a, b in zip(
        (left._last_text[:live], left._last_agent[:live], left._teacher_tokens, left._teacher_provided),
        (right._last_text[:live], right._last_agent[:live], right._teacher_tokens, right._teacher_provided),
        strict=True,
    ):
        assert torch.equal(a.cpu(), b.cpu())
    for key, state in left.sessions.items():
        assert right.sessions[key].sampled_identity == state.sampled_identity


@pytest.mark.cpu
def test_step_commits_exactly_like_record_samples() -> None:
    device = torch.device("cpu")
    depformer = _depformer(device)
    graph_runtime, reference = _runtime(device, 4), _runtime(device, 4)
    graphs = _graphs(depformer, graph_runtime, [1, 2, 4])
    sessions = ["a", "b", "c"]
    request_ids = [*sessions, "a"]

    for seq in (1, 2):
        _prepare(graph_runtime, sessions, seq)
        _prepare(reference, sessions, seq)
        text, hidden = _step_inputs(4, device, seed=seq)
        codes = graphs.run(request_ids, text, hidden)
        expected = _reference_step(reference, depformer, request_ids, text, hidden)
        assert torch.equal(codes, expected)
        _assert_same_state(graph_runtime, reference)


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
    talker.depformer, talker._personaplex_duplex_stage0_runtime = depformer, runtime
    talker._dtype, talker.mtp_hidden_size, talker.num_active_codebooks = torch.float32, HIDDEN, NUM_STEPS
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
