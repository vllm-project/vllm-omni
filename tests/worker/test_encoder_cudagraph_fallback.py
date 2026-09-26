# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Omni encoder CUDA graph manager runs eager instead of raising.

Batches are packed by output-token count and item count only, so a request can
reach replay with more rows than the captured buffers hold. Upstream copies
into the shorter destination and raises; these tests pin the eager answer, the
miss accounting, and that a batch which does fit still replays.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.worker.encoder_cudagraph import BudgetGraphMetadata, EncoderCudaGraphManager
from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphConfig, EncoderItemSpec
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

import vllm_omni.worker.encoder_cudagraph as omni_encoder_cudagraph
import vllm_omni.worker.gpu_model_runner as runner_module
from vllm_omni.worker.encoder_cudagraph import OmniEncoderCudaGraphManager
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CAPTURED_ROWS = 4
OUT_HIDDEN = 3


class RecordingGraph:
    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


class Encoder:
    """Two items, and an embedding whose row count the caller controls."""

    def __init__(self, rows):
        self.rows = rows
        self.eager_calls = []

    def get_encoder_cudagraph_item_specs(self, mm_kwargs):
        return [
            EncoderItemSpec(input_size=1, output_tokens=1),
            EncoderItemSpec(input_size=1, output_tokens=1),
        ]

    def prepare_encoder_cudagraph_replay_buffers(self, mm_kwargs, max_batch_size, max_frames_per_batch, path):
        return SimpleNamespace(
            values={
                "embeddings": torch.ones(self.rows, OUT_HIDDEN),
                "num_frames": torch.tensor(self.rows),
            }
        )

    def encoder_eager_forward(self, mm_kwargs, path="default"):
        self.eager_calls.append(path)
        return torch.full((self.rows, OUT_HIDDEN), 9.0)

    def select_encoder_cudagraph_items(self, mm_kwargs, indices):
        return dict(mm_kwargs)

    def postprocess_encoder_output(
        self, graph_outputs, batch_indices, per_item_out_tokens, outputs_by_orig_idx, clone, batch_mm_kwargs
    ):
        for position, index in enumerate(batch_indices):
            outputs_by_orig_idx[index] = graph_outputs["default"][position]


def make_manager(cls, rows, *, capture_budget=8):
    manager = cls.__new__(cls)
    manager.model = Encoder(rows)
    manager.config = EncoderCudaGraphConfig(
        modalities=["image"], buffer_keys=["embeddings"], out_hidden_size=OUT_HIDDEN
    )
    manager.max_batch_size = 2
    manager.max_frames_per_batch = CAPTURED_ROWS
    manager.graph_hits = 0
    manager.graph_misses = 0
    graph = RecordingGraph()
    manager.budget_graphs = {
        "default": {
            capture_budget: BudgetGraphMetadata(
                token_budget=capture_budget,
                max_batch_size=2,
                max_frames_per_batch=CAPTURED_ROWS,
                graph=graph,
                input_buffers={
                    "embeddings": torch.zeros(CAPTURED_ROWS, OUT_HIDDEN),
                    "num_frames": torch.tensor(0),
                },
                output_buffer=torch.full((CAPTURED_ROWS, OUT_HIDDEN), 5.0),
            )
        }
    }
    return manager, graph


def test_batch_within_the_capture_replays():
    manager, graph = make_manager(OmniEncoderCudaGraphManager, CAPTURED_ROWS - 1)

    output = manager._run_budget_graph({}, 8)

    assert graph.replays == 1
    assert manager.model.eager_calls == []
    assert manager.graph_hits == 2 and manager.graph_misses == 0
    torch.testing.assert_close(output, torch.full((CAPTURED_ROWS, OUT_HIDDEN), 5.0))
    # The graph consumed the real values, padded to the captured height.
    buffers = manager.budget_graphs["default"][8].input_buffers
    torch.testing.assert_close(buffers["embeddings"][: CAPTURED_ROWS - 1], torch.ones(CAPTURED_ROWS - 1, OUT_HIDDEN))
    torch.testing.assert_close(buffers["embeddings"][CAPTURED_ROWS - 1 :], torch.zeros(1, OUT_HIDDEN))


def test_batch_over_the_capture_runs_eager():
    manager, graph = make_manager(OmniEncoderCudaGraphManager, CAPTURED_ROWS + 1)

    output = manager._run_budget_graph({}, 8)

    assert graph.replays == 0
    assert manager.model.eager_calls == ["default"]
    assert manager.graph_hits == 0 and manager.graph_misses == 2
    torch.testing.assert_close(output, torch.full((CAPTURED_ROWS + 1, OUT_HIDDEN), 9.0))


def test_uncaptured_budget_runs_eager():
    manager, graph = make_manager(OmniEncoderCudaGraphManager, CAPTURED_ROWS - 1)

    output = manager._run_budget_graph({}, 16)

    assert graph.replays == 0
    assert manager.model.eager_calls == ["default"]
    assert manager.graph_hits == 0 and manager.graph_misses == 2
    torch.testing.assert_close(output, torch.full((CAPTURED_ROWS - 1, OUT_HIDDEN), 9.0))


def test_upstream_manager_raises_on_the_same_batch():
    """The behaviour this subclass exists to replace."""
    manager, _ = make_manager(EncoderCudaGraphManager, CAPTURED_ROWS + 1)

    with pytest.raises(RuntimeError, match="must match the size"):
        manager._run_budget_graph({}, 8)


@pytest.mark.parametrize(
    ("manager_class", "expectation"),
    [(OmniEncoderCudaGraphManager, "eager"), (EncoderCudaGraphManager, "raises")],
)
def test_execute_local_drives_the_upstream_caller(manager_class, expectation):
    """The caller asserts the budget path returned something, so run it.

    `_execute_local` picks the budget, calls `_run_budget_graph` and asserts on
    its result before postprocessing. Answering with the eager tensor has to
    satisfy that caller, not just the method under test.
    """
    manager, graph = make_manager(manager_class, CAPTURED_ROWS + 1)
    manager.token_budgets = [8]
    manager.path_token_budgets = {"default": [8]}

    if expectation == "raises":
        # The buffer copy is what raises; the caller's `assert output is not
        # None` is a different failure and must not satisfy this test.
        with pytest.raises(RuntimeError, match="must match the size"):
            manager._execute_local({})
        return

    outputs = manager._execute_local({})

    assert graph.replays == 0
    assert manager.model.eager_calls == ["default"]
    assert manager.graph_hits == 0 and manager.graph_misses == 2
    assert len(outputs) == 2
    for row in outputs:
        torch.testing.assert_close(row, torch.full((OUT_HIDDEN,), 9.0))


def test_execute_is_the_production_entry_and_reaches_the_fallback():
    """`execute()` is what the runner calls; `_execute_local` is one level in."""
    manager, graph = make_manager(OmniEncoderCudaGraphManager, CAPTURED_ROWS + 1)
    manager.token_budgets = [8]
    manager.path_token_budgets = {"default": [8]}
    manager.use_dp = False
    manager.log_stats_interval = 100

    outputs = manager.execute({})

    assert graph.replays == 0
    assert manager.model.eager_calls == ["default"]
    assert manager.graph_hits == 0 and manager.graph_misses == 2
    assert manager.get_cumulative_stats()["graph_misses"] == 2
    assert len(outputs) == 2
    for row in outputs:
        torch.testing.assert_close(row, torch.full((OUT_HIDDEN,), 9.0))


def test_runner_installs_the_omni_manager(monkeypatch):
    sentinel = EncoderCudaGraphManager.__new__(EncoderCudaGraphManager)
    monkeypatch.setattr(
        GPUModelRunner,
        "_create_encoder_cudagraph_manager",
        lambda self: sentinel,
        raising=True,
    )
    runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)

    manager = runner._create_encoder_cudagraph_manager()

    assert manager is sentinel
    assert isinstance(manager, OmniEncoderCudaGraphManager)


def test_runner_passes_through_none(monkeypatch):
    monkeypatch.setattr(
        GPUModelRunner,
        "_create_encoder_cudagraph_manager",
        lambda self: None,
        raising=True,
    )
    runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)

    assert runner._create_encoder_cudagraph_manager() is None


def test_a_buffer_key_the_model_does_not_supply_is_left_alone():
    """`replay.values` need not cover every captured buffer; the untouched one
    keeps the contents capture left in it."""
    manager, graph = make_manager(OmniEncoderCudaGraphManager, CAPTURED_ROWS - 1)
    buffers = manager.budget_graphs["default"][8].input_buffers
    buffers["unsupplied"] = torch.full((CAPTURED_ROWS, OUT_HIDDEN), 3.0)

    manager._run_budget_graph({}, 8)

    assert graph.replays == 1
    torch.testing.assert_close(buffers["unsupplied"], torch.full((CAPTURED_ROWS, OUT_HIDDEN), 3.0))


def test_runner_leaves_another_subclass_alone_and_says_so(monkeypatch, caplog):
    """Upstream builds the manager in a second place too, so a manager that is
    not the plain upstream class must not be silently assumed to be ours."""

    class OtherManager(EncoderCudaGraphManager):
        pass

    other = OtherManager.__new__(OtherManager)
    monkeypatch.setattr(
        GPUModelRunner,
        "_create_encoder_cudagraph_manager",
        lambda self: other,
        raising=True,
    )
    runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)

    with caplog.at_level("WARNING", logger=runner_module.logger.name):
        manager = runner._create_encoder_cudagraph_manager()

    assert manager is other
    assert type(manager) is OtherManager
    assert "OtherManager" in "\n".join(record.getMessage() for record in caplog.records)


def test_an_upstream_contract_change_leaves_the_manager_alone(monkeypatch, caplog):
    """A mismatch must not stop the stage: this runs inside
    `profile_cudagraph_memory`, before the cleanup that frees the profiling KV
    cache, so raising would leave the worker unusable. Upstream's manager, which
    raises on an oversized batch, is the right degraded state."""
    monkeypatch.setattr(omni_encoder_cudagraph, "_EXPECTED_PARAMETERS", ("self", "renamed"))
    assert "_run_budget_graph takes" in omni_encoder_cudagraph.upstream_contract_mismatch()

    sentinel = EncoderCudaGraphManager.__new__(EncoderCudaGraphManager)
    monkeypatch.setattr(
        GPUModelRunner,
        "_create_encoder_cudagraph_manager",
        lambda self: sentinel,
        raising=True,
    )
    runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)

    with caplog.at_level("WARNING", logger=runner_module.logger.name):
        manager = runner._create_encoder_cudagraph_manager()

    assert manager is sentinel
    assert type(sentinel) is EncoderCudaGraphManager
    assert "fallback not installed" in "\n".join(record.getMessage() for record in caplog.records)


def test_a_missing_metadata_field_is_named(monkeypatch):
    monkeypatch.setattr(omni_encoder_cudagraph, "_REQUIRED_METADATA_FIELDS", ("input_buffers", "gone"))

    assert "BudgetGraphMetadata is missing" in omni_encoder_cudagraph.upstream_contract_mismatch()


def test_a_missing_manager_attribute_is_named(monkeypatch):
    """The override also calls `_copy_padded_buffer` and friends, so losing one
    of those has to be reported too."""
    monkeypatch.setattr(omni_encoder_cudagraph, "_REQUIRED_MANAGER_ATTRIBUTES", ("_get_item_specs", "gone"))

    assert "EncoderCudaGraphManager is missing" in omni_encoder_cudagraph.upstream_contract_mismatch()


def test_a_model_padding_logic_decides_its_own_key():
    """A model's own padding routine bounds its layout, so the generic
    row-count check must not pre-empt it; it reports by raising."""
    manager, graph = make_manager(OmniEncoderCudaGraphManager, CAPTURED_ROWS + 1)

    def accepts_a_taller_source(dst, src):
        dst.copy_(src[: dst.shape[0]])

    manager.config.padding_logics = {"embeddings": accepts_a_taller_source}

    output = manager._run_budget_graph({}, 8)

    assert graph.replays == 1
    assert manager.graph_hits == 2 and manager.graph_misses == 0
    torch.testing.assert_close(output, torch.full((CAPTURED_ROWS, OUT_HIDDEN), 5.0))


def test_a_model_padding_logic_that_refuses_falls_back_to_eager():
    manager, graph = make_manager(OmniEncoderCudaGraphManager, CAPTURED_ROWS + 1)

    def refuses(dst, src):
        assert src.shape[0] <= dst.shape[0], "source is taller than the captured buffer"

    manager.config.padding_logics = {"embeddings": refuses}

    output = manager._run_budget_graph({}, 8)

    assert graph.replays == 0
    assert manager.model.eager_calls == ["default"]
    assert manager.graph_hits == 0 and manager.graph_misses == 2
    torch.testing.assert_close(output, torch.full((CAPTURED_ROWS + 1, OUT_HIDDEN), 9.0))
