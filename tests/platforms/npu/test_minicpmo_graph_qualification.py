# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Prefill and decode graphs must qualify against the reference path."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

pytest.importorskip("vllm_ascend")

from vllm.config import CUDAGraphMode

from tests.helpers.mark import hardware_marks
from vllm_omni.platforms.npu.worker import prefill_graph, talker_batched_decode, talker_fused_decode

pytestmark = [pytest.mark.core_model, pytest.mark.omni, *hardware_marks(res={"npu": "A3"}, num_cards=1)]


@pytest.mark.parametrize("mismatch", [None, "hidden", "kv"])
def test_startup_prefill_qualifies_before_replay(tmp_path, monkeypatch, mismatch):
    marker = tmp_path / "prefill"
    length = prefill_graph.TALKER_STARTUP_LENGTHS.start
    output = torch.ones(length, 4, dtype=torch.bfloat16)
    cache = torch.zeros_like(output)
    metadata = Mock(
        spec=["attn_state", "causal", "actual_seq_lengths_q", "num_actual_tokens", "slot_mapping", "attn_mask"],
        attn_state=prefill_graph.AscendAttentionState.PrefillNoCache,
        causal=True,
        actual_seq_lengths_q=[length],
        num_actual_tokens=length,
        slot_mapping=torch.arange(length),
        attn_mask=None,
    )

    def reference(**kwargs):
        cache.fill_(2 if mismatch == "kv" else 1)
        return output

    model = Mock(spec=["tts_model", "_flush_deferred_cleanup"], side_effect=reference)
    model.tts_model = model
    runner = Mock(
        spec=["model", "lora_config", "use_aux_hidden_state_outputs", "model_state", "vllm_config"],
        model=model,
        lora_config=None,
        use_aux_hidden_state_outputs=False,
        model_state=Mock(attn_metadata={"layer": metadata}),
    )
    batch = Mock(
        num_reqs=1,
        has_prefill=True,
        is_dummy=False,
        num_tokens=length,
        num_tokens_after_padding=length,
        req_ids=[f"{prefill_graph.TALKER_STARTUP_PREFIX}0"],
    )

    class Graph:
        def __init__(self, *args, **kwargs):
            self.hidden = output + (1 if mismatch == "hidden" else 0)
            self.kv = [("layer", cache)]
            self.metadata = {"layer": metadata}
            self.graph = Mock()
            self.graph.replay.side_effect = lambda: cache.fill_(1)
            self.replays = 0

        def bind(self, *args):
            pass

    monkeypatch.setattr(prefill_graph, "PrefillGraph", Graph)
    monkeypatch.setattr(prefill_graph, "get_forward_context", lambda: Mock(slot_mapping={}))
    monkeypatch.setattr(torch.npu, "Stream", lambda: None)
    monkeypatch.setattr(torch.npu, "mem_get_info", lambda: (8 * 1024**3, 16 * 1024**3))
    inputs = {"model_intermediate_buffer": [{}], "inputs_embeds": output, "positions": torch.arange(length)}
    descriptor = Mock(cg_mode=CUDAGraphMode.NONE)
    assert prefill_graph.run_prefill(runner, inputs, batch, descriptor, startup_path=marker) is None
    marker.touch()
    first = prefill_graph.run_prefill(runner, inputs, batch, descriptor, startup_path=marker)
    assert first is output and model.call_count == 1
    marker.unlink()
    second = prefill_graph.run_prefill(runner, inputs, batch, descriptor, startup_path=marker)
    if mismatch:
        assert runner._exact_prefill_disabled and second is None
        assert not runner._exact_prefill_passed
    else:
        assert runner._exact_prefill_passed == {length}
        assert torch.equal(second, output) and second.data_ptr() != output.data_ptr()
        assert model.call_count == 1


@pytest.mark.parametrize("matches", [False, True])
def test_fused_decode_checks_shadow_before_commit(monkeypatch, matches):
    graph = Mock(spec=["layer_names", "kv", "bind", "run", "snapshot_kv", "restore_kv", "qualified"])
    graph.layer_names = ("layer",)
    graph.kv = [torch.zeros(2, 1)]
    graph.qualified = None
    graph.snapshot_kv.return_value = (torch.tensor([0]),)
    graph.run.return_value = (torch.ones(1), torch.ones(1))
    runner = Mock(spec=["model_config", "model", "vllm_config", "_run_kstep_frames"])
    runner.vllm_config = Mock(cache_config=Mock(block_size=128))
    runner.model_config = Mock(max_model_len=512)
    runner.model = Mock(spec=[])
    runner._run_kstep_frames.return_value = "reference"
    live = Mock(metadata={"layer": object()}, request_id="request")
    monkeypatch.setattr(talker_fused_decode, "prepare_step", lambda *args: live)
    monkeypatch.setattr(talker_fused_decode, "TalkerFusedGraph", lambda *args, **kwargs: graph)
    monkeypatch.setattr(talker_fused_decode, "summarize", lambda _: ([2], [False], False, 1))
    shadow_check = Mock(return_value=matches)
    monkeypatch.setattr(talker_fused_decode, "check_shadow", shadow_check)
    commit = Mock(return_value="optimized")
    monkeypatch.setattr(talker_fused_decode, "commit", commit)
    batch = Mock(spec=["num_reqs"], num_reqs=1)
    assert talker_fused_decode.run_fused_step(runner, None, {}, batch, None) == "reference"
    shadow_check.assert_called_once()
    commit.assert_not_called()
    assert talker_fused_decode.run_fused_step(runner, None, {}, batch, None) == (
        "optimized" if matches else "reference"
    )
    assert commit.call_count == int(matches)
    assert graph.run.call_count == (2 if matches else 1)


@pytest.mark.parametrize(
    ("rows", "wide_capacity", "mismatch"),
    [
        (2, 512, None),
        (12, 256, None),
        (16, 512, None),
        (12, 256, "hidden"),
        (16, 256, "kv"),
    ],
)
def test_compact_batch_checks_the_entire_kstep(monkeypatch, rows, wide_capacity, mismatch):
    monkeypatch.setattr(torch.npu, "Stream", lambda: None)
    monkeypatch.setattr(torch.npu, "graph_pool_handle", lambda: None)
    tokens = rows * 16
    wide = torch.arange(tokens * 2, dtype=torch.bfloat16).reshape(tokens, 2)
    kv = [torch.zeros(rows + 1, 3, dtype=torch.bfloat16) for _ in range(2)]
    metadata = Mock(
        seq_lens_device=torch.full((rows,), 32), block_tables=torch.zeros(rows, 4), slot_mapping=torch.arange(tokens)
    )
    runner = Mock(spec=["model_config", "model_state", "vllm_config"])
    runner.vllm_config = Mock(
        cache_config=Mock(block_size=128),
        additional_config={"minicpmo_talker_wide_graph_capacity": wide_capacity},
    )
    runner.model_config = Mock(max_model_len=512)
    runner.model_state = Mock(attn_metadata={"layer": metadata})
    runner.model_state._request_token_spans.return_value = [(i, i + 16) for i in range(0, tokens, 16)]
    batch = Mock(num_reqs=rows, has_prefill=False, seq_lens_np=np.full(rows, 32))
    inputs = {
        "model_intermediate_buffer": [{} for _ in range(rows)],
        "inputs_embeds": wide,
        "positions": torch.arange(tokens),
    }

    if rows == 12 and mismatch is None:
        batch.seq_lens_np.fill(wide_capacity + 1)
        assert talker_batched_decode.prepare(runner, None, inputs, batch, 16) is None
        batch.seq_lens_np.fill(32)

    class Graph:
        def __init__(self, *args, **kwargs):
            self.metadata = {"layer": metadata}
            self.slots = torch.arange(16 if rows > 8 else rows)
            self.kv = kv
            if rows > 8:
                assert args[-3:] == (16, wide_capacity, 256)

        def prepare_step(self, *args):
            pass

        def bind(self, first, embeds, positions, width, frame, active_rows):
            assert active_rows == rows
            self.frame = frame

        def replay(self):
            result = wide[self.frame : tokens : 16]
            if rows == 12:
                result = torch.cat((result, torch.zeros(4, 2, dtype=result.dtype)))
            if self.frame == 7 and mismatch == "kv":
                self.kv[-1][rows - 1].fill_(1)
            if self.frame == 7 and mismatch == "hidden":
                result = result + 2
            return result

    monkeypatch.setattr(talker_batched_decode, "CompactGraph", Graph)
    forward = talker_batched_decode.prepare(runner, None, inputs, batch, 16)
    assert forward is not None
    reference = Mock(return_value=wide)
    for frame in range(16):
        assert torch.equal(forward(frame, reference), wide[frame:tokens:16])
    assert reference.call_count == 16
    assert all(not cache.any() for cache in kv)
    if mismatch is None:
        assert len(runner._talker_compact_ready) == 1
        forward = talker_batched_decode.prepare(runner, None, inputs, batch, 16)
        assert torch.equal(forward(0, reference), wide[0:tokens:16])
        assert reference.call_count == 16
    else:
        assert not runner._talker_compact_ready and runner._talker_compact_rejected
