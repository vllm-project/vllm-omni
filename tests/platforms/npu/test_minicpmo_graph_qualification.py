# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Active modes must qualify locally and retain reference outputs on failure."""

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
def test_active_prefill_qualifies_before_replay(tmp_path, monkeypatch, mismatch):
    mode = tmp_path / "mode"
    mode.write_text("active_thinker_prefill")
    output = torch.ones(2, 4, dtype=torch.bfloat16)
    cache = torch.zeros(2, 4)
    metadata = Mock(
        spec=["attn_state", "causal", "actual_seq_lengths_q", "num_actual_tokens", "slot_mapping", "attn_mask"],
        attn_state=prefill_graph.AscendAttentionState.PrefillNoCache,
        causal=True,
        actual_seq_lengths_q=[2],
        num_actual_tokens=2,
        slot_mapping=torch.arange(2),
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
    batch = Mock(num_reqs=1, has_prefill=True, is_dummy=False, num_tokens=2, num_tokens_after_padding=2)

    class Graph:
        def __init__(self, *args, **kwargs):
            self.hidden = output + (1 if mismatch == "hidden" else 0)
            self.kv = [("layer", cache)]
            self.graph = Mock()
            self.graph.replay.side_effect = lambda: cache.fill_(1)
            self.replays = 0

        def bind(self, *args):
            pass

    monkeypatch.setattr(prefill_graph, "PrefillGraph", Graph)
    monkeypatch.setattr(torch.npu, "Stream", lambda: None)
    inputs = {"model_intermediate_buffer": [{}], "inputs_embeds": output, "positions": torch.arange(2)}
    descriptor = Mock(cg_mode=CUDAGraphMode.NONE)
    first = prefill_graph.run_prefill(runner, mode, inputs, batch, descriptor)
    assert first is output and model.call_count == 1
    second = prefill_graph.run_prefill(runner, mode, inputs, batch, descriptor)
    if mismatch:
        assert runner._exact_prefill_disabled and second is None
        assert not runner._exact_prefill_passed
    else:
        assert runner._exact_prefill_passed == {2}
        assert torch.equal(second, output) and second.data_ptr() != output.data_ptr()
        assert model.call_count == 1


@pytest.mark.parametrize("matches", [False, True])
def test_active_fused_decode_checks_shadow_before_commit(tmp_path, monkeypatch, matches):
    mode = tmp_path / "mode"
    mode.write_text("active_thinker_prefill")
    graph = Mock(spec=["layer_names", "kv", "replay_ms", "bind", "run", "snapshot_kv", "restore_kv"])
    graph.layer_names = ("layer",)
    graph.kv = [torch.zeros(2, 1)]
    graph.replay_ms = []
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
    assert talker_fused_decode.run_fused_step(runner, mode, None, {}, batch, None) == "reference"
    shadow_check.assert_called_once()
    commit.assert_not_called()
    assert talker_fused_decode.run_fused_step(runner, mode, None, {}, batch, None) == (
        "optimized" if matches else "reference"
    )
    assert commit.call_count == int(matches)
    assert graph.run.call_count == (2 if matches else 1)


@pytest.mark.parametrize("matches", [False, True])
@pytest.mark.parametrize("rows", [2, 12, 16])
def test_active_batch_checks_the_entire_kstep(tmp_path, monkeypatch, matches, rows):
    mode = tmp_path / "mode"
    mode.write_text("active_fixed")
    monkeypatch.setattr(talker_batched_decode, "MODE", str(mode))
    monkeypatch.setattr(torch.npu, "Stream", lambda: None)
    monkeypatch.setattr(torch.npu, "graph_pool_handle", lambda: None)
    tokens = rows * 16
    wide = torch.arange(tokens * 2, dtype=torch.bfloat16).reshape(tokens, 2)
    metadata = Mock(
        seq_lens_device=torch.full((rows,), 32), block_tables=torch.zeros(rows, 4), slot_mapping=torch.arange(tokens)
    )
    runner = Mock(spec=["model_config", "model_state", "vllm_config"])
    runner.vllm_config = Mock(cache_config=Mock(block_size=128))
    runner.model_config = Mock(max_model_len=512)
    runner.model_state = Mock(attn_metadata={"layer": metadata})
    runner.model_state._request_token_spans.return_value = [(i, i + 16) for i in range(0, tokens, 16)]
    batch = Mock(num_reqs=rows, has_prefill=False, seq_lens_np=np.full(rows, 32))
    inputs = {
        "model_intermediate_buffer": [{} for _ in range(rows)],
        "inputs_embeds": wide,
        "positions": torch.arange(tokens),
    }

    if rows > 8:
        assert talker_batched_decode.prepare(runner, None, inputs, batch, 16) is None
        mode.write_text("active_fixed_wide" if rows == 16 else "active_fixed_padded")

    class Graph:
        def __init__(self, *args, **kwargs):
            self.metadata = {"layer": metadata}
            self.slots = torch.arange(16 if rows > 8 else rows)
            self.kv = []
            if rows > 8:
                assert args[-3:] == (16, 512, 256)

        def prepare_step(self, *args):
            pass

        def bind(self, first, embeds, positions, width, frame, active_rows):
            assert active_rows == rows
            self.frame = frame

        def replay(self):
            result = wide[self.frame : tokens : 16]
            if rows == 12:
                result = torch.cat((result, torch.zeros(4, 2, dtype=result.dtype)))
            return result + (0 if matches else 1)

    monkeypatch.setattr(talker_batched_decode, "CompactGraph", Graph)
    forward = talker_batched_decode.prepare(runner, None, inputs, batch, 16)
    assert forward is not None
    reference = Mock(return_value=wide)
    for frame in range(16):
        assert torch.equal(forward(frame, reference), wide[frame:tokens:16])
    assert reference.call_count == 16
    if matches:
        assert len(runner._c8_talker_ready) == 1
        forward = talker_batched_decode.prepare(runner, None, inputs, batch, 16)
        assert torch.equal(forward(0, reference), wide[0:tokens:16])
        assert reference.call_count == 16
    else:
        assert not runner._c8_talker_ready and runner._c8_talker_rejected
