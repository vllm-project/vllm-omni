# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native post-sample identity snapshots, with a CPU depformer double."""

from collections.abc import Callable

import pytest
import torch
from vllm.config import VllmConfig
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from tests.core.test_prefix_cache import make_manager, run_step
from vllm_omni.config.model import OmniModelConfig
from vllm_omni.data_entry_keys import flatten_payload
from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs
from vllm_omni.engine.duplex.contracts import DuplexFence, DuplexOutputContext, DuplexRequestIdentity
from vllm_omni.model_executor.models.personaplex.duplex.plugin import PersonaPlexDuplexPlugin
from vllm_omni.utils.mm_outputs import build_mm_cpu, partition_flat_payload
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.output.payload_build import build_omni_mm_payload

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _PostSampleModel(torch.nn.Module):
    """CPU depformer double with an explicitly typed post-sample seam."""

    def __init__(self, hook: Callable[..., torch.Tensor]) -> None:
        super().__init__()
        self.post_sample_talker_mtp = hook


def test_sampled_append_identity_is_fixed_before_model_hook():
    runner = GPUARModelRunner.__new__(GPUARModelRunner)
    duplex = {"data_plane": True, "epoch": 2, "seq": 4}
    runner.model_intermediate_buffer = {
        "live": {"duplex": duplex},
        "prefill": {"duplex": {"data_plane": True, "epoch": 3, "seq": 5}},
    }
    runner._update_intermediate_buffer = lambda *_: None

    def hook(**kwargs):
        assert kwargs["req_ids"] == ["live"]
        duplex["seq"] = 9
        return torch.arange(8).reshape(1, 8)

    runner.model = _PostSampleModel(hook)
    result = runner._run_post_sample_talker_mtp(
        req_ids=["live", "plain", "prefill"],
        valid_sampled_token_ids=[[11], [12], []],
        sampled_token_ids=torch.tensor([[11], [12], [-1]]),
        invalid_req_indices=[],
        sample_hidden_states=torch.zeros(3, 4),
        multimodal_outputs=None,
    )
    assert [value.tolist() for value in result["meta"]["duplex_epoch"]] == [[2], [-1], [-1]]
    assert [value.tolist() for value in result["meta"]["duplex_generated_seq"]] == [[4], [0], [0]]
    assert result["codes"]["audio"][0].shape == (1, 8)
    assert result["codes"]["audio"][2].numel() == 0


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_invalid_or_unsampled_rows_never_publish_generation_identity(mocker, async_scheduling):
    runner = GPUARModelRunner.__new__(GPUARModelRunner)
    runner.use_async_scheduling = async_scheduling
    runner.model_intermediate_buffer = {"r": {"duplex": {"data_plane": True, "epoch": 2, "seq": 4}}}
    hook = mocker.Mock()
    runner.model = _PostSampleModel(hook)
    result = runner._run_post_sample_talker_mtp(
        req_ids=["r"],
        valid_sampled_token_ids=[[]],
        sampled_token_ids=torch.tensor([[-1]]),
        invalid_req_indices=[0],
        sample_hidden_states=torch.zeros(1, 4),
        multimodal_outputs=None,
    )
    assert result is None
    hook.assert_not_called()


@pytest.mark.parametrize("identity", [{}, {"epoch": 1}, {"epoch": -1, "seq": 2}, {"epoch": 2, "seq": 0}])
def test_missing_or_uncommitted_identity_preserves_existing_metadata(identity):
    runner = GPUARModelRunner.__new__(GPUARModelRunner)
    runner.model_intermediate_buffer = {"r": {"duplex": {"data_plane": True, **identity}}}
    runner._update_intermediate_buffer = lambda *_: None
    runner.model = _PostSampleModel(lambda **_: torch.arange(8).reshape(1, 8))
    result = runner._run_post_sample_talker_mtp(
        req_ids=["r"],
        valid_sampled_token_ids=[[11]],
        sampled_token_ids=torch.tensor([[11]]),
        invalid_req_indices=[],
        sample_hidden_states=torch.zeros(1, 4),
        multimodal_outputs={"meta": {"source": "unchanged"}},
    )
    assert result["meta"] == {"source": "unchanged"}
    assert result["codes"]["audio"][0].shape == (1, 8)


def test_generation_identity_survives_client_metadata_partition_without_codec_rows():
    payload = {
        "meta.duplex_epoch": torch.tensor([2]),
        "meta.duplex_generated_seq": torch.tensor([4]),
        "codes.audio": torch.arange(8).reshape(1, 8),
    }
    inter_stage, client = partition_flat_payload(payload)
    assert "codes.audio" not in client
    assert client.keys() == {"meta.duplex_epoch", "meta.duplex_generated_seq"}
    assert inter_stage.keys() == payload.keys()


@pytest.mark.parametrize("prefix_cache", [False, True])
def test_batched_worker_payload_and_wire_keep_generation_identity_per_request(prefix_cache):
    runner = GPUARModelRunner.__new__(GPUARModelRunner)
    runner.model_intermediate_buffer = {
        "a": {"duplex": {"data_plane": True, "epoch": 2, "seq": 4}},
        "b": {"duplex": {"data_plane": True, "epoch": 3, "seq": 7}},
    }
    runner._update_intermediate_buffer = lambda *_: None
    runner.model = _PostSampleModel(lambda **_: torch.arange(16).reshape(2, 8))
    # This payload test needs only the output-type field, not a model-hub
    # lookup or a fully initialized serving configuration.
    runner.vllm_config = VllmConfig.__new__(VllmConfig)
    runner.vllm_config.model_config = OmniModelConfig.__new__(OmniModelConfig)
    runner.vllm_config.model_config.engine_output_type = "latent"
    result = runner._run_post_sample_talker_mtp(
        req_ids=["a", "b"],
        valid_sampled_token_ids=[[11], [12]],
        sampled_token_ids=torch.tensor([[11], [12]]),
        invalid_req_indices=[],
        sample_hidden_states=torch.zeros(2, 4),
        multimodal_outputs=None,
    )
    flat = flatten_payload(result)
    cpu = build_mm_cpu(flat)
    combined = None
    if prefix_cache:
        manager, view = make_manager()
        try:
            step = run_step(manager, view, {"a": ([0], 0, 1), "b": ([1], 0, 1)}, mm=flat)
            combined = manager.materialize(step, ["a", "b"]).mm_outputs
        finally:
            manager.shutdown()
    payloads = []
    for idx, request_id in enumerate(("a", "b")):
        per_request = build_omni_mm_payload(
            combined_multimodal_outputs=combined,
            mm_cpu=cpu,
            rid=request_id,
            idx=idx,
            start=idx,
            end=idx + 1,
            audio_sparse_output=False,
            sparse_mm_index={},
            hidden_seq_len=2,
            scheduled_seq_len=2,
        )
        _, client = partition_flat_payload(per_request)
        payloads.append(client)
    wire = runner._build_multimodal_outputs(payloads)
    envelope = OmniEngineCoreOutputs(
        outputs=[
            OmniEngineCoreOutput(
                request_id=request_id, new_token_ids=[11 + idx], is_segment_finished=True, multimodal_output=wire[idx]
            )
            for idx, request_id in enumerate(("a", "b"))
        ]
    )
    decoded = MsgpackDecoder(OmniEngineCoreOutputs).decode(MsgpackEncoder().encode(envelope))
    plugin = PersonaPlexDuplexPlugin(lambda *_: "unused")
    for epoch, sequence, output in zip((2, 3), (4, 7), decoded.outputs, strict=True):
        context = DuplexOutputContext(
            identity=DuplexRequestIdentity(
                "s-" + output.request_id, DuplexFence("s-" + output.request_id, epoch=epoch)
            ),
            final_stage_id=1,
            segment_finished=output.is_segment_finished,
            segment_token_ids=tuple(output.new_token_ids),
            segment_output_metadata=output.multimodal_output,
        )
        assert plugin.completed_audio_append(stage_id=0, context=context) == sequence
        assert "codes.audio" not in output.multimodal_output
