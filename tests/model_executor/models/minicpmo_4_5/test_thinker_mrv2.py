# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MRv2 Thinker preserves multimodal encoding and the llm2tts row ledger."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni as omni
from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_mrv2_preprocess_stages_shared_encoders_before_reordered_prefill_rows(mocker):
    model = _model(mocker, session="duplex")
    state = object.__new__(OmniModelState)
    state.model, state.has_preprocess, state._static_inputs_embeds = model, True, None
    state.intermediate_buffer = OmniIntermediateBuffer(3)
    state.intermediate_buffer.buffers = [{"req_id": "r0"}, {"req_id": "decode"}, {"req_id": "r2"}]
    calls = []

    def stage(*, req_ids, model_intermediate_buffer, device):
        calls.append(("batch", req_ids))
        assert device.type == "cpu"
        for request_id in req_ids:
            index = int(request_id[-1])
            assert model_intermediate_buffer[request_id] is state.intermediate_buffer.buffers[index]
            model_intermediate_buffer[request_id]["staged"] = True

    def preprocess(ids, embeds, **info):
        calls.append(("row", info["req_id"]))
        assert info.get("staged", False) is (info["req_id"] != "decode")
        assert info["_omni_is_prefill"] is (info["req_id"] != "decode")
        return ids, embeds, {}

    mocker.patch.object(model, "preprocess_batch", side_effect=stage)
    mocker.patch.object(model, "preprocess", side_effect=preprocess)
    batch = mocker.Mock(
        spec=InputBatch,
        num_reqs=3,
        idx_mapping_np=np.array([2, 0, 1]),
        num_scheduled_tokens=np.array([2, 1, 2]),
        query_start_loc_np=np.array([0, 2, 3]),
    )
    req_states = mocker.Mock(spec=RequestState, prompt_len=np.array([5, 1, 5]), num_computed_tokens=np.array([4, 1, 0]))
    inputs = {"input_ids": torch.arange(5), "inputs_embeds": torch.zeros(5, 4)}
    state.run_preprocess(batch, inputs, req_states)
    assert calls == [("batch", ["r2", "r0"]), ("row", "r2"), ("row", "r0"), ("row", "decode")]
    assert state.intermediate_buffer.buffers[1] == {"req_id": "decode"}


@pytest.mark.parametrize(
    "profile,capacities,kv_gib",
    [
        ("minicpmo_4_5_turn_mrv2.yaml", [16, 8, 8], 2),
        ("minicpmo_4_5_turn_mrv2_h200.yaml", [16, 16, 8], 4),
    ],
)
def test_mrv2_profile_retains_full_thinker_handoff(profile, capacities, kv_gib, monkeypatch):
    from pathlib import Path

    from vllm_omni.config.stage_config import _apply_platform_overrides, load_deploy_config, merge_pipeline_deploy
    from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE
    from vllm_omni.platforms import current_omni_platform

    # merge_pipeline_deploy resolves the platform again; test the CUDA profile
    # consistently even when this CPU test runs on a ROCm host.
    monkeypatch.setattr(current_omni_platform, "device_name", "cuda")
    deploy = Path(__file__).resolve().parents[4] / "vllm_omni/deploy" / profile
    config = _apply_platform_overrides(load_deploy_config(deploy), platform="cuda")
    stages = merge_pipeline_deploy(MINICPMO_4_5_PIPELINE, config)
    assert [s.yaml_engine_args["use_v2_model_runner"] for s in stages] == [True, True, True]
    assert [s.yaml_engine_args["async_chunk"] for s in stages] == [False, True, True]
    assert [s.yaml_engine_args["max_num_seqs"] for s in stages] == capacities
    assert stages[1].yaml_engine_args["kv_cache_memory_bytes"] == kv_gib * 1024**3


def test_gander_mrv2_profile_retains_native_capacity_and_sampling(monkeypatch):
    from pathlib import Path

    from vllm_omni.config.stage_config import _apply_platform_overrides, load_deploy_config, merge_pipeline_deploy
    from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE
    from vllm_omni.platforms import current_omni_platform

    monkeypatch.setattr(current_omni_platform, "device_name", "cuda")
    directory = Path(__file__).resolve().parents[4] / "vllm_omni/deploy"
    configs = [
        _apply_platform_overrides(load_deploy_config(directory / profile), platform="cuda")
        for profile in ["gander_hift_h200.yaml", "gander_mrv2_h200.yaml"]
    ]
    control, optimized = [merge_pipeline_deploy(MINICPMO_4_5_PIPELINE, config) for config in configs]
    assert [stage.yaml_engine_args["use_v2_model_runner"] for stage in optimized] == [True] * 3
    # MRv2 collects the complete Thinker row ledger for llm2tts; only the
    # Talker/codec edge uses chunked output. An inherited True prevents startup.
    assert [stage.yaml_engine_args["async_chunk"] for stage in optimized] == [False, True, True]
    assert [stage.yaml_engine_args["max_num_seqs"] for stage in optimized] == [4] * 3
    assert [stage.yaml_runtime["devices"] for stage in optimized] == ["0"] * 3
    assert [stage.yaml_engine_args["async_scheduling"] for stage in optimized] == [True, False, False]
    for baseline, actual in zip(control, optimized, strict=True):
        assert actual.yaml_extras["default_sampling_params"] == baseline.yaml_extras["default_sampling_params"]
        assert actual.yaml_engine_args["max_model_len"] == baseline.yaml_engine_args["max_model_len"]
    assert configs[1].duplex_session.max_sessions == configs[0].duplex_session.max_sessions == 4
    extra = configs[1].connectors["connector_of_shared_memory"]["extra"]
    assert extra["initial_codec_chunk_frames"] == 10 and extra["codec_chunk_frames"] == 25


def _model(mocker, *, v2=True, session="turn", async_chunk=False):
    thinker = torch.nn.Module()
    thinker.make_empty_intermediate_tensors = lambda: None
    mocker.patch.object(omni, "init_vllm_registered_model", return_value=thinker)
    mocker.patch.object(omni.MiniCPMO45OmniForConditionalGeneration, "_duplex_data_plane_helper", return_value=None)
    mocker.patch("vllm_omni.model_executor.models.minicpmo_4_5.duplex.compat.patch_minicpmo_remote_config")
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(),
            multimodal_config=None,
            model_stage="llm",
            use_v2_model_runner=v2,
            session_mode=session,
            async_chunk=async_chunk,
        )
    )
    return omni.MiniCPMO45OmniForConditionalGeneration(vllm_config=config)


def test_mrv2_thinker_keeps_native_multimodal_embeddings_and_raw_ids(mocker):
    model = _model(mocker)
    batch = SimpleNamespace(input_ids=torch.tensor([11, 22, 33]), num_tokens_after_padding=3)
    embeddings = torch.randn(3, 4)
    runner = SimpleNamespace(
        model=model,
        supports_mm_inputs=True,
        is_first_pp_rank=True,
        model_state=SimpleNamespace(dummy_inputs_embeds=lambda _: embeddings),
    )
    ids, embeds, _ = OmniGPUModelRunner._prepare_mm_inputs(runner, None, batch, dummy_run=True)
    assert ids is batch.input_ids
    assert embeds is embeddings
    hidden = torch.randn(3, 4)
    model.thinker.forward = mocker.Mock(return_value=hidden)
    result = model(ids, torch.tensor([5, 6, 7]), inputs_embeds=embeds)
    assert result is hidden
    torch.testing.assert_close(model.thinker.forward.call_args.kwargs["inputs_embeds"], embeddings)


def test_row_ledger_uses_live_batch_after_replay(mocker):
    model = _model(mocker)
    # Chunked prefill (two tokens), decode (one token), and graph padding.
    batch = SimpleNamespace(
        input_ids=torch.tensor([11, 12, 41, 0]),
        positions=torch.tensor([6, 7, 20, 0]),
    )
    hidden = torch.randn(4, 8)
    for token in [41, 53]:
        batch.input_ids[2] = token
        batch.positions[2] += 1
        out = model.make_omni_output_mrv2(
            hidden,
            input_batch=batch,
            req_states=None,
            model_intermediate_buffer=[{}, {}],
        )
        assert out.text_hidden_states is hidden
        assert out.multimodal_outputs["latent"] is hidden
        torch.testing.assert_close(out.multimodal_outputs["latent_input_ids"], batch.input_ids[:, None])
        torch.testing.assert_close(out.multimodal_outputs["latent_positions"], batch.positions[:, None])
