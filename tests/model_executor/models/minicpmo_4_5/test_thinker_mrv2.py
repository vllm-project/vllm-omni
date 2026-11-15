# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MRv2 Thinker preserves multimodal encoding and the llm2tts row ledger."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni as omni
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_cuda_turn_inherits_mrv2_talker_kstep(monkeypatch):
    from pathlib import Path

    from vllm_omni.config.stage_config import _apply_platform_overrides, load_deploy_config, merge_pipeline_deploy
    from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE
    from vllm_omni.platforms import current_omni_platform

    # merge_pipeline_deploy resolves the platform again; test the CUDA profile
    # consistently even when this CPU test runs on a ROCm host.
    monkeypatch.setattr(current_omni_platform, "device_name", "cuda")
    deploy = Path(__file__).resolve().parents[4] / "vllm_omni/deploy/minicpmo_4_5_turn.yaml"
    config = _apply_platform_overrides(load_deploy_config(deploy), platform="cuda")
    stages = merge_pipeline_deploy(MINICPMO_4_5_PIPELINE, config)
    assert config.session_mode == "turn"
    assert [s.yaml_engine_args["use_v2_model_runner"] for s in stages] == [True, True, True]
    assert [s.yaml_engine_args["async_chunk"] for s in stages] == [True, True, True]
    assert stages[0].yaml_engine_args["enable_prefix_caching"] is True
    talker = stages[1]
    assert talker.yaml_engine_args["attention_backend"] == "FLASH_ATTN"
    assert talker.yaml_engine_args["speculative_config"] == {
        "method": "ngram",
        "num_speculative_tokens": 7,
        "prompt_lookup_min": 1,
        "prompt_lookup_max": 1,
    }
    assert talker.yaml_engine_args["async_scheduling"] is False
    assert talker.yaml_extras["default_sampling_params"]["stop_token_ids"] == [1, 6561]
    assert talker.yaml_runtime["cuda_mps"] is True


def _model(mocker, *, v2=True, session="turn", async_chunk=False, speculative_config=None):
    thinker = torch.nn.Module()
    thinker.make_empty_intermediate_tensors = lambda: None
    mocker.patch.object(omni, "init_vllm_registered_model", return_value=thinker)
    mocker.patch("vllm_omni.model_executor.models.minicpmo_4_5.duplex.compat.patch_minicpmo_remote_config")
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(),
            multimodal_config=None,
            model_stage="llm",
            use_v2_model_runner=v2,
            session_mode=session,
            async_chunk=async_chunk,
        ),
        speculative_config=speculative_config,
    )
    if session is None:
        del config.model_config.session_mode
    return omni.MiniCPMO45OmniForConditionalGeneration(vllm_config=config)


@pytest.mark.parametrize("session", ["turn", "duplex", None])
def test_mrv2_thinker_output_cache_requires_explicit_turn_mode(mocker, session):
    model = _model(mocker, session=session)
    runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
    runner._snapshot_prefix_cache_model_policy(model)
    assert model.supports_mrv2_prefix_cache_outputs is (session == "turn")
    assert runner._omni_cache_policy.native_kv_cache_only is (session != "turn")


@pytest.mark.parametrize("drafts", [False, True], ids=["no_draft", "draft"])
def test_mrv2_thinker_with_a_draft_reuses_native_kv_instead_of_being_refused(mocker, drafts):
    from vllm_omni.core.prefix_cache.group_view import stage_prefix_cache_config
    from vllm_omni.core.prefix_cache.interface import OmniPrefixCacheUnmatchError

    speculative_config = SimpleNamespace(method="dspark", num_speculative_tokens=7) if drafts else None
    model = _model(mocker, speculative_config=speculative_config)
    runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
    runner._snapshot_prefix_cache_model_policy(model)
    assert model.supports_mrv2_prefix_cache_outputs is not drafts
    assert runner._omni_cache_policy.native_kv_cache_only is drafts

    def build(spec):
        return stage_prefix_cache_config(
            kv_cache_config=SimpleNamespace(num_blocks=4, kv_cache_groups=[]),
            cache_config=SimpleNamespace(enable_prefix_caching=True, block_size=16),
            kv_transfer_config=None,
            scheduler_config=SimpleNamespace(max_num_batched_tokens=64, max_num_seqs=2),
            model_config=SimpleNamespace(max_model_len=64),
            is_pooling_model=False,
            speculative_config=spec,
            model_cache_policy=runner._omni_cache_policy,
        )

    if drafts:
        # Native KV reuse alone: no stage-output cache to refuse the draft.
        assert build(speculative_config) is None
    else:
        with pytest.raises(OmniPrefixCacheUnmatchError, match="speculative decoding"):
            build(SimpleNamespace(method="dspark"))


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


@pytest.mark.parametrize("async_chunk", [False, True])
@pytest.mark.parametrize("shared_tokens", [4, 6, 8])
def test_turn_prefix_hit_restores_latent_ledger_before_talker(mocker, monkeypatch, async_chunk, shared_tokens):
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    from tests.core.test_prefix_cache_mrv2 import input_batch, sched_output
    from tests.worker_v2.test_omni_ar_model_runner import _async_output
    from vllm_omni.core.prefix_cache.adapter import PrefixCacheSchedulerAdapter
    from vllm_omni.core.prefix_cache.group_view import MRv2FullAttentionGroupView
    from vllm_omni.core.prefix_cache.interface import PrefixCacheConfig
    from vllm_omni.core.prefix_cache.manager import OmniPrefixCacheManager
    from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import _turn_hidden_rows
    from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner

    # Exercise the real cache manager and MRv2 save/rewrite path on CPU.
    # Only device streams and distributed rank discovery are replaced.
    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)
    mocker.patch("vllm.distributed.parallel_state.get_pp_group", return_value=SimpleNamespace(is_last_rank=True))
    mocker.patch("vllm.distributed.parallel_state.get_tp_group", return_value=SimpleNamespace(is_first_rank=True))
    model = _model(mocker, async_chunk=async_chunk)
    runner = OmniARModelRunner.__new__(OmniARModelRunner)
    runner._snapshot_prefix_cache_model_policy(model)
    assert not runner._omni_cache_policy.native_kv_cache_only
    runner.is_pooling_model = False
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace(engine_output_type="latent"))
    runner._prefix_cache_adapter = PrefixCacheSchedulerAdapter(track_requests=True)
    runner._prefix_cache_group_view = MRv2FullAttentionGroupView(runner._prefix_cache_adapter, 4)
    cache = OmniPrefixCacheManager(PrefixCacheConfig(num_blocks=16, block_size=4), eager=True)
    cache.register_policy(runner._omni_cache_policy)
    runner.omni_prefix_cache = cache

    def payload(ids, positions):
        return torch.stack((ids, positions), dim=1).float()

    def save(rows, ids, positions, **scheduled):
        runner._prefix_cache_step_begin(sched_output(scheduled={r: n for r, _, n in rows}, **scheduled))
        batch = input_batch(rows)
        batch.input_ids = torch.tensor(ids)
        batch.positions = torch.tensor(positions)
        batch.num_tokens = len(ids)
        batch.num_tokens_after_padding = len(ids)
        batch.query_start_loc_np = np.concatenate(([0], np.cumsum(batch.num_scheduled_tokens)))
        output = model.make_omni_output_mrv2(
            payload(batch.input_ids, batch.positions),
            input_batch=batch,
            req_states=None,
            model_intermediate_buffer=None,
        )
        rewrite = runner._prefix_cache_save_mrv2(batch, output.text_hidden_states, output.multimodal_outputs)
        return batch, output, rewrite

    try:
        # Cache two complete blocks, then finish their original request.
        _, _, rewrite = save([("seed", 0, 8)], range(10, 18), range(8), new=[("seed", 0, [0, 1])])
        assert rewrite is None
        runner._prefix_cache_step_begin(sched_output(finished=["seed"]))

        # Only complete matching blocks are reusable. With six shared tokens,
        # recompute positions 4/5 and replace the old seed's positions 6/7.
        # Never report six computed tokens: sub-block hits are unsupported.
        cached_tokens = shared_tokens // 4 * 4
        full_ids = torch.tensor([*range(10, 10 + shared_tokens), 90, 91])
        full_positions = torch.arange(shared_tokens + 2)
        tail_tokens = len(full_ids) - cached_tokens
        # Put a miss before the hit so cache restoration must use the batch row.
        ids = [100, 101, 102, *full_ids[cached_tokens:].tolist()]
        positions = [0, 1, 2, *full_positions[cached_tokens:].tolist()]
        # Allocate fresh blocks after the shared full blocks, as the scheduler
        # does when the rest of the seed's second block no longer matches.
        hit_blocks = [0, 1, 2] if cached_tokens == 8 else [0, 4, 5]
        batch, output, rewrite = save(
            [("miss", 0, 3), ("hit", cached_tokens, tail_tokens)],
            ids,
            positions,
            new=[("hit", cached_tokens, hit_blocks), ("miss", 0, [3])],
        )
        assert rewrite is not None
        assert rewrite.rows == ((1, "hit"),)
        finalize = runner._prefix_cache_output_finalizer(rewrite, None, mergeable=True)
        result = _async_output(
            req_ids=batch.req_ids,
            sampler_output=SamplerOutput(
                torch.tensor([[1], [2]]), None, None, torch.tensor([1, 1]), torch.tensor([0, 0])
            ),
            text_hidden=output.text_hidden_states,
            multimodal_outputs=output.multimodal_outputs,
            input_batch=batch,
            async_chunk=async_chunk,
            partition_async_payload=False,  # llm2tts uses the orchestrator bridge.
            finalize_output=finalize,
        ).get_output()
        miss, hit = result.pooler_output
        expected = payload(full_ids, full_positions)
        torch.testing.assert_close(hit["latent"], expected)
        torch.testing.assert_close(hit["latent_input_ids"].flatten(), full_ids)
        torch.testing.assert_close(hit["latent_positions"].flatten(), full_positions)
        torch.testing.assert_close(miss["latent"], payload(torch.tensor(ids[:3]), torch.tensor(positions[:3])))
        torch.testing.assert_close(miss["latent_input_ids"].flatten(), torch.tensor(ids[:3]))
        torch.testing.assert_close(miss["latent_positions"].flatten(), torch.tensor(positions[:3]))
        # A reply can start inside a reused block. Its Talker condition must
        # match a cold forward, including both cached rows and this step's tail.
        restored = _turn_hidden_rows(hit, hit["latent"], full_ids.tolist(), 2, len(full_ids), "hit")
        torch.testing.assert_close(restored, expected[2:])
    finally:
        cache.shutdown()


def test_row_ledger_uses_live_batch_after_replay(mocker):
    model = _model(mocker)
    # Chunked prefill (two tokens), decode (one token), and graph padding.
    batch = SimpleNamespace(
        req_ids=["a", "b"],
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


def test_mrv2_thinker_duplex_output_and_prompt_rows(mocker):
    model = _model(mocker, session="duplex")
    runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
    runner._snapshot_prefix_cache_model_policy(model)
    assert runner._omni_cache_policy.native_kv_cache_only
    assert model.has_preprocess is True
    # Row "a" completes its append prefill on this step; row "b" decodes.
    batch = SimpleNamespace(
        req_ids=["a", "b"],
        num_reqs=2,
        input_ids=torch.tensor([101, 102]),
        positions=torch.tensor([0, 1]),
        is_prefilling_np=np.array([True, False]),
        num_computed_prefill_tokens_np=np.array([2, 0]),
        num_scheduled_tokens=np.array([1, 1]),
        prefill_len_np=np.array([3, 5]),
    )
    hidden = torch.randn(2, 8)
    buffers = [
        {
            "duplex": {
                "duplex_prompt_token_ids": [1, 2, 3],
                "special_token_ids": {"tts_bos_token_id": 151703, "turn_start_token_id": 151644},
            }
        },
        {},
    ]
    out = model.make_omni_output_mrv2(
        hidden,
        input_batch=batch,
        req_states=None,
        model_intermediate_buffer=buffers,
    )
    assert out.text_hidden_states is hidden
    assert out.multimodal_outputs["latent"] is hidden
    assert out.multimodal_outputs["duplex_prompt_token_ids"] == [[1, 2, 3], None]
    assert "tts_bos_token_id" in out.multimodal_outputs["meta"]
    assert out.multimodal_outputs["meta"]["tts_bos_token_id"][0].item() == 151703
    # Tokenizer constants fill rows that carry none (e.g. an append that built
    # no unit); a None entry would stick in that request's accumulated meta.
    assert out.multimodal_outputs["meta"]["tts_bos_token_id"][1].item() == 151703
    # Host-only metadata: a device tensor would cost a blocking H2D per row and key.
    assert out.multimodal_outputs["meta"]["tts_bos_token_id"][0].device.type == "cpu"

    # Pure decode steps leave the accumulated prompt/meta snapshot untouched.
    batch.is_prefilling_np = np.array([False, False])
    out = model.make_omni_output_mrv2(
        hidden,
        input_batch=batch,
        req_states=None,
        model_intermediate_buffer=buffers,
    )
    assert "duplex_prompt_token_ids" not in out.multimodal_outputs
    assert "meta" not in out.multimodal_outputs
    assert out.multimodal_outputs["latent"] is hidden


def test_mrv2_turn_thinker_retains_native_multimodal_path(mocker):
    assert _model(mocker, session="turn").has_preprocess is False


def test_mrv2_thinker_custom_sampler_and_lifecycle(mocker):
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.mrv2_sampling import (
        MiniCPMO45DuplexSampler,
    )
    from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

    model = _model(mocker, session="duplex")
    base_sampler = mocker.MagicMock()
    sampler, _ = model.mrv2_custom_sampler(base_sampler)
    assert isinstance(sampler, MiniCPMO45DuplexSampler)
    assert model._mrv2_duplex_sampler is sampler

    state_mock = mocker.MagicMock(spec=OmniModelState)
    state_mock.model = model
    resolved = OmniModelState.custom_sampler(state_mock, base_sampler)
    assert resolved is not None
    assert isinstance(resolved[0], MiniCPMO45DuplexSampler)

    model.preprocess(
        torch.tensor([1]),
        torch.randn(1, 4),
        req_id="req-1",
        duplex={"data_plane": True, "session_id": "s1", "epoch": 0, "seq": 1},
    )
    assert "req-1" in model._mrv2_sampling_infos

    model.on_requests_finished({"req-1"})
    assert "req-1" not in model._mrv2_sampling_infos


def test_mrv2_cancel_before_first_prefill_completes_frees_session(mocker):
    """A request cancelled mid-prefill never reaches the sampler, so preprocessing owns its session."""
    model = _model(mocker, session="duplex")
    helper = SimpleNamespace(
        sessions={},
        take_staged_prefill=lambda *_: None,
        _decode_audio_payload=lambda _: None,
        frame_kwargs=mocker.MagicMock(side_effect=ValueError("stop after the session exists")),
    )
    model._minicpmo45_duplex_data_plane_helper = helper
    mocker.patch.object(
        model,
        "_minicpmo45_duplex_session_state",
        side_effect=lambda h, sid, _: h.sessions.setdefault(sid, object()),
    )
    model.preprocess(
        torch.tensor([1]),
        torch.randn(1, 4),
        req_id="req-1",
        duplex={"data_plane": True, "session_id": "s1", "epoch": 0, "seq": 1, "payload": {}},
    )
    assert "s1" in helper.sessions

    model.on_requests_finished({"req-1"})
    assert helper.sessions == {}


def test_mrv2_duplex_thinker_batches_prefill_appends(mocker):
    """MRv2 hands its prefill rows to the V1 cross-session ``preprocess_batch``."""
    model = _model(mocker, session="duplex")
    batch = mocker.patch.object(model, "preprocess_batch")
    infos = [
        {"req_id": "a", "duplex": {"data_plane": True, "session_id": "s1"}},
        {"req_id": "b", "duplex": {"data_plane": True, "session_id": "s2"}},
        {"req_id": "c"},  # not a duplex row
        {},
    ]
    model.preprocess_batch_mrv2(req_infos=infos, device=torch.device("cpu"))
    kwargs = batch.call_args.kwargs
    assert kwargs["req_ids"] == ["a", "b"]
    assert kwargs["model_intermediate_buffer"] == {"a": infos[0], "b": infos[1]}
    batch.reset_mock()
    model.preprocess_batch_mrv2(req_infos=[{"req_id": "c"}], device=torch.device("cpu"))
    batch.assert_not_called()


@pytest.mark.parametrize("v2,session,keeps", [(True, "duplex", True), (True, "turn", False), (False, "duplex", False)])
def test_only_the_mrv2_duplex_thinker_keeps_native_multimodal_inputs(mocker, v2, session, keeps):
    # Chat requests with media share the duplex Thinker; turn MRv2 has no preprocess and V1 encodes first.
    assert _model(mocker, v2=v2, session=session).preprocess_keeps_mm_inputs is keeps


@pytest.mark.parametrize("v2,session", [(True, "turn"), (False, "duplex")])
def test_mrv2_batch_preprocess_hook_is_duplex_mrv2_only(mocker, v2, session):
    assert not hasattr(_model(mocker, v2=v2, session=session), "preprocess_batch_mrv2")


def test_mrv2_thinker_returns_drafter_layers_with_hidden_states(mocker):
    from vllm.model_executor.models.interfaces import EagleModelMixin, SupportsEagle3, supports_eagle3

    class _Decoder(torch.nn.Module, EagleModelMixin):
        start_layer = 0

    model = _model(mocker)
    # As the Thinker's Qwen3ForCausalLM: the mixin decoder sits at ``.model``.
    language_model = torch.nn.Module()
    language_model.model = _Decoder()
    model.thinker.get_language_model = lambda: language_model
    assert supports_eagle3(model)
    # Drafters share the Thinker decoder's embedding and lm_head.
    assert model.get_language_model() is language_model
    # The turn Thinker keeps the runner's samplers, so drafts get verified.
    assert model.mrv2_custom_sampler(object()) is None

    hidden, aux = torch.randn(3, 4), [torch.randn(3, 4) for _ in range(3)]
    model.thinker.forward = mocker.Mock(return_value=hidden)
    assert model(torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2])) is hidden

    # The guard counts the layers the decoder stores, so a repeated id is one layer.
    model.set_aux_hidden_state_layers((17, 9, 1, 9))
    assert language_model.model.aux_hidden_state_layers == (1, 9, 17)
    model.thinker.forward = mocker.Mock(return_value=(hidden, aux))
    out_hidden, out_aux = model(torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2]))
    assert out_hidden is hidden
    assert out_aux is aux
    # A Thinker tuple that isn't (hidden, one tensor per drafter layer) fails loudly.
    model.thinker.forward = mocker.Mock(return_value=(aux[0], hidden))
    with pytest.raises(RuntimeError, match="3 target layers"):
        model(torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2]))
    # A decoder that keeps no layer list fails at load with its type in the message.
    mocker.patch.object(SupportsEagle3, "set_aux_hidden_state_layers")
    model.thinker.get_language_model = lambda: torch.nn.Module()
    with pytest.raises(RuntimeError, match="Module keeps no aux_hidden_state_layers"):
        model.set_aux_hidden_state_layers((1, 9, 17))


def test_v1_thinker_refuses_drafter_layers(mocker):
    model = _model(mocker, v2=False)
    with pytest.raises(NotImplementedError, match="model_runner: v2"):
        model.set_aux_hidden_state_layers((1, 9, 17))
    assert model._returns_aux_hidden_states is False


def test_talker_stage_hands_drafter_hooks_to_the_talker(mocker):
    model = object.__new__(omni.MiniCPMO45OmniForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model_stage = "tts"
    model.thinker = None
    model.talker = mocker.Mock()
    assert model.get_language_model() is model.talker.get_language_model.return_value
    model.set_aux_hidden_state_layers((2, 10, 17))
    model.talker.set_aux_hidden_state_layers.assert_called_once_with((2, 10, 17))
