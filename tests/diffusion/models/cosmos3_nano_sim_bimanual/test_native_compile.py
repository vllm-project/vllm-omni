# SPDX-License-Identifier: Apache-2.0
"""Admission contracts for the native Inductor path and Sim parallel execution."""

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.config import (
    resolve_ar_compile_mode,
    validate_sim_parallel_config,
)


@pytest.mark.parametrize("source", ["direct", "model_config", "custom_pipeline_args"])
@pytest.mark.parametrize("enabled,mode", [(False, "default"), (True, "reduce-overhead")])
def test_cuda_graph_flag_selects_compile_mode(source, enabled, mode):
    options = {"use_cuda_graphs": enabled}
    config = SimpleNamespace(**(options if source == "direct" else {source: options}))
    assert resolve_ar_compile_mode(config) == mode
    assert resolve_ar_compile_mode(SimpleNamespace()) == "default"


@pytest.mark.parametrize("override", [dict(enforce_eager=True), dict(diffusion_compile_granularity="full")])
def test_native_capture_rejects_inactive_or_unvalidated_compilation(override):
    with pytest.raises(ValueError):
        resolve_ar_compile_mode(SimpleNamespace(model_config={"use_cuda_graphs": True}, **override))


@pytest.mark.parametrize("value", ["true", "false", 0, 1])
def test_cuda_graph_flag_requires_boolean(value):
    with pytest.raises(ValueError, match="must be a boolean"):
        resolve_ar_compile_mode(SimpleNamespace(model_config={"use_cuda_graphs": value}))


@pytest.mark.parametrize("mode", ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
def test_old_mode_selector_fails_with_migration_message(mode):
    with pytest.raises(ValueError, match="replaced by the boolean use_cuda_graphs"):
        resolve_ar_compile_mode(SimpleNamespace(model_config={"ar_compile_mode": mode}))


@pytest.mark.parametrize("source", ["direct", "model_config", "custom_pipeline_args"])
def test_hsdp_rejects_cuda_graphs(source):
    options = {"use_cuda_graphs": True}
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(use_hsdp=True, hsdp_shard_size=2),
        **(options if source == "direct" else {source: options}),
    )
    with pytest.raises(ValueError, match="HSDP is currently incompatible with CUDA graphs"):
        resolve_ar_compile_mode(config)


@pytest.mark.parametrize("options", [{}, {"use_cuda_graphs": False}])
def test_hsdp_allows_compilation_without_cuda_graphs(options):
    config = SimpleNamespace(
        model_config=options,
        parallel_config=SimpleNamespace(use_hsdp=True, hsdp_shard_size=2),
    )
    assert resolve_ar_compile_mode(config) == "default"


def test_tensor_parallel_allows_cuda_graphs():
    config = SimpleNamespace(
        model_config={"use_cuda_graphs": True},
        parallel_config=SimpleNamespace(use_hsdp=False, tensor_parallel_size=2),
    )
    assert resolve_ar_compile_mode(config) == "reduce-overhead"


@pytest.mark.parametrize(
    "field",
    [
        "sequence_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "text_encoder_tp_size",
        "vae_patch_parallel_size",
    ],
)
def test_unsupported_parallel_paths_fail_before_loading_weights(field):
    config = SimpleNamespace(parallel_config=SimpleNamespace(**{field: 2}))
    with pytest.raises(ValueError, match=field):
        validate_sim_parallel_config(config)


@pytest.mark.parametrize(
    "parallel",
    [
        dict(tensor_parallel_size=2),
        dict(tensor_parallel_size=4),
        dict(use_hsdp=True, hsdp_shard_size=2),
        dict(data_parallel_size=None),
    ],
)
def test_model_parallel_paths_are_not_blanket_disabled(parallel):
    validate_sim_parallel_config(SimpleNamespace(parallel_config=SimpleNamespace(**parallel)))


def test_dense_model_does_not_claim_expert_parallelism():
    with pytest.raises(ValueError, match="no experts"):
        validate_sim_parallel_config(SimpleNamespace(parallel_config=SimpleNamespace(enable_expert_parallel=True)))


@pytest.mark.parametrize("frame_causal", [False, True])
def test_committed_history_survives_reuse_of_graph_output_storage(frame_causal):
    import torch

    from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.pipeline_cosmos3_nano_sim_bimanual import (
        Cosmos3NanoSimBimanualPipeline,
    )
    from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.transformer_cosmos3_nano_sim_bimanual import (
        Cosmos3NanoSimBimanualTransformerOutput,
    )

    # A graph replay reuses its output buffers. Model that lifetime without a GPU.
    key, value = torch.ones(1, 1, 2), torch.full((1, 1, 2), 2.0)
    video = torch.zeros(1)

    class ReusedOutputs:
        _inductor_cudagraphs = True

        def __call__(self, *args, **kwargs):
            return Cosmos3NanoSimBimanualTransformerOutput(video=video, current_kv=[(key, value)])

    pipeline = Cosmos3NanoSimBimanualPipeline.__new__(Cosmos3NanoSimBimanualPipeline)
    pipeline.transformer = ReusedOutputs()
    pipeline._ar_diffusion_kv_state = None
    pipeline.manifest = SimpleNamespace(conditioning_tokens_per_frame=0, sink_frames=0, window_frames=4)
    state = SimpleNamespace(dense_kv_by_branch={})
    geometry = SimpleNamespace(tokens_per_frame=lambda _: 1)

    def forward(commit):
        return pipeline._transformer_forward(
            state,
            torch.zeros(1, 1, 1, 1, 1),
            torch.zeros(1),
            geometry=geometry,
            text_kv=[],
            real_text_kv_len=1,
            frame_start=0,
            fps=30.0,
            condition_vision=commit,
            commit_current=commit,
            frame_causal=frame_causal,
            action_latents=torch.zeros(1),
            action_domain_ids=torch.zeros(1),
            null_action_frame_indexes=(),
        )

    first = forward(True)
    assert first.video is video  # The projection is outside the compiled blocks.
    key.fill_(3)
    value.fill_(4)
    history = state.dense_kv_by_branch[pipeline._MAIN_BRANCH]
    torch.testing.assert_close(history[0][0], torch.ones_like(key))
    torch.testing.assert_close(history[0][1], torch.full_like(value, 2))
    forward(True)
    key.fill_(5)
    value.fill_(6)
    history = state.dense_kv_by_branch[pipeline._MAIN_BRANCH]
    torch.testing.assert_close(history[0][0], torch.tensor([[[1.0, 1.0], [3.0, 3.0]]]))
    torch.testing.assert_close(history[0][1], torch.tensor([[[2.0, 2.0], [4.0, 4.0]]]))
    denoise = forward(False)
    assert denoise.current_kv[0][0] is key  # Transient denoise outputs need no ownership copy.
    assert state.dense_kv_by_branch[pipeline._MAIN_BRANCH] is history
