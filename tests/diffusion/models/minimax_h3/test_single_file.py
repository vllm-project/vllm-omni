# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def checkpoint_tensors():
    # One DiT block and one text-refiner block; independent hand-sized fixture.
    shapes = {
        "adaln_t_table": (3, 2),
        "adaln_basis": (2, 4),
        "adaln_mean": (4,),
        "video_patch_proj.weight": (8, 8),
        "video_patch_proj.bias": (8,),
        "audio_patch_proj.weight": (8, 2),
        "audio_patch_proj.bias": (8,),
        "condition_proj.weight": (8, 6),
        "condition_proj.bias": (8,),
        "rope.inv_freq": (2,),
        "token_refiner.final_norm.weight": (8,),
        "blocks.0.adaln_proj.linear.weight": (144, 2),
        "blocks.0.adaln_proj.linear.bias": (144,),
        "final_layer.adaln_proj.linear.weight": (16, 2),
        "final_layer.adaln_proj.linear.bias": (16,),
        "final_layer.norm.weight": (8,),
        "final_layer.video_out.weight": (8, 8),
        "final_layer.video_out.bias": (8,),
        "final_layer.audio_out.weight": (2, 8),
        "final_layer.audio_out.bias": (2,),
    }
    for prefix in ("blocks.0", "token_refiner.blocks.0"):
        for suffix, shape in {
            "norm1.weight": (8,),
            "norm2.weight": (8,),
            "attn.q_norm.weight": (4,),
            "attn.k_norm.weight": (4,),
            "attn.qkv_proj.weight": (24, 8),
            "attn.out_proj.weight": (8, 8),
            "mlp.fc1.weight": (32, 8),
            "mlp.fc2.weight": (8, 16),
        }.items():
            shapes[f"{prefix}.{suffix}"] = shape
    return {name: torch.zeros(shape, dtype=torch.bfloat16) for name, shape in shapes.items()}


def test_inspect_compressed_checkpoint_derives_native_architecture(tmp_path, checkpoint_tensors):
    from vllm_omni.diffusion.models.minimax_h3.single_file import MiniMaxH3SingleFileSpec

    path = tmp_path / "hybrid.safetensors"
    save_file(checkpoint_tensors, path)
    spec = MiniMaxH3SingleFileSpec.from_file(path)
    config = spec.transformer_config
    assert config["hidden_size"] == 8
    assert config["num_layers"] == 1
    assert config["token_refiner_num_layers"] == 1
    assert config["num_attention_heads"] == 2
    assert config["attention_head_dim"] == 4
    assert config["time_embed_dim"] == 2
    assert config["adaln_curve_grid"] == 3
    assert config["latents_dim"] == 2
    assert config["audio_latents_dim"] == 2
    assert config["text_dim"] == 6
    assert config["ffn_hidden_size"] == 16


@pytest.mark.parametrize(
    "change, message",
    [
        ("missing_projection", "missing"),
        ("bad_projection_shape", "shape"),
        ("quantized", "quantized"),
        ("unknown_tensor", "unsupported"),
        ("missing_block", "contiguous"),
        ("invalid_grid", "curve"),
        ("dense_embedder", "unsupported"),
        ("incomplete_basis", "basis"),
    ],
)
def test_inspect_rejects_incomplete_or_unsupported_files(tmp_path, checkpoint_tensors, change, message):
    from vllm_omni.diffusion.models.minimax_h3.single_file import MiniMaxH3SingleFileSpec

    tensors = checkpoint_tensors
    if change == "missing_projection":
        del tensors["final_layer.audio_out.weight"]
    elif change == "bad_projection_shape":
        tensors["blocks.0.adaln_proj.linear.weight"] = torch.zeros(144, 4, dtype=torch.bfloat16)
    elif change == "quantized":
        tensors["blocks.0.attn.qkv_proj.weight"] = tensors["blocks.0.attn.qkv_proj.weight"].to(torch.int8)
    elif change == "unknown_tensor":
        tensors["text_encoder.unexpected.weight"] = torch.ones(1, dtype=torch.bfloat16)
    elif change == "missing_block":
        tensors.update(
            {
                key.replace("blocks.0.", "blocks.2."): value.clone()
                for key, value in list(tensors.items())
                if key.startswith("blocks.0.")
            }
        )
    elif change == "invalid_grid":
        tensors["adaln_t_table"] = torch.zeros(1, 2, dtype=torch.bfloat16)
    elif change == "dense_embedder":
        tensors["time_embedder.proj_in.weight"] = torch.ones(4, 4, dtype=torch.bfloat16)
    elif change == "incomplete_basis":
        del tensors["adaln_mean"]
    path = tmp_path / "invalid.safetensors"
    save_file(tensors, path)
    with pytest.raises(ValueError, match=message):
        MiniMaxH3SingleFileSpec.from_file(path)


def test_inspect_accepts_curve_without_optional_adapter_basis(tmp_path, checkpoint_tensors):
    from vllm_omni.diffusion.models.minimax_h3.single_file import MiniMaxH3SingleFileSpec

    del checkpoint_tensors["adaln_basis"]
    del checkpoint_tensors["adaln_mean"]
    path = tmp_path / "comfy.safetensors"
    save_file(checkpoint_tensors, path)
    assert MiniMaxH3SingleFileSpec.from_file(path).transformer_config["time_embed_dim"] == 2


@pytest.mark.parametrize("model_class", ["MiniMaxH3Pipeline", "MiniMaxH3ModularPipeline"])
def test_single_file_config_preserves_h3_architecture_and_reference_inputs(tmp_path, checkpoint_tensors, model_class):
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    path = tmp_path / "hybrid.safetensors"
    save_file(checkpoint_tensors, path)
    config = OmniDiffusionConfig(model=str(path), model_class_name=model_class)
    config.enrich_config()
    assert config.model_class_name == "MiniMaxH3Pipeline"
    assert config.diffusion_load_format == "default"
    assert config.tf_model_config.to_dict()["adaln_curve_grid"] == 3
    assert config.tf_model_config.to_dict()["hidden_size"] == 8
    assert config.supports_multimodal_inputs
    assert config.supports_mixed_reference_inputs


def test_single_file_resolves_serving_stage_without_hub_discovery(tmp_path, checkpoint_tensors, monkeypatch):
    from vllm_omni.config.config_factory import StageConfigFactory
    from vllm_omni.config.resolver import resolve_omni_config

    def unexpected_discovery(*args, **kwargs):
        pytest.fail("single-file H3 must not resolve its local filename as a Hub repository")

    monkeypatch.setattr(StageConfigFactory, "create_from_model", unexpected_discovery)
    path = tmp_path / "hybrid.safetensors"
    save_file(checkpoint_tensors, path)
    resolved = resolve_omni_config(
        str(path),
        trust_remote_code=False,
        deploy_config_path=None,
        cli_overrides={"model_class_name": "MiniMaxH3Pipeline"},
        stage_overrides=None,
        strategy_config_path=None,
    )
    assert len(resolved.stage_configs) == 1


@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va", None])
@pytest.mark.parametrize("as_symlink", [False, True])
@pytest.mark.parametrize(
    ("base_source", "expected_revision"),
    [
        ("local", None),
        ("default", "42ed227ee7df40d41602854ae760620d6eb651fe"),
        ("explicit", "caller-selected-revision"),
        ("custom-remote", None),
    ],
)
def test_pipeline_loads_selected_file_for_each_task(
    tmp_path, checkpoint_tensors, monkeypatch, task, as_symlink, base_source, expected_revision
):
    from vllm.config.load import LoadConfig
    from vllm.distributed import parallel_state

    from tests.diffusion.models.minimax_h3.test_minimax_h3_quantization import _FakeAttention
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
    from vllm_omni.diffusion.model_loader.host_weight_plan import build_checkpoint_binding_plan
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_blocks as blocks
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline_module

    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: SimpleNamespace(world_size=1, rank_in_group=0))
    monkeypatch.setattr(blocks, "Attention", _FakeAttention)
    monkeypatch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: 1)
    # Upstream probes the accelerator for the SM120 AdaLN-offload default; keep these CPU-only tests off CUDA.
    monkeypatch.setattr(h3.current_omni_platform, "get_device_capability", lambda *args, **kwargs: (8, 9))
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(pipeline_module, "get_local_device", lambda: torch.device("cpu"))
    component_paths = []

    def vae(path, *args, **kwargs):
        component_paths.append(path)
        return torch.nn.Module()

    for name in ("MiniMaxH3VideoVAE", "MiniMaxH3AudioVAE"):
        monkeypatch.setattr(pipeline_module, name, vae)
    base = tmp_path / "base"
    for partition, tasks in (("FL2VA", ["t2va", "fl2va"]), ("Ref2VA", ["ref2va"])):
        root = base / partition
        root.mkdir(parents=True)
        (root / "model_index.json").write_text(
            json.dumps({"_minimax_h3": {"partition": partition.lower(), "tasks": tasks}})
        )
    base_download_calls = []

    def download_base(**kwargs):
        base_download_calls.append(kwargs)
        return str(base)

    monkeypatch.setattr(pipeline_module, "download_weights_from_hf_specific", download_base)
    checkpoint_tensors["adaln_t_table"] = torch.tensor([[0, 4], [2, 8], [10, -4]], dtype=torch.bfloat16)
    checkpoint_tensors["blocks.0.attn.qkv_proj.weight"] = torch.arange(192).reshape(24, 8).to(torch.bfloat16)
    path = tmp_path / "hybrid[beta5].safetensors"
    save_file(checkpoint_tensors, path)
    if as_symlink:
        blob = tmp_path / "blobs" / "blob_without_extension"
        blob.parent.mkdir()
        path.rename(blob)
        path.symlink_to(blob)
    save_file(checkpoint_tensors, tmp_path / "unrelated.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "unrelated.safetensors" for key in checkpoint_tensors}})
    )
    config = OmniDiffusionConfig(
        model=str(path),
        model_class_name="MiniMaxH3Pipeline",
        task_type=task,
        model_loaded={"text_encoder": False, "vae_encoder": True},
        custom_pipeline_args={
            "local": {"base_model": str(base)},
            "default": {},
            "explicit": {"base_revision": expected_revision},
            "custom-remote": {"base_model": "Contoso/H3-Components"},
        }[base_source],
    )
    config.enrich_config()
    pipeline = pipeline_module.MiniMaxH3Pipeline(od_config=config)
    loader = DiffusersPipelineLoader(LoadConfig(), config)
    sources = pipeline.weights_sources
    assert len(sources) == (2 if task is None else 1)
    weights = (item for source in sources for item in loader._get_weights_iterator(source, model=pipeline))
    loaded = pipeline.load_weights(weights)
    expected = torch.tensor([[1.0, 6.0]])
    for requested in {"t2va", "fl2va", "ref2va"} if task is None else {task}:
        transformer = pipeline._transformer_for_task(requested)
        torch.testing.assert_close(transformer.time_embedder(torch.tensor([0.25])), expected, rtol=0, atol=0)
    assert "transformer.time_embedder.table" in loaded
    assert all(source.model_or_path == str(path.parent) for source in sources)
    assert set(component_paths) == {str(base / "FL2VA" / component) for component in ("video_vae", "audio_vae")}
    if base_source == "local":
        assert base_download_calls == []
    else:
        assert len(base_download_calls) == 1
        expected_model = (
            "Contoso/H3-Components" if base_source == "custom-remote" else "MiniMaxAI/MiniMax-H3"
        )
        assert base_download_calls[0]["model_name_or_path"] == expected_model
        assert base_download_calls[0]["revision"] == expected_revision
    result = build_checkpoint_binding_plan(
        pipeline,
        dit_modules=tuple((name, getattr(pipeline, name)) for name in pipeline._dit_modules),
        sources=sources,
        model_path=str(path),
        tensor_parallel_size=1,
        online_quantization=False,
    )
    assert result.plan is not None, result.fallback_reason
    targets = dict(pipeline.named_parameters()) | dict(pipeline.named_buffers())
    for name, binding in result.plan.bindings.items():
        assert binding.file_path == str(path)
        stored = checkpoint_tensors[binding.checkpoint_key]
        restored = stored if binding.transform is None else binding.transform(stored)
        assert restored.dtype == targets[name].dtype
        torch.testing.assert_close(restored, targets[name], rtol=0, atol=0)


@pytest.mark.parametrize("partition", ["fl2va", "ref2va", "combined"])
def test_single_file_base_download_excludes_dense_dits(monkeypatch, tmp_path, partition):
    from fnmatch import fnmatch

    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline_module

    captured = {}

    def download(**kwargs):
        captured.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(pipeline_module, "download_weights_from_hf_specific", download)
    monkeypatch.setattr(pipeline_module, "is_minimax_h3_modular", lambda *args: False)
    pipeline_module._resolve_minimax_h3_model_root(
        "MiniMaxAI/MiniMax-H3",
        "pinned-base-revision",
        partition,
        load_text_encoder=True,
        load_transformer=False,
    )
    patterns = captured["allow_patterns"]
    for folder in ("FL2VA", "Ref2VA"):
        assert not any(fnmatch(f"{folder}/transformer/model.safetensors", p) for p in patterns)
    shared = "FL2VA"
    for component in ("text_encoder", "video_vae", "audio_vae", "tokenizer", "processor"):
        assert any(fnmatch(f"{shared}/{component}/config.json", p) for p in patterns)
    assert captured["revision"] == "pinned-base-revision"
