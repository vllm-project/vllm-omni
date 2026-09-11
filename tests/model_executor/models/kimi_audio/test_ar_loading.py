# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AR checkpoint routing using official names and synthetic small CPU tensors.

The real vLLM mapper, recursive loader, and final completeness check run here.
Distributed layer constructors and auxiliary checkpoint I/O are substituted;
this does not exercise the AR forward or pretrained/GPU inference.
"""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config
from vllm.model_executor import model_loader
from vllm.model_executor.layers import layernorm, vocab_parallel_embedding
from vllm.model_executor.models import qwen2

from tests.model_executor.models.kimi_audio.runtime import cpu_pp_group as cpu_pp_group
from tests.model_executor.models.kimi_audio.runtime import registered_model_runtime as registered_model_runtime
from vllm_omni.model_executor.models.kimi_audio.kimi_audio import KimiAudioForConditionalGeneration
from vllm_omni.model_executor.models.kimi_audio.kimi_audio_ar_stage import KimiAudioARStage, KimiAudioInputEncoder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.omni]
MANIFEST = Path(__file__).parent / "fixtures/ar_checkpoint_manifest.json"


@pytest.fixture
def ar_runtime(monkeypatch, tmp_path, registered_model_runtime):
    manifest = json.loads(MANIFEST.read_text())
    config = Qwen2Config(**manifest["config"])
    # Keep the official 28 + 6 layers and branch point; shrink only widths.
    config.hidden_size = 8
    config.intermediate_size = 12
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.vocab_size = 64
    config.pad_token_id, config.bos_token_id, config.eos_token_id = 0, 1, 2
    config.kimia_adaptor_input_dim = 12
    config.kimia_audio_output_vocab = 17
    config.kimia_text_output_vocab = 23
    runtime = SimpleNamespace(constructions=[], loads=[], downloads=[], local_path=str(tmp_path))
    runtime.config = registered_model_runtime(
        model_config=SimpleNamespace(hf_config=config, model_stage="kimi_audio_ar"),
        quant_config=None,
        cache_config=object(),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        additional_config={"kimi_audio": {"glm_tokenizer_path": str(tmp_path)}},
        load_config=SimpleNamespace(download_dir="checkpoint-cache"),
    )

    def packed_linear(input_size, sizes, shards, *, bias=False):
        layer = torch.nn.Linear(input_size, sum(sizes), bias=bias)

        def copy(param, value):
            index = shards.index(value.shard_id)
            start = sum(sizes[:index])
            with torch.no_grad():
                param[start : start + sizes[index]].copy_(value)

        for param in layer.parameters():
            param.weight_loader = copy
        return layer

    class TinyDecoder(torch.nn.Module):
        def __init__(self, *, config, cache_config, quant_config, prefix):
            super().__init__()
            runtime.constructions.append((prefix, config, cache_config))
            hidden, intermediate = config.hidden_size, config.intermediate_size
            kv = hidden // config.num_attention_heads * config.num_key_value_heads
            self.self_attn = torch.nn.ModuleDict(
                {
                    "qkv_proj": packed_linear(hidden, [hidden, kv, kv], ["q", "k", "v"], bias=True),
                    "o_proj": torch.nn.Linear(hidden, hidden, bias=False),
                }
            )
            self.mlp = torch.nn.ModuleDict(
                {
                    "gate_up_proj": packed_linear(hidden, [intermediate, intermediate], [0, 1]),
                    "down_proj": torch.nn.Linear(intermediate, hidden, bias=False),
                }
            )
            self.input_layernorm = torch.nn.RMSNorm(hidden, eps=config.rms_norm_eps)
            self.post_attention_layernorm = torch.nn.RMSNorm(hidden, eps=config.rms_norm_eps)

    class TinyEmbedding(torch.nn.Embedding):
        def __init__(self, vocab_size, hidden_size, **kwargs):
            super().__init__(vocab_size, hidden_size)

    def load_glm(owner, path):
        runtime.loads.append(("glm", path))
        owner.audio_tokenizer = torch.nn.Linear(1, 1, bias=False)
        return {"audio_tokenizer.weight"}

    def load_whisper(owner):
        runtime.loads.append(("whisper", owner.vllm_config))
        owner.whisper_encoder = torch.nn.Linear(1, 1, bias=False)
        return {"whisper_encoder.weight"}

    def download(**kwargs):
        runtime.downloads.append(kwargs)
        return str(tmp_path)

    # Substitute only the auxiliary checkpoint I/O boundary, even when the
    # local runner has no importable vLLM package/native dependencies.
    weight_utils = ModuleType("vllm_omni.model_executor.model_loader.weight_utils")
    weight_utils.download_weights_from_hf_specific = download
    monkeypatch.setitem(sys.modules, weight_utils.__name__, weight_utils)
    monkeypatch.setattr(qwen2, "Qwen2DecoderLayer", TinyDecoder)
    monkeypatch.setattr(layernorm, "RMSNorm", torch.nn.RMSNorm)
    monkeypatch.setattr(vocab_parallel_embedding, "VocabParallelEmbedding", TinyEmbedding)
    monkeypatch.setattr(vocab_parallel_embedding, "ParallelLMHead", TinyEmbedding)
    monkeypatch.setattr(KimiAudioInputEncoder, "load_glm_weights", load_glm)
    monkeypatch.setattr(KimiAudioInputEncoder, "load_whisper_weights", load_whisper)

    # Source shapes follow the official unfused checkpoint, independently of
    # the implementation's packed parameter names. Include every official key.
    shapes = {
        "self_attn.q_proj.weight": (8, 8),
        "self_attn.q_proj.bias": (8,),
        "self_attn.k_proj.weight": (4, 8),
        "self_attn.k_proj.bias": (4,),
        "self_attn.v_proj.weight": (4, 8),
        "self_attn.v_proj.bias": (4,),
        "self_attn.o_proj.weight": (8, 8),
        "self_attn.rotary_emb.inv_freq": (1,),
        "input_layernorm.weight": (8,),
        "post_attention_layernorm.weight": (8,),
        "mlp.gate_proj.weight": (12, 8),
        "mlp.up_proj.weight": (12, 8),
        "mlp.down_proj.weight": (8, 12),
        "model.embed_tokens.weight": (64, 8),
        "lm_head.weight": (64, 8),
        "mimo_output.weight": (64, 8),
        "model.norm.weight": (8,),
        "model.mimo_norm.weight": (8,),
        "model.vq_adaptor.layers.0.weight": (8, 12),
        "model.vq_adaptor.layers.0.bias": (8,),
        "model.vq_adaptor.layers.3.weight": (8, 8),
        "model.vq_adaptor.layers.3.bias": (8,),
        "model.vq_adaptor.layers.4.weight": (8,),
        "model.vq_adaptor.layers.4.bias": (8,),
    }
    generator = torch.Generator().manual_seed(913)
    runtime.weights = {}
    for name in manifest["weight_names"]:
        shape_key = name.split(".", 3)[-1] if name.startswith(("model.layers.", "model.mimo_layers.")) else name
        runtime.weights[name] = torch.randn(shapes[shape_key], generator=generator)
    return runtime


def test_complete_dual_stream_checkpoint_and_input_ownership(ar_runtime):
    runtime = ar_runtime
    model = KimiAudioForConditionalGeneration(vllm_config=runtime.config, prefix="ar")
    stage = model.model
    assert isinstance(stage, KimiAudioARStage)
    assert stage.vllm_config is not runtime.config
    assert stage.config.architectures == ["KimiAudioARStage"]
    assert runtime.config.model_config.model_arch == "KimiAudioForConditionalGeneration"
    assert runtime.config.model_config.hf_config.architectures != ["KimiAudioARStage"]
    assert runtime.loads == runtime.downloads == []
    assert list(stage.input_encoder.parameters()) == []
    assert len(stage.layers) == 28 and stage.branch_layer == 21 and len(stage.mimo_layers) == 6
    prefixes = [prefix for prefix, _, _ in runtime.constructions]
    assert len(prefixes) == len(set(prefixes)) == 34
    assert prefixes[0] == "ar.model.layers.0" and prefixes[-1] == "ar.model.mimo_layers.5"
    assert all(config.rope_parameters["rope_theta"] == 1e6 for _, config, _ in runtime.constructions)
    assert all(cache is runtime.config.cache_config for _, _, cache in runtime.constructions)

    loaded = model.load_weights(iter(runtime.weights.items()))
    model_loader.DefaultModelLoader.track_weights_loading(None, model, loaded)
    assert loaded == set(dict(model.named_parameters()))
    assert [kind for kind, _ in runtime.loads] == ["glm", "whisper"]
    assert runtime.loads[0][1] == runtime.local_path
    assert runtime.downloads == []
    assert stage.input_encoder.prefix == "ar.model.input_encoder"
    ids = torch.tensor([1, 2])
    torch.testing.assert_close(model.embed_input_ids(ids), stage.embed_tokens(ids))
    for name in ("lm_head", "mimo_output"):
        torch.testing.assert_close(getattr(stage, name).weight, runtime.weights[f"{name}.weight"])
    assert stage.lm_head.weight is not stage.mimo_output.weight
    for branch in ("layers", "mimo_layers"):
        for index, layer in enumerate(getattr(stage, branch)):
            source = f"model.{branch}.{index}"
            for suffix in ("weight", "bias"):
                expected = torch.cat(
                    [runtime.weights[f"{source}.self_attn.{q}_proj.{suffix}"] for q in ("q", "k", "v")]
                )
                torch.testing.assert_close(getattr(layer.self_attn.qkv_proj, suffix), expected)
            expected = torch.cat([runtime.weights[f"{source}.mlp.{part}_proj.weight"] for part in ("gate", "up")])
            torch.testing.assert_close(layer.mlp.gate_up_proj.weight, expected)


def test_default_glm_snapshot_resolution(ar_runtime):
    runtime = ar_runtime
    runtime.config.additional_config = {}
    stage = KimiAudioARStage(vllm_config=runtime.config)
    stage.load_weights(iter(runtime.weights.items()))
    assert runtime.downloads == [
        {
            "model_name_or_path": "THUDM/glm-4-voice-tokenizer",
            "revision": "a5f2404e63c84e92f5238908e1706316324ebafa",
            "cache_dir": "checkpoint-cache",
            "allow_patterns": ["config.json", "preprocessor_config.json", "model.safetensors"],
            "require_all": True,
        }
    ]


def test_missing_audio_head_is_visible_to_framework(ar_runtime):
    runtime = ar_runtime
    runtime.weights.pop("mimo_output.weight")
    stage = KimiAudioARStage(vllm_config=runtime.config)
    loaded = stage.load_weights(iter(runtime.weights.items()))
    with pytest.raises(ValueError, match="mimo_output.weight"):
        model_loader.DefaultModelLoader.track_weights_loading(None, stage, loaded)


def test_unexpected_audio_branch_weights_are_not_silently_skipped(ar_runtime):
    stage = KimiAudioARStage(vllm_config=ar_runtime.config)
    with pytest.raises(ValueError, match="unknown"):
        stage.load_weights(iter([("model.mimo_layers.0.unknown.weight", torch.zeros(1))]))
    assert ar_runtime.loads == ar_runtime.downloads == []


def test_dual_stream_rejects_single_stream_prefix_hashing(ar_runtime):
    ar_runtime.config.cache_config = SimpleNamespace(enable_prefix_caching=True)
    with pytest.raises(ValueError, match="enable_prefix_caching=False"):
        KimiAudioARStage(vllm_config=ar_runtime.config)
    assert ar_runtime.constructions == ar_runtime.loads == ar_runtime.downloads == []


def test_pp_weights_belong_to_one_rank(ar_runtime, cpu_pp_group, monkeypatch):
    from vllm.model_executor.models.interfaces import supports_pp
    from vllm.model_executor.models.utils import PPMissingLayer

    runtime = ar_runtime
    cpu_pp_group.world_size = 3
    runtime.config.parallel_config.pipeline_parallel_size = 3
    runtime.config.parallel_config.distributed_executor_backend = "mp"
    runtime.config.scheduler_config = SimpleNamespace(async_scheduling=False)
    runtime.config.compilation_config = SimpleNamespace(pass_config=SimpleNamespace(enable_sp=False))
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "22,3,3")
    owned = set()
    expected_bounds = [(0, 22, 0, 0), (22, 25, 0, 3), (25, 28, 3, 6)]
    for rank, bounds in enumerate(expected_bounds):
        cpu_pp_group.rank_in_group = rank
        cpu_pp_group.is_first_rank, cpu_pp_group.is_last_rank = rank == 0, rank == 2
        runtime.config.model_config.multimodal_config.skip_mm_profiling = False
        model = KimiAudioForConditionalGeneration(vllm_config=runtime.config)
        stage = model.model
        assert runtime.config.model_config.multimodal_config.skip_mm_profiling == (rank != 0)
        assert stage.vllm_config.model_config.multimodal_config.skip_mm_profiling == (rank != 0)
        assert supports_pp(model)
        assert (stage.start_layer, stage.end_layer, stage.mimo_start_layer, stage.mimo_end_layer) == bounds
        assert isinstance(stage.embed_tokens, PPMissingLayer) == (rank != 0)
        assert isinstance(stage.input_encoder, PPMissingLayer) == (rank != 0)
        assert isinstance(stage.vq_adaptor, PPMissingLayer) == (rank != 0)
        assert isinstance(stage.lm_head, PPMissingLayer) == (rank != 2)
        assert isinstance(stage.mimo_output, PPMissingLayer) == (rank != 2)
        loaded = model.load_weights(iter(runtime.weights.items()))
        model_loader.DefaultModelLoader.track_weights_loading(None, model, loaded)
        assert loaded == set(dict(model.named_parameters()))
        assert not owned.intersection(loaded)
        owned.update(loaded)
    assert [kind for kind, _ in runtime.loads] == ["glm", "whisper"]
    assert len(runtime.constructions) == 34
    cpu_pp_group.world_size, cpu_pp_group.rank_in_group = 1, 0
    cpu_pp_group.is_first_rank = cpu_pp_group.is_last_rank = True
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION")
    whole = KimiAudioForConditionalGeneration(vllm_config=runtime.config)
    assert owned == whole.load_weights(iter(runtime.weights.items()))
