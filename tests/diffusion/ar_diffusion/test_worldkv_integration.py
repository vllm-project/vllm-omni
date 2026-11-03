# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real session ownership and LingBot DMD cache routing without CUDA imports."""

import importlib.util
import logging
import sys
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType

import pytest
import torch

from tests.diffusion.ar_diffusion.test_worldkv import worldkv

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
ROOT = Path(__file__).parents[3]
AR = "vllm_omni.experimental.ar_diffusion"


@pytest.fixture
def runtime(monkeypatch):
    def stub(name, **attrs):
        if "." in name and name.rsplit(".", 1)[0] not in sys.modules:
            stub(name.rsplit(".", 1)[0])
        module = ModuleType(name)
        module.__path__ = [str(ROOT / name.replace(".", "/"))]
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def load(name):
        spec = importlib.util.spec_from_file_location(name, ROOT / (name.replace(".", "/") + ".py"))
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    for name in ["vllm", "vllm_omni", "vllm_omni.diffusion", "vllm_omni.experimental", AR, AR + ".kv_cache"]:
        stub(name)
    stub("vllm.logger", init_logger=logging.getLogger)
    stub("vllm.v1.core.single_type_kv_cache_manager", SlidingWindowManager=object)

    @dataclass(frozen=True)
    class SlidingWindowSpec:
        pass

    stub("vllm.v1.kv_cache_interface", SlidingWindowSpec=SlidingWindowSpec)
    stub("vllm.v1.kv_cache_spec_registry", register_kv_cache_spec=lambda **kw: lambda cls: cls)
    monkeypatch.setitem(sys.modules, AR + ".worldkv", worldkv)
    load(AR + ".kv_cache.config")
    load(AR + ".kv_cache.paged")
    load(AR + ".kv_cache.paged_attention")
    stub("vllm_omni.diffusion.attention")
    load("vllm_omni.diffusion.attention.conditioning")
    load(AR + ".kv_cache.conditioning")
    load(AR + ".kv_cache.first_block")
    load(AR + ".kv_cache.segmented")
    return load(AR + ".kv_cache.state")


@dataclass
class Branch:
    name: str


@dataclass
class Adapter:
    request_id: str


class LifecycleCache:
    kv_branches = (Branch("positive"), Branch("negative"))
    cross_attention_lengths = {"text": 1}

    def begin_request(self, request_id):
        return Adapter(request_id)

    def end_request(self, adapter):
        pass

    def release_cross_attention(self, session_id):
        pass

    def retain_cross_attention(self, session_id, names):
        pass


@pytest.mark.parametrize("keep_cross", [(), ("text",)])
def test_runner_memory_is_per_session_branch_and_cleared_on_reset_close(runtime, keep_cross):
    def owner(name):
        cache = LifecycleCache()
        return runtime.ARDiffusionKVState(
            cache,
            name,
            {b.name: Adapter(name + b.name) for b in cache.kv_branches},
            num_layers=2,
        )

    a, b = owner("A"), owner("B")
    config = worldkv.WorldKVConfig(window_frames=6, sink_frames=2)
    original = a.worldkv("positive", config, tokens_per_frame=2)
    assert a.worldkv("positive", config, tokens_per_frame=2) is original
    assert b.worldkv("positive", config, tokens_per_frame=2) is not original
    assert a.worldkv("negative", config, tokens_per_frame=2) is not original
    with pytest.raises(ValueError, match="cannot change"):
        a.worldkv("positive", replace(config, window_frames=8), tokens_per_frame=2)
    with pytest.raises(ValueError, match="cannot change"):
        a.worldkv("positive", config, tokens_per_frame=4)
    a.reset(keep_cross_attention=keep_cross)
    assert original._closed
    replacement = a.worldkv("positive", config, tokens_per_frame=2)
    assert replacement is not original
    a.close()
    a.close()
    assert replacement._closed and not a._worldkv_memories
    with pytest.raises(RuntimeError, match="closed"):
        a.worldkv("positive", config, tokens_per_frame=2)
    assert not b.worldkv("positive", config, tokens_per_frame=2)._closed


@pytest.mark.parametrize("session_bound", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_lingbot_dmd_routes_only_enabled_cache_to_worldkv(runtime, monkeypatch, session_bound, enabled):
    from tests.diffusion.models.lingbot_world.test_lingbot_world_attention import _load_module

    transformer_module = _load_module()
    name = "vllm_omni.diffusion.models.lingbot_world.transformer"
    monkeypatch.setitem(sys.modules, name, transformer_module)
    context = ModuleType("vllm_omni.diffusion.forward_context")
    setattr(context, "set_forward_context_denoise_step_idx", lambda *args: nullcontext())
    monkeypatch.setitem(sys.modules, context.__name__, context)
    module_name = "vllm_omni.diffusion.models.lingbot_world.dmd_block"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / (module_name.replace(".", "/") + ".py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    transformer = transformer_module.CausalLingBotWorldTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=4,
        in_channels=36,
        out_channels=2,
        text_dim=6,
        freq_dim=4,
        ffn_dim=8,
        num_layers=2,
        sink_size=1,
        num_frames_per_block=1,
        sliding_window_num_frames=3,
    ).eval()
    cfg = worldkv.WorldKVConfig(window_frames=3, sink_frames=1) if enabled else None
    transformer.worldkv_config = cfg
    runner = module.LingBotDMDBlockRunner(transformer, device=torch.device("cpu"), enforce_eager=True)
    condition = torch.zeros(1, 36, 1, 4, 4)
    camera = torch.eye(4)

    class State:
        def worldkv(self, branch, config, *, tokens_per_frame):
            assert branch == "positive" and config is cfg and tokens_per_frame == 4
            return memory

        def get_kv_caches(self, branch, *, seq_len, commit_current):
            assert not enabled and branch == "positive" and seq_len == 4
            return [None, None]

    memory = worldkv.WorldKV(cfg, num_layers=2, tokens_per_frame=4) if enabled else None
    cache = transformer_module.LingBotTransformerCache([None, None], [None, None], worldkv=memory)
    ar = module.ARBlockContext(State(), [None, None], "positive") if session_bound else None
    actual = runner._block_cache(condition=condition, cache=cache, ar=ar, commit_current=False, camera_pose=camera)
    assert actual.worldkv is memory
    if enabled:
        torch.testing.assert_close(memory.pose, camera.double())
    else:
        assert actual.self_attention == [None, None]


@pytest.mark.parametrize("ratio", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("score", ["centroid", "pairwise_mean"])
def test_real_lingbot_transformer_retrieves_compressed_bank_and_rolls_back(monkeypatch, ratio, score):
    from tests.diffusion.models.lingbot_world.test_lingbot_world_attention import _load_module

    module = _load_module()
    model = module.CausalLingBotWorldTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=4,
        in_channels=36,
        out_channels=2,
        text_dim=6,
        freq_dim=4,
        ffn_dim=8,
        num_layers=2,
        sink_size=2,
        num_frames_per_block=2,
        sliding_window_num_frames=8,
    ).eval()
    model.worldkv_config = worldkv.WorldKVConfig(
        window_frames=8,
        sink_frames=2,
        retrieval_frames=2,
        bank_bytes=1 << 20,
        world_keep_ratio=ratio,
        world_score=score,
    )
    monkeypatch.setitem(sys.modules, AR + ".worldkv", worldkv)
    cache = model.allocate_cache(
        batch_size=1,
        latent_height=4,
        latent_width=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    memory = cache.worldkv
    assert memory is not None and not cache.self_attention
    generator = torch.Generator().manual_seed(17)
    text = torch.randn(1, 3, 6, generator=generator)
    retrieved = 0
    with torch.inference_mode():
        for start in range(0, 24, 2):
            hidden = torch.randn(1, 36, 2, 4, 4, generator=generator)
            camera = torch.randn(1, 384, 2, 4, 4, generator=generator)
            kwargs = dict(
                hidden_states=hidden,
                camera_hidden_states=camera,
                encoder_hidden_states=text,
                timestep=torch.zeros(1),
                start_frame=start,
                cache=cache,
            )
            outputs = []
            for commit in (False, False, True):
                memory.set_pose(torch.eye(4))
                outputs.append(model(**kwargs, update_cache=commit))
                retrieved += len(memory._retrieved)
                assert torch.isfinite(outputs[-1]).all()
            torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
            torch.testing.assert_close(outputs[0], outputs[2], atol=0, rtol=0)
            assert memory.next_frame == start + 2 and memory.bank_nbytes <= memory.config.bank_bytes
            for entry in memory.bank.values():
                assert all(rows.key.shape[1] == 4 + int(4 * ratio) for rows in entry.layers)
        assert retrieved > 0
        old_history, old_bank = memory.history, memory.bank

        def fail(*args, **kwargs):
            raise RuntimeError("head failed")

        monkeypatch.setattr(model.head, "forward", fail)
        memory.set_pose(torch.eye(4))
        with pytest.raises(RuntimeError, match="head failed"):
            model(**{**kwargs, "start_frame": 24}, update_cache=True)
        assert memory.history is old_history and memory.bank is old_bank and memory.next_frame == 24
