# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Invalid action requests must leave compiled execution and the device usable."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


@pytest.mark.parametrize(
    "device", [pytest.param("cpu", marks=pytest.mark.cpu), pytest.param("cuda", marks=pytest.mark.cuda)]
)
@torch.inference_mode()
def test_invalid_action_request_then_valid_compiled_request(monkeypatch, device):
    if device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
            pytest.skip("Requires Hopper SM90")
        pytest.importorskip("flash_attn.cute")
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear

    from tests.diffusion.models.cosmos3.test_cosmos3_transformer import _tiny_cosmos3_config
    from vllm_omni.diffusion.attention import layer as layer_mod
    from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
    from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import AttentionConfig
    from vllm_omni.diffusion.models.cosmos3 import transformer_cosmos3 as cosmos
    from vllm_omni.diffusion.models.cosmos3.action import resolve_domain_id

    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(cosmos, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(cosmos, "_get_ulysses_state", lambda: (1, 0, None))
    monkeypatch.setattr(layer_mod, "build_parallel_attention_strategy", lambda **kw: NoParallelAttention())
    if device == "cpu":
        monkeypatch.setattr(
            layer_mod, "get_attn_backend_for_role", lambda **kw: (SDPABackend, kw["attention_config"].default)
        )
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    attention_config = AttentionConfig(presets={"dense": {"backend": "FLASH_ATTN"}}, layout={"default": "dense"})
    cfg = SimpleNamespace(
        tf_model_config=_tiny_cosmos3_config(
            num_hidden_layers=1,
            hidden_size=256,
            intermediate_size=512,
            head_dim=128,
            rope_scaling={"mrope_section": [24, 20, 20]},
            action_gen=True,
            max_action_dim=3,
            num_embodiment_domains=4,
        ),
        dtype=dtype,
        diffusion_attention_config=attention_config,
        parallel_config=SimpleNamespace(ring_degree=1, ulysses_degree=1),
        diffusion_kv_cache_dtype=None,
    )
    with set_current_diffusion_config(cfg):
        model = cosmos.Cosmos3VFMTransformer(cfg).to(device=device, dtype=dtype)
    model.post_load_weights()
    model.eval()
    torch.manual_seed(0)
    for weight in model.parameters():
        weight.fill_(1) if weight.ndim == 1 else weight.normal_(std=0.02)
    inputs = dict(
        hidden_states=torch.randn(1, 2, 1, 2, 2, device=device, dtype=dtype),
        timestep=torch.ones(1, device=device, dtype=dtype),
        text_ids=torch.zeros(1, 3, device=device, dtype=torch.long),
        text_mask=torch.ones(1, 3, device=device, dtype=torch.long),
        video_shape=(1, 2, 2),
        action_latents=torch.randn(1, 5, 3, device=device, dtype=dtype),
        action_domain_ids=torch.tensor([2], device=device),
    )
    reference = model(**inputs)
    model.reset_cache()
    graphs = []
    executions = []

    def backend(graph, example_inputs):
        # Check the captured graph too: no destructive device assertions.
        assert not any("_assert_async" in str(node.target) for node in graph.graph.nodes)
        compiled = torch._inductor.compile(graph, example_inputs)
        graphs.append(graph)

        def execute(*args):
            executions.append(graph)
            return compiled(*args)

        return execute

    torch._dynamo.reset()
    runner = model._attention_strategy_runner
    runner.compile(backend=backend, dynamic=True)
    # The public parser accepts this positive integer; model configuration
    # establishes the upper bound, which must be checked before graph entry.
    upper_bound = resolve_domain_id(domain_id=4)
    with override_forward_context(ForwardContext(denoise_step_idx=0, total_denoise_steps=1)):
        for invalid_ids in ([upper_bound], [-1], [0, 1]):
            for warmed in (False, True):
                if not warmed:
                    model.reset_cache()
                before = len(executions)
                before_graphs = len(graphs)
                inputs["action_domain_ids"] = torch.tensor(invalid_ids, device=device)
                with pytest.raises(ValueError, match="domain_id"):
                    model(**inputs)
                assert len(executions) == before
                assert len(graphs) == before_graphs
                if not warmed:
                    assert model.cached_kv is None
                if device == "cuda":
                    torch.accelerator.synchronize()  # No deferred device error.
                inputs["action_domain_ids"] = torch.tensor([2], device=device)
                torch.testing.assert_close(model(**inputs), reference, atol=0.005, rtol=0.005)
                assert len(executions) == before + 1
                if device == "cuda":
                    torch.accelerator.synchronize()
    assert len(graphs) == 1  # All valid requests reuse the full-forward graph.
    torch._dynamo.reset()
