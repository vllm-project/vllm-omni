# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU conformance gate for the planned VibeVoice negative Qwen KV branch."""

from __future__ import annotations

import multiprocessing as mp
import os
import socket
import traceback
from queue import Empty
from typing import Any

import pytest
import torch

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _negative_kv_conformance_worker(port: int, queue: Any) -> None:
    distributed_initialized = False
    try:
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK="0",
            LOCAL_RANK="0",
            WORLD_SIZE="1",
        )

        import gc
        import tempfile
        from types import SimpleNamespace

        from transformers import Qwen2Config, Qwen2ForCausalLM
        from vllm.config import (
            get_layers_from_vllm_config,
            set_current_vllm_config,
        )
        from vllm.distributed import (
            destroy_distributed_environment,
            destroy_model_parallel,
            init_distributed_environment,
            initialize_model_parallel,
        )
        from vllm.forward_context import (
            create_forward_context,
            get_forward_context,
            override_forward_context,
        )
        from vllm.model_executor.layers.attention import Attention
        from vllm.model_executor.model_loader import get_model_loader
        from vllm.v1.attention.backend import CommonAttentionMetadata
        from vllm.v1.worker.gpu.attn_utils import (
            _reshape_attention_kv_cache,
        )

        from vllm_omni.engine.arg_utils import OmniEngineArgs
        from vllm_omni.platforms import current_omni_platform
        from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner
        from vllm_omni.worker.named_kv_branch import NamedKVBranchRequest

        current_omni_platform.set_device(0)
        torch.manual_seed(1234)
        hf_config = Qwen2Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=28,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            attention_dropout=0.0,
            tie_word_embeddings=False,
            use_cache=True,
        )
        hf_model = Qwen2ForCausalLM(hf_config).eval()

        with tempfile.TemporaryDirectory() as model_dir:
            hf_model.save_pretrained(model_dir, safe_serialization=True)
            args = OmniEngineArgs(
                model=model_dir,
                model_arch="Qwen2ForCausalLM",
                worker_type="ar",
                skip_tokenizer_init=True,
                dtype="bfloat16",
                load_format="safetensors",
                trust_remote_code=False,
                max_model_len=64,
                max_num_seqs=1,
                block_size=16,
                enforce_eager=True,
                enable_prefix_caching=False,
            )
            config = args.create_engine_config()
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                backend="nccl",
            )
            distributed_initialized = True
            with set_current_vllm_config(config):
                initialize_model_parallel(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                )
                vllm_model = get_model_loader(config.load_config).load_model(
                    vllm_config=config,
                    model_config=config.model_config,
                )

            hf_model = hf_model.to(device="cuda", dtype=torch.bfloat16)
            language_model = vllm_model.model
            layers = get_layers_from_vllm_config(config, Attention)
            layer_names = list(layers)
            assert len(layer_names) == 28
            first_layer = layers[layer_names[0]]
            spec = first_layer.get_kv_cache_spec(config)
            assert spec is not None
            backend = first_layer.get_attn_backend()
            builder = backend.get_builder_cls()(
                spec,
                layer_names,
                config,
                torch.device("cuda"),
            )

            num_blocks = 4
            with set_current_vllm_config(config):
                stride_order = backend.get_kv_cache_stride_order()
            cache_shape = backend.get_kv_cache_shape(
                num_blocks,
                spec.block_size,
                spec.num_kv_heads,
                spec.head_size,
                cache_dtype_str=config.cache_config.cache_dtype,
            )
            positive_caches = {layer_name: layer.kv_cache for layer_name, layer in layers.items()}
            negative_caches: dict[str, torch.Tensor] = {}
            negative_raw_caches: list[torch.Tensor] = []
            for layer_name in layer_names:
                raw_cache = torch.zeros(
                    num_blocks * spec.page_size_bytes,
                    dtype=torch.uint8,
                    device="cuda",
                )
                negative_raw_caches.append(raw_cache)
                negative_caches[layer_name] = _reshape_attention_kv_cache(
                    raw_cache,
                    spec,
                    cache_shape,
                    stride_order,
                    num_blocks,
                    None,
                )

            hf_past = None
            embeddings: list[torch.Tensor] = []
            max_abs_diff = 0.0
            restored_after_every_step = True
            outer_context = create_forward_context(
                {},
                config,
                slot_mapping={},
                skip_compiled=True,
            )
            with torch.inference_mode(), override_forward_context(outer_context):
                for step in range(4):
                    embedding = torch.randn(
                        1,
                        1,
                        hf_config.hidden_size,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    embeddings.append(embedding.clone())
                    hf_output = hf_model.model(
                        inputs_embeds=embedding,
                        past_key_values=hf_past,
                        use_cache=True,
                        position_ids=torch.tensor([[step]], device="cuda"),
                        cache_position=torch.tensor([step], device="cuda"),
                        return_dict=True,
                    )
                    hf_past = hf_output.past_key_values

                    query_start_cpu = torch.tensor([0, 1], dtype=torch.int32)
                    query_start = query_start_cpu.to("cuda")
                    seq_lens = torch.tensor(
                        [step + 1],
                        dtype=torch.int32,
                        device="cuda",
                    )
                    slot_mapping = torch.tensor(
                        [step],
                        dtype=torch.int64,
                        device="cuda",
                    )
                    block_table = torch.tensor(
                        [[0, 1, 2, 3]],
                        dtype=torch.int32,
                        device="cuda",
                    )
                    common = CommonAttentionMetadata(
                        query_start_loc=query_start,
                        query_start_loc_cpu=query_start_cpu,
                        seq_lens=seq_lens,
                        num_reqs=1,
                        num_actual_tokens=1,
                        max_query_len=1,
                        max_seq_len=step + 1,
                        block_table_tensor=block_table,
                        slot_mapping=slot_mapping,
                        causal=True,
                        positions=torch.tensor(
                            [step],
                            dtype=torch.long,
                            device="cuda",
                        ),
                    )
                    metadata = builder.build(0, common)
                    negative_context = create_forward_context(
                        {name: metadata for name in layer_names},
                        config,
                        slot_mapping={name: slot_mapping for name in layer_names},
                        skip_compiled=True,
                    )

                    try:
                        for layer_name, layer in layers.items():
                            layer.kv_cache = negative_caches[layer_name]
                        with override_forward_context(negative_context):
                            vllm_hidden = language_model(
                                input_ids=None,
                                positions=torch.tensor(
                                    [step],
                                    device="cuda",
                                ),
                                inputs_embeds=embedding.reshape(1, -1),
                            )
                    finally:
                        for layer_name, layer in layers.items():
                            layer.kv_cache = positive_caches[layer_name]

                    restored_after_every_step &= all(
                        layer.kv_cache is positive_caches[layer_name] for layer_name, layer in layers.items()
                    )
                    assert get_forward_context() is outer_context
                    hf_hidden = hf_output.last_hidden_state.reshape(1, -1)
                    diff = float((vllm_hidden.float() - hf_hidden.float()).abs().max())
                    max_abs_diff = max(max_abs_diff, diff)
                    torch.testing.assert_close(
                        vllm_hidden.float(),
                        hf_hidden.float(),
                        rtol=0.04,
                        atol=0.04,
                    )

            # Cross a 16-token page boundary on the production store while
            # keeping the four-step manual mechanism check above lightweight.
            for _ in range(13):
                embeddings.append(
                    torch.randn(
                        1,
                        1,
                        hf_config.hidden_size,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                )

            # Exercise the production fixed-pool store against the same
            # independent Transformers cached reference.
            fake_runner = SimpleNamespace(
                vllm_config=config,
                device=torch.device("cuda"),
                kv_cache_config=SimpleNamespace(
                    kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)],
                    num_blocks=num_blocks,
                ),
                attn_groups=[
                    [
                        SimpleNamespace(
                            backend=backend,
                            layer_names=layer_names,
                        )
                    ]
                ],
                _kernel_block_sizes=[spec.block_size],
            )
            branch_memory_bytes = num_blocks * len(layer_names) * spec.page_size_bytes
            bound_branches: list[Any] = []
            fake_runner.model = SimpleNamespace(
                named_kv_branch_request=NamedKVBranchRequest(
                    name="negative",
                    memory_bytes=branch_memory_bytes,
                ),
                bind_named_kv_branch=bound_branches.append,
            )
            fake_runner.named_kv_branches = {}
            OmniGPUModelRunner._maybe_bind_named_kv_branch(fake_runner)
            branch = fake_runner.named_kv_branches["negative"]
            assert bound_branches == [branch]
            branch.reset("request")
            hf_past = None
            store_max_abs_diff = 0.0
            with torch.inference_mode():
                for step, embedding in enumerate(embeddings):
                    hf_output = hf_model.model(
                        inputs_embeds=embedding,
                        past_key_values=hf_past,
                        use_cache=True,
                        position_ids=torch.tensor([[step]], device="cuda"),
                        cache_position=torch.tensor([step], device="cuda"),
                        return_dict=True,
                    )
                    hf_past = hf_output.past_key_values
                    with branch.append_and_enter("request") as branch_step:
                        assert branch_step.sequence_length == step + 1
                        store_hidden = language_model(
                            input_ids=None,
                            positions=branch_step.position,
                            inputs_embeds=embedding.reshape(1, -1),
                        )
                    hf_hidden = hf_output.last_hidden_state.reshape(1, -1)
                    store_diff = float((store_hidden.float() - hf_hidden.float()).abs().max())
                    store_max_abs_diff = max(
                        store_max_abs_diff,
                        store_diff,
                    )
                    torch.testing.assert_close(
                        store_hidden.float(),
                        hf_hidden.float(),
                        rtol=0.04,
                        atol=0.04,
                    )
            assert branch.get_sequence_length("request") == 17
            branch.free("request")
            all_blocks_freed = branch.num_free_blocks == branch.num_blocks

            # Batched-path conformance: two staggered requests share
            # one batched attention context per step; every row is compared
            # against its own Transformers cached reference.
            branch.reset("batch-a")
            hf_past_a = None
            with torch.inference_mode():
                for step in range(4):
                    embedding = embeddings[step]
                    hf_output = hf_model.model(
                        inputs_embeds=embedding,
                        past_key_values=hf_past_a,
                        use_cache=True,
                        position_ids=torch.tensor([[step]], device="cuda"),
                        cache_position=torch.tensor([step], device="cuda"),
                        return_dict=True,
                    )
                    hf_past_a = hf_output.past_key_values
                    with branch.append_and_enter("batch-a") as branch_step:
                        language_model(
                            input_ids=None,
                            positions=branch_step.position,
                            inputs_embeds=embedding.reshape(1, -1),
                        )
            branch.reset("batch-b")
            hf_past_b = None
            batch_embeddings_b = [
                torch.randn(
                    1,
                    1,
                    hf_config.hidden_size,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                for _ in range(13)
            ]
            batch_max_abs_diff = 0.0
            with torch.inference_mode():
                for index in range(13):
                    step_a = 4 + index
                    step_b = index
                    embedding_a = embeddings[step_a]
                    embedding_b = batch_embeddings_b[step_b]
                    hf_output_a = hf_model.model(
                        inputs_embeds=embedding_a,
                        past_key_values=hf_past_a,
                        use_cache=True,
                        position_ids=torch.tensor([[step_a]], device="cuda"),
                        cache_position=torch.tensor([step_a], device="cuda"),
                        return_dict=True,
                    )
                    hf_past_a = hf_output_a.past_key_values
                    hf_output_b = hf_model.model(
                        inputs_embeds=embedding_b,
                        past_key_values=hf_past_b,
                        use_cache=True,
                        position_ids=torch.tensor([[step_b]], device="cuda"),
                        cache_position=torch.tensor([step_b], device="cuda"),
                        return_dict=True,
                    )
                    hf_past_b = hf_output_b.past_key_values
                    with branch.append_and_enter_batch(["batch-a", "batch-b"]) as branch_step:
                        batched_hidden = language_model(
                            input_ids=None,
                            positions=branch_step.position,
                            inputs_embeds=torch.cat(
                                [embedding_a.reshape(1, -1), embedding_b.reshape(1, -1)],
                                dim=0,
                            ),
                        )
                    for row, hf_hidden in (
                        (batched_hidden[0:1], hf_output_a.last_hidden_state.reshape(1, -1)),
                        (batched_hidden[1:2], hf_output_b.last_hidden_state.reshape(1, -1)),
                    ):
                        batch_max_abs_diff = max(
                            batch_max_abs_diff,
                            float((row.float() - hf_hidden.float()).abs().max()),
                        )
                        torch.testing.assert_close(
                            row.float(),
                            hf_hidden.float(),
                            rtol=0.04,
                            atol=0.04,
                        )
            assert branch.get_sequence_length("batch-a") == 17
            assert branch.get_sequence_length("batch-b") == 13
            branch.free("batch-a")
            branch.free("batch-b")
            all_blocks_freed = all_blocks_freed and branch.num_free_blocks == branch.num_blocks

            branch.reset("active-guard")
            with branch.append_and_enter("active-guard"):
                active_length = branch.get_sequence_length("active-guard")
                active_num_free_blocks = branch.num_free_blocks
                for operation in (branch.reset, branch.free):
                    try:
                        operation("active-guard")
                    except RuntimeError as exc:
                        assert "forward context is active" in str(exc)
                    else:
                        raise AssertionError("External named-KV mutation was accepted in context")
                assert branch.get_sequence_length("active-guard") == active_length
                assert branch.num_free_blocks == active_num_free_blocks
            branch.free("active-guard")
            active_context_guard = (
                branch.get_sequence_length("active-guard") == 0 and branch.num_free_blocks == branch.num_blocks
            )

            branch.reset("fault")
            try:
                with branch.append_and_enter("fault"):
                    raise RuntimeError("injected failure")
            except RuntimeError as exc:
                assert str(exc) == "injected failure"
            else:
                raise AssertionError("Injected named-KV failure was swallowed")
            fault_cleanup = branch.get_sequence_length("fault") == 0 and branch.num_free_blocks == branch.num_blocks
            branch.close()

            queue.put(
                {
                    "steps": 4,
                    "store_steps": len(embeddings),
                    "layers": len(layer_names),
                    "max_abs_diff": max_abs_diff,
                    "store_max_abs_diff": store_max_abs_diff,
                    "batch_max_abs_diff": batch_max_abs_diff,
                    "restored_after_every_step": restored_after_every_step,
                    "all_blocks_freed": all_blocks_freed,
                    "active_context_guard": active_context_guard,
                    "fault_cleanup": fault_cleanup,
                }
            )
            del negative_raw_caches, negative_caches
            del vllm_model, hf_model
            gc.collect()
            torch.accelerator.empty_cache()
    except Exception:
        queue.put({"error": traceback.format_exc()})
    finally:
        if distributed_initialized:
            try:
                destroy_model_parallel()
                destroy_distributed_environment()
            except Exception:
                pass


def test_manual_negative_paged_kv_matches_transformers_cached_qwen() -> None:
    """Pin the private vLLM mechanisms before implementing the branch store."""
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_negative_kv_conformance_worker,
        args=(_free_port(), queue),
    )
    process.start()
    process.join(timeout=180)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("VibeVoice negative-KV conformance subprocess timed out")

    try:
        result = queue.get(timeout=5)
    except Empty:
        pytest.fail(f"Negative-KV conformance subprocess exited without a result: exitcode={process.exitcode}")
    assert "error" not in result, result.get("error")
    assert process.exitcode == 0
    assert result["steps"] == 4
    assert result["store_steps"] == 17
    assert result["layers"] == 28
    assert result["max_abs_diff"] <= 0.04
    assert result["store_max_abs_diff"] <= 0.04
    assert result["batch_max_abs_diff"] <= 0.04
    assert result["restored_after_every_step"] is True
    assert result["all_blocks_freed"] is True
    assert result["active_context_guard"] is True
    assert result["fault_cleanup"] is True
