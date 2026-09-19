# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU conformance test: new append_batch + Qwen2KVBranchAdapter vs old eager path.

Creates a small Qwen2ForCausalLM, saves it, loads via vLLM's model loader,
then runs both paths (old ``append_and_enter_batch`` and new
``append_batch`` + ``Qwen2KVBranchAdapter``) and compares hidden states.

This test reuses the same subprocess pattern as
``test_vibevoice_negative_kv_conformance_gpu.py`` to isolate distributed
state.
"""

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


def _check_batched_replay(executor: Any, branch: Any, hf_model: Any, hidden_size: int) -> None:
    """Small B2–4 replay/recycling gate; no throughput or stress measurement."""
    # B2 must exercise compiled execution before it has a captured graph.
    assert set(executor._graphs) == {1}
    compiled = executor._compiled_fn
    free_before = branch.num_free_blocks
    request_ids = ["uncaptured-0", "uncaptured-1"]
    try:
        for rid in request_ids:
            branch.reset(rid)
        past = None
        for position in range(3):
            embeddings = torch.randn(2, 1, hidden_size, device="cuda", dtype=torch.bfloat16)
            reference = hf_model.model(
                inputs_embeds=embeddings,
                past_key_values=past,
                use_cache=True,
                position_ids=torch.full((2, 1), position, device="cuda"),
                cache_position=torch.tensor([position], device="cuda"),
                return_dict=True,
            )
            past = reference.past_key_values
            with branch.append_batch(request_ids) as step:
                assert step.seq_lens == (position + 1,) * 2
                output = executor.run(step, list(embeddings.unbind(0)))
            assert torch.isfinite(output).all()
            torch.testing.assert_close(
                output.float(), reference.last_hidden_state.reshape(2, -1).float(), rtol=0.04, atol=0.04
            )
            assert set(executor._graphs) == {1}
            assert executor._compiled_fn is compiled
    finally:
        for rid in request_ids:
            branch.free(rid)
    assert branch.num_free_blocks == free_before
    executor.warmup([2, 3, 4])
    assert set(executor._graphs) == {1, 2, 3, 4}
    graphs = dict(executor._graphs)
    pointers = [x.data_ptr() for x in executor.model_adapter._scheduler_metadata]
    executor.warmup([1, 2, 3, 4])
    assert all(executor._graphs[b] is graph for b, graph in graphs.items())
    assert pointers == [x.data_ptr() for x in executor.model_adapter._scheduler_metadata]
    for batch_size in (2, 3, 4):
        previous_blocks: set[int] = set()
        for wave in range(3):
            request_ids = [f"batch-{batch_size}-wave-{wave}-row-{i}" for i in range(batch_size)]
            past = {rid: None for rid in request_ids}
            lengths = {rid: 0 for rid in request_ids}
            free_before = branch.num_free_blocks
            used_blocks: set[int] = set()
            previous_output = previous_snapshot = None
            try:
                for rid in request_ids:
                    branch.reset(rid)
                # Third wave starts with different history lengths per row.
                # Use real B1 appends/replays, not fabricated KV contents.
                if wave == 2:
                    for row_index, rid in enumerate(request_ids):
                        for position in range(row_index):
                            embedding = torch.randn(1, hidden_size, device="cuda", dtype=torch.bfloat16)
                            reference = hf_model.model(
                                inputs_embeds=embedding.unsqueeze(0),
                                past_key_values=past[rid],
                                use_cache=True,
                                position_ids=torch.tensor([[position]], device="cuda"),
                                cache_position=torch.tensor([position], device="cuda"),
                                return_dict=True,
                            )
                            past[rid] = reference.past_key_values
                            with branch.append_batch([rid]) as step:
                                output = executor.run(step, [embedding])
                            torch.testing.assert_close(
                                output.float(),
                                reference.last_hidden_state.reshape(1, -1).float(),
                                rtol=0.04,
                                atol=0.04,
                            )
                            lengths[rid] += 1
                for position in range(branch.block_size + 2):
                    # Reorder rows while keeping each request's own history.
                    rows = request_ids[position % batch_size :] + request_ids[: position % batch_size]
                    embeddings = [torch.randn(1, hidden_size, device="cuda", dtype=torch.bfloat16) for _ in rows]
                    expected = []
                    for rid, embedding in zip(rows, embeddings):
                        reference = hf_model.model(
                            inputs_embeds=embedding.unsqueeze(0),
                            past_key_values=past[rid],
                            use_cache=True,
                            position_ids=torch.tensor([[lengths[rid]]], device="cuda"),
                            cache_position=torch.tensor([lengths[rid]], device="cuda"),
                            return_dict=True,
                        )
                        past[rid] = reference.past_key_values
                        expected.append(reference.last_hidden_state.reshape(1, -1))
                    with branch.append_batch(rows) as step:
                        assert step.positions == tuple(lengths[rid] for rid in rows)
                        assert step.seq_lens == tuple(lengths[rid] + 1 for rid in rows)
                        if wave == 2:
                            assert len(set(step.seq_lens)) == batch_size
                        for rid, blocks_for_row in zip(rows, step.block_ids):
                            assert len(blocks_for_row) == lengths[rid] // branch.block_size + 1
                        blocks = [block for row in step.block_ids for block in row]
                        assert len(blocks) == len(set(blocks)), "Requests share KV blocks"
                        assert len(set(step.slot_values)) == batch_size
                        used_blocks.update(blocks)
                        output = executor.run(step, embeddings)
                    assert torch.isfinite(output).all()
                    torch.testing.assert_close(output.float(), torch.cat(expected).float(), rtol=0.04, atol=0.04)
                    if previous_output is not None:
                        torch.testing.assert_close(previous_output, previous_snapshot, rtol=0, atol=0)
                    previous_output, previous_snapshot = output, output.clone()
                    for rid in rows:
                        lengths[rid] += 1
            finally:
                for rid in request_ids:
                    branch.free(rid)
            assert branch.num_free_blocks == free_before
            if wave:
                assert used_blocks & previous_blocks, "Fixture did not exercise block reuse"
            previous_blocks = used_blocks


def _check_active_request_transitions(executor: Any, branch: Any, hf_model: Any, hidden_size: int) -> None:
    """Retain live histories while requests finish and are replaced across B1–4."""
    schedule = [
        ("a",),
        ("b", "a"),
        ("a", "c", "b"),
        ("d", "b", "a", "c"),
        ("a", "d", "b"),
        ("e", "a", "b", "d"),
        ("d", "a"),
        ("a",),
        ("f", "a", "g"),
        ("g", "h", "a", "f"),
    ]
    active: set[str] = set()
    past: dict[str, Any] = {}
    lengths: dict[str, int] = {}
    blocks_by_request: dict[str, tuple[int, ...]] = {}
    free_before = branch.num_free_blocks
    graphs = dict(executor._graphs)
    retained_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    try:
        for rows in schedule:
            for rid in active - set(rows):
                branch.free(rid)
                del past[rid], lengths[rid], blocks_by_request[rid]
            active.intersection_update(rows)
            for rid in rows:
                if rid not in active:
                    branch.reset(rid)
                    active.add(rid)
                    past[rid], lengths[rid] = None, 0
            for _ in range(2):
                embeddings = [torch.randn(1, hidden_size, device="cuda", dtype=torch.bfloat16) for _ in rows]
                expected = []
                for rid, embedding in zip(rows, embeddings):
                    reference = hf_model.model(
                        inputs_embeds=embedding.unsqueeze(0),
                        past_key_values=past[rid],
                        use_cache=True,
                        position_ids=torch.tensor([[lengths[rid]]], device="cuda"),
                        cache_position=torch.tensor([lengths[rid]], device="cuda"),
                        return_dict=True,
                    )
                    past[rid] = reference.past_key_values
                    expected.append(reference.last_hidden_state.reshape(1, -1))
                with branch.append_batch(list(rows)) as step:
                    assert step.positions == tuple(lengths[rid] for rid in rows)
                    assert step.seq_lens == tuple(lengths[rid] + 1 for rid in rows)
                    blocks = [block for row in step.block_ids for block in row]
                    assert len(set(blocks)) == len(blocks)
                    assert len(set(step.slot_values)) == len(rows)
                    for rid, row_blocks in zip(rows, step.block_ids):
                        old_blocks = blocks_by_request.get(rid, ())
                        assert tuple(row_blocks[: len(old_blocks)]) == old_blocks
                        blocks_by_request[rid] = tuple(row_blocks)
                    output = executor.run(step, embeddings)
                assert torch.isfinite(output).all()
                torch.testing.assert_close(output.float(), torch.cat(expected).float(), rtol=0.04, atol=0.04)
                for previous, snapshot in retained_outputs:
                    torch.testing.assert_close(previous, snapshot, rtol=0, atol=0)
                retained_outputs.append((output, output.clone()))
                for rid in rows:
                    lengths[rid] += 1
                assert all(executor._graphs[b] is graph for b, graph in graphs.items())
        assert lengths["a"] == 20
        assert len(blocks_by_request["a"]) == 2
    finally:
        for rid in active:
            branch.free(rid)
    assert branch.num_free_blocks == free_before


def _adapter_conformance_worker(port: int, queue: Any, use_graph: bool = False) -> None:
    distributed_initialized = False
    branch = None
    adapter = None
    executor = None
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
        )
        from vllm.distributed.parallel_state import initialize_model_parallel
        from vllm.forward_context import (
            create_forward_context,
            override_forward_context,
        )
        from vllm.model_executor.layers.attention import Attention
        from vllm.model_executor.model_loader import get_model_loader

        from vllm_omni.engine.arg_utils import OmniEngineArgs
        from vllm_omni.model_executor.models.vibevoice.negative_qwen_adapter import (
            Qwen2KVBranchAdapter,
        )
        from vllm_omni.platforms import current_omni_platform
        from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner
        from vllm_omni.worker.named_kv_branch import NamedKVBranchRequest

        current_omni_platform.set_device(0)
        torch.manual_seed(1234)
        hf_config = Qwen2Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,  # small for speed
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
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
                max_model_len=256,
                max_num_seqs=4,
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
            assert len(layer_names) == 4
            first_layer = layers[layer_names[0]]
            spec = first_layer.get_kv_cache_spec(config)
            assert spec is not None
            backend = first_layer.get_attn_backend()
            backend.get_builder_cls()(
                spec,
                layer_names,
                config,
                torch.device("cuda"),
            )

            num_blocks = config.scheduler_config.max_num_seqs * (
                (config.model_config.max_model_len + spec.block_size - 1) // spec.block_size
            )
            # vLLM's BlockPool reserves one null block; add one extra physical
            # block so usable capacity still covers the fixed-concurrency set.
            num_blocks += 1

            # Build the production NamedCausalKVBranch.
            fake_runner = SimpleNamespace(
                vllm_config=config,
                device=torch.device("cuda:0"),
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

            # Build the Qwen2KVBranchAdapter.
            adapter = Qwen2KVBranchAdapter(
                language_model=language_model,
                hidden_size=hf_config.hidden_size,
            )
            # Get K/V views from the branch's pool.
            k_caches = []
            v_caches = []
            for name in branch.layer_names:
                cache = branch.kv_caches[name]
                k, v = cache.transpose(1, 2).split(spec.head_size, dim=-1)
                k_caches.append(k)
                v_caches.append(v)
            adapter.bind_kv_caches(
                branch_layer_names=tuple(reversed(branch.layer_names)),
                k_caches=list(reversed(k_caches)),
                v_caches=list(reversed(v_caches)),
            )

            # Keep independent histories for all paths, crossing a block boundary.
            num_steps = spec.block_size + 4
            hf_past = None
            max_abs_diff = 0.0
            if use_graph:
                from vllm_omni.worker.named_kv.executor import NamedKVBranchExecutor
                from vllm_omni.worker.named_kv.flash_attention import FlashAttentionKVBranchAdapter

                # Keep backend eligibility validation before executor construction.
                FlashAttentionKVBranchAdapter(branch)
                executor = NamedKVBranchExecutor(
                    branch,
                    adapter,
                    max_num_seqs=config.scheduler_config.max_num_seqs,
                    max_model_len=config.model_config.max_model_len,
                    device=torch.device("cuda"),
                    dtype=torch.bfloat16,
                )

            outer_context = create_forward_context(
                {},
                config,
                slot_mapping={},
                skip_compiled=True,
            )

            with torch.inference_mode(), override_forward_context(outer_context):
                if executor is not None:
                    executor.warmup([1])
                    assert 1 in executor._graphs, "B1 graph was not captured"
                    assert executor._compiled_fn is not executor._eager_fn
                for request_id in ("old-path", "old-path-2", "new-path"):
                    branch.reset(request_id)
                previous_output = None
                previous_snapshot = None
                for step in range(num_steps):
                    embedding = torch.randn(
                        1,
                        1,
                        hf_config.hidden_size,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )

                    # HF reference.
                    hf_output = hf_model.model(
                        inputs_embeds=embedding,
                        past_key_values=hf_past,
                        use_cache=True,
                        position_ids=torch.tensor([[step]], device="cuda"),
                        cache_position=torch.tensor([step], device="cuda"),
                        return_dict=True,
                    )
                    hf_past = hf_output.past_key_values
                    hf_hidden = hf_output.last_hidden_state.reshape(1, -1)

                    # --- Old path: append_and_enter (run 1) ---
                    with branch.append_and_enter("old-path") as old_step:
                        assert old_step.sequence_length == step + 1
                        torch.testing.assert_close(old_step.position, torch.tensor([step], device="cuda"))
                        old_hidden = language_model(
                            input_ids=None,
                            positions=old_step.position,
                            inputs_embeds=embedding.reshape(1, -1),
                        )
                    old_hidden = old_hidden.detach().clone()

                    # --- Old path again: append_and_enter (run 2) ---
                    with branch.append_and_enter("old-path-2") as old_step2:
                        old_hidden2 = language_model(
                            input_ids=None,
                            positions=old_step2.position,
                            inputs_embeds=embedding.reshape(1, -1),
                        )
                    old_hidden2 = old_hidden2.detach().clone()

                    # Check old vs old (should be identical).
                    old_vs_old_diff = float((old_hidden.float() - old_hidden2.float()).abs().max())
                    assert old_vs_old_diff == 0.0

                    # --- New path: append_batch + adapter ---
                    with branch.append_batch(["new-path"]) as new_step:
                        assert new_step.positions == (step,)
                        assert new_step.seq_lens == (step + 1,)
                        assert len(new_step.block_ids[0]) == step // spec.block_size + 1
                        # Build metadata for adapter forward.
                        slot_mapping = torch.tensor(
                            new_step.slot_values,
                            dtype=torch.int64,
                            device="cuda",
                        )
                        block_table = torch.tensor(
                            [
                                list(new_step.block_ids[0])
                                + [0] * (branch.max_blocks_per_request - len(new_step.block_ids[0]))
                            ],
                            dtype=torch.int32,
                            device="cuda",
                        )
                        query_start_loc = torch.tensor(
                            [0, 1],
                            dtype=torch.int32,
                            device="cuda",
                        )
                        seq_lens = torch.tensor(
                            list(new_step.seq_lens),
                            dtype=torch.int32,
                            device="cuda",
                        )
                        positions = torch.tensor(
                            list(new_step.positions),
                            dtype=torch.long,
                            device="cuda",
                        )
                        if executor is None:
                            new_hidden = adapter.forward(
                                embedding.reshape(1, -1),
                                positions,
                                slot_mapping,
                                block_table,
                                query_start_loc,
                                seq_lens,
                                max_seq_len=step + 1,
                            )
                        else:
                            new_hidden = executor.run(new_step, [embedding.reshape(1, -1)])
                            if previous_output is not None:
                                torch.testing.assert_close(previous_output, previous_snapshot, rtol=0, atol=0)
                            previous_output = new_hidden
                            previous_snapshot = new_hidden.clone()
                    new_hidden = new_hidden.detach().clone()

                    # Compare old vs new (both should match HF reference).
                    old_diff = float((old_hidden.float() - hf_hidden.float()).abs().max())
                    new_diff = float((new_hidden.float() - hf_hidden.float()).abs().max())
                    old_new_diff = float((new_hidden.float() - old_hidden.float()).abs().max())
                    max_abs_diff = max(max_abs_diff, old_diff, new_diff)

                    print(
                        f"Step {step}: old_vs_hf={old_diff:.6f}, "
                        f"new_vs_hf={new_diff:.6f}, "
                        f"new_vs_old={old_new_diff:.6f}"
                    )

                    # Restore the reference gate with matching history. Do not
                    # widen tolerances to accommodate an incorrect fixture.
                    for hidden in (hf_hidden, old_hidden, old_hidden2, new_hidden):
                        assert torch.isfinite(hidden).all(), "Non-finite hidden state"
                    for hidden in (old_hidden, new_hidden):
                        torch.testing.assert_close(hidden.float(), hf_hidden.float(), rtol=0.04, atol=0.04)
                    # Retain the tighter direct old/new comparison.
                    torch.testing.assert_close(
                        new_hidden.float(),
                        old_hidden.float(),
                        rtol=0.001,
                        atol=0.001,
                    )

            for request_id in ("old-path", "old-path-2", "new-path"):
                branch.free(request_id)
            if executor is not None:
                with torch.inference_mode(), override_forward_context(outer_context):
                    _check_batched_replay(executor, branch, hf_model, hf_config.hidden_size)
                    _check_active_request_transitions(executor, branch, hf_model, hf_config.hidden_size)
                executor.close()
            adapter.close()
            branch.close()
            queue.put(
                {
                    "graph": use_graph,
                    "steps": num_steps,
                    "block_size": spec.block_size,
                    "layers": len(layer_names),
                    "max_abs_diff": max_abs_diff,
                    "backend": backend.__name__,
                }
            )
            del vllm_model, hf_model
            gc.collect()
            torch.accelerator.empty_cache()
    except Exception:
        queue.put({"error": traceback.format_exc()})
    finally:
        for resource in (executor, adapter, branch):
            if resource is not None:
                try:
                    resource.close()
                except Exception:
                    traceback.print_exc()
        if distributed_initialized:
            try:
                destroy_model_parallel()
                destroy_distributed_environment()
            except Exception:
                pass


@pytest.mark.parametrize("use_graph", [False, True], ids=["eager", "compiled_graph"])
def test_new_append_batch_adapter_matches_old_eager_path(use_graph: bool) -> None:
    """New append_batch + Qwen2KVBranchAdapter output matches old eager path."""
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_adapter_conformance_worker,
        args=(_free_port(), queue, use_graph),
    )
    process.start()
    process.join(timeout=300)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("Adapter conformance subprocess timed out")

    try:
        result = queue.get(timeout=5)
    except Empty:
        pytest.fail(f"Adapter conformance subprocess exited without a result: exitcode={process.exitcode}")
    assert "error" not in result, result.get("error")
    assert process.exitcode == 0
    assert result["graph"] is use_graph
    assert result["steps"] > result["block_size"]
    assert result["layers"] == 4
    # Numerical gates run per-step in the worker; report the absolute maximum
    # diagnostically rather than adding a different aggregate tolerance.
