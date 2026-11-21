# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Observe real AR backend calls on a local, random-weight Qwen3 MoE.

The test worker adapts only the stock text model's forward signature to the
Omni runner's metadata kwargs. It does not replace model arithmetic, the
scheduler, backend selection, model loading, or worker initialization.
"""

import os
from collections import Counter
from functools import wraps
from typing import Any

import torch
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase

from vllm_omni.worker.gpu_ar_worker import GPUARWorker


class BackendRuntimeWorker(GPUARWorker):
    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        model = self.model_runner.model
        original_forward = model.forward

        # Stock Qwen3MoeForCausalLM has no Omni metadata arguments. Only this
        # test fixture needs the adapter; the four math inputs are unchanged.
        def forward(input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **omni_metadata):
            return original_forward(input_ids, positions, intermediate_tensors, inputs_embeds)

        model.forward = forward
        return result

    def start_backend_runtime_probe(self):
        """Observe after startup; test-owned worker teardown removes the probes."""
        runner = self.model_runner
        self.backend_runtime_events: list[dict[str, Any]] = []
        self.backend_runtime_active_event: dict[str, Any] | None = None
        implementations: dict[str, dict[str, str]] = {
            "linear": {},
            "attention": {},
            "moe": {},
        }

        def observe(owner, attribute, category, name):
            original = getattr(owner, attribute)
            implementations[category][name] = f"{original.__module__}.{original.__qualname__}"

            @wraps(original)
            def counted(*args, **kwargs):
                result = original(*args, **kwargs)
                event = self.backend_runtime_active_event
                if event is not None:
                    event["calls"][category] += 1
                return result

            setattr(owner, attribute, counted)

        for name, module in runner.model.named_modules():
            if isinstance(module, LinearBase):
                observe(module.quant_method, "_gemm_impl", "linear", name)
            elif isinstance(module, Attention):
                observe(module.impl, "forward", "attention", name)
            elif isinstance(module, RoutedExperts):
                experts = module.quant_method.moe_kernel.fused_experts
                observe(experts, "apply", "moe", name)

        assert all(implementations.values()), implementations
        prompt_lengths: dict[str, int] = {}
        original_execute = runner.execute_model
        original_forward = runner.model.forward

        def forward(*args, **kwargs):
            hidden_states = original_forward(*args, **kwargs)
            event = self.backend_runtime_active_event
            if event is not None:
                assert isinstance(hidden_states, torch.Tensor)
                event["finite_hidden_states"] = bool(torch.isfinite(hidden_states).all())
            return hidden_states

        def execute(scheduler_output, *args, **kwargs):
            computed: dict[str, int] = {}
            for request in scheduler_output.scheduled_new_reqs:
                prompt_lengths[request.req_id] = len(request.prompt_token_ids)
                computed[request.req_id] = request.num_computed_tokens
            cached = scheduler_output.scheduled_cached_reqs
            computed.update(zip(cached.req_ids, cached.num_computed_tokens))
            scheduled = scheduler_output.num_scheduled_tokens
            if not scheduled:
                return original_execute(scheduler_output, *args, **kwargs)
            # The acceptance case deliberately schedules one request at a time.
            assert len(scheduled) == 1, scheduled
            request_id, token_count = next(iter(scheduled.items()))
            event = {
                "request_id": request_id,
                "prompt_tokens": prompt_lengths[request_id],
                "num_computed_tokens": int(computed[request_id]),
                "num_scheduled_tokens": int(token_count),
                "phase": "prefill" if computed[request_id] < prompt_lengths[request_id] else "decode",
                "calls": Counter(),
                "finite_hidden_states": False,
            }
            self.backend_runtime_active_event = event
            try:
                result = original_execute(scheduler_output, *args, **kwargs)
                self.backend_runtime_events.append(event)
                return result
            finally:
                self.backend_runtime_active_event = None

        runner.model.forward = forward
        runner.execute_model = execute
        config = self.vllm_config
        self.backend_runtime_config = {
            "linear_backend": config.kernel_config.linear_backend,
            "attention_backend": config.attention_config.backend.name,
            "moe_backend": config.kernel_config.moe_backend,
            "implementations": implementations,
            "model_class": type(runner.model).__name__,
            "runner_class": type(runner).__name__,
            "worker_pid": os.getpid(),
            "model_dtype": str(next(runner.model.parameters()).dtype),
            "skip_tokenizer_init": config.model_config.skip_tokenizer_init,
        }
        return self.backend_runtime_config

    def get_backend_runtime_probe(self):
        return {**self.backend_runtime_config, "events": self.backend_runtime_events}
