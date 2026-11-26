# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Use the first real calls of a requested denoise graph before fixing its head policy."""

from collections import OrderedDict
from functools import partial

from vllm_omni.diffusion.cuda_graph import GraphedForward, kwargs_signature
from vllm_omni.platforms import current_omni_platform

from .head_policy import frozen_graph_policy


class AdaptiveDenoiseGraph(GraphedForward):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix = OrderedDict()
        self.capture_context = partial(frozen_graph_policy, self.capture_context)

    def __call__(self, *, variant=None, keep_alive=(), pin=False, **kwargs):
        if not self.enabled or current_omni_platform.is_graph_capturing():
            return super().__call__(variant=variant, keep_alive=keep_alive, pin=pin, **kwargs)
        pipeline = getattr(getattr(self.module, "_fn", None), "__self__", None)
        component = getattr(pipeline, kwargs.get("component", "transformer"), None)
        policy = getattr(component, "_ulysses_head_policy", None)
        if policy is None:
            return super().__call__(variant=variant, keep_alive=keep_alive, pin=pin, **kwargs)
        # Real outputs from these invocations enter the denoising loop. Keep
        # one graph per resident input signature; capture never benchmarks.
        key = (kwargs_signature(variant), kwargs_signature(kwargs))
        if key not in self.prefix and len(self.prefix) >= self.max_entries:
            self.prefix.popitem(last=False)
        steps, decision = self.prefix.get(key, (0, None))
        if steps < 2:
            result = self.module(**kwargs)
            self.eager_calls += 1
            self.prefix[key] = steps + 1, policy.component_key
            self.prefix.move_to_end(key)
            return result
        if steps == 2:
            # The last layer can still be in flight when its component returns.
            # Poll once before graph preparation. Incomplete trials stay native
            # in this graph. Neither warmup nor capture is a sampling call.
            policy.resolve_pending(decision)
            decision = policy.generation
            self.prefix[key] = 3, decision
        self.prefix.move_to_end(key)
        return super().__call__(variant=(variant, decision), keep_alive=keep_alive, pin=pin, **kwargs)

    def clear(self):
        super().clear()
        self.prefix.clear()
