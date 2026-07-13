# SPDX-License-Identifier: Apache-2.0
"""The AR-Diffusion Engine (AR-Diffusion)."""

from __future__ import annotations

from vllm.logger import init_logger

from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

logger = init_logger(__name__)

#: Import path of the runner the AR-Diffusion engine routes its workers to.
AR_DIFFUSION_MODEL_RUNNER_CLS = "vllm_omni.experimental.ar_diffusion.runner.ARDiffusionModelRunner"


class ARDiffusionEngine(DiffusionEngine):
    """AR-Diffusion engine with engine-level KV cache management.

    AR-Diffusion serves autoregressive / chunked blockwise-causal diffusion models
    (world models, AR-DiT) that materialize persistent attention KV. It reuses
    vLLM's paged KV stack (``KVCacheManager`` / ``BlockPool`` / ``BlockTables``)
    as a library, driven from the engine rather than hand-rolled inside each
    model. See ``BDE_doc/diffusion_kv_cache_management.md`` for the design and
    ``BDE_doc/dreamzero_kv_phase1_plan.md`` for the rollout.

    It is selected per model via ``OmniDiffusionConfig.engine_backend = "ar_diffusion"``
    (resolved by :meth:`DiffusionEngine.make_engine`), so models that do not opt
    in keep using the base ``DiffusionEngine`` unchanged.

    Architecture note: in the multiproc setup the KV cache lives in the worker /
    runner process (GPU side), co-located with the model and KV tensors — so the
    actual KV *body* is :class:`~vllm_omni.experimental.ar_diffusion.kv_cache.manager.ARDiffusionKVCache`, owned
    by :class:`~vllm_omni.experimental.ar_diffusion.runner.ARDiffusionModelRunner`. ``ARDiffusionEngine`` itself is the
    thin selection / injection seam; it wires the AR-Diffusion executor → worker → runner
    so DreamZero's rollout runs against the runner-owned KV cache.
    """

    #: Workers of this engine build the AR-Diffusion runner (unless the config
    #: sets an explicit ``diffusion_model_runner_cls`` override). Declared on
    #: the class so routing needs no od_config mutation at engine init.
    default_diffusion_model_runner_cls = AR_DIFFUSION_MODEL_RUNNER_CLS
