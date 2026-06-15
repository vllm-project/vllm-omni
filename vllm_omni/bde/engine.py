# SPDX-License-Identifier: Apache-2.0
"""The Block Diffusion Engine (BDE)."""

from __future__ import annotations

from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.sched import SchedulerInterface

logger = init_logger(__name__)

#: Import path of the runner the BDE engine routes its workers to.
BDE_MODEL_RUNNER_CLS = "vllm_omni.bde.runner.BDEModelRunner"


def apply_bde_runner_default(od_config: OmniDiffusionConfig) -> None:
    """Route the BDE engine's workers to ``BDEModelRunner``.

    Sets ``od_config.diffusion_model_runner_cls`` in place, unless the caller
    already chose a runner. The worker (in its own process) reads this off the
    propagated ``od_config`` and builds the BDE runner instead of the platform
    default, keeping the swap scoped to BDE-routed models.
    """
    if getattr(od_config, "diffusion_model_runner_cls", None) is None:
        od_config.diffusion_model_runner_cls = BDE_MODEL_RUNNER_CLS


class BDEEngine(DiffusionEngine):
    """AR-Diffusion engine with engine-level KV cache management.

    BDE serves autoregressive / chunked blockwise-causal diffusion models
    (world models, AR-DiT) that materialize persistent attention KV. It reuses
    vLLM's paged KV stack (``KVCacheManager`` / ``BlockPool`` / ``BlockTables``)
    as a library, driven from the engine rather than hand-rolled inside each
    model. See ``BDE_doc/diffusion_kv_cache_management.md`` for the design and
    ``BDE_doc/dreamzero_kv_phase1_plan.md`` for the rollout.

    It is selected per model via ``OmniDiffusionConfig.engine_backend = "bde"``
    (resolved by :meth:`DiffusionEngine.make_engine`), so models that do not opt
    in keep using the base ``DiffusionEngine`` unchanged.

    Architecture note: in the multiproc setup the KV cache lives in the worker /
    runner process (GPU side), co-located with the model and KV tensors — so the
    actual KV *body* is :class:`~vllm_omni.bde.kv_cache.manager.BDEKVCache`, owned
    by :class:`~vllm_omni.bde.runner.BDEModelRunner`. ``BDEEngine`` itself is the
    thin selection / injection seam; it wires the BDE executor → worker → runner
    so DreamZero's rollout runs against the runner-owned KV cache.
    """

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        scheduler: SchedulerInterface | None = None,
    ) -> None:
        # Route this engine's workers to BDEModelRunner before the base __init__
        # builds the executor and spawns workers (od_config is propagated to them).
        apply_bde_runner_default(od_config)
        super().__init__(od_config, scheduler=scheduler)
