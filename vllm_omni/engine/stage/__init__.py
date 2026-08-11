"""vLLM-Omni stage process/replica subpackage.

Home of the refactored stage LLM **and** diffusion core clients / procs /
proc-managers, the shared stage core contract (``stage_core_client``) and data
types (``stage_core_types``), and the head-side replica pool
(``stage_replica_pool``).

Import symbols directly from their submodules, e.g.::

    from vllm_omni.engine.stage.stage_replica_pool import StageReplicaPool
    from vllm_omni.engine.stage.stage_core_types import StageLLMCoreRequest

This ``__init__`` deliberately imports nothing at module scope, so pulling in a
lightweight submodule (e.g. ``stage_core_types`` from ``vllm_omni.patch``) never
transitively loads the heavy engine-client stack (``AsyncMPClient``, the omni
connector modules).
"""
