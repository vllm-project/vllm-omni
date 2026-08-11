"""vLLM-Omni diffusion stage clients/procs.

Home of the diffusion-backend stage surfaces: the in-process
``InlineStageDiffusionClient`` and the out-of-process ``StageDiffusionCoreClient``
/ ``StageDiffusionCoreProc`` / ``StageDiffusionCoreProcManager``. They implement
the shared stage contract declared in ``vllm_omni.engine.stage`` (``StageCoreClientBase``
and the ``stage_core_types`` data types) for the diffusion backend.

Import symbols directly from their submodules, e.g.::

    from vllm_omni.diffusion.stage.stage_diffusion_core_client import StageDiffusionCoreClient
    from vllm_omni.diffusion.stage.inline_stage_diffusion_client import InlineStageDiffusionClient

This ``__init__`` deliberately imports nothing at module scope so the heavy
diffusion-engine stack is only pulled in when a concrete submodule is imported.
"""
