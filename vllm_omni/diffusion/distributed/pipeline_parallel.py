# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import wraps
from typing import Any

import torch
from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors

from vllm_omni.diffusion.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_pipeline_parallel_world_size,
    get_pp_group,
    is_pipeline_first_stage,
)


class AsyncLatents:
    """Transparent async wrapper returned by scheduler_step on rank 0.

    Wraps a pending ``irecv_tensor_dict`` and defers ``handle.wait()`` until the
    underlying tensor is actually consumed — either via attribute access
    (e.g. ``latents.to(dtype)``, ``latents.shape``) or via a torch operation
    (e.g. ``mask * latents``).  This keeps the first PP rank non-blocking after
    posting the receive, matching the async philosophy used everywhere else in
    the PP communication layer.
    """

    __slots__ = ("_tensor_dict", "_handles", "_postproc", "_tensor")

    def __init__(
        self,
        tensor_dict: dict[str, torch.Tensor],
        handles: list[torch.distributed.Work],
        postproc: list,
    ):
        self._tensor_dict = tensor_dict
        self._handles = handles
        self._postproc = postproc
        self._tensor: torch.Tensor | None = None

    def _resolve(self) -> torch.Tensor:
        if self._tensor is not None:
            return self._tensor
        for h in self._handles:
            h.wait()
        for fn in self._postproc:
            fn()
        self._tensor = self._tensor_dict["latents"]
        return self._tensor

    # Attribute access (e.g. .shape, .to(), .dtype) delegates to the resolved tensor.
    def __getattr__(self, name: str):
        return getattr(self._resolve(), name)

    # Torch function protocol: any torch op involving an AsyncLatents resolves it first.
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        def _unwrap(x):
            if isinstance(x, AsyncLatents):
                return x._resolve()
            if isinstance(x, (list, tuple)):
                return type(x)(_unwrap(item) for item in x)  # type(x) return the class of x to preserve its type
            return x

        args = tuple(_unwrap(a) for a in args)
        kwargs = {k: _unwrap(v) for k, v in kwargs.items()}
        return func(*args, **kwargs)


class PipelineParallelMixin:
    """
    Mixin providing Pipeline Parallelism for diffusion pipelines.

    All PP ranks run the full denoising loop in `forward()`.
    `predict_noise_maybe_with_cfg` and `scheduler_step_maybe_with_cfg` encapsulate all inter-rank communication.

    Communication pattern per denoising step:
      Forward chain : rank 0 → 1 → … → N-1  via async isend/irecv (AsyncIntermediateTensors)
      Next timestep : last rank → rank 0     via async isend/irecv (AsyncLatents)

    All communication is asynchronous using isend_tensor_dict/irecv_tensor_dict.
    Only rank 0 needs updated latents for the next forward pass start.

    For sequential CFG (cfg_parallel_size=1) with PP, two full forward chains are
    executed — one for the positive pass and one for the negative pass — so that each
    PP stage operates on the correct encoder_hidden_states.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin

        if not issubclass(cls, CFGParallelMixin):
            raise TypeError(
                f"{cls.__name__} inherits PipelineParallelMixin but not CFGParallelMixin. "
                "Pipeline Parallelism requires CFGParallelMixin for predict_noise(), "
                "predict_noise_maybe_with_cfg(), scheduler_step_maybe_with_cfg(), and combine_cfg_noise(). "
                "Add CFGParallelMixin to the base classes of your pipeline."
            )
        mro = cls.mro()
        if mro.index(PipelineParallelMixin) > mro.index(CFGParallelMixin):
            raise TypeError(
                f"{cls.__name__} must inherit PipelineParallelMixin before CFGParallelMixin so MRO selects "
                f"PP-aware predict/scheduler wrappers and their `super()` calls delegate to CFGParallelMixin."
            )

        init = cls.__dict__.get("__init__")
        if callable(init):

            @wraps(init)
            def wrapped_init(self, *args: Any, **kwargs: Any) -> None:
                init(self, *args, **kwargs)
                vae = getattr(self, "vae", None)
                if vae is not None and hasattr(vae, "decode"):
                    self._wrapped_vae_decode()

            cls.__init__ = wrapped_init

        diffuse = cls.__dict__.get("diffuse")
        if callable(diffuse):

            @wraps(diffuse)
            def wrapped_diffuse(self, *args: Any, **kwargs: Any) -> Any:
                try:
                    latents = diffuse(self, *args, **kwargs)
                    if isinstance(latents, AsyncLatents):
                        latents = torch.as_tensor(latents)  # avoid copying
                    return latents
                finally:
                    self._sync_pp_send()

            cls.diffuse = wrapped_diffuse

    def _wrapped_vae_decode(self) -> None:
        vae, orig_decode = self.vae, self.vae.decode

        @wraps(orig_decode)
        def wrapped_decode(z: torch.Tensor, *args: Any, **kwargs: Any):
            if hasattr(vae, "is_distributed_enabled") and vae.is_distributed_enabled():
                # Middle ranks (world size 3 or more) hold stale latents after the denoising loop.
                # Broadcast from rank 0 so every rank splits identical tiles.
                if get_pipeline_parallel_world_size() > 2:
                    z = get_pp_group().broadcast(z, src=0)
                return orig_decode(z, *args, **kwargs)
            elif is_pipeline_first_stage():
                return orig_decode(z, *args, **kwargs)
            return (None,)  # decoder returns a tuple

        self.vae.decode = wrapped_decode

    @property
    def _pp_send_work(self) -> list[torch.distributed.Work]:
        if not hasattr(self, "_pp_send_work_list"):
            self._pp_send_work_list: list[torch.distributed.Work] = []
        return self._pp_send_work_list

    @_pp_send_work.setter
    def _pp_send_work(self, work: list[torch.distributed.Work]) -> None:
        self._pp_send_work_list = work

    def _sync_pp_send(self) -> None:
        """
        Wait on all pending non-blocking PP sends.

        Must be called after the denoising loop so that the isend handles
        from the last iteration are completed before any subsequent
        collective (e.g. VAE decode broadcast) or tensor reuse.
        """
        if self._pp_send_work:
            for handle in self._pp_send_work:
                handle.wait()
            self._pp_send_work = []

    def predict_noise_maybe_with_cfg(
        self,
        do_true_cfg: bool,
        true_cfg_scale: float,
        positive_kwargs: dict[str, Any],
        negative_kwargs: dict[str, Any] | None,
        cfg_normalize: bool = True,
        output_slice: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        """
        Drop-in replacement for predict_noise_maybe_with_cfg that also handles PP.

        Supports three modes:
          - PP only, sequential CFG: both branches (cond and uncond) run through this PP pipeline.
            This doubles communication volume per denoising step compared to PP + CFG-parallel.
          - PP + CFG-parallel: each PP pipeline carries one branch. The last PP
            rank all-gathers across the CFG group and combines, mirroring
            CFGParallelMixin.predict_noise_maybe_with_cfg exactly.
          - PP only, no CFG: cond branch only.

        Returns:
            noise_pred on the last PP rank (all CFG ranks when CFG-parallel is active).
            None on all other ranks.
        """
        if get_pipeline_parallel_world_size() == 1:
            return super().predict_noise_maybe_with_cfg(
                do_true_cfg, true_cfg_scale, positive_kwargs, negative_kwargs, cfg_normalize, output_slice
            )

        self._sync_pp_send()

        pp_group = get_pp_group()

        cfg_parallel_ready = do_true_cfg and get_classifier_free_guidance_world_size() > 1
        if cfg_parallel_ready:
            # Each PP pipeline carries exactly one CFG branch determined by cfg_rank.
            all_kwargs = [positive_kwargs if get_classifier_free_guidance_rank() == 0 else negative_kwargs]
        else:
            # Sequential CFG (or no CFG): this PP pipeline handles all branches.
            all_kwargs = [positive_kwargs] + ([negative_kwargs] if do_true_cfg else [])

        # Non-first ranks receive intermediate tensors asynchronously
        n = len(all_kwargs)
        its: list[AsyncIntermediateTensors | None] = [None] * n
        if not pp_group.is_first_rank:
            for i in range(n):
                its[i] = AsyncIntermediateTensors(*pp_group.irecv_tensor_dict())

        if not pp_group.is_last_rank:
            # First / middle rank: run partial forwards and propagate ITs downstream.
            for kwargs, it in zip(all_kwargs, its):
                result = self.predict_noise(**kwargs, intermediate_tensors=it)
                self._pp_send_work.extend(pp_group.isend_tensor_dict(result.tensors))
            return None

        # Last rank: run full forward
        noise_preds = [self.predict_noise(**kwargs, intermediate_tensors=it) for kwargs, it in zip(all_kwargs, its)]

        if cfg_parallel_ready:
            # All-gather the single-branch prediction across the CFG group and combine
            # on all CFG ranks so every last PP rank has an identical noise_pred.
            local_pred = noise_preds[0]
            if output_slice is not None:
                local_pred = local_pred[:, :output_slice]
            gathered = get_cfg_group().all_gather(local_pred, separate_tensors=True)
            return self.combine_cfg_noise(gathered[0], gathered[1], true_cfg_scale, cfg_normalize)

        # Sequential CFG or no-CFG path.
        if do_true_cfg:
            pos, neg = noise_preds[0], noise_preds[1]
            if output_slice is not None:
                pos = pos[:, :output_slice]
                neg = neg[:, :output_slice]
            return self.combine_cfg_noise(pos, neg, true_cfg_scale, cfg_normalize)
        pred = noise_preds[0]
        if output_slice is not None:
            pred = pred[:, :output_slice]
        return pred

    def scheduler_step_maybe_with_cfg(
        self,
        noise_pred: torch.Tensor | tuple[torch.Tensor, ...] | None,
        t: torch.Tensor | tuple[torch.Tensor, ...],
        latents: torch.Tensor | tuple[torch.Tensor, ...],
        do_true_cfg: bool,
        per_request_scheduler: Any | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | AsyncLatents:
        """
        Drop-in replacement for scheduler_step_maybe_with_cfg that also handles PP.

        Only the last rank runs the scheduler (it already has noise_pred); the result
        is sent to rank 0 which needs it for the next forward pass.

        Returns a ``AsyncLatents`` on rank 0 that transparently defers
        ``handle.wait()`` until the tensor is actually consumed (via attribute
        access or a torch operation), keeping the rank non-blocking after the
        ``irecv`` is posted.
        """
        if get_pipeline_parallel_world_size() == 1:
            return super().scheduler_step_maybe_with_cfg(
                noise_pred, t, latents, do_true_cfg, per_request_scheduler, generator
            )

        pp_group = get_pp_group()
        if pp_group.is_last_rank:
            latents = super().scheduler_step_maybe_with_cfg(
                noise_pred, t, latents, do_true_cfg, per_request_scheduler, generator
            )
            self._pp_send_work = pp_group.isend_tensor_dict({"latents": latents}, dst=0)
        elif pp_group.is_first_rank:
            latents = AsyncLatents(*pp_group.irecv_tensor_dict(src=pp_group.world_size - 1))
        return latents
