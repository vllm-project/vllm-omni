# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Omni autoregressive runner on Ascend Model Runner V2.

``OmniARModelRunner`` owns the speech lifecycle (frame outputs, connectors).
``NPUModelRunner`` owns Ascend input state (``AscendRequestState``,
``AscendInputBuffers``, ``AscendInputBatch``) and puts ``attn_state`` plus
``seq_lens_np`` on that batch. The model's NPU state reads those fields when
it builds attention metadata, and the forward runs under
``set_ascend_forward_context`` so MoE communication and Ascend attention see
the same step.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vllm.logger import init_logger
from vllm_ascend.worker.v2.model_runner import NPUModelRunner

from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

logger = init_logger(__name__)


def _call_without_unknown_kwargs(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Drop keyword arguments ``function`` does not accept.

    vLLM 0.32 takes graph-support info as ``additional_attn_cg_support``.
    vLLM-Ascend v0.30 also passed ``attn_cg_support``, a different object
    (the narrowed backend support). That name is obsolete and is dropped.
    Renaming it onto ``additional_attn_cg_support`` would overwrite the tuple
    the caller already passes under the accepted name.
    """
    import inspect

    accepted = inspect.signature(function).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in accepted.values()):
        return function(*args, **kwargs)
    return function(*args, **{name: value for name, value in kwargs.items() if name in accepted})


def _install_pcp_partition_stub() -> None:
    """No-op PCP partition when this vLLM build has no helper.

    Ascend's V2 ``prepare_inputs`` always calls
    ``pcp.maybe_partition_pcp_batch``. Drama 3 does not use prefill context
    parallel, so returning the batch unchanged is the right behavior when the
    helper is absent.
    """
    import vllm.v1.worker.gpu.model_runner as model_runner

    pcp = getattr(model_runner, "pcp", None)
    if pcp is None or hasattr(pcp, "maybe_partition_pcp_batch"):
        return

    def maybe_partition_pcp_batch(pcp_manager: Any, input_batch: Any, batch_desc: Any = None) -> Any:
        del pcp_manager, batch_desc
        return input_batch

    pcp.maybe_partition_pcp_batch = maybe_partition_pcp_batch


def v2_extra_forward_attr(ctx: Any, name: str, extra_attrs: tuple[str, ...]) -> Any:
    """Read an Ascend extra from the V2 dict, then from the context object.

    ``set_ascend_forward_context`` assigns ``moe_comm_type`` on the forward
    context. Model Runner V2's proxy only consults ``additional_kwargs``, so
    expert-parallel MoE sees no communication method while an ACL graph is
    captured and aborts. A value stored in the dict still wins.
    """
    if name not in extra_attrs:
        raise AttributeError(
            f"{name} is not extra forward context attribute, please get/set it from vllm's _forward_context directly."
        )
    extra = getattr(ctx, "additional_kwargs", None)
    if isinstance(extra, dict) and name in extra:
        return extra[name]
    return getattr(ctx, name, None)


def resolve_registered_moe_comm_method(
    methods: dict[tuple[Any, tuple[int, ...]], Any],
    moe_comm_type: Any,
    expert_shape: tuple[int, ...],
) -> Any:
    """Find the comm implementation for this expert shape.

    The forward context asks before it has a layer, so the shape is ``(0, 0)``.
    One registered shape for that communication type is unambiguous. Several
    shapes stay unresolved and the layer looks itself up with its own config.
    """
    found = methods.get((moe_comm_type, expert_shape))
    if found is not None or expert_shape != (0, 0):
        return found
    matches = [method for (comm_type, _shape), method in methods.items() if comm_type == moe_comm_type]
    if len(matches) == 1:
        return matches[0]
    return None


def _install_v2_moe_comm_context() -> None:
    """Let V2 reads see MoE fields assigned on the forward context."""
    import vllm_ascend.ops.fused_moe.moe_comm_method as moe_comm
    from vllm_ascend.ascend_forward_context import _ExtraForwardContextProxy

    if not getattr(_ExtraForwardContextProxy.__getattr__, "_omni_moe_comm_fallback", False):
        extra_attrs = _ExtraForwardContextProxy.extra_attrs

        def _proxy_getattr(self: Any, name: str) -> Any:
            return v2_extra_forward_attr(self._ctx(), name, extra_attrs)

        _proxy_getattr._omni_moe_comm_fallback = True  # type: ignore[attr-defined]
        _ExtraForwardContextProxy.__getattr__ = _proxy_getattr  # type: ignore[method-assign]

    if getattr(moe_comm.get_moe_comm_method, "_omni_shape_fallback", False):
        return
    original = moe_comm.get_moe_comm_method

    def get_moe_comm_method(moe_comm_type: Any, moe_config: Any = None) -> Any:
        resolved = original(moe_comm_type, moe_config)
        if resolved is not None or moe_config is not None:
            return resolved
        return resolve_registered_moe_comm_method(moe_comm._MoECommMethods, moe_comm_type, (0, 0))

    get_moe_comm_method._omni_shape_fallback = True  # type: ignore[attr-defined]
    moe_comm.get_moe_comm_method = get_moe_comm_method


def _default_ascend_decode_graph_flag() -> None:
    """Ascend's input batch predates vLLM's required ``decode_graph_eligible``."""
    from vllm_ascend.worker.v2.input_batch import AscendInputBatch

    init = AscendInputBatch.__init__
    if getattr(init, "_omni_default_graph_flag", False):
        return

    def wrapped(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("decode_graph_eligible", False)
        init(self, *args, **kwargs)

    wrapped._omni_default_graph_flag = True  # type: ignore[attr-defined]
    AscendInputBatch.__init__ = wrapped  # type: ignore[method-assign]


class NPUOmniARModelRunner(OmniARModelRunner, NPUModelRunner):
    """Drama 3 (and any other NPU-ready AR model) on Model Runner V2."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _default_ascend_decode_graph_flag()
        _install_pcp_partition_stub()
        _install_v2_moe_comm_context()
        super().__init__(*args, **kwargs)

    def prepare_inputs(
        self,
        scheduler_output: Any,
        batch_req_state: Any,
        batch_desc: Any,
        num_active_loras: int = 0,
    ) -> Any:
        # vLLM 0.32's Ascend batch builder takes the LoRA count itself.
        return NPUModelRunner.prepare_inputs(
            self,
            scheduler_output,
            batch_req_state,
            batch_desc,
            num_active_loras,
        )

    def _make_dummy_input_batch(
        self,
        num_reqs: int,
        num_tokens: int,
        *,
        max_query_len: int | None,
        is_padding: bool,
    ) -> Any:
        del is_padding  # Ascend's dummy batch marks itself and fills seq_lens_np.
        from vllm_ascend.worker.v2.input_batch import AscendInputBatch

        return AscendInputBatch.make_dummy(
            num_reqs,
            num_tokens,
            self.input_buffers,
            max_query_len=max_query_len,
        )

    def _forward_context(self, attn_metadata: Any, vllm_config: Any, **kwargs: Any):
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context

        if "cudagraph_runtime_mode" in kwargs:
            kwargs["aclgraph_runtime_mode"] = kwargs.pop("cudagraph_runtime_mode")
        # Slot maps live in the Ascend attention metadata, not this context.
        kwargs.pop("slot_mapping", None)
        kwargs.pop("is_padding", None)
        kwargs.setdefault("model_instance", getattr(self, "model", None))
        return set_ascend_forward_context(attn_metadata, vllm_config, **kwargs)

    def initialize_kv_cache(self, kv_cache_config: Any, *args: Any, **kwargs: Any) -> None:
        import vllm.v1.worker.gpu.model_runner as model_runner

        original = model_runner.maybe_create_adaptive_verification_manager

        def _compatible(*call_args: Any, **call_kwargs: Any) -> Any:
            return _call_without_unknown_kwargs(original, *call_args, **call_kwargs)

        model_runner.maybe_create_adaptive_verification_manager = _compatible
        try:
            NPUModelRunner.initialize_kv_cache(self, kv_cache_config, *args, **kwargs)
        finally:
            model_runner.maybe_create_adaptive_verification_manager = original

    def _close_omni_aux_output(self) -> None:
        """V1 aux-output streams are not used by this runner."""

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        OmniGPUModelRunner.load_model(self, *args, **kwargs)
        if not getattr(self.model_state, "uses_ascend_attn_metadata", False):
            raise NotImplementedError(
                "NPU Model Runner V2 needs a model state that builds Ascend attention "
                f"metadata; {type(self.model_state).__name__} does not. Keep this stage "
                "on model_runner v1, or provide an NPU state (Drama 3 does)."
            )
        logger.info(
            "NPU Model Runner V2 loaded %s with Ascend attention metadata.",
            type(self.model).__name__,
        )
