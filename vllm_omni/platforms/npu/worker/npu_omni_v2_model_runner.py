# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Omni autoregressive and generation runners on Ascend Model Runner V2.

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
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm_ascend.worker.v2.model_runner import NPUModelRunner

from vllm_omni.model_executor.models.minicpmo_4_5.talker_kstep_contract import (
    MINICPMO_TALKER_FRAMES_MAX,
    talker_kstep_frames,
)
from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.input_batch import AscendInputBuffers

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


class NPUOmniV2RunnerMixin:
    """Ascend input buffers, batches and forward context for Omni V2 runners."""

    input_buffers: AscendInputBuffers

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


class NPUOmniARModelRunner(NPUOmniV2RunnerMixin, OmniARModelRunner, NPUModelRunner):
    """NPU autoregressive models with Ascend attention metadata."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        import os

        if os.environ.get("VLLM_OMNI_STATIC_SHAPE_V2", "1") == "1":
            try:
                from vllm_omni.platforms.npu.attention import static_shape_decode

                static_shape_decode.install_into_ascend_backend()
            except Exception:
                logger.exception("[minicpmo] v2 static-shape install failed; stock backend kept")
        super().__init__(*args, **kwargs)

    def _kstep_active(self) -> bool:
        """Talker K-step block armed, per the shared contract."""
        cfg = getattr(self, "vllm_config", None)
        stage = getattr(getattr(cfg, "model_config", None), "model_stage", None)
        return talker_kstep_frames(stage, getattr(cfg, "speculative_config", None)) > 0

    def _dummy_sampler_run(self, hidden_states: Any) -> None:
        # K-step control rows are two-wide (continue/stop). The upstream
        # profile warmup asks the sampler for 20 logprobs, which makes
        # aclnnTopkV2 fail with "Selected index k out of range" on NPU. The
        # real talker step goes through the model's MRv2 sampler (no logprobs),
        # so skipping this warmup pass only costs first-step kernel compilation.
        if self._kstep_active():
            return
        return super()._dummy_sampler_run(hidden_states)

    def _call_model(self, model_inputs, *, input_batch, batch_desc):
        if self._kstep_active():
            kstep_output = self._run_kstep_frames(model_inputs, input_batch)
            if kstep_output is not None:
                return kstep_output
        return super()._call_model(model_inputs, input_batch=input_batch, batch_desc=batch_desc)

    def _full_graph_step_output(self, batch_desc, *, model_inputs, input_batch):
        """Drive the K-step frame loop on top of FULL graph replays."""
        if not self._kstep_active() or getattr(self, "cudagraph_manager", None) is None:
            return None

        def run_model():
            replayed = self.cudagraph_manager.run_fullgraph(batch_desc)
            if self.use_aux_hidden_state_outputs:
                replayed, _draft_aux = replayed
            hidden, _aux = self._split_fullgraph_output(replayed)
            return hidden

        return self._run_kstep_frames(model_inputs, input_batch, run_model=run_model)

    def _run_kstep_frames(self, model_inputs, input_batch, run_model=None):
        """Replay the talker K-step frame loop on the MRv2 runner.

        V1's ``talker_multiframe.run`` owns the per-frame codec feedback: each
        frame writes the previously sampled codec embedding into the persistent
        inputs_embeds row, re-runs the K-row forward, samples one codec frame
        from that row's hidden state, and finally merges the K frame outputs.
        Re-running the K-row forward each frame rewrites the KV slots in place;
        the causal mask means row k only reads rows 0..k, which all carry their
        real codec embeddings by then. This keeps one host step per K frames
        without touching the Ascend attention metadata layout.

        ``run_model`` defaults to the eager single forward; passing the graph
        replay turns the loop into per-frame replays of the K-row decode graph.
        """
        from vllm_omni.model_executor.models.output_templates import OmniOutput
        from vllm_omni.platforms.npu.worker import talker_multiframe as multiframe

        cfg = getattr(self, "vllm_config", None)
        stage = getattr(getattr(cfg, "model_config", None), "model_stage", None)
        frames = talker_kstep_frames(stage, getattr(cfg, "speculative_config", None))
        if frames <= 1 or getattr(self, "model_state", None) is None:
            return None
        spans = self.model_state._request_token_spans(input_batch)
        infos = model_inputs.get("model_intermediate_buffer") or []
        # Each booked row is one codec frame; replay the booked width to keep
        # stop rows in request-major order.
        widths = [int(e) - int(s) for s, e in spans]
        uniform = bool(widths) and len(set(widths)) == 1
        width = widths[0] if uniform else 0
        prefill = bool(getattr(input_batch, "has_prefill", False))
        if prefill:
            # Single forward is the correct path for prefill steps (the frame
            # loop only replays booked decode rows).
            self._kstep_decline_logs = getattr(self, "_kstep_decline_logs", 0) + 1
            if self._kstep_decline_logs <= 5:
                logger.warning(
                    "[kstep] frame loop declined (prefill step): infos=%d spans=%d width=%d",
                    len(infos),
                    len(spans),
                    width,
                )
            # Disarm the per-step K-step state: this step falls back to the
            # plain forward, so drafts and postprocess must follow suit.
            self._kstep_ran_last = False
            return None
        if not uniform or len(infos) != len(spans):
            # K-step-armed decode step the frame loop cannot run. Falling back
            # to a single forward would sample one frame from the last row of
            # each span and report one stop row for K booked positions,
            # silently corrupting both the codec stream and the scheduler's
            # accounting; V1 raises in the same situation.
            raise RuntimeError(
                "[kstep] multi-token decode step cannot run through the frame "
                f"loop: uniform={uniform} infos={len(infos)} spans={len(spans)} width={width}"
            )
        if width < 2 or width > MINICPMO_TALKER_FRAMES_MAX:
            # Uniform spans outside the K-step window: one-token decode rows or
            # warmup/sizing batches booked at prompt length. A single forward
            # is the correct (and only) path here.
            self._kstep_decline_logs = getattr(self, "_kstep_decline_logs", 0) + 1
            if self._kstep_decline_logs <= 5:
                logger.warning(
                    "[kstep] frame loop declined (width outside K window): width=%d max=%d",
                    width,
                    MINICPMO_TALKER_FRAMES_MAX,
                )
            # Same as the prefill decline: disarm per-step K-step state.
            self._kstep_ran_last = False
            return None
        # One replay per booked row. A width below the configured K would leave
        # rows unsampled; a width above it would write past the request's span.
        frames = width
        if run_model is None:

            def run_model():
                out = self.model(**model_inputs)
                if isinstance(out, OmniOutput):
                    return out.text_hidden_states
                if isinstance(out, (list, tuple)):
                    return out[0]
                return out

        # The K-step control head is two-wide (continue/stop). Keep the input
        # batch's vocab in sync so the rejection sampler and any top-k
        # bookkeeping do not address non-existent token ids.
        prev_vocab_size = None
        had_vocab = True
        try:
            prev_vocab_size = input_batch.vocab_size
        except AttributeError:
            # V2 InputBatch carries no vocab_size field (the sampling vocab
            # lives on model_config): there is nothing on the batch to sync
            # and no reader to mislead. The two-wide control logits are
            # consumed by the talker's device-side sampler stages, and the
            # duplex K-step path installs no rejection speculator
            # (mrv2_custom_sampler returns no speculator for K > 0). Raising
            # here kills every real MRv2 deployment at the first decode step
            # (observed on the 910B3 bench), so keep the original
            # warn-once-and-skip semantics.
            had_vocab = False
            if not getattr(self, "_kstep_vocab_warned", False):
                self._kstep_vocab_warned = True
                logger.warning(
                    "[kstep] input_batch has no vocab_size; skipping stop-row vocab sync"
                )
        else:
            input_batch.vocab_size = 2
        # Arm the per-step K-step state: drafts and postprocess now follow
        # the frame-loop semantics until the next decline re-disarms them.
        self._kstep_ran_last = True
        try:
            # make_omni_output reads its per-request info from the extra kwargs;
            # forwarding the whole forward-input dict duplicates that key.
            model_kwargs = {
                "runtime_additional_information": infos,
                "intermediate_tensors": model_inputs.get("intermediate_tensors"),
            }
            model_kwargs_extra = {
                "request_token_spans": spans,
                "model_intermediate_buffer": infos,
            }
            # The MRv2 stage loads the outer omni wrapper; the codec frame hooks
            # (preprocess/make_omni_output/take_batch_stop_logits/merge_frame_outputs)
            # live on the inner talker, which the wrapper only partially forwards.
            codec_model = getattr(self.model, "talker", self.model)
            return multiframe.run(
                model=codec_model,
                run_model=run_model,
                after_forward=lambda: None,
                inputs_embeds=model_inputs.get("inputs_embeds"),
                input_ids=model_inputs.get("input_ids"),
                frames=frames,
                model_kwargs=model_kwargs,
                model_kwargs_extra=model_kwargs_extra,
                narrow=None,
            )
        finally:
            # Restore before the next step: a declined or non-K-step batch on
            # this runner must not see the two-wide control vocab.
            if had_vocab:
                input_batch.vocab_size = prev_vocab_size

    def _propose_draft_tokens(self, input_batch, draft, num_sampled, num_rejected, mm_inputs):
        # K-step drafts are not predictions: they reserve the K-1 extra KV
        # rows of the next step. V1 feeds CONTINUE_TOKEN_ID (0) for every row;
        # MRv2's speculator (if any) would propose real token ids instead.
        if not getattr(self, "_kstep_ran_last", False):
            # A declined step fell back to the plain forward: leave no
            # speculative CONTINUE rows behind and defer to the base flow.
            return super()._propose_draft_tokens(input_batch, draft, num_sampled, num_rejected, mm_inputs)
        n = int(draft.num_spec_tokens or 0)
        if n > 0:
            self.req_states.draft_tokens[input_batch.idx_mapping, :n] = 0
        self.draft_tokens_handler.set_draft_tokens(
            input_batch,
            self.req_states.draft_tokens[input_batch.idx_mapping, :n],
        )

    def initialize_kv_cache(self, kv_cache_config: Any, *args: Any, **kwargs: Any) -> None:
        super().initialize_kv_cache(kv_cache_config, *args, **kwargs)
        # The Ascend override initializes native KV directly, bypassing the
        # shared MRv2 mixin. Thinker hits also need cached stage-output rows.
        is_profiling = bool(kwargs.get("is_profiling", args[0] if args else False))
        self._init_omni_prefix_cache(is_profiling=is_profiling)
        if self.omni_prefix_cache is not None:
            logger.info("NPU MRv2 stage-output prefix cache initialized.")

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
