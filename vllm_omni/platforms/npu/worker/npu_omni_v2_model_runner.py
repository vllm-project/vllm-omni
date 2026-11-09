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
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.v2.utils import torch_cuda_wrapper

from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner
from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner
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
        kwargs.setdefault("decode_graph_eligible", __import__("os").environ.get("VLLM_NPU_V2_DECODE_GRAPH", "0") == "1")
        init(self, *args, **kwargs)

    wrapped._omni_default_graph_flag = True  # type: ignore[attr-defined]
    AscendInputBatch.__init__ = wrapped  # type: ignore[method-assign]


def _kstep_timing_enabled() -> bool:
    import os

    return os.environ.get("VLLM_OMNI_KSTEP_TIMING", "0") == "1"


# One K-step decode row is one codec frame of that request, so the frame loop
# has to replay exactly the booked width. A step wider than this is carrying a
# prefill-sized span and belongs to the single-forward path.
_KSTEP_MAX_FRAMES = 32


class NPUOmniARModelRunner(OmniARModelRunner, NPUModelRunner):
    """Drama 3 (and any other NPU-ready AR model) on Model Runner V2."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        import os

        if os.environ.get("VLLM_OMNI_STATIC_SHAPE_V2", "1") == "1":
            try:
                from vllm_omni.platforms.npu.attention import static_shape_decode

                static_shape_decode.install_into_ascend_backend()
            except Exception:
                logger.exception("[minicpmo] v2 static-shape install failed; stock backend kept")
        _default_ascend_decode_graph_flag()
        _install_pcp_partition_stub()
        _install_v2_moe_comm_context()
        super().__init__(*args, **kwargs)

    def _kstep_active(self) -> bool:
        """Stage-1 talker has the ngram/K-step block armed."""
        spec = getattr(getattr(self, "vllm_config", None), "speculative_config", None)
        return int(getattr(spec, "num_speculative_tokens", 0) or 0) > 0

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
        """Drive the K-step frame loop on top of FULL graph replays.

        ``cudagraph_mode=FULL_DECODE_ONLY`` sends decode batches down the plain
        replay branch, which never calls :meth:`_call_model`; without this hook
        the K-row batch would be forwarded once and one frame sampled from K
        draft rows. Each frame here re-enters the captured graph after writing
        its own codec embedding into the captured ``inputs_embeds`` buffer.
        """
        if not getattr(self, "_kstep_graph_probe_logged", False):
            self._kstep_graph_probe_logged = True
            logger.info(
                "[kstep-graph] _full_graph_step_output entered kstep_active=%s manager=%s",
                self._kstep_active(),
                type(getattr(self, "cudagraph_manager", None)).__name__,
            )
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

        spec = getattr(getattr(self, "vllm_config", None), "speculative_config", None)
        frames = int(getattr(spec, "num_speculative_tokens", 0) or 0) + 1
        if frames <= 1 or getattr(self, "model_state", None) is None:
            return None
        spans = self.model_state._request_token_spans(input_batch)
        infos = model_inputs.get("model_intermediate_buffer") or []
        # Every booked row is one codec frame of its request: the scheduler
        # advances that many positions, the connector takes one frame per row,
        # and ``merge_frame_outputs`` lays this step's stop rows out one per row
        # in request-major order. Replaying the configured K while the step
        # booked a different width leaves the tail rows without a frame and
        # shifts every later request's stop rows onto another request, which is
        # truncated or repeated audio. So replay the booked width.
        widths = [int(e) - int(s) for s, e in spans]
        uniform = bool(widths) and len(set(widths)) == 1
        width = widths[0] if uniform else 0
        prefill = bool(getattr(input_batch, "has_prefill", False))
        if prefill or not uniform or len(infos) != len(spans) or width < 2 or width > _KSTEP_MAX_FRAMES:
            self._kstep_fallback_logs = getattr(self, "_kstep_fallback_logs", 0) + 1
            if self._kstep_fallback_logs <= 5:
                logger.info(
                    "[kstep-replay] fallback frames=%d spans=%s infos=%d prefill=%s",
                    frames,
                    str(spans[:3]),
                    len(infos),
                    prefill,
                )
            return None
        # One replay per booked row. A width below the configured K would leave
        # rows unsampled; a width above it would write past the request's span.
        frames = width
        self._kstep_engaged_logs = getattr(self, "_kstep_engaged_logs", 0) + 1
        if self._kstep_engaged_logs <= 5:
            logger.info("[kstep-replay] engaged frames=%d spans=%s", frames, str(spans[:3]))
            rows = getattr(input_batch, "logits_indices", None)
            expected = len(spans) * frames
            if rows is not None and int(rows.shape[0]) != expected:
                logger.warning(
                    "[kstep-replay] stop-row mismatch rows=%d expected=%d spans=%s",
                    int(rows.shape[0]),
                    expected,
                    str(spans[:3]),
                )
        if self._kstep_engaged_logs <= 2:
            import os as _os_probe

            if _os_probe.environ.get("VLLM_OMNI_KSTEP_PROBE", "0") == "1":
                _pos = model_inputs.get("positions")
                _ids = model_inputs.get("input_ids")
                _emb = model_inputs.get("inputs_embeds")
                logger.info(
                    "[kstep-probe] inputs keys=%s pos=%s ids=%s embeds=%s",
                    sorted(model_inputs.keys()),
                    _pos.flatten()[:16].tolist() if hasattr(_pos, "flatten") else _pos,
                    _ids.flatten()[:16].tolist() if hasattr(_ids, "flatten") else _ids,
                    None if _emb is None else tuple(_emb.shape),
                )
        # The K-step control head is two-wide (continue/stop). Keep the input
        # batch's vocab in sync so the rejection sampler and any top-k
        # bookkeeping do not address non-existent token ids.
        try:
            input_batch.vocab_size = 2
        except Exception:
            pass

        if run_model is None:

            def run_model():
                out = self.model(**model_inputs)
                if isinstance(out, OmniOutput):
                    return out.text_hidden_states
                if isinstance(out, (list, tuple)):
                    return out[0]
                return out

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
        import time as _time

        _timing = _kstep_timing_enabled()
        _run_seconds = [0.0]
        if _timing:
            _base_run_model = run_model

            def _timed_run_model():
                _start = _time.perf_counter()
                _out = _base_run_model()
                _run_seconds[0] += _time.perf_counter() - _start
                return _out

            run_model = _timed_run_model
        _step_start = _time.perf_counter()
        _step_output = multiframe.run(
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
        if _timing:
            _acc = getattr(self, "_kstep_timing_acc", None)
            if _acc is None:
                _acc = self._kstep_timing_acc = [0, 0.0, 0.0]
            _acc[0] += 1
            _acc[1] += _time.perf_counter() - _step_start
            _acc[2] += _run_seconds[0]
            if _acc[0] % 40 == 0:
                logger.info(
                    "[kstep-timing] steps=%d step_ms=%.3f run_model_ms=%.3f other_ms=%.3f",
                    _acc[0],
                    1000.0 * _acc[1] / _acc[0],
                    1000.0 * _acc[2] / _acc[0],
                    1000.0 * (_acc[1] - _acc[2]) / _acc[0],
                )
        return _step_output

    def _propose_draft_tokens(self, input_batch, draft, num_sampled, num_rejected, mm_inputs):
        # K-step drafts are not predictions: they reserve the K-1 extra KV
        # rows of the next step. V1 feeds CONTINUE_TOKEN_ID (0) for every row;
        # MRv2's speculator (if any) would propose real token ids instead.
        if not self._kstep_active():
            return super()._propose_draft_tokens(input_batch, draft, num_sampled, num_rejected, mm_inputs)
        n = int(draft.num_spec_tokens or 0)
        logger.info_once("[kstep-draft] constant drafts n=%d", n)
        if n > 0:
            self.req_states.draft_tokens[input_batch.idx_mapping, :n] = 0
        self.draft_tokens_handler.set_draft_tokens(
            input_batch,
            self.req_states.draft_tokens[input_batch.idx_mapping, :n],
        )

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


class NPUOmniGenerationModelRunner(OmniGenerationModelRunner):
    """Non-autoregressive generation stage on Ascend Model Runner V2.

    Code2Wav-style stages have no KV cache, sampler or attention metadata, so
    the generic OmniGenerationModelRunner contract applies unchanged. The
    model's own NPU graph capture (minicpmo_4_5_code2wav.py) owns the
    Ascend forward context; this runner only adds the NPU-side entry point so
    the generation worker can honor model_runner: v2 instead of refusing
    the stage.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # The generic V2 base calls torch.cuda.Stream and friends in __init__;
        # Ascend's wrapper maps them onto torch.npu (same pattern as
        # vllm_ascend.worker.v2.model_runner.NPUModelRunner).
        with torch_cuda_wrapper():
            super().__init__(*args, **kwargs)
        # The V1 NPU worker's sample_tokens reads this after the runner call
        # (vllm_ascend/worker/worker.py). AR runners get it from NPUModelRunner;
        # the generic generation base does not initialize it.
        self.ascend_config = get_ascend_config()
