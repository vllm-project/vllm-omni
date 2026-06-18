from dataclasses import fields as dataclass_fields
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.outputs import CompletionOutput, PoolingRequestOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.tokenizers import TokenizerLike
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import OutputProcessor as VLLMOutputProcessor
from vllm.v1.engine.output_processor import (
    OutputProcessorOutput,
    RequestOutputCollector,
    RequestState,
)
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.metrics.stats import IterationStats, RequestStateStats

from vllm_omni.data_entry_keys import unflatten_payload
from vllm_omni.engine.mm_outputs import MultimodalCompletionOutput, MultimodalPayload
from vllm_omni.engine.output_modality import (
    DRAINABLE_MODALITIES,
    OutputModality,
    TensorAccumulationStrategy,
    get_accumulation_strategy,
)
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _to_cpu(x: Any) -> Any:
    """Move a tensor to CPU; pass through non-tensors unchanged."""
    if isinstance(x, torch.Tensor):
        try:
            return x.detach().to("cpu", non_blocking=True).contiguous()
        except (RuntimeError, AttributeError):
            return x
    if isinstance(x, dict):
        return {str(sk): _to_cpu(sv) for sk, sv in x.items()}
    return x


def _merge_payload(target: MultimodalPayload, incoming: MultimodalPayload) -> None:
    """Merge *incoming* into *target* using list-based deferred concatenation."""
    for k, v in incoming.tensors.items():
        if k not in target.tensors:
            target.tensors[k] = v
        else:
            existing = target.tensors[k]
            if isinstance(existing, list):
                existing.append(v)
            elif isinstance(existing, torch.Tensor) and isinstance(v, torch.Tensor):
                target.tensors[k] = [existing, v]
            else:
                target.tensors[k] = v

    for k, v in incoming.metadata.items():
        if k not in target.metadata:
            target.metadata[k] = v
        else:
            existing = target.metadata[k]
            if isinstance(v, torch.Tensor) and isinstance(existing, torch.Tensor):
                target.metadata[k] = [existing, v]
            elif isinstance(v, torch.Tensor) and isinstance(existing, list):
                existing.append(v)
            else:
                # Metadata: replace with latest value
                target.metadata[k] = v


def _cat_tensors(
    tensors: list[torch.Tensor],
    strategy: TensorAccumulationStrategy,
) -> torch.Tensor:
    """Concatenate a list of tensors according to *strategy*."""
    if strategy == TensorAccumulationStrategy.CONCAT_LAST:
        return torch.cat(tensors, dim=-1)
    if strategy == TensorAccumulationStrategy.REPLACE:
        return tensors[-1]
    # CONCAT_DIM0 / APPEND_LIST / default
    return torch.cat(tensors, dim=0)


def _modality_to_type_string(modality: OutputModality) -> str:
    """Convert an OutputModality flag to a lowercase type string."""
    try:
        if OutputModality.AUDIO in modality:
            return "audio"
        if OutputModality.IMAGE in modality:
            return "image"
        if OutputModality.LATENT in modality:
            return "latent"
    except TypeError:
        # Flag identity mismatch (e.g. after module reload in tests).
        name = getattr(modality, "name", "") or ""
        lowered = name.lower()
        if "audio" in lowered:
            return "audio"
        if "image" in lowered:
            return "image"
        if "latent" in lowered:
            return "latent"
    return "text"


class OmniRequestState(RequestState):
    """Request state for omni models, tracking multimodal outputs.

    Extends the base RequestState with support for accumulating
    multimodal tensor outputs (e.g., images, audio, latents) that
    are produced incrementally during generation.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # arrival_time is always passed as a keyword argument by the sole caller
        # (from_new_request).  The *args fallback via positional index was removed
        # because upstream RequestState.__init__ reorders parameters between
        # releases, making args[12] fragile.
        arrival_time = kwargs.get("arrival_time")
        super().__init__(*args, **kwargs)
        self.native_text_stats = RequestStateStats(arrival_time=float(arrival_time or 0.0))
        # Omni-specific: multimodal output accumulation
        # TODO: mm_type is per-request, not per-key. If a model ever produces
        # both audio and latent outputs, the modality type would flip on each
        # add_multimodal_tensor() call. Consider tracking per-key modality
        # types (e.g. dict[str, str]) for future multi-output models.
        self.mm_type: str | None = None
        self.mm_accumulated: MultimodalPayload = MultimodalPayload()

    def apply_streaming_update(self, update) -> None:
        super().apply_streaming_update(update)
        self.native_text_stats.arrival_time = update.arrival_time

    @staticmethod
    def _to_cpu(x):
        """Try to convert to CPU tensor if needed."""
        if isinstance(x, torch.Tensor):
            try:
                return x.detach().to("cpu", non_blocking=True).contiguous()
            except Exception:
                return x
        return x

    def add_multimodal_tensor(self, payload: Any | None, mm_type: str | None) -> None:
        """Accumulate a multimodal tensor payload into the request state.

        Normalizes incoming payloads (dict or raw tensor) into a
        MultimodalPayload and merges with any previously accumulated data.
        Uses list-based deferred concatenation to avoid O(n²) repeated
        torch.cat calls.
        """
        if payload is None:
            return
        try:
            if mm_type:
                self.mm_type = (mm_type or "").lower()

            target_key = self.mm_type or "hidden"

            # Build incoming MultimodalPayload
            if isinstance(payload, dict):
                remapped: dict[str, Any] = {}
                for k, v in payload.items():
                    # Normalize producer keys to the modality name.
                    # AR runners produce {"hidden": ...} and generation
                    # runners produce {"model_outputs": ...}; remap both
                    # to the semantic modality key (e.g. "audio", "latent").
                    if k == "model_outputs":
                        k = target_key
                    elif k == "hidden" and target_key != "hidden":
                        k = target_key
                    remapped[k] = _to_cpu(v)
                incoming = MultimodalPayload.from_dict(remapped)
            elif isinstance(payload, MultimodalPayload):
                incoming = payload
            else:
                incoming = MultimodalPayload.from_dict({target_key: _to_cpu(payload)})

            if incoming is None:
                return

            if self.mm_accumulated.is_empty:
                self.mm_accumulated = incoming
            else:
                _merge_payload(self.mm_accumulated, incoming)
        except (ValueError, TypeError, RuntimeError):
            logger.exception("Error accumulating multimodal tensor")

    def _consolidate_multimodal_tensors(self) -> None:
        """Consolidate accumulated tensor lists into single tensors.

        Uses TensorAccumulationStrategy derived from the output modality
        to determine concatenation behavior. Metadata values always use
        REPLACE (keep latest).
        """
        if self.mm_accumulated.is_empty:
            return

        try:
            modality = OutputModality.from_string(self.mm_type)
        except (ValueError, KeyError):
            modality = OutputModality.TEXT
        strategy = get_accumulation_strategy(modality)

        try:
            for k, v in list(self.mm_accumulated.tensors.items()):
                if isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                    try:
                        self.mm_accumulated.tensors[k] = _cat_tensors(v, strategy)
                    except RuntimeError:
                        if k == "audio":
                            try:
                                self.mm_accumulated.tensors[k] = torch.cat(v, dim=-1)
                            except RuntimeError:
                                self.mm_accumulated.tensors[k] = torch.cat([t.reshape(-1) for t in v], dim=0)
                        else:
                            logger.warning(
                                "Error concatenating tensor for key %s; keeping last tensor",
                                k,
                            )
                            self.mm_accumulated.tensors[k] = v[-1]

            # Metadata: consolidate any deferred tensor lists (REPLACE semantics)
            for k, v in list(self.mm_accumulated.metadata.items()):
                if isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                    self.mm_accumulated.metadata[k] = v[-1]
                elif isinstance(v, dict):
                    for sk, sv in list(v.items()):
                        if isinstance(sv, list) and sv and isinstance(sv[0], torch.Tensor):
                            v[sk] = sv[-1]
        except (RuntimeError, TypeError, KeyError):
            logger.exception("Error consolidating multimodal tensors")

    # Override: do not route to pooling-only path; always create completion
    # outputs, and attach pooling_result into the CompletionOutput.
    def make_request_output(
        self,
        new_token_ids: list[int],
        pooling_output: torch.Tensor | None,
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
        kv_transfer_params: dict[str, Any] | None = None,
    ) -> OmniRequestOutput | PoolingRequestOutput | None:
        """Create a request output from generation results.

        Creates a RequestOutput or PoolingRequestOutput from the generated
        tokens and accumulated multimodal outputs. Attaches multimodal
        tensors to the completion output if available.

        Args:
            new_token_ids: List of newly generated token IDs
            pooling_output: Optional pooling output tensor
            finish_reason: Optional finish reason indicating why generation stopped
            stop_reason: Optional stop reason (token ID or stop string)
            kv_transfer_params: Optional KV cache transfer parameters

        Returns:
            OmniRequestOutput or PoolingRequestOutput if output should be
            emitted (based on finish status and output kind), None otherwise
        """
        # Pooling-only requests should follow base behavior.
        if self.detokenizer is None and pooling_output is not None:
            return super().make_request_output(
                new_token_ids,
                pooling_output,
                finish_reason,
                stop_reason,
                kv_transfer_params,
            )

        finished = finish_reason is not None
        final_only = self.output_kind == RequestOutputKind.FINAL_ONLY
        is_delta = self.output_kind == RequestOutputKind.DELTA

        if not finished and final_only:
            return None

        # Consolidate accumulated tensors when finishing, or for every
        # CUMULATIVE step so the consumer always sees a single tensor.
        if finished or not is_delta:
            self._consolidate_multimodal_tensors()

        if self.stream_interval > 1 and self.detokenizer is not None:
            # Send output request only when
            # 1. It has finished, or
            # 2. It is the first token, or
            # 3. It has reached the stream interval number of tokens
            if not (
                finished
                or self.sent_tokens_offset == 0
                or self.detokenizer.num_output_tokens() - self.sent_tokens_offset >= self.stream_interval
            ):
                return None

            if self.output_kind == RequestOutputKind.DELTA:
                # Send tokens from the offset in DELTA mode, otherwise all
                # tokens are sent.
                new_token_ids = self.detokenizer.output_token_ids[self.sent_tokens_offset :]
                self.sent_tokens_offset = self.detokenizer.num_output_tokens()

        external_req_id = self.external_req_id

        output = self._new_completion_output(new_token_ids, finish_reason, stop_reason)

        if self.parent_req is None:
            outputs = [output]
        else:
            outputs, finished = self.parent_req.get_outputs(self.request_id, output)
            if not outputs:
                return None
            external_req_id = self.parent_req.external_req_id

        return self._new_request_output(
            external_req_id,
            outputs,
            finished,
            kv_transfer_params,
        )

    def _new_completion_output(
        self,
        token_ids: list[int],
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
    ) -> MultimodalCompletionOutput | CompletionOutput:
        """Create a completion output with multimodal data attached.

        Returns a MultimodalCompletionOutput when multimodal data has been
        accumulated, otherwise returns the base CompletionOutput.  Snapshots
        the accumulated payload before draining DELTA-mode modality keys so
        that callers receive an immutable view.
        """
        # When there is no detokenizer (generation stages), build a minimal
        # CompletionOutput directly since upstream asserts detokenizer != None.
        if self.detokenizer is None:
            finished = finish_reason is not None
            base_output = CompletionOutput(
                index=self.request_index,
                text="",
                token_ids=token_ids,
                logprobs=None,
                cumulative_logprob=None,
                finish_reason=str(finish_reason) if finished else None,
                stop_reason=stop_reason if finished else None,
            )
        else:
            base_output = super()._new_completion_output(token_ids, finish_reason, stop_reason)

        # Always provide cumulative token IDs for inter-stage processors.
        if self.detokenizer is not None:
            base_output.cumulative_token_ids = list(self.detokenizer.output_token_ids)
        else:
            base_output.cumulative_token_ids = list(token_ids)

        # Attach cumulative_text only at the final step for inter-stage use.
        if finish_reason is not None and hasattr(self.detokenizer, "output_text"):
            base_output.cumulative_text = self.detokenizer.output_text

        try:
            if self.mm_accumulated and not self.mm_accumulated.is_empty:
                # Snapshot: copy current tensors/metadata so drain doesn't
                # mutate the payload already handed to the caller.
                # Unflatten dotted keys (e.g. "hidden_states.layer_0") back
                # to nested dicts so downstream consumers (thinker2talker etc.)
                # can access with .get("hidden_states", {}).get("layers", {}).
                merged = {**self.mm_accumulated.tensors, **self.mm_accumulated.metadata}
                nested = unflatten_payload(merged)
                snapshot = MultimodalPayload.from_dict(nested) or MultimodalPayload()
                kwargs = {f.name: getattr(base_output, f.name) for f in dataclass_fields(CompletionOutput)}
                output = MultimodalCompletionOutput(
                    multimodal_output=snapshot,
                    **kwargs,
                )
                output.cumulative_token_ids = base_output.cumulative_token_ids
                if hasattr(base_output, "cumulative_text"):
                    output.cumulative_text = base_output.cumulative_text

                # DELTA mode: drain modality keys (e.g. audio) so the next
                # step only sees freshly accumulated data for those keys.
                if self.output_kind == RequestOutputKind.DELTA:
                    for modality_key in DRAINABLE_MODALITIES:
                        self.mm_accumulated.tensors.pop(modality_key, None)

                return output
        except (RuntimeError, TypeError, AttributeError):
            logger.exception("Error creating MultimodalCompletionOutput")
        return base_output

    def _new_request_output(
        self,
        external_req_id: str,
        outputs: list,
        finished: bool,
        kv_transfer_params: dict[str, Any] | None = None,
    ) -> RequestOutput | PoolingRequestOutput:
        """Create request output, handling no-detokenizer generation stages.

        Upstream asserts ``self.logprobs_processor is not None`` which fails
        for generation stages (talker, code2wav) that have no tokenizer.
        When the logprobs processor is absent we build the RequestOutput
        directly with ``prompt_logprobs=None``.
        """
        if self.logprobs_processor is not None:
            return super()._new_request_output(
                external_req_id,
                outputs,
                finished,
                kv_transfer_params,
            )

        # No-detokenizer path: build RequestOutput directly.
        prompt_token_ids = self.prompt_token_ids
        if prompt_token_ids is None and self.prompt_embeds is not None:
            prompt_token_ids = [0] * len(self.prompt_embeds)
        if prompt_token_ids is None:
            prompt_token_ids = []

        return RequestOutput(
            request_id=external_req_id,
            lora_request=self.lora_request,
            prompt=self.prompt,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None,
            outputs=outputs,
            finished=finished,
            kv_transfer_params=kv_transfer_params,
            num_cached_tokens=self.num_cached_tokens,
            metrics=self.stats,
        )


class MultimodalOutputProcessor(VLLMOutputProcessor):
    """Handles multimodal output processing.

    Captures multimodal outputs from OmniEngineCoreOutput and accumulates
    them as MultimodalPayload in OmniRequestState, before delegating to
    the base vLLM OutputProcessor for text handling.

    The data flow is:
    1. For each EngineCoreOutput with multimodal_output:
       - Capture into OmniRequestState.add_multimodal_tensor()
    2. Base vLLM OutputProcessor handles text detokenization
    3. On finish, _consolidate_multimodal_tensors() concatenates accumulated
       tensors using strategy-based dispatch
    4. _new_completion_output() returns MultimodalCompletionOutput
    """

    def __init__(
        self,
        tokenizer: TokenizerLike | None,
        *,
        log_stats: bool,
        stream_interval: int = 1,
        tracing_enabled: bool = False,
        engine_core_output_type: str | None = None,
        output_modality: OutputModality = OutputModality.TEXT,
    ):
        """Initialize the multimodal output processor.

        Args:
            tokenizer: Tokenizer for detokenizing text outputs
            log_stats: Whether to log statistics
            stream_interval: Stream interval for output generation
            engine_core_output_type: Optional output type string (e.g.,
                "image", "audio", "latent"). Converted to OutputModality
                internally. Kept for backward compatibility with
                stage_init_utils.
            output_modality: Type-safe output modality flag. Used to tag
                multimodal outputs with the correct modality key when
                per-output type info is unavailable.
        """
        super().__init__(
            tokenizer=tokenizer,
            log_stats=log_stats,
            stream_interval=stream_interval,
            tracing_enabled=tracing_enabled,
        )
        # Convert string-based engine_core_output_type to OutputModality
        if engine_core_output_type is not None:
            self.output_modality = OutputModality.from_string(engine_core_output_type)
        else:
            self.output_modality = output_modality
        self.engine_core_output_type = engine_core_output_type
        self._native_text_metrics_by_request: dict[str, dict[str, Any]] = {}

    def _native_text_metric_record(self, request_id: str) -> dict[str, Any]:
        return self._native_text_metrics_by_request.setdefault(
            request_id,
            {
                "vllm_ttft_ms": 0.0,
                "vllm_tpot_ms": 0.0,
                "vllm_itl_ms": 0.0,
                "vllm_itls_ms": [],
            },
        )

    def pop_native_text_metrics(self, request_id: str) -> dict[str, Any]:
        return self._native_text_metrics_by_request.pop(request_id, {})

    def abort_requests(self, request_ids, internal: bool) -> list[str]:
        request_ids = list(request_ids)
        for request_id in request_ids:
            if internal:
                req_state = self.request_states.get(request_id)
                if req_state is not None:
                    self._native_text_metrics_by_request.pop(req_state.external_req_id, None)
            else:
                self._native_text_metrics_by_request.pop(request_id, None)
        return super().abort_requests(request_ids, internal)

    def add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None = None,
        request_index: int = 0,
        queue: RequestOutputCollector | None = None,
    ) -> None:
        """Add a new request to be processed.

        Creates an OmniRequestState for the request and registers it
        for output processing.

        Args:
            request: Engine core request to add
            prompt: Optional prompt string for the request
            parent_req: Optional parent request for parallel sampling
            request_index: Index of the request in the batch
            queue: Optional queue for collecting outputs

        Raises:
            ValueError: If the request ID is already registered
        """
        request_id = request.request_id
        req_state = self.request_states.get(request_id)
        if req_state is not None:
            self._update_streaming_request_state(req_state, request, prompt)
            return

        req_state = OmniRequestState.from_new_request(
            tokenizer=self.tokenizer,
            request=request,
            prompt=prompt,
            parent_req=parent_req,
            request_index=request_index,
            queue=queue,
            log_stats=self.log_stats,
            stream_interval=self.stream_interval,
        )
        self.request_states[request_id] = req_state
        if parent_req:
            self.parent_requests[parent_req.request_id] = parent_req
        self.external_req_ids[req_state.external_req_id].append(request_id)

    def remove_request(self, request_id: str) -> None:
        """Rollback one previously registered request if it was never submitted."""
        req_state = self.request_states.pop(request_id, None)
        if req_state is None:
            return

        external_req_id = getattr(req_state, "external_req_id", None)
        if external_req_id is not None:
            self._native_text_metrics_by_request.pop(external_req_id, None)
            request_ids = self.external_req_ids.get(external_req_id)
            if request_ids is not None:
                self.external_req_ids[external_req_id] = [rid for rid in request_ids if rid != request_id]
                if not self.external_req_ids[external_req_id]:
                    self.external_req_ids.pop(external_req_id, None)

        parent_req = getattr(req_state, "parent_req", None)
        if parent_req is not None:
            self.parent_requests.pop(parent_req.request_id, None)

    def process_outputs(
        self,
        engine_core_outputs: list[EngineCoreOutput],
        engine_core_timestamp: float | None = None,
        iteration_stats: IterationStats | None = None,
    ) -> OutputProcessorOutput:
        default_mm_type = _modality_to_type_string(self.output_modality)

        # Separate outputs that upstream can handle (has detokenizer or
        # pooling) from multimodal-only outputs (no detokenizer, no pooling)
        # that would trigger upstream's `assert detokenizer is not None`.
        upstream_outputs: list[EngineCoreOutput] = []
        mm_only_outputs: list[EngineCoreOutput] = []

        for eco in engine_core_outputs:
            req_state = self.request_states.get(eco.request_id)
            if req_state is None:
                continue

            # Accumulate multimodal tensors regardless of path.
            if isinstance(req_state, OmniRequestState):
                mm_output = getattr(eco, "multimodal_output", None)
                if mm_output is not None:
                    mm_type = getattr(eco, "output_type", None) or default_mm_type
                    req_state.add_multimodal_tensor(mm_output, mm_type)

            # Route: if no detokenizer and no pooling output, handle locally
            # to avoid upstream's assert on detokenizer.
            if req_state.detokenizer is None and eco.pooling_output is None:
                mm_only_outputs.append(eco)
            else:
                upstream_outputs.append(eco)

        # Handle multimodal-only outputs (generation stages) locally.
        self._process_mm_only_outputs(mm_only_outputs)

        # Delegate text/pooling outputs to upstream.
        return super().process_outputs(
            upstream_outputs,
            engine_core_timestamp=engine_core_timestamp,
            iteration_stats=iteration_stats,
        )

    def _process_mm_only_outputs(
        self,
        engine_core_outputs: list[EngineCoreOutput],
    ) -> None:
        """Handle outputs from generation stages that have no detokenizer.

        These cannot go through upstream process_outputs because it asserts
        detokenizer is not None when pooling_output is None.
        """
        for eco in engine_core_outputs:
            req_state = self.request_states.get(eco.request_id)
            if req_state is None or not isinstance(req_state, OmniRequestState):
                continue

            new_token_ids = eco.new_token_ids
            finish_reason = eco.finish_reason
            stop_reason = eco.stop_reason
            kv_transfer_params = eco.kv_transfer_params
            routed_experts = eco.routed_experts
            req_state.num_cached_tokens = eco.num_cached_tokens
            req_state.is_prefilling = False

            if request_output := req_state.make_request_output(
                new_token_ids,
                None,  # pooling_output
                finish_reason,
                stop_reason,
                kv_transfer_params,
                routed_experts,
            ):
                if req_state.queue is not None:
                    req_state.queue.put(request_output)

            if finish_reason is not None:
                self._finish_request(req_state)

    def _update_stats_from_output(
        self,
        req_state: RequestState,
        engine_core_output: EngineCoreOutput,
        engine_core_timestamp: float | None,
        iteration_stats: IterationStats | None,
    ):
        was_prefilling = req_state.is_prefilling
        native_stats = req_state.native_text_stats if isinstance(req_state, OmniRequestState) else None
        previous_last_token_ts = native_stats.last_token_ts if native_stats is not None else 0.0

        # NOTE: We pass ``None`` for  *iteration_stats* to the parent so that
        # the upstream's ``_update_stats_from_output`` logs stats via its own
        # ``RequestStateStats`` (req_state.stats) without touching the omni
        # ``IterationStats`` object.  Below we re-apply the same update using
        # *native_stats* (the omni-specific ``RequestStateStats`` attached to
        # ``OmniRequestState``), which correctly feeds the omni metrics path.
        # If the upstream adds new logic in ``_update_stats_from_output`` (e.g.
        # additional stat fields), this override must be kept in sync.
        super()._update_stats_from_output(
            req_state,
            engine_core_output,
            engine_core_timestamp,
            None,
        )

        if iteration_stats is None or engine_core_timestamp is None or native_stats is None:
            return

        iteration_stats.update_from_output(
            engine_core_output,
            engine_core_timestamp,
            was_prefilling,
            native_stats,
            self.lora_states,
            req_state.lora_name,
        )
        record = self._native_text_metric_record(req_state.external_req_id)
        if was_prefilling:
            record["vllm_ttft_ms"] = max(float(native_stats.first_token_latency) * 1000.0, 0.0)
            return

        if previous_last_token_ts > 0:
            itl_ms = max((engine_core_timestamp - previous_last_token_ts) * 1000.0, 0.0)
            itls_ms = record.setdefault("vllm_itls_ms", [])
            itls_ms.append(itl_ms)
            record["vllm_itl_ms"] = sum(itls_ms) / float(len(itls_ms))

    def _update_stats_from_finished(
        self,
        req_state: RequestState,
        finish_reason: FinishReason | None,
        iteration_stats: IterationStats | None,
    ):
        super()._update_stats_from_finished(req_state, finish_reason, None)

        native_stats = req_state.native_text_stats if isinstance(req_state, OmniRequestState) else None
        if iteration_stats is None or native_stats is None or finish_reason is None:
            return

        iteration_stats.update_from_finished_request(
            finish_reason=finish_reason,
            request_id=req_state.external_req_id,
            num_prompt_tokens=req_state.prompt_len,
            max_tokens_param=req_state.max_tokens_param,
            req_stats=native_stats,
            num_cached_tokens=req_state.num_cached_tokens,
        )
        if not iteration_stats.finished_requests:
            return

        finished_request = iteration_stats.finished_requests[-1]
        if finished_request.request_id != req_state.external_req_id:
            return
        self._native_text_metric_record(req_state.external_req_id)["vllm_tpot_ms"] = (
            float(finished_request.mean_time_per_output_token) * 1000.0
        )
