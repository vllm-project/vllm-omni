from ast import Dict
from collections.abc import Callable
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.outputs import PoolingRequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.tokenizers import TokenizerLike
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import OutputProcessor as VLLMOutputProcessor
from vllm.v1.engine.output_processor import OutputProcessorOutput, RequestOutputCollector, RequestState
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.metrics.stats import IterationStats

from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


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
        super().__init__(*args, **kwargs)
        self.mm_type: str | None = None
        self.mm_accumulated: Dict[str, Any] | None = None

    def add_multimodal_tensor(self, payload: Any | None, mm_type: str | None) -> None:
        if payload is None:
            return
        try:
            if mm_type:
                self.mm_type = (mm_type or "").lower()

            # Normalize incoming payload to dict on CPU
            def _to_cpu(x):
                if isinstance(x, torch.Tensor):
                    try:
                        return x.detach().to("cpu", non_blocking=True).contiguous()
                    except Exception:
                        return x
                return x

            if isinstance(payload, dict):
                incoming: Dict[str, Any] = {}
                target_key = self.mm_type or "hidden"

                # Iterate directly without unnecessary dict copy
                for k, v in payload.items():
                    # Optional remap: if producer used "model_outputs" or "hidden", rename to mm_type
                    if k == "model_outputs":
                        k = target_key
                    elif k == "hidden" and target_key != "hidden":
                        k = target_key

                    if isinstance(v, dict):
                        incoming[k] = {str(sk): _to_cpu(sv) for sk, sv in v.items()}
                    else:
                        incoming[k] = _to_cpu(v)
            else:
                key = self.mm_type or "hidden"
                incoming = {key: _to_cpu(payload)}

            if self.mm_accumulated is None:
                self.mm_accumulated = incoming
            else:
                # Merge keys; accumulate tensors in lists for deferred concatenation
                for k, v in incoming.items():
                    if k not in self.mm_accumulated:
                        self.mm_accumulated[k] = v
                    else:
                        existing = self.mm_accumulated[k]
                        if isinstance(v, torch.Tensor) and isinstance(existing, torch.Tensor):
                            # Use list accumulation to avoid O(n²) repeated concatenation
                            self.mm_accumulated[k] = [existing, v]
                        elif isinstance(v, torch.Tensor) and isinstance(existing, list):
                            # Append to existing list
                            existing.append(v)
                        elif isinstance(v, dict) and isinstance(existing, dict):
                            # Merge nested dicts with list accumulation for tensors
                            for sk, sv in v.items():
                                if sk not in existing:
                                    existing[sk] = sv
                                elif isinstance(sv, torch.Tensor) and isinstance(existing[sk], torch.Tensor):
                                    existing[sk] = [existing[sk], sv]
                                elif isinstance(sv, torch.Tensor) and isinstance(existing[sk], list):
                                    existing[sk].append(sv)
                                else:
                                    existing[sk] = sv
                        else:
                            self.mm_accumulated[k] = v
        except Exception:
            # Log and continue without crashing the output pipeline
            logger.exception("Error accumulating multimodal tensor")

    def _consolidate_multimodal_tensors(self) -> None:
        """Consolidate accumulated tensor lists into single tensors via concatenation."""
        if self.mm_accumulated is None:
            return
        try:
            for k, v in self.mm_accumulated.items():
                if isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                    try:
                        self.mm_accumulated[k] = torch.cat(v, dim=0)
                    except Exception:
                        # Keep last tensor on failure
                        self.mm_accumulated[k] = v[-1]
                elif isinstance(v, dict):
                    for sk, sv in v.items():
                        if isinstance(sv, list) and sv and isinstance(sv[0], torch.Tensor):
                            try:
                                v[sk] = torch.cat(sv, dim=0)
                            except Exception:
                                v[sk] = sv[-1]
        except Exception:
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
        finished = finish_reason is not None
        final_only = self.output_kind == RequestOutputKind.FINAL_ONLY

        if not finished and final_only:
            return None

        if self.stream_interval > 1:
            assert self.detokenizer is not None

            # Send output request only when
            # 1. It has finished, or
            # 2. It is the first token, or
            # 3. It has reached the stream interval number of tokens
            if not (
                finished
                or self.sent_tokens_offset == 0
                or len(self.detokenizer.output_token_ids) - self.sent_tokens_offset >= self.stream_interval
            ):
                return None

            if self.output_kind == RequestOutputKind.DELTA:
                # Send tokens from the offset in DELTA mode, otherwise all
                # tokens are sent.
                new_token_ids = self.detokenizer.output_token_ids[self.sent_tokens_offset :]
                self.sent_tokens_offset = len(self.detokenizer.output_token_ids)

        request_id = self.request_id
        output = self._new_completion_output(new_token_ids, finish_reason, stop_reason)

        if self.parent_req is None:
            outputs = [output]
        else:
            request_id, outputs, finished = self.parent_req.get_outputs(request_id, output)
            if not outputs:
                return None

        return self._new_request_output(request_id, outputs, finished, kv_transfer_params)

    def _new_completion_output(
        self,
        token_ids: list[int],
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
    ) -> Any:
        # Reuse base text/logprobs logic, then annotate with pooling_result.
        base_output = super()._new_completion_output(token_ids, finish_reason, stop_reason)
        try:
            if self.mm_accumulated is not None:
                # Attach accumulated multimodal dict on the completion output
                if not hasattr(base_output, "multimodal_output"):
                    setattr(base_output, "multimodal_output", {})
                mm_out = getattr(base_output, "multimodal_output")
                if isinstance(mm_out, dict):
                    for k, v in self.mm_accumulated.items():
                        mm_out[k] = v
                else:
                    setattr(base_output, "multimodal_output", self.mm_accumulated)
        except Exception:
            logger.exception("Error in _new_completion_output")
        return base_output


class MultimodalOutputProcessor(VLLMOutputProcessor):
    """Handles multimodal output processing by normalizing EngineCoreOutput
    before delegating to the base vLLM OutputProcessor.

    Strategy:
    - Route by EngineCoreOutput.output_type when present
      ("image", "text+image", "latents", "text").
    - Fallback to pooling/text heuristics when output_type is absent.
    - Mutate EngineCoreOutput in-place to ensure vLLM's base processor can
      produce the correct RequestOutput/PoolingRequestOutput.
    - Allow custom per-modality handlers via register_handler().
    """

    def __init__(
        self,
        tokenizer: TokenizerLike,
        log_stats: bool,
        engine_core_output_type: str | None = None,
    ):
        """Initialize the multimodal output processor.

        Args:
            tokenizer: Tokenizer for detokenizing text outputs
            log_stats: Whether to log statistics
            engine_core_output_type: Optional output type specification
                (e.g., "image", "audio", "latents"). Used to route outputs
                to appropriate processors. If None, output type is inferred.
        """
        super().__init__(tokenizer=tokenizer, log_stats=log_stats)
        self.output_handlers: dict[str, Callable[[EngineCoreOutput], None]] = {}
        self._reqid_to_mm_type: dict[str, str] = {}
        self.request_states: dict[str, OmniRequestState] = {}
        self.engine_core_output_type = engine_core_output_type

    def register_handler(self, modality: str, handler: Callable[[EngineCoreOutput], None]) -> None:
        """Register a custom handler for a specific modality.

        Allows custom processing logic for specific output modalities.
        The handler is called before default processing for outputs
        matching the specified modality.

        Args:
            modality: Modality name (e.g., "image", "audio", "latents")
            handler: Callable that takes an EngineCoreOutput and processes it
        """
        self.output_handlers[modality.lower()] = handler

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
        if request_id in self.request_states:
            raise ValueError(f"Request id {request_id} already running.")

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

    def process_outputs(
        self,
        engine_core_outputs: list[EngineCoreOutput],
        engine_core_timestamp: float | None = None,
        iteration_stats: IterationStats | None = None,
    ) -> OutputProcessorOutput:
        """Process engine core outputs into request outputs.

        Converts EngineCoreOutput objects into RequestOutput objects,
        handling multimodal outputs by routing them through appropriate
        processors and accumulating tensors in request states.

        Args:
            engine_core_outputs: List of engine core outputs to process
            engine_core_timestamp: Optional timestamp for the outputs
            iteration_stats: Optional iteration statistics

        Returns:
            OutputProcessorOutput containing processed request outputs
            and list of request IDs to abort
        """
        self._reqid_to_mm_type.clear()
        for eco in engine_core_outputs:
            mm_type = (self.engine_core_output_type or "").lower()
            if mm_type:
                self._reqid_to_mm_type[eco.request_id] = mm_type
            self._route_and_normalize(eco)

        # Build RequestOutputs without delegating to base, so we can keep ids
        request_outputs: list[Any] = []
        reqs_to_abort: list[str] = []
        for eco in engine_core_outputs:
            req_id = eco.request_id
            req_state = self.request_states.get(req_id)
            if req_state is None:
                continue

            # 1) Stats
            self._update_stats_from_output(req_state, eco, engine_core_timestamp, iteration_stats)

            new_token_ids = eco.new_token_ids
            pooling_output = eco.pooling_output
            finish_reason = eco.finish_reason
            stop_reason = eco.stop_reason
            kv_transfer_params = eco.kv_transfer_params
            req_state.num_cached_tokens = eco.num_cached_tokens
            req_state.is_prefilling = False

            # 2) Detokenize and logprobs when text path
            assert req_state.detokenizer is not None
            assert req_state.logprobs_processor is not None
            stop_string = req_state.detokenizer.update(new_token_ids, finish_reason == FinishReason.STOP)
            if stop_string:
                finish_reason = FinishReason.STOP
                stop_reason = stop_string
            req_state.logprobs_processor.update_from_output(eco)

            # 2.5) Accumulate multimodal tensors in RequestState
            try:
                mm_type = (getattr(eco, "output_type", self.engine_core_output_type) or "").lower()
                if pooling_output is not None and isinstance(req_state, OmniRequestState):
                    req_state.add_multimodal_tensor(pooling_output, mm_type)
            except Exception:
                logger.debug(
                    "Failed to accumulate multimodal tensor for request %s",
                    req_id,
                    exc_info=True,
                )

            # 3) Create RequestOutput objects, forcing combined mode to keep ids
            pooling_for_make = pooling_output
            if pooling_output is not None and new_token_ids:
                # Do not consume pooling path now; keep ids and attach mm later
                pooling_for_make = None

            # Consolidate any accumulated tensor lists before creating output
            if isinstance(req_state, OmniRequestState) and finish_reason is not None:
                req_state._consolidate_multimodal_tensors()

            ro = req_state.make_request_output(
                new_token_ids,
                pooling_for_make,
                finish_reason,
                stop_reason,
                kv_transfer_params,
            )
            if ro:
                # Attach accumulated multimodal payload if any
                try:
                    if isinstance(req_state, OmniRequestState) and req_state.mm_accumulated is not None:
                        if not hasattr(ro, "multimodal_output"):
                            setattr(ro, "multimodal_output", {})
                        ro.multimodal_output = req_state.mm_accumulated
                except Exception:
                    logger.exception("Error attaching multimodal payload in process_outputs")
                if req_state.queue is not None:
                    req_state.queue.put(ro)
                else:
                    request_outputs.append(ro)

            # 4) Free completed
            if finish_reason is not None:
                self.request_states.pop(req_id)
                parent_req = req_state.parent_req
                if parent_req and not parent_req.child_requests:
                    self.parent_requests.pop(parent_req.request_id, None)
                if not self.request_states:
                    self._requests_drained.set()
                if not eco.finished:
                    reqs_to_abort.append(req_id)
                self._update_stats_from_finished(req_state, finish_reason, iteration_stats)
                if self.tracer:
                    self.do_tracing(eco, req_state, iteration_stats)
                # Cleanup per-request mm state
                if isinstance(req_state, OmniRequestState):
                    req_state.mm_accumulated = None
                    req_state.mm_type = None

        return OutputProcessorOutput(
            request_outputs=request_outputs,
            reqs_to_abort=reqs_to_abort,
        )

    # ---- routing helpers ----
    def _route_and_normalize(self, eco: EngineCoreOutput) -> None:
        output_type = (getattr(eco, "output_type", self.engine_core_output_type) or "").lower()

        # Custom handler first (if registered)
        if output_type in self.output_handlers:
            try:
                self.output_handlers[output_type](eco)
                # Fall through to default fixups in case the handler left gaps
            except Exception:
                logger.exception("Error in custom output handler for %s", output_type)

        if output_type == "image":
            self._process_image_output(eco)
        elif output_type in ("text+image", "text,image", "image+text"):
            self._process_text_image_output(eco)
        elif output_type in ("latents", "latent"):
            self._process_latents_output(eco)
        elif output_type in ("audio", "speech"):
            self._process_audio_output(eco)
        elif output_type == "text":
            self._process_text_output(eco)
        else:
            # Fallback heuristic
            if eco.pooling_output is not None:
                self._process_pooling_output(eco)
            else:
                self._process_text_output(eco)

    # ---- modality processors ----
    def _process_image_output(self, eco: EngineCoreOutput) -> None:
        """Ensure image tensors are surfaced via pooling_output for vLLM."""
        if eco.pooling_output is None:
            tensor = self._extract_from_multimodal_outputs(eco, keys=("image", "images", "pixel_values", "pixels"))
            if tensor is not None:
                eco.pooling_output = tensor

    def _process_text_image_output(self, eco: EngineCoreOutput) -> None:
        """Allow text+image outputs. Text path stays as new_token_ids;
        image/latents route via pooling_output."""
        # Preserve text tokens as-is; ensure pooling_output carries image/latents
        if eco.pooling_output is None:
            tensor = self._extract_from_multimodal_outputs(
                eco,
                keys=(
                    "image",
                    "images",
                    "pixel_values",
                    "pixels",
                    "latent",
                    "latents",
                    "z",
                ),
            )
            if tensor is not None:
                eco.pooling_output = tensor

    def _process_latents_output(self, eco: EngineCoreOutput) -> None:
        """Ensure latent tensors are surfaced via pooling_output."""
        if eco.pooling_output is None:
            tensor = self._extract_from_multimodal_outputs(eco, keys=("latent", "latents", "z", "posterior"))
            if tensor is not None:
                eco.pooling_output = tensor

    def _process_audio_output(self, eco: EngineCoreOutput) -> None:
        """Ensure audio tensors are surfaced via pooling_output."""
        if eco.pooling_output is None:
            tensor = self._extract_from_multimodal_outputs(
                eco, keys=("audio", "audios", "wav", "waveform", "audio_pcm", "pcm")
            )
            if tensor is not None:
                eco.pooling_output = tensor

    def _process_text_output(self, eco: EngineCoreOutput) -> None:
        """No-op; base processor will detokenize new_token_ids → text."""
        return

    def _process_pooling_output(self, eco: EngineCoreOutput) -> None:
        """Optional sanity checks for pooling tensor."""
        if eco.pooling_output is None:
            return
        if not isinstance(eco.pooling_output, torch.Tensor):
            # Best-effort: convert to tensor if it's a list/ndarray-like
            try:
                eco.pooling_output = torch.as_tensor(eco.pooling_output)
            except Exception:
                pass

    def _extract_from_multimodal_outputs(self, eco: EngineCoreOutput, keys: tuple[str, ...]) -> torch.Tensor | None:
        mm = getattr(eco, "multimodal_outputs", None)
        if not isinstance(mm, dict):
            return None
        for k in keys:
            v = mm.get(k)
            if isinstance(v, torch.Tensor):
                return v
        # Try the first tensor in the dict as a fallback
        for v in mm.values():
            if isinstance(v, torch.Tensor):
                return v
        return None
