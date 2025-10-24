from typing import Dict, Callable, Optional, Any, Union

import torch

from vllm.v1.engine.output_processor import OutputProcessor as VLLMOutputProcessor
from vllm.v1.engine.output_processor import OutputProcessorOutput, RequestState, RequestOutputCollector
from vllm.transformers_utils.tokenizer_group import TokenizerGroup
from vllm.v1.engine import FinishReason
from vllm.v1.metrics.stats import IterationStats
from vllm.sampling_params import RequestOutputKind
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.detokenizer import IncrementalDetokenizer
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreOutput


logger = init_logger(__name__)


class OmniRequestState(RequestState):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mm_type: Optional[str] = None
        self.mm_accumulated: Optional[torch.Tensor] = None

    @classmethod
    def from_new_request(
        cls,
        tokenizer: AnyTokenizer,
        request: EngineCoreRequest,
        prompt: Optional[str],
        parent_req: Optional[ParentRequest],
        request_index: int,
        queue: Optional[Any],
        log_stats: bool,
    ) -> "OmniRequestState":

        if sampling_params := request.sampling_params:
            if not sampling_params.detokenize:
                tokenizer = None
            output_kind = sampling_params.output_kind
            logprobs_processor = LogprobsProcessor.from_new_request(
                tokenizer=tokenizer,
                request=request,
            )
            detokenizer = IncrementalDetokenizer.from_new_request(
                tokenizer=tokenizer,
                request=request,
            )
            max_tokens_param = sampling_params.max_tokens
        else:
            logprobs_processor = None
            detokenizer = None
            max_tokens_param = None
            assert request.pooling_params is not None
            output_kind = request.pooling_params.output_kind

        return cls(
            request_id=request.request_id,
            parent_req=parent_req,
            request_index=request_index,
            lora_name=(request.lora_request.name
                       if request.lora_request is not None else None),
            output_kind=output_kind,
            prompt=prompt,
            prompt_token_ids=request.prompt_token_ids,
            logprobs_processor=logprobs_processor,
            detokenizer=detokenizer,
            max_tokens_param=max_tokens_param,
            arrival_time=request.arrival_time,
            queue=queue,
            log_stats=log_stats,
        )

    def add_multimodal_tensor(self, tensor: Optional[torch.Tensor],
                               mm_type: Optional[str]) -> None:
        if tensor is None:
            return
        try:
            if mm_type:
                self.mm_type = (mm_type or "").lower()
            t = tensor.detach()
            try:
                t = t.to("cpu")
            except Exception:
                # Best-effort CPU move; keep original device if conversion fails
                logger.debug("Failed to move multimodal tensor to CPU", exc_info=True)
            if self.mm_accumulated is None:
                self.mm_accumulated = t
            else:
                self.mm_accumulated = torch.cat([self.mm_accumulated, t], dim=0)
        except Exception:
            # Log and continue without crashing the output pipeline
            logger.exception("Error accumulating multimodal tensor")

    # Override: do not route to pooling-only path; always create completion
    # outputs, and attach pooling_result into the CompletionOutput.
    def make_request_output(
        self,
        new_token_ids: list[int],
        pooling_output: Optional[torch.Tensor],
        finish_reason: Optional[FinishReason],
        stop_reason: Optional[Union[int, str]],
        kv_transfer_params: Optional[dict[str, Any]] = None,
        num_cached_tokens: Optional[int] = None,
    ) -> Optional[Any]:
        finished = finish_reason is not None
        final_only = self.output_kind == RequestOutputKind.FINAL_ONLY

        if not finished and final_only:
            return None

        if num_cached_tokens is not None:
            # Keep num_cached_tokens in RequestOutput for compatibility
            try:
                self.num_cached_tokens = num_cached_tokens  # type: ignore[attr-defined]
            except Exception:
                pass

        request_id = self.request_id
        output = self._new_completion_output(new_token_ids, finish_reason,
                                             stop_reason)

        if self.parent_req is None:
            outputs = [output]
        else:
            request_id, outputs, finished = self.parent_req.get_outputs(
                request_id, output)
            if not outputs:
                return None

        return self._new_request_output(request_id, outputs, finished,
                                        kv_transfer_params)

    def _new_completion_output(
        self,
        token_ids: list[int],
        finish_reason: Optional[FinishReason],
        stop_reason: Optional[Union[int, str]]
    ) -> Any:
        # Reuse base text/logprobs logic, then annotate with pooling_result.
        base_output = super()._new_completion_output(token_ids, finish_reason,
                                                     stop_reason)
        try:
            if self.mm_accumulated is not None:
                tensor = self.mm_accumulated
                try:
                    tensor = tensor.detach().to("cpu")
                except Exception:
                    logger.debug("Failed to move accumulated multimodal tensor to CPU", exc_info=True)
                # Attach on the completion output for downstream consumers.
                if not hasattr(base_output, "multimodal_output"):
                    setattr(base_output, "multimodal_output", {})
                setattr(base_output, "multimodal_output", {self.mm_type: tensor})
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
    def __init__(self, tokenizer: TokenizerGroup, log_stats: bool, engine_core_output_type: Optional[str] = "text"):
        super().__init__(tokenizer=tokenizer, log_stats=log_stats)
        self.output_handlers: Dict[str, Callable[[EngineCoreOutput], None]] = {}
        self._reqid_to_mm_type: Dict[str, str] = {}
        self.request_states: dict[str, OmniRequestState] = {}
        self.engine_core_output_type = engine_core_output_type
        
    def register_handler(self, modality: str,
                         handler: Callable[[EngineCoreOutput], None]) -> None:
        self.output_handlers[modality.lower()] = handler
    
    def add_request(
        self,
        request: EngineCoreRequest,
        prompt: Optional[str],
        parent_req: Optional[ParentRequest] = None,
        request_index: int = 0,
        queue: Optional[RequestOutputCollector] = None,
    ) -> None:
        request_id = request.request_id
        if request_id in self.request_states:
            raise ValueError(f"Request id {request_id} already running.")

        tokenizer = None if not self.tokenizer else \
            self.tokenizer.get_lora_tokenizer(request.lora_request)

        req_state = OmniRequestState.from_new_request(tokenizer=tokenizer,
                                                  request=request,
                                                  prompt=prompt,
                                                  parent_req=parent_req,
                                                  request_index=request_index,
                                                  queue=queue,
                                                  log_stats=self.log_stats)
        self.request_states[request_id] = req_state
        self.lora_states.add_request(req_state)
        if parent_req:
            self.parent_requests[parent_req.request_id] = parent_req
    
    def process_outputs(
        self,
        engine_core_outputs: list[EngineCoreOutput],
        engine_core_timestamp: Optional[float] = None,
        iteration_stats: Optional[IterationStats] = None,
    ) -> OutputProcessorOutput:
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
            self._update_stats_from_output(req_state, eco,
                                           engine_core_timestamp,
                                           iteration_stats)

            new_token_ids = eco.new_token_ids
            pooling_output = eco.pooling_output
            finish_reason = eco.finish_reason
            stop_reason = eco.stop_reason
            kv_transfer_params = eco.kv_transfer_params
            num_cached_tokens = eco.num_cached_tokens
            req_state.is_prefilling = False

            # 2) Detokenize and logprobs when text path
            assert req_state.detokenizer is not None
            assert req_state.logprobs_processor is not None
            stop_string = req_state.detokenizer.update(
                new_token_ids, finish_reason == FinishReason.STOP)
            if stop_string:
                finish_reason = FinishReason.STOP
                stop_reason = stop_string
            req_state.logprobs_processor.update_from_output(eco)

            # 2.5) Accumulate multimodal tensors in RequestState
            try:
                mm_type = (self.engine_core_output_type or "").lower()
                if pooling_output is not None and isinstance(req_state, OmniRequestState):
                    req_state.add_multimodal_tensor(pooling_output, mm_type)
            except Exception:
                logger.debug("Failed to accumulate multimodal tensor for request %s", req_id, exc_info=True)

            # 3) Create RequestOutput objects, forcing combined mode to keep ids
            pooling_for_make = pooling_output
            if pooling_output is not None and new_token_ids:
                # Do not consume pooling path now; keep ids and attach mm later
                pooling_for_make = None

            ro = req_state.make_request_output(new_token_ids, pooling_for_make,
                                               finish_reason, stop_reason,
                                               kv_transfer_params,
                                               num_cached_tokens)
            if ro:
                # Attach accumulated multimodal payload if any
                try:
                    if isinstance(req_state, OmniRequestState) and req_state.mm_accumulated is not None:
                        mm_key = req_state.mm_type or "latents"
                        if not hasattr(ro, "multimodal_output"):
                            setattr(ro, "multimodal_output", {})
                        ro.multimodal_output[mm_key] = req_state.mm_accumulated
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
                if not eco.finished:
                    reqs_to_abort.append(req_id)
                self._update_stats_from_finished(req_state, finish_reason,
                                                 iteration_stats)
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
        output_type = (self.engine_core_output_type or "").lower()

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
            tensor = self._extract_from_multimodal_outputs(
                eco, keys=("image", "images", "pixel_values", "pixels"))
            if tensor is not None:
                eco.pooling_output = tensor

    def _process_text_image_output(self, eco: EngineCoreOutput) -> None:
        """Allow text+image outputs. Text path stays as new_token_ids;
        image/latents route via pooling_output."""
        # Preserve text tokens as-is; ensure pooling_output carries image/latents
        if eco.pooling_output is None:
            tensor = self._extract_from_multimodal_outputs(
                eco, keys=("image", "images", "pixel_values", "pixels",
                           "latent", "latents", "z"))
            if tensor is not None:
                eco.pooling_output = tensor

    def _process_latents_output(self, eco: EngineCoreOutput) -> None:
        """Ensure latent tensors are surfaced via pooling_output."""
        if eco.pooling_output is None:
            tensor = self._extract_from_multimodal_outputs(
                eco, keys=("latent", "latents", "z", "posterior"))
            if tensor is not None:
                eco.pooling_output = tensor

    def _process_audio_output(self, eco: EngineCoreOutput) -> None:
        """Ensure audio tensors are surfaced via pooling_output."""
        if eco.pooling_output is None:
            tensor = self._extract_from_multimodal_outputs(
                eco, keys=("audio", "audios", "wav", "waveform",
                           "audio_pcm", "pcm"))
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

    def _extract_from_multimodal_outputs(
        self, eco: EngineCoreOutput, keys: tuple[str, ...]
    ) -> Optional[torch.Tensor]:
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