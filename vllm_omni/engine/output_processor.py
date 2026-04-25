from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.outputs import PoolingRequestOutput
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
from vllm.v1.metrics.stats import IterationStats

from vllm_omni.data_entry_keys import unflatten_payload
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
        # Omni-specific: multimodal output accumulation
        self.mm_type: str | None = None
        self.mm_accumulated: dict[str, Any] | None = None

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
                # Keep payload flat (dotted keys like "hidden_states.layer_0")
                # during accumulation so that all values are tensors/scalars and
                # the merge logic below works correctly.  Unflatten happens
                # later in _consolidate_multimodal_tensors after concatenation.

                incoming: dict[str, Any] = {}
                target_key = self.mm_type or "hidden"

                for k, v in payload.items():
                    if k == "model_outputs":
                        k = target_key
                    elif k == "hidden" and target_key != "hidden":
                        k = target_key

                    incoming[k] = _to_cpu(v)
            else:
                key = self.mm_type or "hidden"
                incoming = {key: _to_cpu(payload)}

            if self.mm_accumulated is None:
                self.mm_accumulated = incoming
            else:
                for k, v in incoming.items():
                    if k not in self.mm_accumulated:
                        self.mm_accumulated[k] = v
                    else:
                        existing = self.mm_accumulated[k]
                        if isinstance(v, torch.Tensor) and isinstance(existing, torch.Tensor):
                            self.mm_accumulated[k] = [existing, v]
                        elif isinstance(v, torch.Tensor) and isinstance(existing, list):
                            existing.append(v)
                        elif isinstance(v, dict) and isinstance(existing, dict):
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
                        if k == "audio":
                            # When the audio tensor shape is inconsistent, torch.cat will fail.
                            # We need to use torch.cat in -1 dimension.
                            continue
                        elif k == "sr":
                            # Sample rate is a constant scalar, keep last value.
                            self.mm_accumulated[k] = v[-1]
                        else:
                            self.mm_accumulated[k] = torch.cat(v, dim=0)
                    except Exception:
                        # Keep last tensor on failure
                        logger.warning(f"Error concatenating tensor for key {k}; keeping last tensor")
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

        # Restore nested structure from flat dotted keys now that all tensor
        # lists have been concatenated into single tensors.
        try:
            self.mm_accumulated = unflatten_payload(self.mm_accumulated)
        except Exception:
            logger.exception("Error unflattening consolidated multimodal tensors")

    # Override: do not route to pooling-only path; always create completion
    # outputs, and attach pooling_result into the CompletionOutput.
    def make_request_output(
        self,
        new_token_ids: list[int],
        pooling_output: torch.Tensor | None,
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
        kv_transfer_params: dict[str, Any] | None = None,
        routed_experts: np.ndarray | None = None,
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
                routed_experts,
            )

        finished = finish_reason is not None
        final_only = self.output_kind == RequestOutputKind.FINAL_ONLY

        if not finished and final_only:
            return None

        # Consolidate accumulated tensors when finishing.
        if finished:
            self._consolidate_multimodal_tensors()

        if self.stream_interval > 1:
            assert self.detokenizer is not None

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
        output = self._new_completion_output(new_token_ids, finish_reason, stop_reason, routed_experts)

        if self.parent_req is None:
            outputs = [output]
        else:
            outputs, finished = self.parent_req.get_outputs(self.request_id, output)
            if not outputs:
                return None
            external_req_id = self.parent_req.external_req_id

        return self._new_request_output(external_req_id, outputs, finished, kv_transfer_params)

    def _new_completion_output(
        self,
        token_ids: list[int],
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
        routed_experts: np.ndarray | None = None,
    ) -> Any:
        # Reuse base text/logprobs logic, then annotate with pooling_result.
        base_output = super()._new_completion_output(token_ids, finish_reason, stop_reason, routed_experts)
        try:
            if not hasattr(base_output, "multimodal_output"):
                setattr(base_output, "multimodal_output", {})
            if self.mm_accumulated is not None:
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
    """Handles multimodal output processing by capturing pooling_output
    from EngineCoreOutput and accumulating it as multimodal tensors,
    before delegating to the base vLLM OutputProcessor for text handling.

    The actual data flow is:
    1. For each EngineCoreOutput with pooling_output and a detokenizer:
       - Capture pooling_output into OmniRequestState.add_multimodal_tensor()
       - Clear eco.pooling_output to force text path in base processor
    2. Base vLLM OutputProcessor handles text detokenization
    3. On finish, _consolidate_multimodal_tensors() concatenates accumulated tensors
    4. _new_completion_output() attaches mm_accumulated to CompletionOutput
    """

    def __init__(
        self,
        tokenizer: TokenizerLike | None,
        *,
        log_stats: bool,
        stream_interval: int = 1,
        tracing_enabled: bool = False,
        engine_core_output_type: str | None = None,
    ):
        """Initialize the multimodal output processor.

        Args:
            tokenizer: Tokenizer for detokenizing text outputs
            log_stats: Whether to log statistics
            stream_interval: Stream interval for output generation
            engine_core_output_type: Optional output type specification
                (e.g., "image", "audio", "latent"). Used to tag multimodal
                outputs with the correct modality key.
        """
        super().__init__(
            tokenizer=tokenizer,
            log_stats=log_stats,
            stream_interval=stream_interval,
            tracing_enabled=tracing_enabled,
        )
        self.engine_core_output_type = engine_core_output_type

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

    def process_outputs(
        self,
        engine_core_outputs: list[EngineCoreOutput],
        engine_core_timestamp: float | None = None,
        iteration_stats: IterationStats | None = None,
    ) -> OutputProcessorOutput:
        for eco in engine_core_outputs:
            req_state = self.request_states.get(eco.request_id)
            if req_state is None or not isinstance(req_state, OmniRequestState):
                continue
            if eco.pooling_output is not None and req_state.detokenizer is not None:
                mm_type = (getattr(eco, "output_type", self.engine_core_output_type) or "").lower()
                req_state.add_multimodal_tensor(eco.pooling_output, mm_type)
                # Force text path in base processor for multimodal outputs.
                eco.pooling_output = None

        return super().process_outputs(
            engine_core_outputs,
            engine_core_timestamp=engine_core_timestamp,
            iteration_stats=iteration_stats,
        )
