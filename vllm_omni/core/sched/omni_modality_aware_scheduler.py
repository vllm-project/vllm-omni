"""Modality-aware scheduler for multimodal LLM inference.

This module implements a scheduler that optimizes encoder utilization
by batching requests with compatible modality combinations together.
"""

from .omni_ar_base_scheduler import BaseOmniARScheduler
from vllm.v1.request import Request, RequestStatus
from vllm.v1.core.sched.request_queue import RequestQueue
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
import time
from vllm.logger import init_logger
from typing import List, Literal, TypedDict, Dict, Set
from .modality_aware_request_queue import (
    OmniSchedulingPolicy,
    MaskFilterPolicy,
    ModalityAwareRequestQueue,
    create_request_queue
)
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput
)
from vllm.v1.engine import (
    EngineCoreEventType,
    EngineCoreOutput,
    EngineCoreOutputs
)
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.structured_output import StructuredOutputManager

_ScheduleResult = Literal["scheduled", "skipped", "exhausted"]
logger = init_logger(__name__)


class SchedulingContext(TypedDict):
    """Context dictionary holding scheduling state for a single schedule() call.
    
    Attributes:
        scheduled_new_reqs: Newly scheduled requests from waiting queue.
        scheduled_resumed_reqs: Resumed requests (previously preempted).
        scheduled_running_reqs: Continuing requests from running queue.
        preempted_reqs: Requests preempted during this scheduling round.
        skipped_waiting_requests: Requests skipped due to async operations.
        token_budget: Remaining token budget for this step.
        encoder_compute_budget: Remaining encoder compute budget.
        req_to_new_blocks: Mapping of request IDs to allocated KV blocks.
        num_scheduled_tokens: Tokens scheduled per request this step.
        scheduled_encoder_inputs: Encoder inputs to process per request.
        scheduled_spec_decode_tokens: Speculative decode tokens per request.
        scheduled_loras: Set of LoRA adapter IDs in this batch.
        scheduled_timestamp: Monotonic timestamp of this scheduling step.
        hot_modality_mask: Bitmask of currently active encoder modalities.
    """
    scheduled_new_reqs: List[Request]
    scheduled_resumed_reqs: List[Request]
    scheduled_running_reqs: List[Request]
    preempted_reqs: List[Request]
    skipped_waiting_requests: RequestQueue
    token_budget: int
    encoder_compute_budget: int
    req_to_new_blocks: Dict[str, KVCacheBlocks]
    num_scheduled_tokens: Dict[str, int]
    scheduled_encoder_inputs: Dict[str, list[int]]
    scheduled_spec_decode_tokens: Dict[str, list[int]]
    scheduled_loras: Set[int]
    scheduled_timestamp: float
    hot_modality_mask: int


class OmniModalityAwareScheduler(BaseOmniARScheduler):
    """Scheduler with modality-aware batching for multimodal models.
    
    This scheduler optimizes encoder utilization by:
    1. Grouping requests with compatible modality combinations
    2. Dynamically activating encoders based on workload
    3. Preventing starvation of long-waiting requests
    
    The scheduling algorithm proceeds in 5 phases:
    1. Schedule running requests (maintain continuity)
    2. Starvation rescue (FCFS for old requests)
    3. Hot modality piggy-backing (fill encoder capacity)
    4. Cold encoder activation (based on workload threshold)
    5. Pure text request scheduling
    """
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the modality-aware scheduler.
        
        Args:
            vllm_config: Global vLLM configuration.
            kv_cache_config: KV cache configuration.
            structured_output_manager: Manager for structured output.
            mm_registry: Multimodal registry for modality info.
            include_finished_set: Whether to track finished requests.
            log_stats: Whether to log scheduling statistics.
        """
        super().__init__(
            vllm_config,
            kv_cache_config,
            structured_output_manager,
            block_size,
            mm_registry,
            include_finished_set,
            log_stats
        )
        # change the policy back to omni_modality_aware
        # it was previously set to fcfs by OmniEngineArgs to bypass the check in SchedulerConfig
        self.policy = OmniSchedulingPolicy.OMNI_MODALITY_AWARE
        # initialize the waiting queue as ModalityAwareRequestQueue
        self.waiting = create_request_queue(self.policy)

        # Initialize modality bitmask mapping
        self.modality_to_flag = {}
        self.all_modalities_mask = 0
        self.encoder_modalities_count = 0
        model_config = self.vllm_config.model_config
        
        if model_config.is_multimodal_model:
            # Get modality limits (e.g., {"audio": 1, "image": 1, "video": 1})
            mm_limits = mm_registry.get_mm_limits_per_prompt(model_config)

            # Sort modality names for consistent bitmask assignment across nodes
            sorted_modality_names = sorted(mm_limits.keys())

            bit_index = 0
            for raw_name in sorted_modality_names:
                mod_name_lower = raw_name.lower()
                # Embedding types (e.g., image_embeds) don't need encoder
                if "embed" in mod_name_lower:
                    self.modality_to_flag[mod_name_lower] = 0
                    logger.info(
                        f"Modality '{raw_name}' identified as EMBEDDING type "
                        "(No Encoder)."
                    )
                else:
                    # Assign bit position (1, 2, 4, 8, ...)
                    mask_val = 1 << bit_index
                    self.modality_to_flag[mod_name_lower] = mask_val
                    self.all_modalities_mask |= mask_val
                    bit_index += 1
                    logger.info(
                        f"Modality '{raw_name}' assigned mask bit: "
                        f"{mask_val:#05b}."
                    )
            self.encoder_modalities_count = bit_index
        else:
            logger.info("Model is not a multimodal model")

        # Bind modality config to waiting queue
        if isinstance(self.waiting, ModalityAwareRequestQueue):
            if self.all_modalities_mask > 0:
                self.waiting.bind_modality_config(self.all_modalities_mask)
        else:
            raise TypeError("Expected ModalityAwareRequestQueue for this policy.")
        
        # Scheduling parameters from additional config
        extra_config = vllm_config.additional_config or {}
        # Threshold for encoder activation (fraction of step budget)
        self.encoder_gain_threshold = extra_config.get("encoder_gain_threshold", 1024)
        # Starvation threshold in seconds
        self.starve_threshold = extra_config.get("starve_threshold", 0.1)


    @classmethod
    def validate_stage_config(
        cls,
        global_policy,
        stage_id,
        is_comprehension,
        is_dit
    ):
        """Validate that stage configuration is compatible with this scheduler.
        
        Args:
            global_policy: The global scheduling policy.
            stage_id: ID of the pipeline stage.
            is_comprehension: Whether this is the comprehension stage.
            is_dit: Whether this is a DiT stage.
        
        Raises:
            ValueError: If configuration is invalid for this scheduler.
        """
        if not is_comprehension:
            raise ValueError(
                f"Stage {stage_id} Configuration Error: {cls.__name__} "
                "(Modality-Aware) requires 'is_comprehension: true' to "
                "function. It should only be assigned to the primary "
                "understanding stage."
            )
        if global_policy == "priority" or global_policy == "fcfs":
            raise ValueError(
                f"Stage {stage_id} Configuration Error: {cls.__name__} "
                "(Modality-Aware) can only be used along with "
                "omni_modality_aware policy."
            )

    def _initialize_scheduling_context(self) -> SchedulingContext:
        """Initialize the scheduling context for a new scheduling round.
        
        Creates a fresh context dictionary with empty result lists,
        full budgets, and hot_modality_mask set to 0.
        
        Returns:
            Initialized SchedulingContext dictionary.
        """
        ctx: SchedulingContext = {
            'scheduled_new_reqs': [],
            'scheduled_resumed_reqs': [],
            'scheduled_running_reqs': [],
            'preempted_reqs': [],
            'skipped_waiting_requests': create_request_queue(
                self.policy, self.all_modalities_mask
            ),
            'token_budget': self.max_num_scheduled_tokens,
            'encoder_compute_budget': self.max_num_encoder_input_tokens,
            'req_to_new_blocks': {},
            'num_scheduled_tokens': {},
            'scheduled_encoder_inputs': {},
            'scheduled_spec_decode_tokens': {},
            'scheduled_loras': set(),
            'scheduled_timestamp': time.monotonic(),
            'hot_modality_mask': 0
        }
        return ctx

    def _compute_num_new_tokens(
        self,
        request: Request,
        token_budget: int,
    ) -> int:
        """Calculate the number of new tokens to process for a request.
        
        Considers:
        - Request's actual token needs
        - Long prefill threshold for chunking
        - Available token budget
        - Maximum model length
        
        Args:
            request: The request to compute tokens for.
            token_budget: Available token budget.
        
        Returns:
            Number of new tokens to schedule.
        """
        num_new_tokens = (request.num_tokens_with_spec +
                         request.num_output_placeholders -
                         request.num_computed_tokens)
        
        # Apply long prefill chunking threshold
        if (0 < self.scheduler_config.long_prefill_token_threshold <
                num_new_tokens):
            num_new_tokens = self.scheduler_config.long_prefill_token_threshold
        
        # Apply token budget limit
        num_new_tokens = min(num_new_tokens, token_budget)
        
        # Ensure we don't exceed max model length
        num_new_tokens = min(
            num_new_tokens,
            self.max_model_len - 1 - request.num_computed_tokens
        )
        
        return num_new_tokens

    def _preempt_running_request(self, ctx: SchedulingContext) -> Request:
        """Preempt the lowest priority running request to free resources.
        
        Removes the last request from running queue (FCFS order),
        frees its KV and encoder cache, and returns it to waiting queue.
        
        Args:
            ctx: Current scheduling context.
        
        Returns:
            The preempted request.
        """
        # Preempt last arrived request in FCFS
        preempted_req = self.running.pop()
        
        # Free resources
        self.kv_cache_manager.free(preempted_req)
        self.encoder_cache_manager.free(preempted_req)
        preempted_req.status = RequestStatus.PREEMPTED
        preempted_req.num_computed_tokens = 0
        preempted_req.num_preemptions += 1
        
        if self.log_stats:
            preempted_req.record_event(
                EngineCoreEventType.PREEMPTED, ctx['scheduled_timestamp']
            )
        
        # Return to front of waiting queue
        self.prepend_request(preempted_req, self.waiting)
        ctx['preempted_reqs'].append(preempted_req)
        
        return preempted_req

    def _process_spec_decode_tokens(
        self,
        request: Request,
        num_new_tokens: int,
        ctx: SchedulingContext,
    ) -> None:
        """Process speculative decoding tokens for a request.
        
        Trims spec_token_ids to match actually scheduled tokens and
        records them in the context.
        
        Args:
            request: Request with potential speculative tokens.
            num_new_tokens: Number of tokens being scheduled.
            ctx: Current scheduling context.
        """
        if not request.spec_token_ids:
            return
        
        num_scheduled_spec_tokens = (num_new_tokens +
                                     request.num_computed_tokens -
                                     request.num_tokens -
                                     request.num_output_placeholders)
        if num_scheduled_spec_tokens > 0:
            # Trim spec_token_ids to actual scheduled count
            del request.spec_token_ids[num_scheduled_spec_tokens:]
            ctx['scheduled_spec_decode_tokens'][request.request_id] = (
                request.spec_token_ids
            )
        # New spec tokens will be set in `update_draft_token_ids` before the
        # next step when applicable.
        request.spec_token_ids = []

    def _allocate_encoder_inputs(
        self,
        request: Request,
        encoder_inputs_to_schedule: list[int],
        ctx: SchedulingContext,
    ) -> None:
        """Allocate encoder cache for scheduled multimodal inputs.
        
        Updates the hot_modality_mask to include modalities of
        newly allocated encoder inputs.
        
        Args:
            request: Request with multimodal features.
            encoder_inputs_to_schedule: Indices of MM features to encode.
            ctx: Current scheduling context.
        """
        ctx['scheduled_encoder_inputs'][request.request_id] = (
            encoder_inputs_to_schedule
        )
        for i in encoder_inputs_to_schedule:
            self.encoder_cache_manager.allocate(request, i)
            ctx['hot_modality_mask'] |= self._get_modality_mask(
                request.mm_features[i].modality
            )

    def _collect_lora_info(self, ctx: SchedulingContext) -> None:
        """Collect LoRA adapter IDs from all scheduled running requests.
        
        Args:
            ctx: Scheduling context to update with LoRA info.
        """
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in ctx['scheduled_running_reqs']
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras
            ctx['scheduled_loras'].update(scheduled_loras)

    def _move_waiting_request_to_skipped(
        self,
        request: Request,
        ctx: SchedulingContext
    ) -> None:
        """Move a waiting request to the skipped queue.
        
        Used when a request cannot be scheduled due to async operations
        or resource constraints.
        
        Args:
            request: Request to skip.
            ctx: Current scheduling context.
        
        Raises:
            AttributeError: If request has no request_id.
        """
        request_id = getattr(request, 'request_id', None)
        if not request_id:
            raise AttributeError("current request has no request_id")
        self.waiting.pop_request_by_id(request_id)
        self.prepend_request(request, ctx['skipped_waiting_requests'])

    def _ensure_waiting_request_readiness(
        self,
        request: Request,
        ctx: SchedulingContext
    ) -> bool:
        """Check if a waiting request is ready for scheduling.
        
        Handles:
        - KV transfer waiting state
        - FSM compilation waiting state
        
        Args:
            request: Request to check.
            ctx: Current scheduling context.
        
        Returns:
            True if request can proceed, False if it should be skipped.
        """
        # Check KV transfer status
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            is_ready = self._update_waiting_for_remote_kv(request)
            if is_ready:
                request.status = RequestStatus.WAITING
            else:
                self._move_waiting_request_to_skipped(request, ctx)
                return False
        
        # Check FSM compilation status
        if request.status == RequestStatus.WAITING_FOR_FSM:
            structured_output_req = request.structured_output_request
            if structured_output_req and structured_output_req.grammar:
                request.status = RequestStatus.WAITING
            else:
                self._move_waiting_request_to_skipped(request, ctx)
                return False
        
        return True

    def _ensure_lora_limit_or_skip(
        self,
        request: Request,
        ctx: SchedulingContext
    ) -> bool:
        """Check if scheduling this request stays within LoRA limits.
        
        If the request would exceed max_loras, it is moved to skipped.
        
        Args:
            request: Request to check.
            ctx: Current scheduling context.
        
        Returns:
            True if within limits, False if request was skipped.
        """
        if not self.lora_config or not request.lora_request:
            return True
        
        scheduled_loras = ctx['scheduled_loras']
        exceed_flag = (
            len(scheduled_loras) == self.lora_config.max_loras and
            request.lora_request.lora_int_id not in scheduled_loras
        )
        
        if exceed_flag:
            self._move_waiting_request_to_skipped(request, ctx)
            return False
        
        return True

    def _process_waiting_request(
        self,
        request: Request,
        ctx: SchedulingContext
    ) -> _ScheduleResult:
        """Process a waiting request: check cache, allocate resources, update state.
        
        This is the core scheduling logic for a single waiting request.
        Handles cache hit detection, KV block allocation, async KV loading,
        and state transitions.
        
        Args:
            request: Request to process.
            ctx: Current scheduling context.
        
        Returns:
            "scheduled": Request successfully scheduled.
            "skipped": Request skipped (async not ready, etc.).
            "exhausted": Resources exhausted, stop scheduling new requests.
        """
        num_external_computed_tokens = 0
        load_kv_async = False

        # 1) Check cache hits: get locally and remotely cached token blocks
        if request.num_computed_tokens == 0:
            # Get locally cached tokens
            new_computed_blocks, num_new_local_computed_tokens = \
                self.kv_cache_manager.get_computed_blocks(request)

            # Query remote server for cache hits if using disaggregated arch
            if self.connector is not None:
                num_external_computed_tokens, load_kv_async = (
                    self.connector.get_num_new_matched_tokens(
                        request, num_new_local_computed_tokens
                    )
                )

                if num_external_computed_tokens is None:
                    # KVConnector cannot determine matched tokens
                    self._move_waiting_request_to_skipped(request, ctx)
                    return "skipped"

            # Total computed tokens (local + remote)
            num_computed_tokens = (num_new_local_computed_tokens +
                                   num_external_computed_tokens)

        # Handle resumed requests after async KV receive
        else:
            new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
            num_new_local_computed_tokens = 0
            num_computed_tokens = request.num_computed_tokens

        # 2) Calculate new tokens to schedule this step
        encoder_inputs_to_schedule = None
        external_load_encoder_input: list[int] = []
        new_encoder_compute_budget = ctx['encoder_compute_budget']

        # Handle async KV loading - don't allocate new compute tokens
        if load_kv_async:
            assert num_external_computed_tokens > 0
            num_new_tokens = 0
        else:
            # Use num_tokens instead of num_prompt_tokens to handle resumed requests
            num_new_tokens = request.num_tokens - num_computed_tokens
            if (0 < self.scheduler_config.long_prefill_token_threshold <
                    num_new_tokens):
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold

            # If chunked_prefill disabled and budget insufficient, skip
            if (not self.scheduler_config.enable_chunked_prefill and
                    num_new_tokens > ctx['token_budget']):
                self._move_waiting_request_to_skipped(request, ctx)
                return "skipped"

            num_new_tokens = min(num_new_tokens, ctx['token_budget'])
            assert num_new_tokens > 0

            # Schedule encoder inputs for multimodal requests
            if request.has_encoder_inputs:
                (encoder_inputs_to_schedule, num_new_tokens,
                 new_encoder_compute_budget,
                 external_load_encoder_input) = self._try_schedule_encoder_inputs(
                    request, num_computed_tokens, num_new_tokens,
                    ctx['encoder_compute_budget'],
                    shift_computed_tokens=1 if self.use_eagle else 0
                )
                if num_new_tokens == 0:
                    return "exhausted"

        # Handle edge case with spec decoding and P/D disaggregation
        effective_lookahead_tokens = (
            0 if request.num_computed_tokens == 0
            else self.num_lookahead_tokens
        )

        # Determine cross-attention cache blocks for encoder-decoder models
        if self.is_encoder_decoder and request.has_encoder_inputs:
            # For Whisper, input is always padded to max_len
            num_encoder_tokens = self.scheduler_config.max_num_encoder_input_tokens
        else:
            num_encoder_tokens = 0

        # 3) Allocate physical KV cache blocks
        new_blocks = self.kv_cache_manager.allocate_slots(
            request,
            num_new_tokens + num_external_computed_tokens,
            num_new_local_computed_tokens,
            new_computed_blocks,
            num_lookahead_tokens=effective_lookahead_tokens,
            delay_cache_blocks=load_kv_async,
            num_encoder_tokens=num_encoder_tokens,
        )
        
        # Resources exhausted - stop scheduling
        if new_blocks is None:
            return "exhausted"

        # Notify KVConnector of allocation for potential remote loading
        if self.connector is not None:
            self.connector.update_state_after_alloc(
                request,
                new_computed_blocks + new_blocks,
                num_external_computed_tokens,
            )

        # 4) Handle async KV loading - transition to WAITING_FOR_REMOTE state
        if load_kv_async:
            self._move_waiting_request_to_skipped(request, ctx)
            request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
            return "skipped"

        # 5) Add to running queue and update status
        request_id = getattr(request, 'request_id', None)
        if not request_id:
            raise AttributeError("current request has no request_id")
        self.waiting.pop_request_by_id(request_id)

        self._update_connector_prefix_cache_stats(request)

        self.running.append(request)
        
        if self.log_stats:
            request.record_event(
                EngineCoreEventType.SCHEDULED, ctx['scheduled_timestamp']
            )

        if request.status == RequestStatus.WAITING:
            ctx['scheduled_new_reqs'].append(request)
        elif request.status == RequestStatus.PREEMPTED:
            ctx['scheduled_resumed_reqs'].append(request)
        else:
            raise RuntimeError(f"Invalid request status: {request.status}")

        # Record LoRA, KV blocks, and token counts
        if self.lora_config and request.lora_request:
            ctx['scheduled_loras'].add(request.lora_request.lora_int_id)
        ctx['req_to_new_blocks'][request.request_id] = (
            self.kv_cache_manager.get_blocks(request.request_id)
        )
        ctx['num_scheduled_tokens'][request.request_id] = num_new_tokens
        ctx['token_budget'] -= num_new_tokens
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = num_computed_tokens

        # Update prefix cache hit token count
        if request.num_cached_tokens < 0:
            request.num_cached_tokens = num_computed_tokens

        # Allocate encoder cache for multimodal inputs
        if encoder_inputs_to_schedule:
            self._allocate_encoder_inputs(
                request, encoder_inputs_to_schedule, ctx
            )
            ctx['encoder_compute_budget'] = new_encoder_compute_budget
        # Allocate for external load encoder cache (EC Connector)
        if external_load_encoder_input:
            for i in external_load_encoder_input:
                self.encoder_cache_manager.allocate(request, i)
                if self.ec_connector is not None:
                    self.ec_connector.update_state_after_alloc(request, i)

        return "scheduled"

    def _try_dispatch_single_waiting_request(
        self,
        request: Request,
        ctx: SchedulingContext,
    ) -> _ScheduleResult:
        """Attempt to schedule a single waiting request.
        
        Wraps readiness checks, LoRA limits, and resource allocation.
        
        Args:
            request: Request to attempt scheduling.
            ctx: Current scheduling context.
        
        Returns:
            Schedule result indicating outcome.
        """
        # Check readiness (async operations, FSM compilation)
        if not self._ensure_waiting_request_readiness(request, ctx):
            return "skipped"
        
        # Check LoRA limits
        if not self._ensure_lora_limit_or_skip(request, ctx):
            return "skipped"
        
        # Core scheduling logic
        return self._process_waiting_request(request, ctx)

    def _build_scheduler_output(self, ctx: SchedulingContext) -> SchedulerOutput:
        """Build the final SchedulerOutput from scheduling context.
        
        Performs integrity checks and constructs the output object
        with all scheduled request data.
        
        Args:
            ctx: Completed scheduling context.
        
        Returns:
            SchedulerOutput ready for model execution.
        """
        # Integrity checks
        total_num_scheduled_tokens = sum(ctx['num_scheduled_tokens'].values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert ctx['token_budget'] >= 0
        assert len(self.running) <= self.max_num_running_reqs
        assert (len(ctx['scheduled_new_reqs']) +
                len(ctx['scheduled_resumed_reqs']) +
                len(ctx['scheduled_running_reqs']) <= len(self.running))
        
        # Get common prefix block counts
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request.request_id
                )
            )
        
        # Build new request data
        # For V2 model runner, merge resumed requests into new requests
        if self.use_v2_model_runner:
            scheduled_new_reqs = (ctx['scheduled_new_reqs'] +
                                  ctx['scheduled_resumed_reqs'])
            ctx['scheduled_resumed_reqs'] = []
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    ctx['req_to_new_blocks'][req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req, ctx['req_to_new_blocks'][req.request_id].get_block_ids()
                )
                for req in ctx['scheduled_new_reqs']
            ]
        
        # Build cached request data
        cached_reqs_data = self._make_cached_request_data(
            ctx['scheduled_running_reqs'],
            ctx['scheduled_resumed_reqs'],
            ctx['num_scheduled_tokens'],
            ctx['scheduled_spec_decode_tokens'],
            ctx['req_to_new_blocks'],
        )
        
        # Record the request ids that were scheduled in this step.
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(ctx['num_scheduled_tokens'].keys())
        
        # Build SchedulerOutput
        total_num_scheduled_tokens = sum(ctx['num_scheduled_tokens'].values())
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=ctx['num_scheduled_tokens'],
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=ctx['scheduled_spec_decode_tokens'],
            scheduled_encoder_inputs=ctx['scheduled_encoder_inputs'],
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids={req.request_id for req in ctx['preempted_reqs']},
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
        )
        
        return scheduler_output

    def _post_schedule_processing(
        self,
        scheduler_output: SchedulerOutput
    ) -> None:
        """Perform post-scheduling tasks.
        
        Handles KV connector metadata, EC connector metadata, and state updates.
        Note: KV event publishing is now done in update_from_output().
        
        Args:
            scheduler_output: The scheduler output to finalize.
        """
        # Build KV Connector metadata
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # Build EC Connector metadata
        if self.ec_connector is not None:
            ec_meta = self.ec_connector.build_connector_meta(scheduler_output)
            scheduler_output.ec_connector_metadata = ec_meta

        # Update request computed tokens for model runner
        self._update_after_schedule(scheduler_output)

    def _get_modality_mask(self, modality_str: str) -> int:
        """Map a modality name to its bitmask value.
        
        Supports case-insensitive matching and handles unknown modalities.
        
        Args:
            modality_str: Name of the modality (e.g., "image", "audio").
        
        Returns:
            Bitmask value for the modality, or 0 if not found/embedding type.
        """
        if not modality_str:
            return 0
        
        mod_key = modality_str.lower()
        
        # Check registered modality mapping
        if mod_key in self.modality_to_flag:
            return self.modality_to_flag[mod_key]
        
        # Fallback for unregistered modalities
        if "embed" in mod_key:
            return 0

        return 0

    def get_cold_encoder_masks(self, cold_mm_mask: int) -> List[int]:
        """Extract individual encoder masks from a combined cold mask.
        
        For example, mm_mask=0b110 produces [0b010, 0b100].
        
        Args:
            cold_mm_mask: Combined bitmask of cold (inactive) encoders.
        
        Returns:
            List of individual encoder bitmasks.
        """
        cold_encoder_masks = []
        m = cold_mm_mask
        while m:
            lsb = m & -m  # Extract least significant bit
            cold_encoder_masks.append(lsb)
            m -= lsb
        return cold_encoder_masks

    def add_request(self, request: Request) -> None:
        """Add a new request to the waiting queue.
        
        Enriches the request with modality metadata before adding.
        
        Args:
            request: The request to add.
        """
        request = self._request_data_enrichment(request)
        self.waiting.add_request(request)
        self.requests[request.request_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

    def prepend_request(
        self,
        request: Request,
        request_queue: RequestQueue
    ) -> None:
        """Add a request to the front of a queue.
        
        Args:
            request: The request to prepend.
            request_queue: Target queue (waiting or skipped).
        """
        request = self._request_data_enrichment(request)
        request_queue.prepend_request(request)

    def _get_waiting_duration(
        self,
        request: Request,
        ctx: SchedulingContext
    ) -> float:
        """Calculate how long a request has been waiting.
        
        Args:
            request: Request to check.
            ctx: Current scheduling context with timestamp.
        
        Returns:
            Wait duration in seconds.
        
        Raises:
            AttributeError: If request lacks arrival_time_mono.
        """
        if hasattr(request, "arrival_time_mono"):
            return ctx['scheduled_timestamp'] - request.arrival_time_mono
        else:
            raise AttributeError("request has no arrival_time_mono")

    def _request_data_enrichment(self, request: Request):
        """Enrich request with modality metadata for scheduling.
        
        Calculates the modality bitmask and token counts for multimodal
        features that still need encoding.
        
        Args:
            request: Request to enrich.
        
        Returns:
            Enriched request with mm_mask_to_prefill and mm_tokens_to_prefill.
        """
        # Only inject arrival time for truly new requests
        if not hasattr(request, "arrival_time_mono"):
            request.arrival_time_mono = time.monotonic()
        
        # Calculate modality mask and token stats
        combined_mask = 0
        all_prefilled_mm_tokens = 0
        
        for mm_feature in request.mm_features:
            mod_name_lower = mm_feature.modality.lower()
            
            # Skip embedding types (don't need encoding)
            if "embed" in mod_name_lower:
                continue
            
            # Check if already in encoder cache (e.g., after preemption)
            mm_hash = mm_feature.identifier
            is_in_encoder_cache = mm_hash in self.encoder_cache_manager.cached
            if is_in_encoder_cache:
                continue
            
            # Accumulate tokens and mask for features needing encoding
            all_prefilled_mm_tokens += mm_feature.mm_position.length
            combined_mask |= self._get_modality_mask(mod_name_lower)
        
        request.mm_mask_to_prefill = combined_mask
        request.mm_tokens_to_prefill = all_prefilled_mm_tokens
        return request

    def schedule(self) -> SchedulerOutput:
        """Main scheduling method implementing modality-aware batching.
        
        Scheduling proceeds in 5 phases:
        1. Schedule running requests (maintain continuity, may preempt)
        2. Starvation rescue (FCFS for requests exceeding wait threshold)
        3. Hot modality piggy-backing (fill active encoder capacity)
        4. Cold encoder activation (based on workload threshold)
        5. Pure text request scheduling
        
        Returns:
            SchedulerOutput containing all scheduling decisions.
        """
        # Initialize scheduling context
        ctx = self._initialize_scheduling_context()
        total_step_token_budget = ctx['token_budget']
        total_step_encoder_budget = ctx['encoder_compute_budget']


        # Phase 1: Schedule running requests (maintain FCFS continuity)
        req_index = 0
        can_schedule_mm_request = True
        can_schedule_any_request = True

        while req_index < len(self.running) and ctx['token_budget'] > 0:
            request = self.running[req_index]

            # Async scheduling: Avoid scheduling an extra step when we are sure
            # that the previous step has reached request.max_tokens.
            if (
                request.num_output_placeholders > 0
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                req_index += 1
                continue
            
            # Calculate tokens for this step
            num_new_tokens = self._compute_num_new_tokens(
                request, ctx['token_budget']
            )
            
            # Schedule encoder inputs for multimodal requests
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            new_encoder_compute_budget = ctx['encoder_compute_budget']
            if request.has_encoder_inputs:
                (encoder_inputs_to_schedule, num_new_tokens,
                 new_encoder_compute_budget,
                 external_load_encoder_input) = self._try_schedule_encoder_inputs(
                    request, request.num_computed_tokens, num_new_tokens,
                    ctx['encoder_compute_budget'],
                    shift_computed_tokens=1 if self.use_eagle else 0
                )
            
            # Skip if no tokens can be scheduled (resource constraints)
            if num_new_tokens == 0:
                req_index += 1
                continue

            # Allocate KV cache, preempting if necessary
            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens
                )
                if new_blocks is None:
                    preempted_req = self._preempt_running_request(ctx)
                    if preempted_req == request:
                        can_schedule_any_request = False
                        can_schedule_mm_request = False
                        break
                else:
                    break
            
            if not can_schedule_any_request:
                break
            assert new_blocks is not None

            # Record scheduling decision
            ctx['scheduled_running_reqs'].append(request)
            ctx['req_to_new_blocks'][request.request_id] = new_blocks
            ctx['num_scheduled_tokens'][request.request_id] = num_new_tokens
            ctx['token_budget'] -= num_new_tokens
            req_index += 1

            # Handle speculative decoding
            self._process_spec_decode_tokens(request, num_new_tokens, ctx)
            
            # Allocate encoder cache
            if encoder_inputs_to_schedule:
                self._allocate_encoder_inputs(
                    request, encoder_inputs_to_schedule, ctx
                )
                ctx['encoder_compute_budget'] = new_encoder_compute_budget
            # Allocate for external load encoder cache (EC Connector)
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)

        # Collect LoRA info from running requests
        self._collect_lora_info(ctx)

        # Phase 2: Starvation rescue - schedule old waiting requests
        if not ctx['preempted_reqs']:
            while self.waiting and ctx['token_budget'] > 0:
                if len(self.running) == self.max_num_running_reqs:
                    can_schedule_mm_request = False
                    can_schedule_any_request = False
                    break
                
                request = self.waiting.peek_request()
                if self._get_waiting_duration(request, ctx) < self.starve_threshold:
                    break
                
                schedule_result = self._try_dispatch_single_waiting_request(
                    request, ctx
                )

                if schedule_result == "skipped":
                    continue
                elif schedule_result == "exhausted":
                    can_schedule_mm_request = False
                    can_schedule_any_request = False
                    break
                elif schedule_result == "scheduled":
                    req_index += 1

        
        
        # Phase 3: Hot modality piggy-backing
        if can_schedule_mm_request and ctx['hot_modality_mask'] != 0:
            while self.waiting.compatible_buckets_not_empty(
                ctx['hot_modality_mask'],
                filter_policy=MaskFilterPolicy.COMPATIBLE_MULTI_MODAL
            ):
                if (ctx['token_budget'] <= 0 or
                        len(self.running) == self.max_num_running_reqs):
                    can_schedule_mm_request = False
                    can_schedule_any_request = False
                    break
                if ctx['encoder_compute_budget'] <= 0:
                    can_schedule_mm_request = False
                    break

                request = self.waiting.peek_request_by_mm_mask(
                    ctx['hot_modality_mask'],
                    filter_policy=MaskFilterPolicy.COMPATIBLE_MULTI_MODAL
                )
                schedule_result = self._try_dispatch_single_waiting_request(
                    request, ctx
                )

                if schedule_result == "skipped":
                    continue
                elif schedule_result == "exhausted":
                    can_schedule_mm_request = False
                    can_schedule_any_request = False
                    break
                elif schedule_result == "scheduled":
                    req_index += 1
        
        

        # Phase 4: Cold encoder activation (one at a time)
        while (can_schedule_mm_request and
               ctx['hot_modality_mask'] < self.all_modalities_mask):
            # Find inactive encoders
            inactive_mm_mask = ctx['hot_modality_mask'] ^ self.all_modalities_mask
            cold_encoder_masks = self.get_cold_encoder_masks(inactive_mm_mask)
            
            longest_waiting_time = float('inf')
            longest_waiting_encoder_combo_mask = 0
            
            # Evaluate potential gain for each cold encoder
            for cold_encoder_mask in cold_encoder_masks:
                encoders_combo_mask = cold_encoder_mask | ctx['hot_modality_mask']
                
                # Calculate encoder budget utilization
                encoder_mm_tokens = self.waiting.get_compatible_mm_tokens(
                    encoders_combo_mask,
                    filter_policy=MaskFilterPolicy.COMPATIBLE_MULTI_MODAL
                )
                encoder_gain = min(
                    encoder_mm_tokens,
                    ctx['encoder_compute_budget'],
                    ctx['token_budget']
                )
                
                # Check if gain exceeds activation threshold
                threshold = self.encoder_gain_threshold 
                if encoder_gain >= threshold:
                    # Select encoder combo with oldest waiting request
                    oldest_req = self.waiting.peek_request_by_mm_mask(
                        encoders_combo_mask,
                        filter_policy=MaskFilterPolicy.COMPATIBLE_MULTI_MODAL
                    )
                    wait_time = self._get_waiting_duration(oldest_req, ctx)
                    if wait_time < longest_waiting_time:
                        longest_waiting_encoder_combo_mask = encoders_combo_mask

            # No encoder combo meets threshold - switch to pure text
            if longest_waiting_encoder_combo_mask == 0:
                can_schedule_mm_request = False
                break

            # Activate the selected encoder combo
            ctx['hot_modality_mask'] |= longest_waiting_encoder_combo_mask

            # Schedule requests for newly activated encoder
            while self.waiting.compatible_buckets_not_empty(
                longest_waiting_encoder_combo_mask,
                filter_policy=MaskFilterPolicy.COMPATIBLE_MULTI_MODAL
            ):
                if (ctx['token_budget'] <= 0 or
                        len(self.running) == self.max_num_running_reqs):
                    can_schedule_mm_request = False
                    can_schedule_any_request = False
                    break
                if ctx['encoder_compute_budget'] <= 0:
                    can_schedule_mm_request = False
                    break
                
                request = self.waiting.peek_request_by_mm_mask(
                    longest_waiting_encoder_combo_mask,
                    filter_policy=MaskFilterPolicy.COMPATIBLE_MULTI_MODAL
                )

                schedule_result = self._try_dispatch_single_waiting_request(
                    request, ctx
                )
                if schedule_result == "skipped":
                    continue
                elif schedule_result == "exhausted":
                    can_schedule_mm_request = False
                    can_schedule_any_request = False
                    break
                elif schedule_result == "scheduled":
                    req_index += 1

       
        # Phase 5: Schedule pure text requests (mask=0)
        if can_schedule_any_request:
            while self.waiting.compatible_buckets_not_empty(
                0, filter_policy=MaskFilterPolicy.EXACT
            ):
                if (ctx['token_budget'] <= 0 or
                        len(self.running) == self.max_num_running_reqs):
                    break
                
                request = self.waiting.peek_request_by_mm_mask(
                    0, filter_policy=MaskFilterPolicy.EXACT
                )
                schedule_result = self._try_dispatch_single_waiting_request(
                    request, ctx
                )

                if schedule_result == "skipped":
                    continue
                elif schedule_result == "exhausted":
                    can_schedule_mm_request = False
                    can_schedule_any_request = False
                    break
                elif schedule_result == "scheduled":
                    req_index += 1

        # Return skipped requests to waiting queue
        if ctx['skipped_waiting_requests']:
            self.waiting.prepend_requests(ctx['skipped_waiting_requests'])

        # Build final scheduler output
        scheduler_output = self._build_scheduler_output(ctx)

        # Post-processing: KV connector, events, state updates
        self._post_schedule_processing(scheduler_output)

        # Enrich output with request-level payloads
        scheduler_output = self._enrich_scheduler_output(scheduler_output)
        return scheduler_output
