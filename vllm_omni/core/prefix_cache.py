"""
Utilities for Prefix Caching in Omni models.
"""

import torch
from vllm.logger import init_logger
from vllm.v1.worker.gpu_input_batch import InputBatch

from vllm_omni.utils.mm_outputs import build_mm_cpu, to_payload_element

logger = init_logger(__name__)


class OmniTensorPrefixCache:
    """Prefix cache for hidden states (model outputs) and model specific
    multimodal outputs.

    This class implements prefix caching in a non-invasive way on top of
    vLLM by leveraging the same slot mappings that the vLLM scheduler uses
    for the KV Cache.

    Conceptually, this means we are mapping vLLM's cache mapping:
                        (num_blocks, block_size)

    to 3D tensors of shape:
                   (num_blocks, block_size, feature_size)

    Note that feature_size may vary across multimodal_outputs.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        hidden_size: int,
        hs_dtype: torch.dtype,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.default_hidden_size = hidden_size

        # Initialize the hidden states cache immediately
        self.hidden_states_cache = self._get_cache_tensor(dtype=hs_dtype)

        # Defer initialization of the mm_outputs_cache until we
        # actually see mm output tensors dependent on num tokens.
        self.mm_outputs_cache = {}
        self.mm_cache_keys = set()
        self._new_req_cache_hit_ids: set[str] = set()
        self._deferred_mm_outputs: dict[str, dict[str, list[tuple[int, torch.Tensor]]]] = {}

    def maybe_init_missing_mm_cache_keys(self, multimodal_outputs: dict, seq_len: int):
        """Given multimodal outputs from executing the model, dynamically
        determine which multimodal outputs are tensors depending on sequence
        length and should be cached, and initialize the cache tensors
        accordingly.

        NOTE: This is done to avoid the need for explicit specification of
        cache keys for every model/stage and aligns with the current way
        that we slice the multimodal outputs based on the first dimension.

        This will usually be called by the first forward pass, i.e.,
        determined by the warmup.
        """
        for key, val in multimodal_outputs.items():
            # Only cache per-token feature tensors: 2D+ with first dim == seq_len.
            # A 1D tensor of shape (seq_len,) is a broadcast scalar (per-request
            # metadata such as ref_code_len / codec_streaming), not per-token data;
            # caching it by slot causes a shape mismatch when a later request has a
            # different scheduled seq length.
            if (
                isinstance(val, torch.Tensor)
                and val.ndim >= 2
                and val.shape[0] == seq_len
                and key not in self.mm_cache_keys
            ):
                feat_dim = val.shape[-1]
                self.mm_outputs_cache[key] = self._get_cache_tensor(
                    dtype=val.dtype,
                    hidden_size=feat_dim,
                )
                self.mm_cache_keys.add(key)
                new_tensor_shape = self.mm_outputs_cache[key].shape
                logger.info("Initializing multimodal output cache of size %s for key: %s", list(new_tensor_shape), key)

    def _get_cache_tensor(self, dtype: torch.dtype, hidden_size: int | None = None) -> torch.Tensor:
        """Allocate a CPU cache tensor for a specific key."""
        actual_hidden_size = hidden_size if hidden_size is not None else self.default_hidden_size
        return torch.zeros(
            (self.num_blocks, self.block_size, actual_hidden_size),
            dtype=dtype,
            device="cpu",
        )

    def add_prefix_cached_new_req_id(self, req_id: str):
        """Adds a new request ID to the set of prefix cache hits on the batch."""
        self._new_req_cache_hit_ids.add(req_id)

    def reset_prefix_cached_new_req_ids(self):
        """Clears the cache hit IDs to prepare for a new engine step."""
        self._new_req_cache_hit_ids.clear()

    def has_prefix_cached_new_req_ids(self) -> bool:
        """Return True when this step contains a newly scheduled prefix hit."""
        return bool(self._new_req_cache_hit_ids)

    @staticmethod
    def _coerce_to_cpu_tensor(maybe_gpu_tensor: torch.Tensor) -> torch.Tensor:
        """Convert GPU tensors -> contiguous CPU tensors if needed."""
        return maybe_gpu_tensor.detach().cpu().contiguous()

    @staticmethod
    def _resolve_hidden_states_cpu(
        hidden_states: torch.Tensor,
        num_tokens: int,
        hidden_states_cpu: torch.Tensor | None,
    ) -> torch.Tensor:
        if hidden_states_cpu is None:
            return OmniTensorPrefixCache._coerce_to_cpu_tensor(hidden_states[:num_tokens])
        if hidden_states_cpu.device.type != "cpu":
            raise RuntimeError("hidden_states_cpu must be a CPU tensor.")
        if not hidden_states_cpu.is_contiguous():
            raise RuntimeError("hidden_states_cpu must be contiguous.")
        if hidden_states_cpu.dtype != hidden_states.dtype:
            raise RuntimeError(
                "hidden_states_cpu has an incompatible dtype: "
                f"got {hidden_states_cpu.dtype}, expected {hidden_states.dtype}."
            )
        if hidden_states_cpu.shape[1:] != hidden_states.shape[1:]:
            raise RuntimeError(
                "hidden_states_cpu has an incompatible feature shape: "
                f"got {tuple(hidden_states_cpu.shape[1:])}, expected {tuple(hidden_states.shape[1:])}."
            )
        if hidden_states_cpu.shape[0] < num_tokens:
            raise RuntimeError(
                "hidden_states_cpu does not cover the requested hidden states "
                f"slice: got {hidden_states_cpu.shape[0]} tokens, need {num_tokens}."
            )
        return hidden_states_cpu[:num_tokens]

    def update_omni_tensor_prefix_cache(
        self,
        hidden_states: torch.Tensor | None,
        multimodal_outputs: dict[str, torch.Tensor] | None,
        num_tokens_unpadded: int,
        slot_mapping: torch.Tensor,
        num_tokens_padded: int | None = None,
        skip_mm_cache_keys: set[str] | None = None,
        hidden_states_cpu: torch.Tensor | None = None,
    ):
        """Updates the hidden cache state for the provided hidden states and multimodal outputs.

        Args:
            hidden_states: Hidden states tensor to cache (if any)
            multimodal_outputs: Multimodal dict whose tensors may be cached
            num_tokens_unpadded: Number of tokens without padding
            slot_mapping: Slot mapping for the input sequence
            num_tokens_padded: Total number of tokens including padding
            skip_mm_cache_keys: Multimodal keys whose CPU cache writes are deferred
            hidden_states_cpu: Optional pre-staged CPU view of hidden_states.
                When provided, it must be contiguous, live on CPU, match the
                feature shape of hidden_states, and cover num_tokens_unpadded.
        """
        unpadded_slot_mapping = slot_mapping[:num_tokens_unpadded]
        if num_tokens_padded is None:
            num_tokens_padded = num_tokens_unpadded
        skip_mm_cache_keys = skip_mm_cache_keys or set()

        if hidden_states is not None:
            # Slice to unpadded portion before caching
            hidden_states = OmniTensorPrefixCache._resolve_hidden_states_cpu(
                hidden_states,
                num_tokens_unpadded,
                hidden_states_cpu,
            )
            # View the cache as 2D so that we can treat our slots as row indices
            flat_cache = self.hidden_states_cache.view(-1, self.hidden_states_cache.shape[-1])
            flat_cache[unpadded_slot_mapping] = hidden_states
            logger.debug("Writing to hidden states for %s tokens", num_tokens_unpadded)

        # Do the same for the stage's cached multimodal outputs
        if multimodal_outputs is not None:
            # If we haven't initialized the keys already, do it now
            # We check against the padded token count since we haven't sliced yet
            self.maybe_init_missing_mm_cache_keys(
                multimodal_outputs,
                seq_len=num_tokens_padded,
            )

            for mm_out_key, mm_cache in self.mm_outputs_cache.items():
                if mm_out_key in multimodal_outputs:
                    if mm_out_key in skip_mm_cache_keys:
                        continue
                    # Slice to unpadded portion before caching
                    mm_state = multimodal_outputs[mm_out_key][:num_tokens_unpadded]
                    mm_state = OmniTensorPrefixCache._coerce_to_cpu_tensor(mm_state)
                    flat_cache = mm_cache.view(-1, mm_cache.shape[-1])
                    flat_cache[unpadded_slot_mapping] = mm_state
            logger.debug("Writing to mm output cache for %s tokens", num_tokens_unpadded)

    def stage_deferred_mm_outputs(
        self,
        query_start_loc: torch.Tensor,
        input_batch: InputBatch,
        multimodal_outputs: dict[str, torch.Tensor] | None,
        num_scheduled_tokens: dict[str, int],
        deferred_mm_cache_keys: set[str],
    ) -> None:
        """Keep GPU multimodal chunks until a request finishes.

        The normal prefix-cache write path copies every cached multimodal tensor
        to CPU on every step.  For model outputs that are only needed by future
        full-block prefix hits, we can keep detached GPU chunks and materialize
        the CPU cache once when the request completes.
        """
        if not multimodal_outputs or not deferred_mm_cache_keys:
            return

        for mm_key in deferred_mm_cache_keys:
            mm_state = multimodal_outputs.get(mm_key)
            if not isinstance(mm_state, torch.Tensor) or mm_state.ndim < 2:
                continue
            for req_id in input_batch.req_ids:
                req_idx = input_batch.req_id_to_index[req_id]
                sched = int(num_scheduled_tokens.get(req_id, 0))
                if sched <= 0:
                    continue
                start = int(query_start_loc[req_idx])
                end = start + sched
                if start >= int(mm_state.shape[0]):
                    continue
                end = min(end, int(mm_state.shape[0]))
                if end <= start:
                    continue

                computed_end = int(input_batch.num_computed_tokens_cpu[req_idx])
                token_start = max(0, computed_end - sched)
                chunk = mm_state[start:end].detach()
                self._deferred_mm_outputs.setdefault(req_id, {}).setdefault(mm_key, []).append((token_start, chunk))

    def commit_deferred_mm_outputs(
        self,
        finished_req_ids: set[str] | list[str],
        input_batch: InputBatch,
    ) -> None:
        """Write deferred multimodal chunks into the CPU prefix cache.

        This must run before finished requests are removed from ``input_batch``,
        because the block table is needed to map logical token positions to KV
        cache slots.
        """
        if not finished_req_ids or not self._deferred_mm_outputs:
            return

        for req_id in tuple(finished_req_ids):
            per_key = self._deferred_mm_outputs.pop(req_id, None)
            if not per_key:
                continue
            req_idx = input_batch.req_id_to_index.get(req_id)
            if req_idx is None:
                continue
            for mm_key, chunks in per_key.items():
                self._commit_deferred_mm_key(req_idx, input_batch, mm_key, chunks)

    def discard_deferred_mm_outputs(self, req_id: str) -> None:
        """Drop deferred chunks for requests that leave without a cache commit."""
        self._deferred_mm_outputs.pop(req_id, None)

    def _commit_deferred_mm_key(
        self,
        req_idx: int,
        input_batch: InputBatch,
        mm_key: str,
        chunks: list[tuple[int, torch.Tensor]],
    ) -> None:
        if not chunks:
            return

        chunks = sorted(chunks, key=lambda item: item[0])
        first_start = chunks[0][0]
        expected_start = first_start
        tensors: list[torch.Tensor] = []
        for token_start, tensor in chunks:
            if tensor.numel() == 0:
                continue
            if token_start != expected_start:
                if tensors:
                    self._write_deferred_mm_tensor(req_idx, input_batch, mm_key, tensors, first_start)
                first_start = token_start
                expected_start = token_start
                tensors = []
            tensors.append(tensor)
            expected_start += int(tensor.shape[0])

        if tensors:
            self._write_deferred_mm_tensor(req_idx, input_batch, mm_key, tensors, first_start)

    def _write_deferred_mm_tensor(
        self,
        req_idx: int,
        input_batch: InputBatch,
        mm_key: str,
        tensors: list[torch.Tensor],
        token_start: int,
    ) -> None:
        mm_state = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
        if mm_state.ndim < 2 or mm_state.shape[0] == 0:
            return

        if mm_key not in self.mm_cache_keys:
            self.mm_outputs_cache[mm_key] = self._get_cache_tensor(
                dtype=mm_state.dtype,
                hidden_size=int(mm_state.shape[-1]),
            )
            self.mm_cache_keys.add(mm_key)
            logger.info(
                "Initializing deferred multimodal output cache of size %s for key: %s",
                list(self.mm_outputs_cache[mm_key].shape),
                mm_key,
            )

        slots = self._get_slot_ids_for_token_range(
            req_idx=req_idx,
            input_batch=input_batch,
            token_start=token_start,
            num_tokens=int(mm_state.shape[0]),
        )
        if slots.numel() == 0:
            return

        mm_state_cpu = OmniTensorPrefixCache._coerce_to_cpu_tensor(mm_state[: slots.numel()])
        mm_cache = self.mm_outputs_cache[mm_key]
        flat_cache = mm_cache.view(-1, mm_cache.shape[-1])
        flat_cache[slots] = mm_state_cpu

    def _get_slot_ids_for_token_range(
        self,
        req_idx: int,
        input_batch: InputBatch,
        token_start: int,
        num_tokens: int,
    ) -> torch.Tensor:
        if num_tokens <= 0:
            return torch.empty((0,), dtype=torch.long)

        block_table = input_batch.block_table[0].block_table.cpu
        token_positions = torch.arange(token_start, token_start + num_tokens, dtype=torch.long)
        block_offsets = token_positions // self.block_size
        max_blocks = int(block_table.shape[1])
        valid = block_offsets < max_blocks
        if not bool(valid.all()):
            token_positions = token_positions[valid]
            block_offsets = block_offsets[valid]
        if token_positions.numel() == 0:
            return torch.empty((0,), dtype=torch.long)

        block_ids = block_table[req_idx, block_offsets].to(torch.long)
        return block_ids * self.block_size + (token_positions % self.block_size)

    def _coerce_to_payload_dict(
        self,
        element: object,
        query_start_loc: torch.Tensor,
        input_batch: InputBatch,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, object]:
        """Build the multimodal passthrough data per request for
        the object under consideration. This mirrors the no-prefix-cache
        path: a tensor whose first dimension matches the total scheduled
        token count is sliced per request, so 1D (seq_len,) metadata that
        is intentionally not cached (e.g. ref_code_len, codec_streaming)
        is still split per request instead of leaking the whole batch.
        """
        total_scheduled_tokens = sum(int(num_scheduled_tokens[r]) for r in input_batch.req_ids)
        elem_dict = {}
        for req_id in input_batch.req_ids:
            req_idx = input_batch.req_id_to_index[req_id]
            start = query_start_loc[req_idx]
            end = start + num_scheduled_tokens[req_id]
            elem_dict[req_id] = to_payload_element(
                element,
                req_idx,
                start=start,
                end=end,
                pass_lists_through=True,
                seq_len=total_scheduled_tokens,
            )
        return elem_dict

    def get_merged_multimodal_states(
        self,
        query_start_loc: torch.Tensor,
        input_batch: InputBatch,
        multimodal_outputs: dict,
        num_scheduled_tokens: dict[str, int],
    ):
        """Get the merged multimodal states if hidden state prefix caching is enabled."""
        combined_multimodal_outputs = {}
        # First get the prefix cached tensors that are present in the mm data
        for mm_key in self.mm_cache_keys:
            if mm_key in multimodal_outputs:
                combined_multimodal_outputs[mm_key] = self._get_merged_tensors(
                    query_start_loc=query_start_loc,
                    input_batch=input_batch,
                    cache=self.mm_outputs_cache[mm_key],
                    hidden_states=multimodal_outputs[mm_key],
                    num_scheduled_tokens=num_scheduled_tokens,
                )

        # Then, get everything else (passthrough data); first, convert to CPU
        # tensors similarly to the non prefix cached path, and then populate
        # the subdicts mapping request IDs -> payload objects
        passthrough_keys = set(multimodal_outputs.keys()) - self.mm_cache_keys
        passthrough_mm_data = {k: v for k, v in multimodal_outputs.items() if k in passthrough_keys}
        mm_cpu = build_mm_cpu(multimodal_outputs=passthrough_mm_data)

        for mm_key, mm_val in mm_cpu.items():
            combined_multimodal_outputs[mm_key] = self._coerce_to_payload_dict(
                element=mm_val,
                query_start_loc=query_start_loc,
                input_batch=input_batch,
                num_scheduled_tokens=num_scheduled_tokens,
            )
        return combined_multimodal_outputs

    def get_merged_hidden_states(
        self,
        query_start_loc: torch.Tensor,
        input_batch: InputBatch,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
        hidden_states_cpu: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Get merged hidden states, optionally reusing pre-staged CPU states.

        When provided, hidden_states_cpu follows the same contract as
        update_omni_tensor_prefix_cache: CPU, contiguous, same dtype and feature
        shape as hidden_states, and covering every scheduled-token span derived
        from query_start_loc and num_scheduled_tokens.
        """
        return self._get_merged_tensors(
            query_start_loc=query_start_loc,
            input_batch=input_batch,
            hidden_states=hidden_states,
            num_scheduled_tokens=num_scheduled_tokens,
            staged_cpu_tensor=hidden_states_cpu,
            cache=self.hidden_states_cache,
        )

    def _get_merged_tensors(
        self,
        query_start_loc: torch.Tensor,
        input_batch: InputBatch,
        cache: torch.Tensor,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
        staged_cpu_tensor: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """When hidden state caching is enabled, takes the input hidden_states,
        which only correspond to the scheduled tokens, and returns a mapping
        from request IDs to their full hidden states. This is accomplished by
        looking up the block IDs & scheduled token counts to split the
        hidden_states. staged_cpu_tensor can supply the already materialized CPU
        scheduled-token view; None preserves the legacy per-call CPU conversion.
        """
        # We do not support hybrid caches at the moment.
        if len(input_batch.block_table.block_tables) > 1:
            logger.warning_once(
                "Omni prefix caching is enabled, but the batch block table appears to"
                " have multiple kv groups; only the first group will be used!"
            )

        combined_hidden_states = {}
        required_tokens = 0
        for req_id in input_batch.req_ids:
            req_idx = input_batch.req_id_to_index[req_id]
            start = query_start_loc[req_idx]
            end = start + num_scheduled_tokens[req_id]
            required_tokens = max(required_tokens, int(end))
        hidden_states = OmniTensorPrefixCache._resolve_hidden_states_cpu(
            hidden_states,
            required_tokens,
            staged_cpu_tensor,
        )
        for req_id in input_batch.req_ids:
            req_idx = input_batch.req_id_to_index[req_id]

            if req_id in self._new_req_cache_hit_ids:
                block_ids = self._get_cached_block_ids(req_idx, input_batch)
                cached_hs = cache[block_ids].reshape(-1, cache.shape[-1])

                # Slice the hidden states corresponding to this request;
                # we do this by using the query start
                start = query_start_loc[req_idx]
                new_hs = hidden_states[start : start + num_scheduled_tokens[req_id]]
                combined_hidden_states[req_id] = torch.cat([cached_hs, new_hs], dim=0)
            else:
                # cache miss for this request, pass through normally
                start = query_start_loc[req_idx]
                new_hs = hidden_states[start : start + num_scheduled_tokens[req_id]]
                combined_hidden_states[req_id] = new_hs

        return combined_hidden_states

    def _get_cached_block_ids(self, req_idx: int, input_batch: InputBatch) -> torch.Tensor:
        """Given an input batch and request index in the batch (not ID), get the
        block IDs corresponding to the cache hit.
        """
        num_computed = input_batch.num_computed_tokens_cpu[req_idx]
        # NOTE: vLLM only caches full blocks
        num_cached_blocks = num_computed // self.block_size
        # Get the block IDs attached to this cache hit and reindex into
        # the flattened cached hidden states (i.e., 1 row per token).
        return input_batch.block_table[0].block_table.cpu[req_idx, :num_cached_blocks]
