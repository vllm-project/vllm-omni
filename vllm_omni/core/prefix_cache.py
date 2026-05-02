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
            if isinstance(val, torch.Tensor) and val.shape[0] == seq_len and key not in self.mm_cache_keys:
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

    @staticmethod
    def _coerce_to_cpu_tensor(maybe_gpu_tensor: torch.Tensor) -> torch.Tensor:
        """Convert GPU tensors -> contiguous CPU tensors if needed."""
        return maybe_gpu_tensor.detach().cpu().contiguous()

    def update_omni_tensor_prefix_cache(
        self,
        hidden_states: torch.Tensor | None,
        multimodal_outputs: dict[str, torch.Tensor] | None,
        num_tokens_unpadded: int,
        slot_mapping: torch.Tensor,
        num_tokens_padded: int | None = None,
    ):
        """Updates the hidden cache state for the provided hidden states and multimodal outputs.

        Args:
            hidden_states: Hidden states tensor to cache (if any)
            multimodal_outputs: Multimodal dict whose tensors may be cached
            num_tokens_unpadded: Number of tokens without padding
            slot_mapping: Slot mapping for the input sequence
            num_tokens_padded: Total number of tokens including padding
        """
        unpadded_slot_mapping = slot_mapping[:num_tokens_unpadded]
        if num_tokens_padded is None:
            num_tokens_padded = num_tokens_unpadded

        if hidden_states is not None:
            # Slice to unpadded portion before caching
            hidden_states = hidden_states[:num_tokens_unpadded]
            # Ensure that hidden states are on the CPU
            hidden_states = OmniTensorPrefixCache._coerce_to_cpu_tensor(hidden_states)
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
                    # Slice to unpadded portion before caching
                    mm_state = multimodal_outputs[mm_out_key][:num_tokens_unpadded]
                    mm_state = OmniTensorPrefixCache._coerce_to_cpu_tensor(mm_state)
                    flat_cache = mm_cache.view(-1, mm_cache.shape[-1])
                    flat_cache[unpadded_slot_mapping] = mm_state
            logger.debug("Writing to mm output cache for %s tokens", num_tokens_unpadded)

    def _coerce_to_payload_dict(
        self,
        element: object,
        query_start_loc: torch.Tensor,
        input_batch: InputBatch,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, object]:
        """Build the multimodal passthrough data per request for
        the object under consideration. This is identical to the case
        for no prefix cache when we tensor does have a first dimension
        matching the seq len.
        """
        elem_dict = {}
        for req_id in input_batch.req_ids:
            req_idx = input_batch.req_id_to_index[req_id]
            start = query_start_loc[req_idx]
            end = start + num_scheduled_tokens[req_id]
            elem_dict[req_id] = to_payload_element(
                element, req_idx, start=start, end=end, pass_lists_through=True, seq_len=None
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

    def get_merged_hidden_states(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Get the merged hidden states."""
        return self._get_merged_tensors(
            *args,
            **kwargs,
            cache=self.hidden_states_cache,
        )

    def _get_merged_tensors(
        self,
        query_start_loc: torch.Tensor,
        input_batch: InputBatch,
        cache: torch.Tensor,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, torch.Tensor]:
        """When hidden state caching is enabled, takes the input hidden_states,
        which only correspond to the scheduled tokens, and returns a mapping
        from request IDs to their full hidden states. This is accomplished by
        looking up the block IDs & scheduled token counts to split the
        hidden_states.
        """
        # We do not support hybrid caches at the moment.
        if len(input_batch.block_table.block_tables) > 1:
            logger.warning_once(
                "Omni prefix caching is enabled, but the batch block table appears to"
                " have multiple kv groups; only the first group will be used!"
            )

        combined_hidden_states = {}
        hidden_states = OmniTensorPrefixCache._coerce_to_cpu_tensor(hidden_states)
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
