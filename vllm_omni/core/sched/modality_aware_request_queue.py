"""Modality-aware request queue for multimodal scheduling.

This module provides a specialized request queue that organizes requests
by their modality combinations using bitmask-based bucketing.
"""

import heapq
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm.v1.core.sched.request_queue import RequestQueue
from vllm.v1.request import Request


class MaskFilterPolicy(Enum):
    """Policy for filtering modality buckets during scheduling.

    Attributes:
        EXACT: Match only the exact modality mask.
        COMPATIBLE: Match all subset masks including pure text (0b000).
        COMPATIBLE_MULTI_MODAL: Match all subset masks excluding pure text.
    """

    EXACT = "exact"
    COMPATIBLE = "compatible"
    COMPATIBLE_MULTI_MODAL = "compatible_multi_modal"


class OmniSchedulingPolicy(Enum):
    """Scheduling policy options for the Omni scheduler.

    Attributes:
        FCFS: First-come-first-served scheduling.
        PRIORITY: Priority-based scheduling.
        OMNI_MODALITY_AWARE: Modality-aware scheduling with encoder optimization.
    """

    FCFS = "fcfs"
    PRIORITY = "priority"
    OMNI_MODALITY_AWARE = "omni_modality_aware"


def create_request_queue(policy: OmniSchedulingPolicy, all_modalities_mask: int | None = 0) -> RequestQueue:
    """Create a request queue based on the omni_modality_aware policy.

    Args:
        policy: The scheduling policy to use.
        all_modalities_mask: Bitmask representing all supported modalities.

    Returns:
        A ModalityAwareRequestQueue instance.

    """
    return ModalityAwareRequestQueue(all_modalities_mask)


class ModalityAwareRequestQueue(RequestQueue):
    """Request queue that organizes requests by modality combination.

    This queue uses bitmask-based bucketing to efficiently group requests
    by their modality combinations (e.g., text-only, image+text, audio+text).
    Each bucket maintains FCFS ordering using a deque.

    Attributes:
        _modality_buckets: Dict mapping modality masks to request deques.
        _modality_stats: Dict mapping modality masks to total mm_tokens counts.
    """

    def __init__(self, all_modalities_mask: int) -> None:
        """Initialize the modality-aware request queue.

        Args:
            all_modalities_mask: Bitmask representing all supported modalities.
                For example, 0b111 (7) indicates 3 modality types.
        """
        # Initialize with pure text bucket (mask=0)
        self._modality_buckets: dict[int, deque[Request]] = {0: deque()}
        self._modality_stats: dict[int, int] = {0: 0}

        if all_modalities_mask > 0:
            self.bind_modality_config(all_modalities_mask)

    def bind_modality_config(self, all_modalities_mask: int) -> None:
        """Initialize buckets for all possible modality combinations.

        Args:
            all_modalities_mask: Must be a contiguous bitmask (e.g., 0b111).
                Creates buckets for all values from 1 to all_modalities_mask.

        Raises:
            AssertionError: If mask is not valid (not all trailing 1s).
        """
        # Validate mask format: must be contiguous 1s (e.g., 0b111)
        assert all_modalities_mask > 0 and (all_modalities_mask & (all_modalities_mask + 1)) == 0

        for mask in range(1, all_modalities_mask + 1):
            self._modality_buckets[mask] = deque()
            self._modality_stats[mask] = 0

    def add_request(self, request: Request) -> None:
        """Add a request to the appropriate modality bucket.

        The request must have been enriched with modality metadata by the
        scheduler before calling this method.

        Args:
            request: Request with mm_mask_to_prefill and mm_tokens_to_prefill
                attributes set.

        Raises:
            AssertionError: If required attributes are missing.
        """
        assert hasattr(request, "request_id")
        assert hasattr(request, "mm_mask_to_prefill")
        assert hasattr(request, "mm_tokens_to_prefill")
        assert hasattr(request, "arrival_time_mono")

        mask_key = int(request.mm_mask_to_prefill)
        self._modality_buckets[mask_key].append(request)
        self._modality_stats[mask_key] += request.mm_tokens_to_prefill

    def pop_request(self) -> Request:
        """Pop and return the globally oldest request across all buckets.

        Finds the request with the smallest arrival_time_mono across all
        non-empty buckets and removes it.

        Returns:
            The oldest request in the queue.

        Raises:
            IndexError: If all buckets are empty.
        """
        min_req: Request | None = None
        target_mask: int = -1

        for mask, bucket in self._modality_buckets.items():
            if not bucket:
                continue
            current_head = bucket[0]
            if min_req is None or current_head.arrival_time_mono < min_req.arrival_time_mono:
                min_req = current_head
                target_mask = mask

        if min_req is None:
            raise IndexError("pop from an empty queue")

        popped_req = self._modality_buckets[target_mask].popleft()
        self._modality_stats[target_mask] -= popped_req.mm_tokens_to_prefill
        return popped_req

    def pop_request_by_id(self, request_id: str) -> Request:
        """Pop a specific request by its ID from the bucket head.

        This ensures consistency between peek and pop operations by
        only allowing removal of requests at bucket heads.

        Args:
            request_id: Unique identifier of the request to pop.

        Returns:
            The popped request.

        Raises:
            AssertionError: If request_id is not found at any bucket head.
        """
        popped_req = None
        found = False

        for mask, bucket in self._modality_buckets.items():
            if not bucket:
                continue
            if bucket[0].request_id == request_id:
                popped_req = bucket.popleft()
                self._modality_stats[mask] -= popped_req.mm_tokens_to_prefill
                found = True
                break

        assert found, f"Request {request_id} not found at any bucket head"
        return popped_req

    def peek_request(self) -> Request:
        """Peek at the globally oldest request without removing it.

        Used in starvation prevention to check arrival times.

        Returns:
            The oldest request in the queue.

        Raises:
            IndexError: If all buckets are empty.
        """
        min_req: Request | None = None
        target_mask: int = -1

        for mask, bucket in self._modality_buckets.items():
            if not bucket:
                continue
            current_head = bucket[0]
            if min_req is None or current_head.arrival_time_mono < min_req.arrival_time_mono:
                min_req = current_head
                target_mask = mask

        if min_req is None:
            raise IndexError("peek from an empty queue")

        return self._modality_buckets[target_mask][0]

    def peek_request_by_mm_mask(
        self, mm_mask: int, filter_policy: MaskFilterPolicy = MaskFilterPolicy.EXACT
    ) -> Request:
        """Peek at the oldest request matching the modality mask criteria.

        Args:
            mm_mask: The modality mask to match against.
            filter_policy: How to interpret the mask matching:
                - EXACT: Only exact mask match
                - COMPATIBLE: Include all subsets and pure text
                - COMPATIBLE_MULTI_MODAL: Include subsets but not pure text

        Returns:
            The oldest request matching the criteria.

        Raises:
            KeyError: If modality combination is not supported.
            IndexError: If no matching requests found.
        """
        if mm_mask not in self._modality_buckets:
            raise KeyError("this modality combination is not supported by the model")

        if filter_policy == MaskFilterPolicy.EXACT:
            if not self._modality_buckets[mm_mask]:
                raise IndexError("peek from an empty modality bucket")
            return self._modality_buckets[mm_mask][0]

        # Build target mask set based on filter policy
        target_masks_set = self.get_compatible_mm_submask(mm_mask)
        if filter_policy == MaskFilterPolicy.COMPATIBLE:
            if self._modality_buckets[0]:
                target_masks_set.add(0)

        if not target_masks_set:
            raise IndexError("peek from empty modality buckets")

        # Find oldest request across matching buckets
        min_req: Request | None = None
        target_mask: int = -1

        for mask, bucket in self._modality_buckets.items():
            if mask not in target_masks_set or not bucket:
                continue
            current_head = bucket[0]
            if min_req is None or current_head.arrival_time_mono < min_req.arrival_time_mono:
                min_req = current_head
                target_mask = mask

        return self._modality_buckets[target_mask][0]

    def prepend_request(self, request: Request) -> None:
        """Add a request to the front of its modality bucket.

        Used for:
        1. Returning preempted running requests to waiting queue
        2. Adding skipped requests to temporary skip queue

        Args:
            request: Request with modality metadata attributes.
        """
        assert hasattr(request, "request_id")
        assert hasattr(request, "mm_mask_to_prefill")
        assert hasattr(request, "mm_tokens_to_prefill")
        assert hasattr(request, "arrival_time_mono")

        mask_key = int(request.mm_mask_to_prefill)
        self._modality_buckets[mask_key].appendleft(request)
        self._modality_stats[mask_key] += request.mm_tokens_to_prefill

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to this queue.

        Used to merge skipped requests back to the waiting queue.

        Args:
            requests: Another ModalityAwareRequestQueue to merge from.

        Raises:
            TypeError: If requests is not a ModalityAwareRequestQueue.
            KeyError: If bucket masks don't match between queues.
        """
        if not isinstance(requests, ModalityAwareRequestQueue):
            raise TypeError("Omni-Scheduler requires ModalityAwareRequestQueue for consistency.")

        for mask, bucket in self._modality_buckets.items():
            if mask not in requests._modality_buckets or mask not in requests._modality_stats:
                raise KeyError("the two ModalityAwareRequestQueue should have the same bucket masks")
            bucket.extendleft(requests.get_reversed_bucket(mask))
            self._modality_stats[mask] += requests._modality_stats[mask]

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue.

        Args:
            request: The request to remove.

        Raises:
            ValueError: If request cannot be found in queue.
        """
        mask = request.mm_mask_to_prefill
        if mask not in self._modality_buckets:
            raise ValueError(
                "this request cannot be in this queue as its modality is not supported by the model config"
            )

        if not self._modality_buckets[mask]:
            raise ValueError("this request cannot be in this queue as its bucket is empty")

        try:
            self._modality_buckets[mask].remove(request)
            self._modality_stats[mask] -= request.mm_tokens_to_prefill
        except ValueError:
            raise ValueError("this request is not in the queue")

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple requests from the queue efficiently.

        Groups requests by modality mask and performs batch removal
        per bucket to minimize overhead.

        Args:
            requests: Iterable of requests to remove.
        """
        if not requests:
            return

        # Group requests by modality mask
        grouped_to_remove = defaultdict(list)
        for req in requests:
            mask = req.mm_mask_to_prefill
            grouped_to_remove[mask].append(req)

        # Process each affected bucket
        for mask, to_remove_list in grouped_to_remove.items():
            if mask not in self._modality_buckets:
                continue

            bucket = self._modality_buckets[mask]
            if not bucket:
                continue

            remove_set = set(to_remove_list)

            # Filter bucket and calculate token deduction
            new_bucket_elements = []
            tokens_to_deduct = 0

            for req in bucket:
                if req in remove_set:
                    tokens_to_deduct += req.mm_tokens_to_prefill
                else:
                    new_bucket_elements.append(req)

            # Atomic update of bucket and statistics
            self._modality_buckets[mask] = deque(new_bucket_elements)
            self._modality_stats[mask] -= tokens_to_deduct

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return any(self._modality_buckets.values())

    def __len__(self) -> int:
        """Get total number of requests across all buckets."""
        return sum(len(b) for b in self._modality_buckets.values())

    def __iter__(self) -> Iterator[Request]:
        """Iterate over all requests in global FCFS order.

        Uses heap-based multi-way merge across buckets.
        Complexity: O(N log K) time, O(K) space where K is bucket count.
        """
        active_bucket_iters = [iter(bucket) for bucket in self._modality_buckets.values() if bucket]
        yield from heapq.merge(*active_bucket_iters, key=lambda r: r.arrival_time_mono)

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over all requests from newest to oldest.

        Uses heap-based multi-way merge with reverse ordering.
        Complexity: O(N log K) time, O(K) space.
        """
        active_bucket_reversed_iters = [reversed(bucket) for bucket in self._modality_buckets.values() if bucket]
        yield from heapq.merge(*active_bucket_reversed_iters, key=lambda r: r.arrival_time_mono, reverse=True)

    def get_reversed_bucket(self, mm_mask: int) -> deque[Request]:
        """Get a reversed iterator for a specific modality bucket.

        Args:
            mm_mask: The modality mask of the bucket.

        Returns:
            Reversed iterator over the bucket's requests.

        Raises:
            KeyError: If mask is not in the queue.
        """
        if mm_mask not in self._modality_buckets:
            raise KeyError("cannot reverse a bucket when its mask is not in the queue")
        return reversed(self._modality_buckets[mm_mask])

    def compatible_buckets_not_empty(
        self, mm_mask: int, filter_policy: MaskFilterPolicy = MaskFilterPolicy.EXACT
    ) -> bool:
        """Check if any compatible modality bucket has requests.

        Args:
            mm_mask: The modality mask representing available encoder resources.
            filter_policy: How to filter compatible buckets:
                - EXACT: Only check the exact mask bucket
                - COMPATIBLE: Check all subset buckets including pure text
                - COMPATIBLE_MULTI_MODAL: Check subsets excluding pure text

        Returns:
            True if at least one matching bucket is non-empty.

        Raises:
            KeyError: If modality combination is not supported.
        """
        if mm_mask not in self._modality_buckets:
            raise KeyError("this modality combination is not supported by the model")

        if filter_policy == MaskFilterPolicy.EXACT:
            return bool(self._modality_buckets[mm_mask])

        # Build target mask set based on filter policy
        target_masks_set = self.get_compatible_mm_submask(mm_mask)
        if filter_policy == MaskFilterPolicy.COMPATIBLE:
            if self._modality_buckets[0]:
                target_masks_set.add(0)

        return bool(target_masks_set)

    def get_compatible_mm_submask(self, current_resource_mask: int) -> set[int]:
        """Find all non-empty bucket masks that are subsets of the given mask.

        Uses bit manipulation to enumerate all submasks efficiently.
        Excludes mask=0 (pure text) from results.

        Args:
            current_resource_mask: Bitmask of available encoder resources.

        Returns:
            Set of non-empty bucket masks that are subsets of current_resource_mask.
        """
        compatible_masks = set()
        submask = current_resource_mask

        while submask > 0:
            if submask in self._modality_buckets and self._modality_buckets[submask]:
                compatible_masks.add(submask)
            # Iterate to next submask using bit trick
            submask = (submask - 1) & current_resource_mask

        return compatible_masks

    def get_compatible_mm_tokens(self, mm_mask: int, filter_policy: MaskFilterPolicy = MaskFilterPolicy.EXACT) -> int:
        """Get total multimodal tokens across compatible buckets.

        Used to estimate encoder compute benefit when deciding whether
        to activate a new encoder.

        Args:
            mm_mask: The modality mask to match.
            filter_policy: How to filter compatible buckets.

        Returns:
            Total mm_tokens_to_prefill across matching buckets.

        Raises:
            KeyError: If modality combination is not supported.
        """
        if mm_mask not in self._modality_buckets:
            raise KeyError("this modality combination is not supported by the model")

        if filter_policy == MaskFilterPolicy.EXACT:
            return self._modality_stats[mm_mask]

        # Build target mask set based on filter policy
        target_masks_set = self.get_compatible_mm_submask(mm_mask)
        if filter_policy == MaskFilterPolicy.COMPATIBLE:
            if self._modality_buckets[0]:
                target_masks_set.add(0)

        # Sum tokens across matching buckets
        total_tokens = 0
        for mask in target_masks_set:
            total_tokens += self._modality_stats.get(mask, 0)

        return total_tokens
