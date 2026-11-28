# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Strict scheduled-row H50 handoff, separate from either model's KV cache.

This first connector deliberately requires APC off: an APC hit without a
validated auxiliary sidecar cannot reconstruct H50. The final sampled token
has no hidden row until another forward actually computes it.
"""

from dataclasses import dataclass

import torch

H50_KEY = "hidden_states.layer_50"
IDS_KEY = "realization_token_ids"
POSITIONS_KEY = "realization_positions"
H50_SIDECAR_KEYS = (H50_KEY, IDS_KEY, POSITIONS_KEY)


def attach_scheduled_metadata(payload, token_ids, positions, valid_rows):
    """Called by the AR runner after forward, outside any captured graph."""
    if token_ids is None or valid_rows < 1:
        raise ValueError("Realization H50 requires scheduled input token identities")
    ids = token_ids[:valid_rows].detach().to(dtype=torch.long).clone()
    pos = positions[..., :valid_rows].detach().to(dtype=torch.long).clone()
    if ids.shape != (valid_rows,) or pos.ndim not in (1, 2) or (pos.ndim == 2 and pos.shape[0] != 3):
        raise ValueError("Invalid scheduled Intent token/MRoPE geometry")
    # Wire row dimension must be first for the generic per-request slicer.
    pos = pos[:, None] if pos.ndim == 1 else pos.T.contiguous()
    return {**payload, IDS_KEY: ids, POSITIONS_KEY: pos}


@dataclass(frozen=True)
class H50Selection:
    hidden: torch.Tensor
    token_ids: torch.Tensor
    positions: torch.Tensor
    offsets: tuple[int, ...]
    request_id: str
    fence: int


class H50Connector:
    """One fenced AR request's bounded, computed-only row ledger.

    Explicit absolute offsets (not token value matching) select shared/body
    rows. Callers must supply the expected token IDs so off-by-one sampled vs
    forwarded rows fail closed. Selected tensors are snapshots, never mutable
    graph output views. CPU transport is intentional in this first integration.
    """

    def __init__(self, request_id, fence, *, width=5120, max_rows=32768):
        if not request_id or type(fence) is not int or fence < 0 or max_rows < 1:
            raise ValueError("Invalid H50 connector identity/limit")
        self.request_id, self.fence = request_id, fence
        self.width, self.max_rows = width, max_rows
        self._chunks = []
        self.computed_rows = 0
        self.snapshot_rows = 0
        self.closed = False
        self._shared = None

    def append(self, payload, *, request_id, fence):
        if self.closed or (request_id, fence) != (self.request_id, self.fence):
            raise ValueError("Closed or stale H50 connector")
        hidden, ids, pos = (payload[k] for k in H50_SIDECAR_KEYS)
        n = ids.numel()
        if (
            hidden.shape != (n, self.width)
            or ids.shape != (n,)
            or ids.dtype != torch.long
            or pos.shape not in ((n, 1), (n, 3))
            or pos.dtype != torch.long
            or not n
            or not torch.isfinite(hidden).all()
        ):
            raise ValueError("Malformed/nonfinite scheduled H50 handoff")
        if self.computed_rows + n > self.max_rows:
            raise BufferError("H50 row ledger capacity exceeded")
        self._chunks.append((hidden.detach().cpu().clone(), ids.detach().cpu().clone(), pos.detach().cpu().clone()))
        self.computed_rows += n

    def append_snapshot(self, payload, *, request_id, fence):
        """Consume native cumulative client payload without duplicating rows.

        Native output processing reconstructs hidden_states.layers[50]. Other
        modalities and sampled token IDs are deliberately not used here.
        """
        if self.closed or (request_id, fence) != (self.request_id, self.fence):
            raise ValueError("Closed or stale H50 connector")
        from vllm_omni.data_entry_keys import flatten_payload

        flat = flatten_payload(payload)
        tensors = {key: flat[key] for key in H50_SIDECAR_KEYS}
        total = tensors[IDS_KEY].numel()
        if total < self.snapshot_rows:
            raise ValueError("H50 cumulative snapshot moved backwards")
        if total == self.snapshot_rows:
            return 0
        delta = {key: value[self.snapshot_rows :] for key, value in tensors.items()}
        self.append(delta, request_id=request_id, fence=fence)
        self.snapshot_rows = total
        return len(delta[IDS_KEY])

    def remove_window_rows(self, prefix, count):
        end = prefix + count
        kept, pos = [], 0
        for chunk in self._chunks:
            stop = pos + len(chunk[1])
            if stop <= prefix or pos >= end:
                kept.append(chunk)
            else:
                if pos < prefix:
                    kept.append(tuple(x[: prefix - pos] for x in chunk))
                if stop > end:
                    kept.append(tuple(x[end - pos :] for x in chunk))
            pos = stop
        self._chunks = kept
        self.computed_rows -= count

    def drop_tail(self):
        """Remove one logical row while the native wire stays cumulative."""
        hidden, ids, positions = self._chunks.pop()
        if len(ids) > 1:
            self._chunks.append((hidden[:-1], ids[:-1], positions[:-1]))
        self.computed_rows -= 1

    def select(self, offsets, expected_token_ids):
        offsets = tuple(offsets)
        if self.closed or not offsets or len(offsets) != len(expected_token_ids):
            raise ValueError("Invalid H50 row selection")
        if any(type(i) is not int or i < 0 or i >= self.computed_rows for i in offsets):
            raise ValueError("Requested H50 row has not been computed (sampled tail is not H50)")
        if any(a >= b for a, b in zip(offsets, offsets[1:])):
            raise ValueError("H50 offsets must strictly increase")
        # Gather only selected rows, not a full concatenated conversation.
        out, tokens, positions = [], [], []
        base = 0
        remaining = iter(offsets)
        wanted = next(remaining, None)
        for h, ids, pos in self._chunks:
            while wanted is not None and wanted < base + len(ids):
                local = wanted - base
                out.append(h[local])
                tokens.append(ids[local])
                positions.append(pos[local])
                wanted = next(remaining, None)
            base += len(ids)
        token_ids = torch.stack(tokens)
        if not torch.equal(token_ids, torch.tensor(expected_token_ids, dtype=torch.long)):
            raise ValueError("H50 selected token identities differ from committed text")
        return H50Selection(torch.stack(out), token_ids, torch.stack(positions), offsets, self.request_id, self.fence)

    def select_shared_once(self, offsets, expected_token_ids):
        if self._shared is not None:
            if self._shared.offsets != tuple(offsets) or self._shared.token_ids.tolist() != list(expected_token_ids):
                raise ValueError("Shared H50 identity changed within session")
            return self._shared
        self._shared = self.select(offsets, expected_token_ids)
        return self._shared

    def close(self):
        self.closed = True
        self._chunks.clear()
        self._shared = None
