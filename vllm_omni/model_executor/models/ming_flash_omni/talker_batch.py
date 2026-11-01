# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cross-request batched decoding for the Ming talker.

The single-request generator runs one Python AR loop per request. Here one
producer thread advances every active request by one latent per step: the
flow head, the aggregator, the stop head and the next Qwen2 decode run as one
CUDA graph over a padded batch of KV rows, followed by a single host read of
the stop probabilities for the whole batch.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from threading import Condition, Thread
from typing import TypedDict

import torch
from transformers import Qwen2Model
from transformers.models.qwen2.modeling_qwen2 import rotate_half
from vllm.logger import init_logger

from vllm_omni.model_executor.models.common.ming.audio_dsp import ISTFTEnvelopeError
from vllm_omni.model_executor.models.common.ming.cfm_solver import get_epss_timesteps
from vllm_omni.model_executor.models.common.ming.fused_istft import deferred_envelope_checks

logger = init_logger(__name__)

_BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256)
# Packed tokens per captured PackedQwen2 forward: talker prompts (a larger
# prefill batch is split) and AudioVAE frames of one runner step.
_PREFILL_BUCKETS = (64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048)
_VAE_BUCKETS = (16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)


class VAEUpsampleState(TypedDict, total=False):
    """Buffers used by the legacy or frame-holdback upsampler."""

    prev_chunk: torch.Tensor | None
    history_last: torch.Tensor | None
    is_first: bool
    held: torch.Tensor | None
    left: torch.Tensor | None


class BatchedVAEState(TypedDict, total=False):
    """Request-owned decoder state; row membership denotes an active lease.

    Empty states initialize on first decode. Releasing removes only the row;
    the next decode replaces the retained position and buffers before use.
    """

    row: int
    position: int
    upsample: VAEUpsampleState | None
    istft: tuple[torch.Tensor | None, torch.Tensor | None]


def _capture(body, *, pool, stream) -> tuple[torch.cuda.CUDAGraph, object]:
    """Capture an already warmed-up body; propagate capture failures unchanged."""
    # Return the warmups' cached blocks so the graph's private pool fits.
    torch.accelerator.empty_cache()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool, stream=stream, capture_error_mode="thread_local"):
        output = body()
    return graph, output


def _layer_qkv(layer, hidden, cos, sin, num_heads: int, num_kv_heads: int, head_dim: int):
    attn = layer.self_attn
    x = layer.input_layernorm(hidden)
    q = attn.q_proj(x).view(-1, num_heads, head_dim)
    k = attn.k_proj(x).view(-1, num_kv_heads, head_dim)
    v = attn.v_proj(x).view(-1, num_kv_heads, head_dim)
    # Same arithmetic as transformers' apply_rotary_pos_emb.
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k, v


def _layer_out(layer, hidden, attn_out):
    hidden = hidden + layer.self_attn.o_proj(attn_out.reshape(attn_out.shape[0], -1))
    return hidden + layer.mlp(layer.post_attention_layernorm(hidden))


class PackedQwen2:
    """Qwen2 forward over packed tokens with one KV row per request.

    Each row owns the KV blocks in its block-table entry. Prefill and decode
    use that same mapping for writes and attention reads, so a row can move
    without copying its KV. Full-attention rows retain contiguous blocks.
    The HF modules supply the weights and the norm, rotary and MLP code;
    only attention and the cache layout are replaced.

    With ``window`` (a sliding-window model), a query attends to itself and
    the ``window - 1`` previous positions, as in transformers' sliding mask.
    ``ring_blocks`` then bounds each row to that many physical KV blocks:
    logical block ``j`` maps to physical block ``j % ring_blocks``, so a call
    may write at most ``ring_blocks * BLOCK - window + 1`` positions per row.

    With ``compile_layers``, the work around attention in each layer (norms,
    projections, rotary, residuals and MLP) runs as torch.compile'd code shared
    by every layer; attention and the KV writes stay eager.

    ``capture_graphs`` records ``forward`` for packed batches up to a set of
    token counts; ``run`` then replays the smallest fitting graph, which
    removes the per-layer host work of an eager pass. Padding tokens and
    unused sequence slots become empty sequences on an extra scratch row.
    """

    BLOCK = 256

    def __init__(
        self,
        model: Qwen2Model,
        *,
        rows: int,
        max_len: int,
        window: int | None = None,
        ring_blocks: int | None = None,
        compile_layers: bool = False,
    ):
        from vllm.vllm_flash_attn import flash_attn_varlen_func, is_fa_version_supported

        cfg = model.config
        if window is None and getattr(cfg, "use_sliding_window", False):
            raise ValueError("Sliding-window Qwen2 needs an explicit window")
        if ring_blocks is not None and (window is None or ring_blocks * self.BLOCK <= window):
            raise ValueError("A KV ring must hold more than one attention window")
        self.model = model
        self.rows = rows
        self.window = window
        self.max_len = -(-max_len // self.BLOCK) * self.BLOCK
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        param = next(model.parameters())
        self.dtype = param.dtype
        logical_blocks = self.max_len // self.BLOCK
        self.ring_blocks = ring_blocks
        physical_blocks = ring_blocks or logical_blocks
        layers = len(model.layers)
        self.scratch_row = rows
        shape = ((rows + 1) * physical_blocks, self.BLOCK, self.num_kv_heads, self.head_dim)
        self.k_cache = [torch.zeros(shape, device=param.device, dtype=param.dtype) for _ in range(layers)]
        self.v_cache = [torch.zeros(shape, device=param.device, dtype=param.dtype) for _ in range(layers)]
        row_base = torch.arange(rows + 1, device=param.device, dtype=torch.int32)[:, None] * physical_blocks
        logical = torch.arange(logical_blocks, device=param.device, dtype=torch.int32)[None, :]
        self.block_table = row_base + (logical % physical_blocks if ring_blocks else logical)
        self._attn = flash_attn_varlen_func
        self._fa_version = 3 if is_fa_version_supported(3) else 2
        self._qkv = torch.compile(_layer_qkv, dynamic=True) if compile_layers else _layer_qkv
        self._out = torch.compile(_layer_out, dynamic=True) if compile_layers else _layer_out
        self.token_buckets: list[int] = []
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._graph_outputs: dict[int, torch.Tensor] = {}

    def capture_graphs(self, token_buckets, *, stream=None, pool=None) -> None:
        """Capture ``forward`` for up to ``rows`` sequences of each bucket's total tokens."""
        buckets = sorted(b for b in token_buckets if b <= self.max_len)
        if not buckets:
            return
        device, tokens, seqs = self.block_table.device, buckets[-1], self.rows
        # Staged per call: positions, token rows, cu_seqlens, KV lengths, rows.
        sections = [tokens, tokens, seqs + 1, seqs, seqs]
        self._meta_host = torch.zeros(sum(sections), dtype=torch.long, pin_memory=True)
        self._meta = torch.zeros(sum(sections), dtype=torch.long, device=device)
        self._meta_host_views = self._meta_host.split(sections)
        self._meta_views = self._meta.split(sections)
        self._embeds = torch.zeros(tokens, self.model.config.hidden_size, device=device, dtype=self.dtype)
        self._staged = torch.cuda.Event()
        pool = pool if pool is not None else torch.cuda.graph_pool_handle()
        stream = stream if stream is not None else torch.cuda.Stream(device)
        stream.wait_stream(torch.cuda.current_stream(device))
        # Without inference mode the captured activations would be kept for
        # autograd (the weights require grad when called from load_weights).
        with torch.inference_mode(), torch.cuda.stream(stream):
            # Largest first so smaller buckets reuse its pool blocks; each
            # capture runs one full-length sequence on the scratch row.
            for n in reversed(buckets):
                self._stage([self.scratch_row], [0], [n], n)
                for _ in range(2):
                    self._graph_body(n)
                self.graphs[n], self._graph_outputs[n] = _capture(
                    lambda n=n: self._graph_body(n), pool=pool, stream=stream
                )
        torch.cuda.current_stream(device).wait_stream(stream)
        self.token_buckets = buckets

    def _graph_body(self, tokens: int) -> torch.Tensor:
        positions, token_rows, cu, seq_lens, rows = self._meta_views
        return self.forward(
            self._embeds[:tokens],
            positions[:tokens],
            token_rows[:tokens],
            cu.to(torch.int32),
            seq_lens.to(torch.int32),
            self.block_table.index_select(0, rows),
            tokens,
        )

    def _stage(self, rows: list[int], starts: list[int], lengths: list[int], tokens: int) -> None:
        self._staged.synchronize()  # the previous copy has read the host buffer
        positions, token_rows, cu, seq_lens, row_ids = self._meta_host_views
        count, total = len(rows), sum(lengths)
        rows_t = torch.tensor(rows, dtype=torch.long)
        lengths_t = torch.tensor(lengths, dtype=torch.long)
        positions[:tokens] = 0
        positions[:total] = torch.cat([torch.arange(s, s + n) for s, n in zip(starts, lengths, strict=True)])
        token_rows[:tokens] = self.scratch_row
        token_rows[:total] = torch.repeat_interleave(rows_t, lengths_t)
        cu.fill_(total)
        cu[0] = 0
        torch.cumsum(lengths_t, 0, out=cu[1 : count + 1])
        seq_lens.zero_()
        seq_lens[:count] = torch.tensor(starts, dtype=torch.long) + lengths_t
        row_ids.fill_(self.scratch_row)
        row_ids[:count] = rows_t
        self._meta.copy_(self._meta_host, non_blocking=True)
        self._staged.record()

    def run(self, embeds: list[torch.Tensor], rows: list[int], starts: list[int]) -> torch.Tensor:
        """Forward one sequence per row, each ``[n, hidden]`` continuing at its start.

        Returns final-norm hidden states [total tokens, hidden]. A replayed
        graph returns a view of its output, overwritten by the next replay.
        """
        lengths = [x.shape[0] for x in embeds]
        total = sum(lengths)
        tokens = next((b for b in self.token_buckets if b >= total), None)
        if tokens is not None:
            self._stage(rows, starts, lengths, tokens)
            self._embeds[:total].copy_(torch.cat(embeds))
            self.graphs[tokens].replay()
            return self._graph_outputs[tokens][:total]
        device = self.block_table.device
        rows_t = torch.tensor(rows, dtype=torch.long)
        lengths_t = torch.tensor(lengths, dtype=torch.long)
        cu = torch.zeros(len(rows) + 1, dtype=torch.int32)
        cu[1:] = torch.cumsum(lengths_t, 0)
        positions = torch.cat([torch.arange(s, s + n) for s, n in zip(starts, lengths, strict=True)])
        seq_lens = torch.tensor([s + n for s, n in zip(starts, lengths, strict=True)], dtype=torch.int32)
        return self.forward(
            torch.cat(embeds).to(self.dtype),
            positions.to(device, non_blocking=True),
            torch.repeat_interleave(rows_t, lengths_t).to(device, non_blocking=True),
            cu.to(device, non_blocking=True),
            seq_lens.to(device, non_blocking=True),
            self.block_table.index_select(0, rows_t.to(device, non_blocking=True)),
            max(lengths),
        )

    def forward(
        self,
        embeds: torch.Tensor,
        positions: torch.Tensor,
        token_rows: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_k: torch.Tensor,
        block_table: torch.Tensor,
        max_seqlen_q: int,
    ) -> torch.Tensor:
        """Return final-norm hidden states [tokens, hidden] for packed ``embeds``.

        ``positions`` and ``token_rows`` give each token's position and KV
        row; ``cu_seqlens_q``/``seqused_k``/``block_table`` describe the
        sequences (one per row) in FlashAttention's varlen layout.
        """
        model = self.model
        if self.ring_blocks:
            block = self.block_table[token_rows, positions // self.BLOCK].long()
            slots = block * self.BLOCK + positions % self.BLOCK
        else:
            slots = self.block_table[token_rows, 0].long() * self.BLOCK + positions
        cos, sin = model.rotary_emb(embeds, positions.unsqueeze(0))
        cos, sin = cos[0].unsqueeze(1), sin[0].unsqueeze(1)
        # A fresh contiguous tensor, whichever buffer view the caller passes,
        # so the compiled layer code sees one input layout (fewer recompiles).
        hidden = embeds.clone(memory_format=torch.contiguous_format)
        for index, layer in enumerate(model.layers):
            q, k, v = self._qkv(layer, hidden, cos, sin, self.num_heads, self.num_kv_heads, self.head_dim)
            k_cache, v_cache = self.k_cache[index], self.v_cache[index]
            k_cache.view(-1, self.num_kv_heads, self.head_dim).index_copy_(0, slots, k)
            v_cache.view(-1, self.num_kv_heads, self.head_dim).index_copy_(0, slots, v)
            out = self._attn(
                q,
                k_cache,
                v_cache,
                max_seqlen_q,
                cu_seqlens_q,
                self.max_len,
                seqused_k=seqused_k,
                block_table=block_table,
                causal=True,
                softmax_scale=layer.self_attn.scaling,
                window_size=None if self.window is None else [self.window - 1, 0],
                # Talker and AudioVAE contexts are short and each call has many
                # rows: one KV split, no split accumulators to allocate.
                num_splits=1,
                fa_version=self._fa_version,
            )
            hidden = self._out(layer, hidden, out)
        return model.norm(hidden)


@dataclass(eq=False)
class TalkerSegment:
    embeds: torch.Tensor  # [1, prompt_len, hidden]
    max_steps: int


class BatchedTalkerRequest:
    """Engine-side state of one request: its text segments and decode cursor.

    ``publish(latent, last)`` receives each latent once the step that produced
    it has completed on the GPU; ``fail(exc)`` ends the request instead.
    """

    def __init__(
        self,
        segments: list[TalkerSegment],
        *,
        publish,
        fail,
        prompt_wav_lat: torch.Tensor | None,
        min_steps: int,
        stop_threshold: float,
        cfg: float,
        sigma: float,
        temperature: float,
        seed: int | None,
    ):
        self.segments = segments
        self.publish = publish
        self.fail = fail
        self.prompt_wav_lat = prompt_wav_lat
        self.ready_event: torch.cuda.Event | None = None
        self.min_steps = min_steps
        self.stop_threshold = stop_threshold
        self.sde_args = (cfg, sigma, temperature)
        self.seed = seed
        self.generator: torch.Generator | None = None
        self.cancelled = False
        self.row: int | None = None
        self.segment_index = 0
        self.step = 0
        self.num_steps = 0


class MingTalkerBatchEngine:
    """Producer thread that decodes all active talker requests together."""

    def __init__(
        self,
        generator,
        *,
        max_rows: int,
        device: torch.device,
        use_cuda_graphs: bool = True,
        compile_flow: bool = False,
    ):
        self.generator = generator
        self.use_cuda_graphs = use_cuda_graphs
        self.cfm = generator._cfm
        self.aggregator = generator._aggregator
        # A step is ~3k small kernels, most of them DiT elementwise ops; with
        # compile_flow, Inductor fuses them before the step graph is captured.
        # Compiling one DiT forward, shared by every ODE step, keeps
        # compilation short; its 3-token attention becomes fusable math and
        # its MLP GELU a GEMM epilogue.
        if compile_flow:
            for module in (*self.cfm.modules(), *self.aggregator.modules()):
                if hasattr(module, "short_attention"):
                    module.short_attention = True
                if hasattr(module, "fused_gelu"):
                    module.fused_gelu = True
        self._velocity = torch.compile(self.cfm.model.forward_with_cfg, dynamic=True) if compile_flow else None
        self._aggregate = torch.compile(self.aggregator, dynamic=True) if compile_flow else self.aggregator
        self.stop_head = generator._stop_head
        self.device = device
        param = next(generator._model.parameters())
        self.dtype = param.dtype
        self.max_cache_len = generator._max_cache_len
        # Inference tensors, like the activations the compiled step code
        # also receives, so its guards hold across ODE steps and buckets.
        with torch.inference_mode():
            self.llm = PackedQwen2(
                generator._model, rows=max_rows, max_len=self.max_cache_len, compile_layers=compile_flow
            )
            self.buckets = sorted({b for b in _BUCKETS if b < max_rows} | {max_rows})

            hidden = generator._model.config.hidden_size
            steps = generator._config.steps
            patch, his, z_dim = generator.patch_size, generator.his_patch_size, generator.latent_dim
            self.patch_size, self.his_patch_size, self.latent_dim = patch, his, z_dim
            self.steps = steps

            def buf(*shape, dtype=self.dtype):
                return torch.zeros(shape, device=device, dtype=dtype)

            self.hs = buf(max_rows, 1, hidden)
            self.his_lat = buf(max_rows, his, z_dim)
            self.y0 = buf(max_rows, patch, z_dim)
            self.sde_rnd = buf(max_rows, steps, patch, z_dim)
            self.sde_args = buf(3, max_rows, 1, 1)
            self.positions = buf(max_rows, dtype=torch.long)
            self.advance = buf(max_rows, dtype=torch.long)
            self.gen_lat = buf(max_rows, patch, z_dim)
            self.stop = buf(max_rows, 2)
            self.row_ids = torch.arange(max_rows, device=device, dtype=torch.long)
            self.cu_seqlens = torch.arange(max_rows + 1, device=device, dtype=torch.int32)
            self.t = get_epss_timesteps(steps, device=device, dtype=self.dtype)
        self.stop_host = torch.zeros(max_rows, dtype=torch.float32, device="cpu", pin_memory=True)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}

        self.cond = Condition()
        self.output_cond = Condition()
        self.pending: deque[BatchedTalkerRequest] = deque()
        self.free_rows = list(range(max_rows))
        self.rows: dict[int, BatchedTalkerRequest] = {}
        self.prefill: list[BatchedTalkerRequest] = []
        self.error: BaseException | None = None
        self.closed = False
        self.stream = torch.cuda.Stream(device)
        self.thread: Thread | None = None

    # ------------------------------------------------------------------
    # Client side (engine worker thread)

    def capture(self) -> None:
        """Capture before starting the producer or accepting requests."""
        if self.use_cuda_graphs:
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.inference_mode(), torch.cuda.stream(self.stream):
                self._capture_graphs()
            self.stream.synchronize()

    def start(self) -> None:
        """Start the producer once, after capture and memory profiling."""
        if self.closed:
            raise RuntimeError("Ming talker batch engine is closed")
        if self.thread is not None:
            return
        self.thread = Thread(target=self._run, name="ming-talker-batch", daemon=True)
        self.thread.start()

    def close(self) -> None:
        """Stop the producer and finish device work before releasing buffers."""
        with self.cond:
            self.closed = True
            self.cond.notify_all()
        if self.thread is not None:
            self.thread.join()
        self.stream.synchronize()

    def submit(self, request: BatchedTalkerRequest) -> None:
        """Publish a request after recording all caller-stream input preparation."""
        if self.error is not None:
            raise RuntimeError("Ming talker batch engine failed") from self.error
        request.ready_event = torch.cuda.Event()
        request.ready_event.record(torch.cuda.current_stream(self.device))
        with self.cond:
            self.pending.append(request)
            self.cond.notify()

    def cancel(self, request: BatchedTalkerRequest) -> None:
        request.cancelled = True
        with self.cond:
            self.cond.notify()

    def wait_for_output(self, predicate, timeout: float) -> None:
        """Block until ``predicate()`` holds, a step publishes, or timeout."""
        with self.output_cond:
            self.output_cond.wait_for(lambda: predicate() or self.error is not None, timeout=timeout)

    # ------------------------------------------------------------------
    # Graphs

    def _step_body(self, rows: int) -> None:
        hs = self.hs[:rows]
        his_lat = self.his_lat[:rows]
        gen_lat = self.cfm.sample(
            hs,
            his_lat,
            self.y0[:rows],
            self.t,
            self.sde_args[:, :rows],
            self.sde_rnd[:rows].transpose(0, 1),
            forward_with_cfg=self._velocity,
        )
        inputs_embeds = self._aggregate(gen_lat)
        self.stop[:rows].copy_(self.stop_head(hs[:, -1, :]).softmax(dim=-1))
        self.gen_lat[:rows].copy_(gen_lat)
        if self.his_patch_size == self.patch_size:
            his_lat.copy_(gen_lat)
        else:
            his_lat.copy_(torch.cat([his_lat[:, self.patch_size :], gen_lat], dim=1))
        positions = self.positions[:rows]
        hidden = self.llm.forward(
            inputs_embeds.reshape(rows, -1),
            positions,
            self.row_ids[:rows],
            self.cu_seqlens[: rows + 1],
            (positions + 1).to(torch.int32),
            self.llm.block_table[:rows],
            1,
        )
        hs.copy_(hidden.unsqueeze(1))
        positions.add_(self.advance[:rows])

    def _capture_graphs(self) -> None:
        pool = torch.cuda.graph_pool_handle()
        self.advance.zero_()
        if self._velocity is not None:
            # Populate the DiT's lazily built fused-QKV weights and rope caches
            # eagerly, so compilation traces no module mutation.
            self.cfm.sample(
                self.hs[:2],
                self.his_lat[:2],
                self.y0[:2],
                self.t,
                self.sde_args[:, :2],
                self.sde_rnd[:2].transpose(0, 1),
            )
            self.aggregator(self.gen_lat[:2])
        # Largest first so smaller buckets reuse its pool blocks.
        for rows in sorted(self.buckets, reverse=True):
            for _ in range(2):
                self._step_body(rows)
            self.graphs[rows], _ = _capture(lambda rows=rows: self._step_body(rows), pool=pool, stream=self.stream)
        self.positions.zero_()
        self.llm.capture_graphs(_PREFILL_BUCKETS, stream=self.stream, pool=pool)
        logger.info(
            "Captured Ming talker batch graphs for rows=%s, prefill tokens=%s", self.buckets, self.llm.token_buckets
        )

    # ------------------------------------------------------------------
    # Producer thread

    def _run(self) -> None:
        try:
            with torch.inference_mode(), torch.cuda.stream(self.stream):
                while True:
                    with self.cond:
                        while not self.closed and not self.pending and not self.rows:
                            self.cond.wait()
                        if self.closed:
                            return
                        incoming = list(self.pending)
                        self.pending.clear()
                    self._admit(incoming)
                    self._drop_cancelled()
                    if self.prefill:
                        self._run_prefill()
                    if any(request.num_steps for request in self.rows.values()):
                        self._decode_step()
                    with self.output_cond:
                        self.output_cond.notify_all()
        except BaseException as exc:  # noqa: BLE001 - reported to every request
            logger.exception("Ming talker batch engine failed")
            self.error = exc
            for request in [*self.rows.values(), *self.pending]:
                request.fail(exc)
            with self.output_cond:
                self.output_cond.notify_all()

    def _admit(self, incoming: list[BatchedTalkerRequest]) -> None:
        waiting = deque(incoming)
        while waiting:
            request = waiting.popleft()
            if request.cancelled:
                continue
            if not self.free_rows:
                # More requests than rows: keep the rest for a later step.
                with self.cond:
                    self.pending.extendleft(reversed([request, *waiting]))
                return
            assert request.ready_event is not None
            self.stream.wait_event(request.ready_event)
            # Inputs were allocated on the submitting stream; cancellation or
            # row release must not recycle them while this stream uses them.
            for segment in request.segments:
                segment.embeds.record_stream(self.stream)
            if request.prompt_wav_lat is not None and request.prompt_wav_lat.is_cuda:
                request.prompt_wav_lat.record_stream(self.stream)
            request.row = heapq.heappop(self.free_rows)
            if request.seed is not None:
                request.generator = torch.Generator(device=self.device)
                request.generator.manual_seed(request.seed)
            self.rows[request.row] = request
            self.prefill.append(request)

    def _release(self, request: BatchedTalkerRequest) -> None:
        row = request.row
        if row is None:
            return
        self.rows.pop(row, None)
        self.advance[row] = 0
        self.positions[row] = 0
        heapq.heappush(self.free_rows, row)
        request.row = None
        request.num_steps = 0

    def _drop_cancelled(self) -> None:
        for request in [r for r in self.rows.values() if r.cancelled]:
            if request in self.prefill:
                self.prefill.remove(request)
            self._release(request)

    def _run_prefill(self) -> None:
        batch, self.prefill = self.prefill, []
        ready = []
        for request in batch:
            segment = request.segments[request.segment_index]
            prompt_len = segment.embeds.shape[1]
            if prompt_len >= self.max_cache_len:
                request.fail(
                    ValueError(f"TTS prompt has {prompt_len} positions, exceeding cache limit {self.max_cache_len}")
                )
                self._release(request)
                continue
            request.step = 0
            request.num_steps = min(segment.max_steps, self.max_cache_len - prompt_len)
            ready.append(request)
        if not ready:
            return
        # Without prefill graphs one eager pass takes the whole batch.
        limit = max(self.llm.token_buckets, default=float("inf"))
        chunk: list[BatchedTalkerRequest] = []
        tokens = 0
        for request in ready:
            length = request.segments[request.segment_index].embeds.shape[1]
            if chunk and tokens + length > limit:
                self._prefill(chunk)
                chunk, tokens = [], 0
            chunk.append(request)
            tokens += length
        self._prefill(chunk)

    def _prefill(self, ready: list[BatchedTalkerRequest]) -> None:
        lengths = [r.segments[r.segment_index].embeds.shape[1] for r in ready]
        rows = [r.row for r in ready]
        hidden = self.llm.run([r.segments[r.segment_index].embeds[0] for r in ready], rows, [0] * len(ready))
        device = self.device
        last = torch.tensor(lengths).cumsum(0) - 1
        rows_d, lengths_d, last_d = torch.stack([torch.tensor(rows), torch.tensor(lengths), last]).to(
            device, non_blocking=True
        )
        self.hs.index_copy_(0, rows_d, hidden.index_select(0, last_d).unsqueeze(1))
        self.positions.index_copy_(0, rows_d, lengths_d)
        self.advance.index_fill_(0, rows_d, 1)
        his_lat = torch.cat(
            [self.generator._init_his_lat(r.prompt_wav_lat, device, self.dtype) for r in ready],
        )
        self.his_lat.index_copy_(0, rows_d, his_lat)
        sde_args = torch.tensor([r.sde_args for r in ready], dtype=torch.float32).T.contiguous()
        self.sde_args.index_copy_(1, rows_d, sde_args.to(device, self.dtype).view(3, -1, 1, 1))

    def _fill_noise(self, rows: int) -> None:
        self.y0[:rows].normal_()
        self.sde_rnd[:rows].normal_()
        shape = (1, self.patch_size, self.latent_dim)
        for row, request in self.rows.items():
            if request.generator is None or row >= rows:
                continue
            # Same draws, in the same order, as the single-request loop.
            self.y0[row : row + 1].copy_(
                torch.randn(shape, generator=request.generator, device=self.device, dtype=self.dtype)
            )
            self.sde_rnd[row].copy_(
                torch.randn(
                    (self.steps, *shape), generator=request.generator, device=self.device, dtype=self.dtype
                ).squeeze(1)
            )

    def _compact_rows(self) -> None:
        """Move live rows into a smaller graph bucket without copying KV.

        Run between steps on the producer stream. Graph buffers keep their
        addresses; only persistent state and KV ownership move. Swap the
        block-table entries so padding and reused rows cannot alias live KV.
        """
        count = len(self.rows)
        bucket = next(b for b in self.buckets if b >= count)
        if max(self.rows) < bucket:
            return
        sources = sorted((row for row in self.rows if row >= bucket), reverse=True)
        destinations = [row for row in range(bucket) if row not in self.rows][: len(sources)]
        dst, src = torch.tensor([destinations, sources], device=self.device, dtype=torch.long)
        for state in (self.hs, self.his_lat, self.positions, self.advance):
            state.index_copy_(0, dst, state.index_select(0, src))
        self.sde_args.index_copy_(1, dst, self.sde_args.index_select(1, src))
        swap_dst, swap_src = torch.cat([dst, src]), torch.cat([src, dst])
        table = self.llm.block_table
        table.index_copy_(0, swap_dst, table.index_select(0, swap_src))
        self.positions.index_fill_(0, src, 0)
        self.advance.index_fill_(0, src, 0)
        for destination, source in zip(destinations, sources, strict=True):
            request = self.rows.pop(source)
            request.row = destination
            self.rows[destination] = request
        self.free_rows = [row for row in range(self.hs.shape[0]) if row not in self.rows]

    def _decode_step(self) -> None:
        self._compact_rows()
        live = max(self.rows) + 1
        rows = next(b for b in self.buckets if b >= live)
        self._fill_noise(rows)
        if self.graphs:
            self.graphs[rows].replay()
        else:
            self._step_body(rows)
        latents = self.gen_lat[:rows].clone()
        self.stop_host[:rows].copy_(self.stop[:rows, 1].float(), non_blocking=True)
        torch.cuda.current_stream().synchronize()
        stop = self.stop_host[:rows].tolist()

        for row, request in list(self.rows.items()):
            if request.num_steps == 0 or request.cancelled:
                continue
            step = request.step
            stopped = step > request.min_steps and stop[row] > request.stop_threshold
            segment_done = stopped or step + 1 == request.num_steps
            final = segment_done and request.segment_index + 1 == len(request.segments)
            request.publish(latents[row : row + 1], final)
            request.step = step + 1
            if final:
                self._release(request)
            elif segment_done:
                # The next segment starts from a fresh prompt in the same row.
                request.segment_index += 1
                request.num_steps = 0
                self.advance[row] = 0
                self.prefill.append(request)


class BatchedAudioVAEDecoder:
    """Streaming AudioVAE decode of every request's new latents in one pass.

    The per-request VAE path runs the 24-layer sliding-window decoder once per
    request and runner step. Here each request keeps its own latent
    projection, streaming-upsampler and ISTFT overlap-add state, while the
    decoder runs once over the packed frames of all requests. Its keys live in
    a ring of KV blocks per request, since attention reaches back only one
    window.
    """

    RING_BLOCKS = 4
    PASS_FRAMES = max(_VAE_BUCKETS)

    def __init__(
        self,
        vae,
        *,
        rows: int,
        max_frames: int = 1 << 18,
        compile_layers: bool = False,
        use_cuda_graphs: bool = False,
    ):
        self.decoder = vae.decoder
        self.head = self.decoder.head
        backbone = self.decoder.decoder
        window = int(backbone.config.sliding_window)
        with torch.inference_mode():
            self.llm = PackedQwen2(
                backbone,
                rows=rows,
                max_len=max_frames,
                window=window,
                ring_blocks=self.RING_BLOCKS,
                compile_layers=compile_layers,
            )
        self.max_frames = self.llm.max_len
        self.max_chunk = self.RING_BLOCKS * PackedQwen2.BLOCK - window + 1
        param = next(backbone.parameters())
        self.device, self.dtype = param.device, param.dtype
        self.free_rows = list(range(rows))
        if use_cuda_graphs:
            self.llm.capture_graphs(_VAE_BUCKETS)

    def release(self, state: BatchedVAEState) -> None:
        row = state.pop("row", None)
        if row is not None:
            heapq.heappush(self.free_rows, row)

    @torch.inference_mode()
    def decode(self, items: list[tuple[BatchedVAEState, torch.Tensor, bool]]) -> list[torch.Tensor | ValueError]:
        """Decode ``(state, latents [1, n, latent_dim], last)`` items into audio.

        Invalid requests return a ValueError in their output slot. Device and
        internal failures propagate; advanced KV state cannot be retried.
        """
        audio: list[torch.Tensor | ValueError] = [
            torch.empty(1, 1, 0, device=self.device, dtype=self.dtype) for _ in items
        ]
        audio_parts: list[list[torch.Tensor]] = [[] for _ in items]
        frames: list[tuple[int, torch.Tensor]] = []
        for index, (state, latents, last) in enumerate(items):
            if latents.ndim != 3 or latents.shape[0] != 1 or latents.shape[1] == 0:
                audio[index] = ValueError("AudioVAE expects a nonempty [1, frames, latent_dim] chunk")
                self.release(state)
                continue
            position = state["position"] if "row" in state else 0
            upsample = state["upsample"] if "row" in state else None
            # The legacy upsampler mutates its dict. Commit it only after this
            # request passes validation; tensor contents are read-only.
            upsample = None if upsample is None else upsample.copy()
            x = self.decoder.fc1(latents.to(device=self.device, dtype=self.dtype))
            if self.decoder.patch_size != -1:
                x, upsample = self.decoder.upsampling(x, state=upsample, is_last=last)
            if x is not None and position + x.shape[1] > self.max_frames:
                audio[index] = ValueError(f"AudioVAE stream exceeds {self.max_frames} frames")
                self.release(state)
                continue
            if "row" not in state:
                if not self.free_rows:
                    raise RuntimeError("No free AudioVAE stream row")
                state.update(row=heapq.heappop(self.free_rows), position=0, upsample=None, istft=(None, None))
            state["upsample"] = upsample
            if x is not None:
                frames.append((index, x[0]))

        chunk_size = min(self.max_chunk, self.PASS_FRAMES)
        check_owners: list[int] = []
        invalid: set[int] = set()
        try:
            with deferred_envelope_checks() as checks:
                while frames:
                    take = [(index, x[:chunk_size]) for index, x in frames]
                    frames = [(index, x[chunk_size:]) for index, x in frames if x.shape[0] > chunk_size]
                    remaining = {index for index, _ in frames}
                    for group in self._groups(take):
                        lengths = [x.shape[0] for _, x in group]
                        states = [items[index][0] for index, _ in group]
                        hidden = self.llm.run(
                            [x for _, x in group],
                            [state["row"] for state in states],
                            [state["position"] for state in states],
                        )
                        # Consume this graph output before another replay overwrites it.
                        spectrum, _ = self.head.spectrum(hidden)
                        spectra = spectrum.split(lengths, dim=1)
                        for (index, _), spec, length in zip(group, spectra, lengths, strict=True):
                            state, _, last = items[index]
                            audio_buffer, window_buffer = state["istft"]
                            check_start = len(checks)
                            try:
                                out, audio_buffer, window_buffer = self.head.istft(
                                    spec.unsqueeze(0),
                                    audio_buffer=audio_buffer,
                                    window_buffer=window_buffer,
                                    streaming=True,
                                    last_chunk=last and index not in remaining,
                                )
                            except ISTFTEnvelopeError:
                                # Native fallback checks immediately. Discard this
                                # request's remaining frames without retrying its KV.
                                invalid.add(index)
                                frames = [(i, x) for i, x in frames if i != index]
                                continue
                            check_owners.extend([index] * (len(checks) - check_start))
                            audio_parts[index].append(out.unsqueeze(1))
                            state["istft"] = (audio_buffer, window_buffer)
                            state["position"] += length
        except ISTFTEnvelopeError as exc:
            if exc.indices is None:
                raise
            invalid.update(check_owners[index] for index in exc.indices)
        for index, (state, _, last) in enumerate(items):
            if index in invalid:
                audio[index] = ValueError(str(ISTFTEnvelopeError()))
            elif audio_parts[index]:
                audio[index] = torch.cat(audio_parts[index], dim=-1)
            if last or index in invalid:
                self.release(state)
        return audio

    def _groups(self, entries: list[tuple[int, torch.Tensor]]) -> list[list[tuple[int, torch.Tensor]]]:
        """Group already bounded chunks without exceeding the spectrum budget."""
        limit = self.PASS_FRAMES
        groups: list[list[tuple[int, torch.Tensor]]] = []
        total = 0
        for entry in entries:
            assert entry[1].shape[0] <= limit
            if groups and groups[-1] and total + entry[1].shape[0] <= limit:
                groups[-1].append(entry)
                total += entry[1].shape[0]
            else:
                groups.append([entry])
                total = entry[1].shape[0]
        return groups
