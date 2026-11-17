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

import torch
from transformers import Qwen2Model
from transformers.models.qwen2.modeling_qwen2 import rotate_half
from vllm.logger import init_logger

from vllm_omni.model_executor.models.common.ming.cfm_solver import get_epss_timesteps

logger = init_logger(__name__)

_BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256)


class PackedQwen2:
    """Qwen2 forward over packed tokens with one KV row per request.

    Row ``r`` owns cache slots ``[r * max_len, (r + 1) * max_len)``. Prefill
    and decode both write K/V at ``row * max_len + position`` and attend
    through FlashAttention's paged interface, so rows of different lengths
    share one call. The HF modules supply the weights and the norm, rotary and
    MLP code; only attention and the cache layout are replaced.
    """

    BLOCK = 256

    def __init__(self, model: Qwen2Model, *, rows: int, max_len: int):
        from vllm.vllm_flash_attn import flash_attn_varlen_func, is_fa_version_supported

        cfg = model.config
        if getattr(cfg, "use_sliding_window", False):
            raise ValueError("Batched talker decode does not support sliding-window attention")
        self.model = model
        self.rows = rows
        self.max_len = -(-max_len // self.BLOCK) * self.BLOCK
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        param = next(model.parameters())
        blocks_per_row = self.max_len // self.BLOCK
        shape = (rows * blocks_per_row, self.BLOCK, self.num_kv_heads, self.head_dim)
        layers = len(model.layers)
        self.k_cache = [torch.zeros(shape, device=param.device, dtype=param.dtype) for _ in range(layers)]
        self.v_cache = [torch.zeros(shape, device=param.device, dtype=param.dtype) for _ in range(layers)]
        self.block_table = torch.arange(rows * blocks_per_row, device=param.device, dtype=torch.int32).view(rows, -1)
        self._attn = flash_attn_varlen_func
        self._fa_version = 3 if is_fa_version_supported(3) else 2

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
        slots = token_rows * self.max_len + positions
        cos, sin = model.rotary_emb(embeds, positions.unsqueeze(0))
        cos, sin = cos[0].unsqueeze(1), sin[0].unsqueeze(1)
        hidden = embeds
        for index, layer in enumerate(model.layers):
            attn = layer.self_attn
            x = layer.input_layernorm(hidden)
            q = attn.q_proj(x).view(-1, self.num_heads, self.head_dim)
            k = attn.k_proj(x).view(-1, self.num_kv_heads, self.head_dim)
            v = attn.v_proj(x).view(-1, self.num_kv_heads, self.head_dim)
            # Same arithmetic as transformers' apply_rotary_pos_emb.
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)
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
                softmax_scale=attn.scaling,
                fa_version=self._fa_version,
            )
            hidden = hidden + attn.o_proj(out.reshape(-1, self.num_heads * self.head_dim))
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
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

    def __init__(self, generator, *, max_rows: int, device: torch.device, use_cuda_graphs: bool = True):
        self.generator = generator
        self.use_cuda_graphs = use_cuda_graphs
        self.cfm = generator._cfm
        self.aggregator = generator._aggregator
        self.stop_head = generator._stop_head
        self.device = device
        param = next(generator._model.parameters())
        self.dtype = param.dtype
        self.max_rows = max_rows
        self.max_cache_len = generator._max_cache_len
        self.llm = PackedQwen2(generator._model, rows=max_rows, max_len=self.max_cache_len)
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
        self.stop_host = torch.zeros(max_rows, dtype=torch.float32, pin_memory=True)
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

    def start(self) -> None:
        """Capture every bucket's graph, then start the producer thread."""
        if self.use_cuda_graphs:
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.inference_mode(), torch.cuda.stream(self.stream):
                self._capture_graphs()
            self.stream.synchronize()
        self.thread = Thread(target=self._run, name="ming-talker-batch", daemon=True)
        self.thread.start()

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
        )
        inputs_embeds = self.aggregator(gen_lat)
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
        # Largest first so smaller buckets reuse its pool blocks.
        for rows in sorted(self.buckets, reverse=True):
            for _ in range(2):
                self._step_body(rows)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                self._step_body(rows)
            self.graphs[rows] = graph
        self.positions.zero_()
        logger.info("Captured Ming talker batch graphs for rows=%s", self.buckets)

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
        lengths = [r.segments[r.segment_index].embeds.shape[1] for r in ready]
        rows = torch.tensor([r.row for r in ready], dtype=torch.long)
        lengths_t = torch.tensor(lengths, dtype=torch.long)
        cu = torch.zeros(len(ready) + 1, dtype=torch.int32)
        cu[1:] = torch.cumsum(lengths_t, 0)
        embeds = torch.cat([r.segments[r.segment_index].embeds[0] for r in ready]).to(self.dtype)
        positions = torch.cat([torch.arange(n) for n in lengths])
        token_rows = torch.repeat_interleave(rows, lengths_t)

        device = self.device
        rows_d = rows.to(device, non_blocking=True)
        cu_d = cu.to(device, non_blocking=True)
        hidden = self.llm.forward(
            embeds,
            positions.to(device, non_blocking=True),
            token_rows.to(device, non_blocking=True),
            cu_d,
            lengths_t.to(device=device, dtype=torch.int32, non_blocking=True),
            self.llm.block_table.index_select(0, rows_d),
            max(lengths),
        )
        self.hs.index_copy_(0, rows_d, hidden.index_select(0, (cu_d[1:] - 1).long()).unsqueeze(1))
        self.positions.index_copy_(0, rows_d, lengths_t.to(device, non_blocking=True))
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

    def _decode_step(self) -> None:
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
