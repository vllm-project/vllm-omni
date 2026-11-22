# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MOSS Local request-slot state, independent of each step's batch order.

State writes, gathers and MTP replay run on the runner stream. Output gathers
own their storage; the runner retains its normal asynchronous output snapshot.
No request dictionary holds a view into a mutable slot or CUDA graph buffer.
"""

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.sampler import Sampler

from vllm_omni.model_executor.output_snapshot import PackedOutputSnapshot
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

logger = init_logger(__name__)


def _metadata_to_device(data: np.ndarray, device: torch.device) -> torch.Tensor:
    # CPU state tests need no CUDA allocator. The serving CUDA path retains
    # vLLM's asynchronous pinned upload and its ownership/lifetime handling.
    if device.type == "cpu":
        return torch.from_numpy(data)
    return async_tensor_h2d(data, device=device)


class _CodeRowsSnapshot(PackedOutputSnapshot):
    """Owned per-request code rows backed by one gathered device tensor.

    The generic snapshot flattens every per-request leaf, concatenates them
    into a slab and slices each back, then slices each again after the D2H.
    Here the gathered ``[decode_rows, n_vq]`` tensor is already owned, so the
    copy is a single D2H and the per-request views come from one ``split``.
    """

    def __init__(self, codes: torch.Tensor, decode_rows: list[int], num_reqs: int) -> None:
        dict.__init__(self, {"codes": {"audio": self._rows(codes, decode_rows, num_reqs)}})
        self._codes = codes
        self._decode_rows = decode_rows
        self._num_reqs = num_reqs
        self.producer_event = None

    @staticmethod
    def _rows(codes: torch.Tensor, decode_rows: list[int], num_reqs: int) -> list[torch.Tensor]:
        outputs = [codes.new_empty((0, codes.shape[-1]))] * num_reqs
        for batch_row, row in zip(decode_rows, codes.split(1), strict=True):
            outputs[batch_row] = row
        return outputs

    def copy_to_cpu(self, copy_tensor):
        cpu = copy_tensor(self._codes)
        return {"codes": {"audio": self._rows(cpu, self._decode_rows, self._num_reqs)}}


class MossLocalModelState(OmniModelState):
    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        self._init_slot_buffers(self.scheduler_config.max_num_seqs, model.hidden_size, device, self.dtype)
        self._batch_prefill = bool(getattr(model.config, "mrv2_batch_prefill", False))
        self._direct_tokens = bool(getattr(model.config, "mrv2_direct_tokens", False))
        self._local_eager_mtp = bool(getattr(model.config, "mrv2_eager_mtp", False))
        logger.info(
            "MOSS Local MRV2 batch prefill=%s, determined tokens=%s, eager MTP=%s",
            self._batch_prefill,
            self._direct_tokens,
            self._local_eager_mtp,
        )
        logger.info("MOSS Local MRV2 GPU slot state enabled: capacity=%d", self.scheduler_config.max_num_seqs)

    def _init_slot_buffers(self, capacity, hidden_size, device, dtype):
        self._hidden_pool = torch.zeros((capacity, hidden_size), dtype=dtype, device=device)
        self._codes_pool = torch.full(
            (capacity, self.model.n_vq), self.model.audio_pad_token_id, dtype=torch.long, device=device
        )
        # Execution eligibility mirrors audio_state.is_stopping. Binary MTP
        # continuation is a per-step output, not a sticky request-state update.
        self._active_pool = torch.zeros(capacity, dtype=torch.bool, device=device)
        self._active_host = [False] * capacity
        self._decode_rows = []
        self._prefill_staging = []
        self._batch_prefill = False
        self._direct_tokens = False
        # Eager MTP: a frame is sampled right after the forward that produced
        # its hidden state, one step earlier than the canonical preprocess
        # path. Its input-embedding contribution (audio embedding times emit
        # mask) is added to the next step's text embedding; ``_keep_pool``
        # chains emission so a stopped stream cannot emit again.
        self._local_eager_mtp = False
        self._eager_emb_pool = torch.zeros((capacity, hidden_size), dtype=dtype, device=device)
        self._keep_pool = torch.zeros(capacity, dtype=torch.bool, device=device)
        self._completing_rows = []
        self._local_eager_rows = []
        # A slot takes the eager path only after its final prefill chunk ran
        # an eager MTP in this state; others keep the canonical path, so a
        # reused slot never reads a previous request's frame state.
        self._eager_host = [False] * capacity
        self._mtp_dispatcher = None

    def sample_determined_tokens(self, batch, sampler):
        """Skip full-vocabulary sampling only for unconstrained Local tokens.

        The audio/binary MTP sampler has already made the random decision. The
        outer text distribution has exactly one finite logit. Keep the original
        path for processors that can mask it or request distribution metadata.
        """
        if not self._direct_tokens or type(sampler) is not Sampler:
            return None
        rows = batch.idx_mapping_np
        if (
            sampler.compute_nans
            or sampler.return_sampling_mask
            or sampler.trace_replay_state is not None
            or sampler.get_logprobs_dims(rows) is not None
            or (
                sampler.thinking_budget_state.enabled
                and np.any(sampler.thinking_budget_state.use_thinking_budget[rows])
            )
        ):
            return None
        for processor in sampler.logits_processors:
            if type(processor) is LogitBiasState:
                active = np.any(processor.use_logit_bias[rows])
            elif type(processor) is PenaltiesState:
                active = np.any(processor.use_penalty[rows])
            elif type(processor) is BadWordsState:
                active = np.any(processor.num_bad_words.np[rows])
            else:
                # Custom processors can alter even a single finite logit.
                return None
            if active:
                return None
        keep = self.model._batch_should_continue
        if keep is None or keep.numel() != batch.num_reqs:
            return None
        # Owned storage: subsequent forwards can replace the continue mask
        # while these token IDs are being copied or delivered asynchronously.
        tokens = torch.where(keep, self.model.audio_assistant_slot_token_id, self.model.im_end_token_id)
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            batch.seq_lens.new_ones(batch.num_reqs),
            batch.seq_lens,
            batch.cu_num_logits,
            batch.idx_mapping,
            sampler.req_states.prefill_len.gpu,
        )
        return SamplerOutput(tokens.reshape(-1, 1), None, None, num_sampled, num_rejected)

    def _apply_reference_batch(self, embeds, references):
        """One owned pinned upload for reference codes and their token positions.

        Host storage may be reused only after its H2D event completes. Device
        storage is consumed and overwritten in runner-stream order. If all
        cached slots are busy, allocate a transient snapshot instead of waiting.
        """
        if not references:
            return
        count = sum(codes.shape[0] for _, codes in references)
        capacity = max(count, self._static_inputs_embeds.shape[0])
        staging = None
        for item in self._prefill_staging:
            if item[0].shape[0] >= count and (item[2] is None or item[2].query()):
                staging = item
                break
        if staging is None:
            host = torch.empty((capacity, self.model.n_vq + 1), dtype=torch.long, pin_memory=embeds.is_cuda)
            device = torch.empty_like(host, device=embeds.device) if embeds.is_cuda else host
            staging = [host, device, torch.cuda.Event() if embeds.is_cuda else None]
            if len(self._prefill_staging) < 3:
                self._prefill_staging.append(staging)
        host, device, event = staging
        offset = 0
        for start, codes in references:
            end = offset + codes.shape[0]
            host[offset:end, 0].numpy()[:] = np.arange(start, start + codes.shape[0])
            host[offset:end, 1:].copy_(codes)
            offset = end
        if embeds.is_cuda:
            device[:count].copy_(host[:count], non_blocking=True)
            event.record(torch.cuda.current_stream(embeds.device))
        packed = device[:count]
        positions = packed[:, 0]
        audio = self.model._audio_embed(packed[:, 1:])
        embeds.index_copy_(0, positions, embeds.index_select(0, positions) + audio)

    def add_request(self, req_index, new_req_data):
        super().add_request(req_index, new_req_data)
        self._hidden_pool[req_index].zero_()
        self._codes_pool[req_index].fill_(self.model.audio_pad_token_id)
        self._active_pool[req_index].fill_(True)
        self._active_host[req_index] = True
        self._eager_host[req_index] = False
        self._keep_pool[req_index].fill_(False)
        self._eager_emb_pool[req_index].zero_()

    def remove_request(self, req_index_or_id):
        # Freeing is CPU bookkeeping. Reinitialization happens on admission,
        # in stream order, so pending output snapshots remain independent.
        super().remove_request(req_index_or_id)

    def _select_rows(self, tensor, rows):
        if rows == list(range(len(rows))):
            return tensor[: len(rows)]
        indices = _metadata_to_device(np.asarray(rows, dtype=np.int64), tensor.device)
        return tensor.index_select(0, indices)

    def run_preprocess(self, input_batch, model_inputs, req_states=None, mtp_batch_descriptor_dispatcher=None):
        input_ids = model_inputs.get("input_ids")
        if input_ids is None:
            input_ids = input_batch.input_ids
        embeds = model_inputs.get("inputs_embeds")
        if embeds is None:
            embeds = self.model.embed_input_ids(input_ids[: input_batch.num_tokens])
            model_inputs["inputs_embeds"] = embeds
        elif self._static_inputs_embeds is not None and embeds.data_ptr() == self._static_inputs_embeds.data_ptr():
            embeds[: input_batch.num_tokens].copy_(self.model.embed_input_ids(input_ids[: input_batch.num_tokens]))

        decode_rows = []
        completing_rows = []
        references = []
        for row in range(input_batch.num_reqs):
            slot = int(input_batch.idx_mapping_np[row])
            buf = self.intermediate_buffer.buffers[slot]
            if not buf or str(buf.get("req_id", "")).startswith("_warmup_"):
                continue
            start = int(input_batch.query_start_loc_np[row])
            count = int(input_batch.num_scheduled_tokens[row])
            prompt_len = (
                None if req_states is None else self._get_req_state_value(getattr(req_states, "prompt_len", None), slot)
            )
            computed = (
                None
                if req_states is None
                else self._get_req_state_value(getattr(req_states, "num_computed_tokens", None), slot)
            )
            if computed is None:
                computed = self._get_input_batch_num_computed(input_batch, slot, row)
            prefill = computed < prompt_len if computed is not None and prompt_len is not None else count > 1
            if prefill and computed is not None and prompt_len is not None and computed + count >= prompt_len:
                completing_rows.append(row)
            if count == 1 and not prefill and isinstance(buf.get("audio_state"), dict):
                active = not bool(buf["audio_state"].get("is_stopping"))
                if active != self._active_host[slot]:
                    self._active_pool[slot].fill_(active)
                    self._active_host[slot] = active
                decode_rows.append(row)
                continue
            if self._batch_prefill:
                ref = (buf.get("codes", {}) or {}).get("ref")
                ref_offset = int(buf.get("ref_offset", 0))
                # Device-side references keep the canonical path; do not add
                # an implicit D2H just to place them in a host staging buffer.
                if not isinstance(ref, torch.Tensor) or ref.device.type == "cpu":
                    if isinstance(ref, torch.Tensor) and ref.numel():
                        if ref.dim() == 1 and ref.numel() % self.model.n_vq == 0:
                            ref = ref.view(-1, self.model.n_vq)
                        if ref.dim() == 2:
                            chunk = ref[ref_offset : ref_offset + count]
                            if chunk.numel() and chunk.shape[0] == count:
                                references.append((start, chunk))
                    # Text embeddings have already been computed for all input
                    # tokens. Match the canonical prefill state transition.
                    self.intermediate_buffer.update(
                        slot,
                        {
                            "audio_state": {"is_stopping": False},
                            "ref_offset": ref_offset + count,
                        },
                        self.model.gpu_resident_buffer_keys,
                    )
                    if not self._active_host[slot]:
                        self._active_pool[slot].fill_(True)
                        self._active_host[slot] = True
                    continue
            # Reference conditioning and chunked-prefill coordinates retain
            # the model's canonical scalar implementation.
            info = dict(buf)
            info["_omni_is_prefill"] = prefill
            if prompt_len is not None:
                info["_omni_prompt_len"] = prompt_len
            if computed is not None:
                info["_omni_num_computed_tokens"] = computed
            ids = input_ids[start : start + count]
            previous = embeds[start : start + count]
            new_ids, new_embeds, updates = self.model.preprocess(ids, previous, **info)
            if "mtp_inputs" in updates:
                raise RuntimeError("MOSS slot state requires an admitted prefill before decode")
            if self._preprocess_result_needs_writeback(ids, new_ids):
                ids.copy_(new_ids)
            if self._preprocess_result_needs_writeback(previous, new_embeds):
                previous.copy_(new_embeds)
            self.intermediate_buffer.update(slot, updates, self.model.gpu_resident_buffer_keys)
            active = not bool(updates.get("audio_state", {}).get("is_stopping", False))
            if active != self._active_host[slot]:
                self._active_pool[slot].fill_(active)
                self._active_host[slot] = active

        self._apply_reference_batch(embeds, references)
        self._decode_rows = decode_rows
        if self._local_eager_mtp:
            self._completing_rows = completing_rows
            self._mtp_dispatcher = mtp_batch_descriptor_dispatcher
            eager_rows = [row for row in decode_rows if self._eager_host[int(input_batch.idx_mapping_np[row])]]
            canonical_rows = [row for row in decode_rows if not self._eager_host[int(input_batch.idx_mapping_np[row])]]
            self._local_eager_rows = eager_rows
            if eager_rows:
                # Their frame was sampled after the previous forward.
                slots = self._select_rows(input_batch.idx_mapping, eager_rows)
                offsets = self._mtp_offsets[: len(eager_rows)]
                offsets.copy_(self._select_rows(input_batch.query_start_loc, eager_rows))
                embeds.index_copy_(
                    0, offsets, embeds.index_select(0, offsets) + self._eager_emb_pool.index_select(0, slots)
                )
            if canonical_rows:
                self._run_slot_mtp(input_batch, input_ids, embeds, canonical_rows, mtp_batch_descriptor_dispatcher)
        elif decode_rows:
            self._run_slot_mtp(input_batch, input_ids, embeds, decode_rows, mtp_batch_descriptor_dispatcher)

    def _run_slot_mtp(self, batch, input_ids, embeds, rows, dispatcher):
        bsz = len(rows)
        slots = self._select_rows(batch.idx_mapping, rows)
        # Upstream query_start_loc is int32; index_copy_ requires int64.
        offsets = self._mtp_offsets[:bsz]
        offsets.copy_(self._select_rows(batch.query_start_loc, rows))
        ids, emb = self._mtp_input_ids[:bsz], self._mtp_input_embeds[:bsz]
        hidden, step = self._mtp_hidden[:bsz], self._mtp_text_step[:bsz]
        ids.copy_(input_ids.index_select(0, offsets))  # MRV2 token inputs are int32.
        torch.index_select(embeds, 0, offsets, out=emb)
        torch.index_select(self._hidden_pool, 0, slots, out=hidden)
        step[:, 0].copy_(self._active_pool.index_select(0, slots))

        buffers = [self.intermediate_buffer.buffers[int(batch.idx_mapping_np[row])] for row in rows]
        req_ids = [str(buf["req_id"]) for buf in buffers]
        generators = [
            self._get_mtp_generator(req_id, buf.get("sampling_params"), ids.device)
            for req_id, buf in zip(req_ids, buffers, strict=True)
        ]
        can_graph = self._is_mtp_graph_runner() and not any(g is not None for g in generators)
        desc = dispatcher(bsz) if can_graph and dispatcher is not None else None
        mode = getattr(desc, "cg_mode", CUDAGraphMode.FULL) if can_graph else CUDAGraphMode.NONE
        num_tokens = int(desc.num_tokens) if desc is not None else bsz
        with set_forward_context(
            None, self.vllm_config, num_tokens=num_tokens, cudagraph_runtime_mode=mode, batch_descriptor=desc
        ):
            new_emb, codes = self._call_mtp_with_sampling(
                self._mtp_input_ids[:num_tokens],
                self._mtp_input_embeds[:num_tokens],
                self._mtp_hidden[:num_tokens],
                self._mtp_text_step[:num_tokens],
                buffers=buffers,
                req_ids=req_ids,
                generators=generators,
            )
        embeds.index_copy_(0, offsets, new_emb[:bsz])
        self._codes_pool.index_copy_(0, slots, codes[:bsz])

    def _run_local_eager_mtp(self, batch, rows, completing):
        """Sample the next frame from this forward's hidden state.

        Same MTP inputs as the canonical next-step preprocess (hidden state
        and active flag); the input embedding is zero, so the returned
        embedding is exactly the audio contribution the next step adds to its
        text embedding. A decode row emits only if its previous frame did.
        """
        bsz = len(rows)
        slots = self._select_rows(batch.idx_mapping, rows)
        ids, emb = self._mtp_input_ids[:bsz], self._mtp_input_embeds[:bsz]
        hidden, step = self._mtp_hidden[:bsz], self._mtp_text_step[:bsz]
        ids.zero_()  # the MTP does not read token ids
        emb.zero_()
        torch.index_select(self._hidden_pool, 0, slots, out=hidden)
        gate = self._keep_pool.index_select(0, slots)
        if completing:
            first = _metadata_to_device(
                np.asarray([row in completing for row in rows], dtype=np.bool_),
                gate.device,
            )
            gate = gate | first
        step[:, 0].copy_(self._active_pool.index_select(0, slots) & gate)

        buffers = [self.intermediate_buffer.buffers[int(batch.idx_mapping_np[row])] for row in rows]
        req_ids = [str(buf["req_id"]) for buf in buffers]
        generators = [
            self._get_mtp_generator(req_id, buf.get("sampling_params"), ids.device)
            for req_id, buf in zip(req_ids, buffers, strict=True)
        ]
        dispatcher = self._mtp_dispatcher
        can_graph = self._is_mtp_graph_runner() and not any(g is not None for g in generators)
        desc = dispatcher(bsz) if can_graph and dispatcher is not None else None
        mode = getattr(desc, "cg_mode", CUDAGraphMode.FULL) if can_graph else CUDAGraphMode.NONE
        num_tokens = int(desc.num_tokens) if desc is not None else bsz
        with set_forward_context(
            None, self.vllm_config, num_tokens=num_tokens, cudagraph_runtime_mode=mode, batch_descriptor=desc
        ):
            contribution, codes = self._call_mtp_with_sampling(
                self._mtp_input_ids[:num_tokens],
                self._mtp_input_embeds[:num_tokens],
                self._mtp_hidden[:num_tokens],
                self._mtp_text_step[:num_tokens],
                buffers=buffers,
                req_ids=req_ids,
                generators=generators,
            )
        codes = codes[:bsz]
        self._eager_emb_pool.index_copy_(0, slots, contribution[:bsz])
        self._codes_pool.index_copy_(0, slots, codes)
        self._keep_pool.index_copy_(0, slots, codes.ne(self.model.audio_pad_token_id).any(dim=-1))

    def run_postprocess(self, hidden_states, input_batch):
        # query_start_loc is already device-resident and authoritative for
        # mixed/chunked-prefill batches. index_copy snapshots graph outputs.
        last = input_batch.query_start_loc[1 : input_batch.num_reqs + 1] - 1
        self._hidden_pool.index_copy_(
            0, input_batch.idx_mapping[: input_batch.num_reqs], hidden_states.index_select(0, last)
        )
        if self._local_eager_mtp:
            completing = self._completing_rows
            rows = sorted(self._local_eager_rows + completing)
            # Canonical decode rows emit their preprocess frame; eager rows
            # and completed prefills emit the frame sampled below.
            self._decode_rows = sorted(set(self._decode_rows) | set(completing))
            if rows:
                self._run_local_eager_mtp(input_batch, rows, set(completing))
                for row in completing:
                    self._eager_host[int(input_batch.idx_mapping_np[row])] = True

    def postprocess_model_output(self, model_output, input_batch, req_states):
        # Local backbone returns a tensor; keep the standard runner's output
        # snapshot/materializer contract and omit hidden payload transport.
        if not isinstance(model_output, torch.Tensor):
            raise TypeError("MOSS Local slot state requires a tensor backbone output")
        model = self.model
        model._batch_state = None
        model._batch_state_spans = None
        model._batch_should_continue = torch.ones(input_batch.num_reqs, device=model_output.device, dtype=torch.bool)
        if not self._decode_rows:
            return model_output, {}
        # This gather is an owned snapshot. Later slot writes cannot change
        # data already handed to the runner, even before its D2H completes.
        slots = self._select_rows(input_batch.idx_mapping, self._decode_rows)
        codes = self._codes_pool.index_select(0, slots)
        keep = codes.ne(model.audio_pad_token_id).any(dim=-1)
        if self._decode_rows == list(range(input_batch.num_reqs)):
            model._batch_should_continue = keep
        else:
            rows = _metadata_to_device(
                np.asarray(self._decode_rows, dtype=np.int64),
                codes.device,
            )
            model._batch_should_continue.index_copy_(0, rows, keep)
        return model_output, _CodeRowsSnapshot(codes, list(self._decode_rows), input_batch.num_reqs)
