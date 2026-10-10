# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Request-local MRV2 state and post-primary continuation for Lychee-FD."""

from __future__ import annotations

import binascii
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import pybase64 as base64
import torch

from vllm_omni.model_executor.models.lychee_fd.audio_features import (
    SAMPLE_RATE_HZ,
    WINDOW_SAMPLES,
    log_mel_spectrogram,
    valid_mel_frames,
)
from vllm_omni.model_executor.models.lychee_fd.sampling import (
    LycheeControlMode,
    sample_control_tokens,
    sample_speech_tokens,
    speech_audio_token_id_max,
    update_control_modes,
    update_speaking_steps,
)
from vllm_omni.worker_v2.model_states.lychee_history import LycheePromptHistory, LycheeResidentHistory
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState


@dataclass(frozen=True)
class LycheeEncodedAudioWindow:
    seq: int
    start_tick: int
    embeddings: torch.Tensor


class LycheePoisonedRequestError(RuntimeError):
    """Raised before execution when a row contains an uncommitted Lychee tick."""


class LycheeModelState(OmniModelState):
    """Own Lychee's same-tick text -> merge -> speech continuation.

    The main forward produces text, control, and pre-merge speech hidden
    states. The standard MRV2 sampler chooses the text token. This state then
    builds the exact right-shifted text stream used by the released model,
    advances the merge branch under the original forward context, and emits a
    request-local three-channel token payload.
    """

    structured_output_via_multimodal_only = True
    _request_row_tensors = (
        "_last_text_tokens",
        "_text_eos_seen",
        "_text_generated_steps",
        "_last_stoken_tokens",
        "_last_control_tokens",
        "_control_modes",
        "_control_ticks",
        "_speaking_steps",
        "_tail_padding_until",
        "_tail_detect_enabled",
        "_speech_history",
        "_speech_history_lengths",
        "_prepared_ticks",
        "_committed_ticks",
        "_execution_epochs",
        "_audio_window_seqs",
        "_poisoned",
    )

    def custom_sampler(self, sampler: Any) -> tuple[Any, None]:
        from vllm_omni.worker_v2.model_states.lychee_sampler import LycheeSampler

        return LycheeSampler(sampler, self), None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        config = self.model.config
        self._last_text_tokens = torch.full(
            (self.max_num_reqs,),
            config.text_pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        self._text_eos_seen = torch.zeros(self.max_num_reqs, dtype=torch.bool, device=self.device)
        self._text_generated_steps = torch.zeros(self.max_num_reqs, dtype=torch.long, device=self.device)
        self._last_stoken_tokens = torch.full(
            (self.max_num_reqs,),
            config.stoken_pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        self._last_control_tokens = torch.full(
            (self.max_num_reqs,),
            config.sleep_token_id,
            dtype=torch.long,
            device=self.device,
        )
        self._control_modes = torch.full(
            (self.max_num_reqs,),
            int(LycheeControlMode.LISTENING),
            dtype=torch.int8,
            device=self.device,
        )
        self._control_ticks = torch.zeros(
            (self.max_num_reqs,),
            dtype=torch.long,
            device=self.device,
        )
        self._speaking_steps = torch.full(
            (self.max_num_reqs,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self._tail_padding_until = torch.full((self.max_num_reqs,), -1, dtype=torch.long, device=self.device)
        self._tail_detect_enabled = torch.zeros(self.max_num_reqs, dtype=torch.bool, device=self.device)
        speech_history_capacity = config.stoken_delay_num + 1 + config.stoken_max_tokens
        self._speech_history = torch.full(
            (self.max_num_reqs, speech_history_capacity),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self._speech_history_lengths = torch.zeros(self.max_num_reqs, dtype=torch.long, device=self.device)
        self._prepared_ticks = torch.full(
            (self.max_num_reqs,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self._committed_ticks = torch.full(
            (self.max_num_reqs,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self._execution_epochs = torch.full(
            (self.max_num_reqs,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self._audio_window_seqs = torch.full(
            (self.max_num_reqs,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self._poisoned = torch.zeros(
            (self.max_num_reqs,),
            dtype=torch.bool,
            device=self.device,
        )
        self._poisoned_rows: set[int] = set()
        self._next_execution_epoch = 0
        self._speech_generators: dict[str, torch.Generator] = {}
        self._rebind_state: dict[str, tuple[dict[str, torch.Tensor], bool]] = {}
        self._pending_append_text: set[str] = set()
        self._prompt_histories: dict[str, tuple[dict[str, Any], LycheePromptHistory | LycheeResidentHistory]] = {}
        self._resident_histories: dict[str, LycheeResidentHistory] = {}
        self._history_audio_cache: dict[str, OrderedDict[int, LycheeEncodedAudioWindow]] = {}
        self._audio_feature_owners: dict[str, int] = {}
        self._audio_feature_lru: OrderedDict[tuple[str, int], None] = OrderedDict()
        # Context bounds retained features, including suspended owners. A cache
        # miss always reuses original CPU PCM and the unchanged one-window encoder.
        # 8192 context / ten ticks retains at most 820 matrices per owner.
        # At 3584 BF16 this is 56.05 MiB/owner, 224.22 MiB for four slots.
        self._audio_feature_owner_limit = math.ceil(self.max_model_len / config.control_token_chunk_size)
        self._audio_feature_total_limit = self.max_num_reqs * self._audio_feature_owner_limit
        self._audio_feature_window_bytes = (
            config.control_token_chunk_size * config.text_config.hidden_size * torch.finfo(self.dtype).bits // 8
        )
        self._history_rng_draws: dict[str, int] = {}
        self._prepared_text_ids: torch.Tensor | None = None
        self._forced_listen_bindings: set[str] = set()
        # req_id -> (duplex append seq, control tick at window start,
        #             ten alternating audio-step embeddings)
        self._audio_window_cache: dict[str, tuple[int, torch.Tensor, torch.Tensor]] = {}

    def _reset_request_row(self, req_index: int) -> None:
        """Release slot-local state without touching a suspended session."""
        self._last_text_tokens[req_index] = self.model.config.text_pad_token_id
        self._text_eos_seen[req_index] = False
        self._text_generated_steps[req_index] = 0
        self._last_stoken_tokens[req_index] = self.model.config.stoken_pad_token_id
        self._last_control_tokens[req_index] = self.model.config.sleep_token_id
        self._control_modes[req_index] = int(LycheeControlMode.LISTENING)
        self._control_ticks[req_index] = 0
        self._speaking_steps[req_index] = -1
        self._tail_padding_until[req_index] = -1
        self._tail_detect_enabled[req_index] = False
        self._speech_history[req_index].fill_(-1)
        self._speech_history_lengths[req_index] = 0
        self._prepared_ticks[req_index] = -1
        self._committed_ticks[req_index] = -1
        self._execution_epochs[req_index] = -1
        self._audio_window_seqs[req_index] = -1
        self._poisoned[req_index] = False
        self._poisoned_rows.discard(req_index)

    def on_request_rebind(self, req_id: str, req_index: int) -> None:
        """Keep session state across MRV2's same-id streaming remove/add.

        Clones remain on the model device. The next binding may use another
        slot, so retaining views into the released row would be unsafe.
        """
        self._rebind_state[req_id] = (
            {name: getattr(self, name)[req_index].detach().clone() for name in self._request_row_tensors},
            req_index in self._poisoned_rows,
        )

    def add_request(self, req_index: int, new_req_data: Any) -> None:
        super().add_request(req_index, new_req_data)
        self._reset_request_row(req_index)
        req_id = new_req_data.req_id
        saved = self._rebind_state.pop(req_id, None)
        buffer = self.intermediate_buffer.buffers[req_index]
        duplex = buffer.get("duplex")
        meta = buffer.get("meta")
        replacing = isinstance(meta, dict) and meta.get("replace_streaming_prompt") is True
        if saved is None and not replacing:
            self._drop_audio_feature_owner(req_id)
        if replacing:
            rebuild = duplex.get("lychee_kv_rebuild") if isinstance(duplex, dict) else None
            if not isinstance(rebuild, dict) or rebuild.get("reason") != "natural_speech_eos":
                raise ValueError("Lychee KV replacement requires natural EOS history evidence")
            history = self._get_prompt_history(req_index, req_id)
            if not isinstance(history, LycheePromptHistory):
                raise ValueError("Lychee KV replacement requires full canonical history")
            eos_tick, frontier = rebuild.get("eos_tick"), rebuild.get("frontier_tick")
            if (
                type(eos_tick) is not int
                or type(frontier) is not int
                or not 0 < eos_tick <= frontier == history.logical_ticks[-1]
                or history.raw_speech[history.prefix_len + eos_tick] != self.model.config.tts_end_token_id
                or history.force_listen_at_frontier
            ):
                raise ValueError("Lychee KV replacement has invalid natural EOS/frontier evidence")
            # Native scheduler discarded every AR KV row. Do not consume the
            # saved streaming sample as a one-row append: replay all committed
            # inputs and let the normal chunked prefill restore their cursor.
            self._pending_append_text.discard(req_id)
        if saved is not None and not replacing:
            tensors, poisoned = saved
            self._ensure_speech_history_capacity(tensors["_speech_history"].numel())
            for name, tensor in tensors.items():
                target = getattr(self, name)[req_index]
                if name == "_speech_history":
                    target[: tensor.numel()].copy_(tensor)
                else:
                    target.copy_(tensor)
            if poisoned:
                self._poisoned_rows.add(req_index)
            duplex = self.intermediate_buffer.buffers[req_index].get("duplex")
            if isinstance(duplex, dict) and duplex.get("data_plane") is True:
                self._pending_append_text.add(req_id)
            return
        self._speech_history[req_index, 0] = self.model.config.stoken_pad_token_id
        self._speech_history_lengths[req_index] = 1
        self._execution_epochs[req_index] = self._next_execution_epoch
        self._next_execution_epoch += 1
        seed = getattr(getattr(new_req_data, "sampling_params", None), "seed", None)
        if not replacing or req_id not in self._speech_generators:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(0 if seed is None else int(seed))
            self._speech_generators[req_id] = generator
            self._history_rng_draws[req_id] = 0
        self._forced_listen_bindings.discard(req_id)
        duplex = self.intermediate_buffer.buffers[req_index].get("duplex")
        buffer = self.intermediate_buffer.buffers[req_index]
        has_bootstrap = "lychee_history" in buffer or (isinstance(duplex, dict) and "lychee_history" in duplex)
        history = self._get_prompt_history(req_index, req_id) if has_bootstrap else None
        if history is not None:
            self._execution_epochs[req_index] = history.execution_epoch
            self._next_execution_epoch = max(self._next_execution_epoch, history.execution_epoch + 1)
        self._audio_window_cache.pop(req_id, None)
        self._pending_append_text.discard(req_id)

    def remove_request(self, req_index_or_id: int | str) -> None:
        req_index = self._resolve_req_index(req_index_or_id)
        if req_index is not None:
            req_id = self.intermediate_buffer.buffers[req_index].get("req_id")
            self._reset_request_row(req_index)
            if req_id is not None and req_id not in self._rebind_state:
                self._speech_generators.pop(req_id, None)
                self._audio_window_cache.pop(req_id, None)
                self._pending_append_text.discard(req_id)
                self._prompt_histories.pop(req_id, None)
                self._resident_histories.pop(req_id, None)
                self._drop_audio_feature_owner(req_id)
                self._history_rng_draws.pop(req_id, None)
                self._forced_listen_bindings.discard(req_id)
        super().remove_request(req_index_or_id)

    def on_requests_finished(self, req_ids: set[str]) -> None:
        """Discard request-owned state even if the streaming slot was released."""
        super().on_requests_finished(req_ids)
        for req_id in req_ids:
            self._rebind_state.pop(req_id, None)
            self._speech_generators.pop(req_id, None)
            self._audio_window_cache.pop(req_id, None)
            self._pending_append_text.discard(req_id)
            self._prompt_histories.pop(req_id, None)
            self._resident_histories.pop(req_id, None)
            self._drop_audio_feature_owner(req_id)
            self._history_rng_draws.pop(req_id, None)
            self._forced_listen_bindings.discard(req_id)

    @staticmethod
    def _decode_audio_window(payload: object) -> torch.Tensor:
        if not isinstance(payload, dict):
            raise ValueError("Lychee duplex request is missing its audio payload")
        if payload.get("format") != "pcm_f32le" or payload.get("sample_rate_hz") != SAMPLE_RATE_HZ:
            raise ValueError(f"Lychee worker requires pcm_f32le audio at {SAMPLE_RATE_HZ} Hz")
        encoded = payload.get("audio")
        if not isinstance(encoded, str):
            raise ValueError("Lychee worker requires base64 audio")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Lychee worker received invalid base64 audio") from exc
        if len(raw) != WINDOW_SAMPLES * 4:
            raise ValueError(
                f"Lychee worker requires one padded {WINDOW_SAMPLES}-sample window, got {len(raw) // 4} samples"
            )
        return torch.frombuffer(bytearray(raw), dtype=torch.float32)

    def _audio_pad_rows(self, count: int) -> torch.Tensor:
        ids = torch.full((count,), self.model.config.audio_pad_token_id, dtype=torch.long, device=self.device)
        return self.model.embed_input_ids(ids).to(dtype=self.dtype)

    def _encode_audio_steps(self, payload: object) -> torch.Tensor:
        waveform = self._decode_audio_window(payload).to(device=self.device)
        mel = log_mel_spectrogram(waveform)
        # Feature extraction stays FP32; the model boundary follows the
        # configured inference dtype so BF16 checkpoints do not promote the
        # entire encoder and repeatedly materialize FP32 parameter copies.
        features = mel.unsqueeze(0).to(dtype=self.dtype)
        feature_lengths = torch.tensor(
            [valid_mel_frames(int(mel.shape[-1]))],
            dtype=torch.int32,
            device=self.device,
        )
        encoded, lengths = self.model.encode_audio(features, feature_lengths)
        encoded_length = int(lengths[0].item())
        expected_features = self.model.config.control_token_chunk_size // 2
        if encoded_length != expected_features:
            raise RuntimeError(
                "Lychee audio adaptor/window contract changed: "
                f"expected {expected_features} features, got {encoded_length}"
            )
        # The reference embeds the full alternating AudioPatch/AudioPad
        # token layout, replacing only AudioPatch rows with encoder features.
        # AudioPad is a learned shared embedding, not a zero feature row.
        step_embeddings = self._audio_pad_rows(self.model.config.control_token_chunk_size)
        step_embeddings[0::2] = encoded[0, :encoded_length].to(dtype=self.dtype)
        return step_embeddings

    def _prepare_audio_embeddings(self, input_batch: Any) -> torch.Tensor | None:
        """Select the request-local audio vector for each scheduled Lychee tick."""

        prepared: torch.Tensor | None = None
        total_rows = input_batch.num_tokens_after_padding
        for batch_index in range(input_batch.num_reqs):
            start = int(input_batch.query_start_loc_np[batch_index])
            end = int(input_batch.query_start_loc_np[batch_index + 1])
            req_state_index = int(input_batch.idx_mapping_np[batch_index])
            info = self.intermediate_buffer.buffers[req_state_index]
            req_id = input_batch.req_ids[batch_index]
            if info.get("req_id") != req_id:
                continue
            duplex = info.get("duplex")
            if not isinstance(duplex, dict):
                continue
            if end - start != 1:
                raise RuntimeError("Lychee single-GPU duplex expects one scheduled row per logical tick")
            seq = duplex.get("seq")
            payload = duplex.get("payload")
            ledger = payload.get("lychee_audio_ledger") if isinstance(payload, dict) else None
            if isinstance(ledger, dict) and "audio_window_seq" in ledger:
                seq = ledger["audio_window_seq"]
            if not isinstance(seq, int) or seq <= 0:
                raise ValueError(f"Lychee duplex append seq must be positive, got {seq!r}")
            cached = self._audio_window_cache.get(req_id)
            if cached is None or cached[0] != seq:
                step_embeddings = self._encode_audio_steps(payload)
                if isinstance(ledger, dict):
                    absolute_start = ledger.get("consumable_tick_start")
                    chunk_size = self.model.config.control_token_chunk_size
                    if type(absolute_start) is not int or absolute_start < 0 or absolute_start % chunk_size:
                        raise ValueError("Lychee audio ledger lacks an aligned absolute input start")
                    window_start_tick = self._control_ticks.new_tensor(absolute_start)
                else:
                    window_start_tick = self._control_ticks[req_state_index].detach().clone()
                cached = (seq, window_start_tick, step_embeddings)
                self._audio_window_cache[req_id] = cached
                self._audio_window_seqs[req_state_index] = seq
            _, window_start_tick, step_embeddings = cached
            relative_tick = self._control_ticks[req_state_index] - window_start_tick
            valid = (relative_tick >= 0) & (relative_tick < step_embeddings.shape[0])
            if not bool(valid):
                raise ValueError("Lychee input tick is outside its audio window; full channel history is required")
            selected = step_embeddings[relative_tick]
            if prepared is None:
                prepared = torch.zeros(
                    (total_rows, self.model.config.text_config.hidden_size),
                    dtype=self.dtype,
                    device=self.device,
                )
            prepared[start] = selected
        return prepared

    def _get_prompt_history(self, index: int, req_id: str) -> LycheePromptHistory | LycheeResidentHistory | None:
        info = self.intermediate_buffer.buffers[index]
        if info.get("req_id") != req_id:
            return None
        duplex = info.get("duplex")
        if not isinstance(duplex, dict):
            duplex = {}
        payload = info.get("lychee_history", duplex.get("lychee_history"))
        delta = duplex.get("lychee_audio_delta")
        if payload is not None and delta is not None:
            raise ValueError("Lychee request cannot combine full bootstrap and resident delta")
        cache = getattr(self, "_prompt_histories", None)
        if cache is None:
            cache = self._prompt_histories = {}
        resident = getattr(self, "_resident_histories", None)
        if resident is None:
            resident = self._resident_histories = {}
        saved = cache.get(req_id)
        if delta is not None:
            if saved is not None and saved[0] is duplex:
                return saved[1]
            binding = resident.get(req_id)
            if binding is None or req_id not in self._pending_append_text:
                raise ValueError("Lychee resident delta lacks its retained bootstrap owner; full rebuild required")
            history = binding.append_packet(
                duplex, req_id=req_id, window_ticks=self.model.config.control_token_chunk_size
            )
            resident[req_id] = history
            cache[req_id] = (duplex, history)
            return history
        if payload is None:
            # Legacy model-only inputs have no scheduler ledger. Native Duplex
            # packets must explicitly bootstrap or append a retained owner.
            audio = duplex.get("payload")
            if duplex.get("data_plane") is True and isinstance(audio, dict):
                raise ValueError("Native Lychee duplex append requires complete three-channel/audio history or delta")
            return None
        if saved is None or saved[0] is not payload:
            history = LycheePromptHistory.from_payload(
                payload,
                window_ticks=self.model.config.control_token_chunk_size,
                vocab_size=getattr(self.model.config.text_config, "vocab_size", None),
            )
            binding = LycheeResidentHistory.from_bootstrap(
                history, op_seq=duplex.get("seq", 1), session_epoch=duplex.get("epoch", history.execution_epoch)
            )
            self._bind_audio_feature_owner(req_id, history.execution_epoch)
            resident[req_id] = binding
            cache[req_id] = (payload, history)
            return history
        return saved[1]

    def _ensure_speech_history_capacity(self, required: int) -> None:
        """Keep the whole session n-gram evidence as the conversation grows."""
        capacity = self._speech_history.shape[1]
        if required <= capacity:
            return
        expanded = self._speech_history.new_full((self.max_num_reqs, max(required, capacity * 2)), -1)
        expanded[:, :capacity].copy_(self._speech_history)
        self._speech_history = expanded

    def _restore_history_cursor(self, req_id: str, index: int, history: LycheePromptHistory, tick: int) -> None:
        """Restore sampler state from committed evidence, without sampling it again."""
        config = self.model.config
        mode, step = int(LycheeControlMode.LISTENING), -1
        tail_padding_until = -1
        tail_detect_enabled = False
        required_draws = 0
        text_eos_seen, text_generated_steps = False, 0
        last_text, last_speech, last_control = (
            config.text_pad_token_id,
            config.stoken_pad_token_id,
            config.sleep_token_id,
        )
        for known_tick in range(1, tick + 1):
            position = history.prefix_len + known_tick
            last_text, last_speech, last_control = (
                history.text[position],
                history.speech[position],
                history.control[position],
            )
            raw_text = history.raw_text[position]
            raw_speech, raw_control = history.raw_speech[position], history.raw_control[position]
            if mode != int(LycheeControlMode.LISTENING) and not text_eos_seen:
                text_generated_steps += 1
                text_eos_seen = raw_text == getattr(config, "eos_token_id", -1)
            if (
                config.stoken_do_sample
                and config.stoken_temperature > 0
                and mode != int(LycheeControlMode.LISTENING)
                and config.stoken_delay_num < step < config.stoken_delay_num + config.stoken_max_tokens
            ):
                required_draws += 1
            new_mode = mode
            if raw_control == config.start_speaking_token_id:
                new_mode = int(LycheeControlMode.SPEAKING)
            elif raw_control == config.start_listening_token_id:
                new_mode = int(LycheeControlMode.LISTENING)
            elif raw_control == config.start_bc_token_id:
                new_mode = int(LycheeControlMode.BACKCHANNEL)
            if raw_speech == config.tts_end_token_id:
                new_mode = int(LycheeControlMode.LISTENING)
                tail_padding_until = (
                    (known_tick + 1) // config.control_token_chunk_size + 1
                ) * config.control_token_chunk_size - 1
                tail_detect_enabled = tail_padding_until - known_tick > 2
            if mode != new_mode:
                text_eos_seen, text_generated_steps = False, 0
            if new_mode == int(LycheeControlMode.LISTENING):
                step = -1
            elif mode != new_mode:
                step = 0
            else:
                step += 1
            mode = new_mode
        self._last_text_tokens[index] = last_text
        self._text_eos_seen[index], self._text_generated_steps[index] = text_eos_seen, text_generated_steps
        self._last_stoken_tokens[index] = last_speech
        self._last_control_tokens[index] = last_control
        self._control_modes[index], self._speaking_steps[index] = mode, step
        self._tail_padding_until[index] = tail_padding_until
        self._tail_detect_enabled[index] = tail_detect_enabled
        self._control_ticks[index] = tick + 1
        self._committed_ticks[index] = tick
        self._prepared_ticks[index] = tick
        speech_history = [value for value in history.speech[: history.prefix_len + tick + 1] if value is not None]
        self._ensure_speech_history_capacity(len(speech_history))
        self._speech_history[index].fill_(-1)
        self._speech_history[index, : len(speech_history)] = torch.tensor(
            speech_history, dtype=torch.long, device=self.device
        )
        self._speech_history_lengths[index] = len(speech_history)
        draws = self._history_rng_draws.get(req_id, 0)
        if required_draws < draws:
            # Native preemption/recompute replays earlier positions. Reset the
            # request stream so later sampling follows the same draw frontier.
            params = self.intermediate_buffer.buffers[index].get("sampling_params")
            seed = getattr(params, "seed", None)
            self._speech_generators[req_id].manual_seed(0 if seed is None else int(seed))
            draws = 0
        if required_draws > draws:
            probabilities = torch.ones(
                (1, 1 + speech_audio_token_id_max(config) - config.stoken_audio_token_id_min), device=self.device
            )
            for _ in range(draws, required_draws):
                # Only actual stochastic codec/EOS samples consume draws;
                # forced pad/delay/start/max prefixes leave the stream unchanged.
                torch.multinomial(probabilities, num_samples=1, generator=self._speech_generators[req_id])
        self._history_rng_draws[req_id] = required_draws

    def _audio_feature_cache_mutable(self) -> bool:
        return self.device.type != "cuda" or not torch.cuda.is_current_stream_capturing()

    def _drop_audio_feature_owner(self, req_id: str) -> None:
        for seq in self._history_audio_cache.pop(req_id, {}):
            self._audio_feature_lru.pop((req_id, seq), None)
        self._audio_feature_owners.pop(req_id, None)

    def _bind_audio_feature_owner(self, req_id: str, execution_epoch: int) -> None:
        if not self._audio_feature_cache_mutable():
            return
        previous = self._audio_feature_owners.get(req_id)
        if previous is not None and previous != execution_epoch:
            self._drop_audio_feature_owner(req_id)
        self._audio_feature_owners[req_id] = execution_epoch

    def _evict_audio_feature(self, req_id: str, seq: int) -> None:
        cache = self._history_audio_cache.get(req_id)
        if cache is not None:
            cache.pop(seq, None)
            if not cache:
                self._history_audio_cache.pop(req_id, None)
        self._audio_feature_lru.pop((req_id, seq), None)

    def _encoded_history_window(
        self, req_id: str, history: LycheePromptHistory | LycheeResidentHistory, window: dict[str, Any]
    ) -> torch.Tensor:
        """Retain immutable whole-window features; apply cancellation only at gather."""
        mutable = self._audio_feature_cache_mutable()
        if mutable:
            self._bind_audio_feature_owner(req_id, history.execution_epoch)
        elif self._audio_feature_owners.get(req_id) != history.execution_epoch:
            raise RuntimeError("Lychee graph capture requires an already prepared audio owner")
        seq, first = window["seq"], window["start_tick"]
        cache = self._history_audio_cache.get(req_id)
        entry = cache.get(seq) if cache is not None else None
        if entry is not None:
            if entry.seq != seq or entry.start_tick != first:
                raise ValueError("Lychee encoded audio sequence/start mismatch; full rebuild required")
            if mutable and cache is not None:
                cache.move_to_end(seq)
                self._audio_feature_lru.move_to_end((req_id, seq))
            return entry.embeddings
        if not mutable:
            raise RuntimeError("Lychee audio encode/cache insertion must run before CUDA graph capture")
        # Evict before encoding so retained ownership never exceeds either bound.
        if cache is not None and len(cache) >= self._audio_feature_owner_limit:
            self._evict_audio_feature(req_id, next(iter(cache)))
        while len(self._audio_feature_lru) >= self._audio_feature_total_limit:
            self._evict_audio_feature(*next(iter(self._audio_feature_lru)))
        embeddings = self._encode_audio_steps(window["payload"])
        expected = (self.model.config.control_token_chunk_size, self.model.config.text_config.hidden_size)
        if embeddings.shape != expected or embeddings.dtype != self.dtype or embeddings.device != self.device:
            raise ValueError("Lychee encoded audio matrix violates its configured shape/dtype/device")
        cache = self._history_audio_cache.setdefault(req_id, OrderedDict())
        # Own exactly this matrix's storage, even if an encoder returns a
        # view of a larger/reused output buffer. No feature copy crosses to CPU.
        retained = embeddings.detach().clone(memory_format=torch.contiguous_format)
        cache[seq] = LycheeEncodedAudioWindow(seq, first, retained)
        self._audio_feature_lru[(req_id, seq)] = None
        return cache[seq].embeddings

    def _history_audio_rows(
        self, history: LycheePromptHistory | LycheeResidentHistory, req_id: str, ticks: tuple[int, ...], index: int
    ) -> torch.Tensor:
        rows = torch.zeros(
            (len(ticks), self.model.config.text_config.hidden_size), dtype=self.dtype, device=self.device
        )
        # Dummy/capture batches may map to a live slot while carrying another
        # request identity. They must not consume that owner's history or pool.
        if self.intermediate_buffer.buffers[index].get("req_id") != req_id:
            return rows
        chunk_size = self.model.config.control_token_chunk_size
        selected: dict[int, list[tuple[int, int]]] = {}
        pad_embedding: torch.Tensor | None = None
        for row, tick in enumerate(ticks):
            if tick < 0:
                continue
            first = tick // chunk_size * chunk_size
            if first not in history.audio_by_start:
                raise ValueError(f"No committed audio evidence for input tick {tick}")
            window = history.audio_by_start[first]
            self._audio_window_seqs[index] = window["seq"]
            # Input cancellation preserves the original PCM for every encoded
            # KV row, but uncomputed input ticks become AUDIO_PAD. The cutoff
            # is the sampled output frontier minus one, not the output tick.
            cutoff = window.get("discard_after_tick")
            if cutoff is not None and tick > cutoff:
                if pad_embedding is None:
                    pad_embedding = self._audio_pad_rows(1)[0]
                rows[row] = pad_embedding
                continue
            selected.setdefault(first, []).append((row, tick - first))
        for first, matching in selected.items():
            window = history.audio_by_start[first]
            embeddings = self._encoded_history_window(req_id, history, window)
            for row, offset in matching:
                rows[row] = embeddings[offset]
        return rows

    def prepare_inputs(self, input_batch: Any, req_states: Any) -> dict[str, Any]:
        """Build aligned request-local speech/control inputs for this tick."""

        poisoned_req_ids = [
            input_batch.req_ids[batch_index]
            for batch_index in range(input_batch.num_reqs)
            if int(input_batch.idx_mapping_np[batch_index]) in self._poisoned_rows
            and self.intermediate_buffer.buffers[int(input_batch.idx_mapping_np[batch_index])].get("req_id")
            == input_batch.req_ids[batch_index]
        ]
        if poisoned_req_ids:
            raise LycheePoisonedRequestError(
                "Refusing to reuse Lychee rows with an uncommitted tick: " + ", ".join(poisoned_req_ids)
            )
        inputs = super().prepare_inputs(input_batch, req_states)
        config = self.model.config
        total_rows = input_batch.num_tokens_after_padding
        stoken_ids = torch.full(
            (total_rows,),
            config.stoken_pad_token_id,
            dtype=torch.long,
            device=input_batch.input_ids.device,
        )
        control_ids = torch.full(
            (total_rows,),
            config.sleep_token_id,
            dtype=torch.long,
            device=input_batch.input_ids.device,
        )
        stoken_mask = torch.ones(total_rows, dtype=torch.bool, device=input_batch.input_ids.device)
        control_mask = stoken_mask.clone()
        audio_embeddings = None
        text_ids = input_batch.input_ids.clone()
        uses_history = False
        for batch_index in range(input_batch.num_reqs):
            computed = int(input_batch.num_computed_tokens_np[batch_index])
            prefill_len = int(input_batch.prefill_len_np[batch_index])
            start = int(input_batch.query_start_loc_np[batch_index])
            end = int(input_batch.query_start_loc_np[batch_index + 1])
            req_state_index = int(input_batch.idx_mapping_np[batch_index])
            req_id = input_batch.req_ids[batch_index]
            if self.intermediate_buffer.buffers[req_state_index].get("req_id") != req_id:
                continue
            history = self._get_prompt_history(req_state_index, req_id)
            info = self.intermediate_buffer.buffers[req_state_index]
            duplex = info.get("duplex")
            payload = duplex.get("payload") if isinstance(duplex, dict) else None
            if (
                history is None
                and isinstance(duplex, dict)
                and duplex.get("data_plane") is True
                and isinstance(payload, dict)
                and isinstance(payload.get("lychee_audio_ledger"), dict)
            ):
                raise ValueError("Native Lychee duplex append requires complete three-channel/audio history")
            if history is not None and req_id in self._pending_append_text and computed < prefill_len:
                # The host may still be materializing the previous segment.
                # Its new buffer can contain a lagging committed snapshot, but
                # this same request owns retained KV and the live channel/RNG
                # cursor. Upstream replaces the uncomputed final sample with
                # this one placeholder; consume the saved sample exactly once.
                if computed <= 0 or end - start != 1 or computed + 1 != prefill_len:
                    raise RuntimeError("Lychee native append expects one new token over its retained context")
                tick = computed - history.prefix_len
                if tick < 0:
                    raise ValueError(
                        "Lychee retained KV does not align with its live channel frontier: "
                        f"computed={computed}, prefix={history.prefix_len}, tick={tick}"
                    )
                uses_history = True
                text_ids[start:end] = self._last_text_tokens[req_state_index]
                stoken_ids[start:end] = self._last_stoken_tokens[req_state_index]
                control_ids[start:end] = self._last_control_tokens[req_state_index]
                if audio_embeddings is None:
                    audio_embeddings = torch.zeros(
                        (total_rows, config.text_config.hidden_size), dtype=self.dtype, device=self.device
                    )
                audio_embeddings[start:end] = self._history_audio_rows(history, req_id, (tick,), req_state_index)
                self._pending_append_text.discard(req_id)
                continue
            if history is not None and computed < prefill_len:
                if isinstance(history, LycheeResidentHistory):
                    raise ValueError("Lychee resident delta cannot reconstruct missing KV; full rebuild required")
                uses_history = True
                if end - start + computed > len(history.text):
                    raise ValueError(
                        "Lychee scheduled prefill exceeds its explicit channel evidence: "
                        f"req_id={req_id}, computed={computed}, rows={end - start}, "
                        f"prefill_len={prefill_len}, history_rows={len(history.text)}, "
                        f"history_frontier={history.logical_ticks[-1]}"
                    )
                stop = computed + end - start
                text_ids[start:end] = torch.tensor(history.text[computed:stop], device=text_ids.device)
                for values, target, mask, pad in (
                    (history.speech, stoken_ids, stoken_mask, config.stoken_pad_token_id),
                    (history.control, control_ids, control_mask, config.sleep_token_id),
                ):
                    target[start:end] = torch.tensor(
                        [pad if value is None else value for value in values[computed:stop]], device=target.device
                    )
                    mask[start:end] = torch.tensor(
                        [value is not None for value in values[computed:stop]], device=target.device
                    )
                ticks = history.logical_ticks[computed:stop]
                if ticks[-1] >= 0:
                    self._restore_history_cursor(req_id, req_state_index, history, ticks[-1])
                if audio_embeddings is None:
                    audio_embeddings = torch.zeros(
                        (total_rows, config.text_config.hidden_size), dtype=self.dtype, device=self.device
                    )
                audio_embeddings[start:end] = self._history_audio_rows(history, req_id, ticks, req_state_index)
                if (
                    history.force_listen_at_frontier
                    and stop == prefill_len
                    and req_id not in self._forced_listen_bindings
                ):
                    # Only the frontier input has no KV yet. Preserve every
                    # historical model row and its raw merge sample, then apply
                    # the session owner's cancel/recovery policy at this row.
                    frontier = end - 1
                    text_ids[frontier] = config.text_pad_token_id
                    stoken_ids[frontier] = config.stoken_pad_token_id
                    control_ids[frontier] = config.sleep_token_id
                    stoken_mask[frontier] = control_mask[frontier] = True
                    self._last_text_tokens[req_state_index] = config.text_pad_token_id
                    self._last_stoken_tokens[req_state_index] = config.stoken_pad_token_id
                    self._last_control_tokens[req_state_index] = config.sleep_token_id
                    self._control_modes[req_state_index] = int(LycheeControlMode.LISTENING)
                    self._speaking_steps[req_state_index] = -1
                    self._text_eos_seen[req_state_index] = False
                    self._text_generated_steps[req_state_index] = 0
                    # Cancel changes only this uncomputed frontier input.
                    # Earlier speech IDs remain part of session n-gram evidence.
                    history_length = int(self._speech_history_lengths[req_state_index].item())
                    self._speech_history[req_state_index, history_length - 1] = config.stoken_pad_token_id
                    self._tail_padding_until[req_state_index] = -1
                    self._tail_detect_enabled[req_state_index] = False
                    self._forced_listen_bindings.add(req_id)
                self._pending_append_text.discard(req_id)
                continue
            if history is not None:
                # A decode row consumes the previous sampled token. Its audio
                # tick is one behind the output/control sampling frontier.
                uses_history = True
                if end - start != 1:
                    raise ValueError("Lychee decode must consume one model position per request")
                tick = computed - history.prefix_len
                if audio_embeddings is None:
                    audio_embeddings = torch.zeros(
                        (total_rows, config.text_config.hidden_size), dtype=self.dtype, device=self.device
                    )
                audio_embeddings[start:end] = self._history_audio_rows(history, req_id, (tick,), req_state_index)
            pending_append = req_id in getattr(self, "_pending_append_text", set())
            append_frontier = pending_append and computed < prefill_len
            if pending_append and computed >= prefill_len:
                self._pending_append_text.discard(req_id)
            if append_frontier:
                if computed <= 0 or end - start != 1 or computed + 1 != prefill_len:
                    raise RuntimeError("Lychee native append expects one new token over its retained context")
                # The scheduler drops the last uncomputed sample at a streaming
                # boundary and appends a placeholder. Consume the saved sample
                # once so text, speech and control retain their right shift.
                if "input_ids" not in inputs:
                    inputs["input_ids"] = input_batch.input_ids.clone()
                inputs["input_ids"][start] = self._last_text_tokens[req_state_index]
            if computed < prefill_len and not append_frontier:
                continue
            stoken_ids[start:end] = self._last_stoken_tokens[req_state_index]
            control_ids[start:end] = self._last_control_tokens[req_state_index]
        inputs["stoken_input_ids"] = stoken_ids
        inputs["control_input_ids"] = control_ids
        if uses_history:
            inputs["input_ids"] = text_ids
            inputs["stoken_input_mask"] = stoken_mask
            inputs["control_input_mask"] = control_mask
        else:
            audio_embeddings = self._prepare_audio_embeddings(input_batch)
        self._prepared_text_ids = inputs.get("input_ids", input_batch.input_ids)
        if audio_embeddings is not None:
            inputs["audio_embeddings"] = audio_embeddings
        return inputs

    def _runtime_sampling_config(self, index: int) -> dict[str, Any]:
        info = self.intermediate_buffer.buffers[index]
        duplex = info.get("duplex")
        configured = duplex.get("runtime_config", {}) if isinstance(duplex, dict) else {}
        if not isinstance(configured, dict):
            raise ValueError("Lychee request sampling configuration must be a mapping")
        defaults = {
            "allowing_backchannel": True,
            "start_speak_token_factor": 1.2,
            "start_listen_token_factor": 1.2,
            "backchannel_token_bias": 1.0,
            "end_speak_token_factor": 1.0,
            "text_max_tokens": 128,
        }
        values = {name: configured.get(name, default) for name, default in defaults.items()}
        if type(values["allowing_backchannel"]) is not bool:
            raise ValueError("Lychee allowing_backchannel must be a bool")
        if type(values["text_max_tokens"]) is not int or values["text_max_tokens"] <= 0:
            raise ValueError("Lychee text_max_tokens must be a positive integer")
        for name in (
            "start_speak_token_factor",
            "start_listen_token_factor",
            "backchannel_token_bias",
            "end_speak_token_factor",
        ):
            value = values[name]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Lychee {name} must be finite")
            if name != "backchannel_token_bias" and value <= 0:
                raise ValueError(f"Lychee {name} must be positive")
        return values

    def constrain_primary_logits(self, logits: torch.Tensor, input_batch: Any) -> torch.Tensor:
        """Apply released per-response text EOS/pad policy before native sampling."""
        if logits.shape[0] != input_batch.num_reqs or not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Invalid Lychee primary logits; abort the native transaction")
        config = self.model.config
        settings = [
            self._runtime_sampling_config(int(index)) for index in input_batch.idx_mapping_np[: input_batch.num_reqs]
        ]
        indices = input_batch.idx_mapping[: input_batch.num_reqs]
        limits = torch.tensor([value["text_max_tokens"] for value in settings], dtype=torch.long, device=logits.device)
        # BF16 scalar multiplication uses FP32 opmath. Storing the factor in
        # BF16 first would change the released policy's rounding.
        factor_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        factors = torch.tensor(
            [value["end_speak_token_factor"] for value in settings], dtype=factor_dtype, device=logits.device
        )
        forced = torch.where(
            self._control_modes[indices] == int(LycheeControlMode.LISTENING),
            config.text_pad_token_id,
            torch.where(
                self._text_eos_seen[indices],
                config.tts_pad_token_id,
                torch.where(self._text_generated_steps[indices] + 1 >= limits, config.eos_token_id, -1),
            ),
        )
        constrained = logits.clone()
        constrained[:, config.eos_token_id] = torch.where(
            forced < 0, logits[:, config.eos_token_id].to(factor_dtype) * factors, logits[:, config.eos_token_id]
        ).to(logits.dtype)
        columns = forced.clamp_min(0).unsqueeze(1)
        restored = torch.where(forced.unsqueeze(1) >= 0, logits.gather(1, columns), constrained[:, :1])
        constrained.masked_fill_(forced.unsqueeze(1) >= 0, float("-inf"))
        constrained.scatter_(1, columns, restored)
        return constrained

    def constrain_primary_sample(
        self,
        *,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        input_batch: Any,
    ) -> torch.Tensor:
        """Force listening rows to emit the released text-pad token."""

        self._prepare_tick(input_batch=input_batch, num_sampled=num_sampled)
        constrained = sampled_token_ids.clone()
        req_state_indices = input_batch.idx_mapping[: input_batch.num_reqs]
        listening = self._control_modes[req_state_indices] == int(LycheeControlMode.LISTENING)
        forced = torch.where(
            listening,
            constrained.new_tensor(self.model.config.text_pad_token_id),
            constrained[:, 0],
        )
        constrained[:, 0] = torch.where(
            num_sampled[: input_batch.num_reqs] > 0,
            forced,
            constrained[:, 0],
        )
        return constrained

    def _prepare_tick(self, *, input_batch: Any, num_sampled: torch.Tensor) -> None:
        """Mark sampled rows prepared after main KV has advanced."""

        req_state_indices = input_batch.idx_mapping[: input_batch.num_reqs]
        active = num_sampled[: input_batch.num_reqs] > 0
        next_ticks = self._committed_ticks[req_state_indices] + 1
        self._prepared_ticks[req_state_indices] = torch.where(
            active,
            next_ticks,
            self._prepared_ticks[req_state_indices],
        )

    def mark_primary_continuation_failed(
        self,
        *,
        input_batch: Any,
        **_: Any,
    ) -> None:
        """Invalidate every scheduled row after a post-main failure."""

        req_state_indices = input_batch.idx_mapping[: input_batch.num_reqs]
        self._poisoned[req_state_indices] = True
        epoch = self._next_execution_epoch
        self._next_execution_epoch += 1
        self._execution_epochs[req_state_indices] = epoch
        for batch_index in range(input_batch.num_reqs):
            self._poisoned_rows.add(int(input_batch.idx_mapping_np[batch_index]))

    def abort_failed_step(self, *, req_ids: list[str], exception: Exception, phase: str) -> dict[str, str] | None:
        """Quarantine the entire touched batch and expose a terminal failure.

        Python/model errors can be isolated after stream synchronization.
        Device context corruption cannot be recovered by freeing a request.
        """
        fatal_device_markers = (
            "cuda error",
            "device-side assert",
            "illegal memory access",
            "unspecified launch failure",
            "nccl",
            "cublas_status_execution_failed",
        )
        if not req_ids or any(marker in str(exception).lower() for marker in fatal_device_markers):
            return None
        epoch = self._next_execution_epoch
        self._next_execution_epoch += 1
        for req_id in req_ids:
            index = self.intermediate_buffer.req_id_to_index.get(req_id)
            if index is not None:
                self._poisoned[index] = True
                self._poisoned_rows.add(index)
                self._execution_epochs[index] = epoch
        return {
            req_id: f"Lychee {phase} transaction aborted; rebuild required; execution_epoch={epoch}: "
            f"{type(exception).__name__}: {exception}"
            for req_id in req_ids
        }

    def _commit_tick(
        self,
        *,
        req_state_indices: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        """Publish the prepared tick only after every continuation step succeeds."""

        committed = self._committed_ticks[req_state_indices]
        prepared = self._prepared_ticks[req_state_indices]
        self._committed_ticks[req_state_indices] = torch.where(
            active,
            prepared,
            committed,
        )

    @staticmethod
    def build_merge_text_token_ids(
        *,
        input_batch: Any,
        req_states: Any,
        sampled_text_token_ids: torch.Tensor,
        text_pad_token_id: int,
        total_rows: int,
        prepared_input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build next-text conditioning for every scheduled model position.

        Rows inside a prefill chunk use the following known prompt token. The
        last row of an incomplete chunk also uses the next prompt token from
        request state. Only a row that reaches the current sequence frontier
        consumes the text token sampled in this MRV2 step.
        """

        merge_ids = torch.full(
            (total_rows,),
            text_pad_token_id,
            dtype=torch.long,
            device=sampled_text_token_ids.device,
        )
        input_ids = input_batch.input_ids if prepared_input_ids is None else prepared_input_ids
        all_token_ids = req_states.all_token_ids.gpu
        for batch_index in range(input_batch.num_reqs):
            start = int(input_batch.query_start_loc_np[batch_index])
            count = int(input_batch.num_scheduled_tokens[batch_index])
            if count <= 0:
                continue
            end = start + count
            if count > 1:
                merge_ids[start : end - 1] = input_ids[start + 1 : end].to(torch.long)

            computed = int(input_batch.num_computed_tokens_np[batch_index])
            prefill_len = int(input_batch.prefill_len_np[batch_index])
            if computed + count < prefill_len:
                req_state_index = int(input_batch.idx_mapping_np[batch_index])
                merge_ids[end - 1] = all_token_ids[req_state_index, computed + count].to(torch.long)
            else:
                merge_ids[end - 1] = sampled_text_token_ids[batch_index, 0].to(torch.long)
        return merge_ids

    @staticmethod
    def sample_legal_greedy(
        logits: torch.Tensor,
        *,
        token_id_min: int,
        token_id_max: int,
    ) -> torch.Tensor:
        """Greedily sample an exclusive-upper-bound Lychee token range."""

        if token_id_min < 0 or token_id_max <= token_id_min:
            raise ValueError(f"Invalid legal token range [{token_id_min}, {token_id_max})")
        if token_id_max > logits.shape[-1]:
            raise ValueError(f"Legal token range ends at {token_id_max}, vocab is {logits.shape[-1]}")
        return logits[:, token_id_min:token_id_max].argmax(dim=-1) + token_id_min

    @staticmethod
    def _request_tokens_on_model_rows(
        token_ids: torch.Tensor,
        *,
        input_batch: Any,
        num_sampled: torch.Tensor,
        total_rows: int,
    ) -> torch.Tensor:
        """Place one request token at its final model row; use -1 elsewhere."""

        result = torch.full(
            (total_rows,),
            -1,
            dtype=torch.int32,
            device=token_ids.device,
        )
        for batch_index in range(input_batch.num_reqs):
            end = int(input_batch.query_start_loc_np[batch_index + 1])
            active_token = torch.where(
                num_sampled[batch_index] > 0,
                token_ids[batch_index].to(torch.int32),
                token_ids.new_tensor(-1, dtype=torch.int32),
            )
            result[end - 1] = active_token
        return result

    def continue_after_primary_sample(
        self,
        *,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        multimodal_outputs: dict[str, Any],
        input_batch: Any,
        req_states: Any,
    ) -> dict[str, Any]:
        """Finish one Lychee tick after MRV2's standard text sample."""

        if input_batch.num_draft_tokens != 0:
            raise NotImplementedError("Lychee-FD post-primary continuation does not support speculative decoding")
        try:
            stoken_hidden = multimodal_outputs["lychee_stoken_hidden"]
            control_hidden = multimodal_outputs["lychee_control_hidden"]
        except KeyError as exc:
            raise RuntimeError("Lychee main forward did not return both continuation hidden streams") from exc

        total_rows = int(stoken_hidden.shape[0])
        if control_hidden.shape[0] != total_rows:
            raise RuntimeError(
                "Lychee continuation streams disagree on the model-row axis: "
                f"speech={total_rows}, control={control_hidden.shape[0]}"
            )

        config = self.model.config
        merge_text_ids = self.build_merge_text_token_ids(
            input_batch=input_batch,
            req_states=req_states,
            sampled_text_token_ids=sampled_token_ids,
            text_pad_token_id=config.text_pad_token_id,
            total_rows=total_rows,
            prepared_input_ids=getattr(self, "_prepared_text_ids", None),
        )
        for batch_index in range(input_batch.num_reqs):
            computed = int(input_batch.num_computed_tokens_np[batch_index])
            count = int(input_batch.num_scheduled_tokens[batch_index])
            history = self._get_prompt_history(
                int(input_batch.idx_mapping_np[batch_index]), input_batch.req_ids[batch_index]
            )
            if isinstance(history, LycheePromptHistory):
                first_row = int(input_batch.query_start_loc_np[batch_index])
                for offset in range(count):
                    next_position = computed + offset + 1
                    if next_position >= int(input_batch.prefill_len_np[batch_index]):
                        break
                    raw_next = history.raw_text[next_position]
                    merge_text_ids[first_row + offset] = history.text[next_position] if raw_next is None else raw_next
        _, speech_logits = self.model.continue_after_primary_sample(
            positions=input_batch.positions[:total_rows],
            stoken_hidden=stoken_hidden,
            sampled_text_token_ids=merge_text_ids,
        )

        logits_rows = input_batch.logits_indices
        req_state_indices = input_batch.idx_mapping[: input_batch.num_reqs]
        frontier_rows = [
            int(input_batch.num_computed_tokens_np[index]) + int(input_batch.num_scheduled_tokens[index])
            >= int(input_batch.prefill_len_np[index])
            for index in range(input_batch.num_reqs)
        ]
        if not any(frontier_rows):
            empty = sampled_token_ids.new_full((total_rows,), -1, dtype=torch.int32)
            return {
                "lychee_text_token_ids": empty,
                "lychee_speech_token_ids": empty.clone(),
                "lychee_control_token_ids": empty.clone(),
            }
        modes = self._control_modes[req_state_indices]
        ticks = self._control_ticks[req_state_indices]
        speaking_steps = self._speaking_steps[req_state_indices]
        control_tokens = self._last_control_tokens[req_state_indices].clone()
        speech_tokens = self._last_stoken_tokens[req_state_indices].clone()
        for batch_index, frontier in enumerate(frontier_rows):
            if not frontier or int(num_sampled[batch_index].item()) <= 0:
                continue
            selection = slice(batch_index, batch_index + 1)
            runtime = self._runtime_sampling_config(int(input_batch.idx_mapping_np[batch_index]))
            control_tokens[selection] = sample_control_tokens(
                self.model.compute_control_logits(control_hidden[logits_rows[selection]]),
                modes=modes[selection],
                ticks=ticks[selection],
                config=config,
                allowing_backchannel=runtime["allowing_backchannel"],
                start_speak_token_factor=runtime["start_speak_token_factor"],
                start_listen_token_factor=runtime["start_listen_token_factor"],
                backchannel_token_bias=runtime["backchannel_token_bias"],
            )
            req_id = input_batch.req_ids[batch_index]
            speech_tokens[selection] = sample_speech_tokens(
                speech_logits[logits_rows[selection]],
                modes=modes[selection],
                speaking_steps=speaking_steps[selection],
                config=config,
                generator=self._speech_generators[req_id],
                speech_history=self._speech_history[req_state_indices[selection]],
                history_lengths=self._speech_history_lengths[req_state_indices[selection]],
            )
            if req_id in getattr(self, "_history_rng_draws", {}) and (
                config.stoken_do_sample
                and config.stoken_temperature > 0
                and bool(modes[batch_index] != int(LycheeControlMode.LISTENING))
                and bool(config.stoken_delay_num < speaking_steps[batch_index])
                and bool(speaking_steps[batch_index] < config.stoken_delay_num + config.stoken_max_tokens)
            ):
                self._history_rng_draws[req_id] += 1
        tail_padding_until = self._tail_padding_until[req_state_indices]
        forced_tail = ticks <= tail_padding_until
        forced_control = torch.full_like(control_tokens, config.sleep_token_id)
        forced_control = torch.where(
            (ticks == tail_padding_until - 1) & self._tail_detect_enabled[req_state_indices],
            forced_control.new_tensor(config.detect_token_id),
            forced_control,
        )
        forced_control = torch.where(
            ticks == tail_padding_until, forced_control.new_tensor(config.start_listening_token_id), forced_control
        )
        control_tokens = torch.where(forced_tail, forced_control, control_tokens)
        speech_tokens = torch.where(forced_tail, speech_tokens.new_tensor(config.stoken_pad_token_id), speech_tokens)
        text_tokens = sampled_token_ids[:, 0]

        active = num_sampled[: input_batch.num_reqs] > 0
        self._last_text_tokens[req_state_indices] = torch.where(
            active,
            text_tokens,
            self._last_text_tokens[req_state_indices],
        )
        self._last_stoken_tokens[req_state_indices] = torch.where(
            active,
            speech_tokens,
            self._last_stoken_tokens[req_state_indices],
        )
        self._last_control_tokens[req_state_indices] = torch.where(
            active,
            control_tokens,
            self._last_control_tokens[req_state_indices],
        )
        was_text_active = active & (modes != int(LycheeControlMode.LISTENING)) & ~self._text_eos_seen[req_state_indices]
        self._text_generated_steps[req_state_indices] += was_text_active.to(torch.long)
        self._text_eos_seen[req_state_indices] |= was_text_active & (text_tokens == getattr(config, "eos_token_id", -1))
        new_modes = update_control_modes(modes, control_tokens, config=config)
        speech_ended = active & (modes != int(LycheeControlMode.LISTENING)) & (speech_tokens == config.tts_end_token_id)
        new_modes = torch.where(speech_ended, new_modes.new_tensor(int(LycheeControlMode.LISTENING)), new_modes)
        new_tail_end = ((ticks + 1) // config.control_token_chunk_size + 1) * config.control_token_chunk_size - 1
        self._tail_padding_until[req_state_indices] = torch.where(speech_ended, new_tail_end, tail_padding_until)
        self._tail_detect_enabled[req_state_indices] = torch.where(
            speech_ended, new_tail_end - ticks > 2, self._tail_detect_enabled[req_state_indices]
        )
        reset_tail_inputs = active & (modes != new_modes) & ~speech_ended
        self._last_text_tokens[req_state_indices] = torch.where(
            reset_tail_inputs,
            text_tokens.new_tensor(config.text_pad_token_id),
            self._last_text_tokens[req_state_indices],
        )
        self._last_stoken_tokens[req_state_indices] = torch.where(
            reset_tail_inputs,
            speech_tokens.new_tensor(config.stoken_pad_token_id),
            self._last_stoken_tokens[req_state_indices],
        )
        changed_mode = active & (modes != new_modes)
        self._text_eos_seen[req_state_indices] &= ~changed_mode
        self._text_generated_steps[req_state_indices] = torch.where(
            changed_mode, ticks.new_zeros(()), self._text_generated_steps[req_state_indices]
        )
        new_speaking_steps = update_speaking_steps(
            speaking_steps,
            old_modes=modes,
            new_modes=new_modes,
        )
        history_positions = self._speech_history_lengths[req_state_indices]
        self._ensure_speech_history_capacity(int((history_positions + active.to(torch.long)).max().item()))
        safe_positions = history_positions.clamp_max(self._speech_history.shape[1] - 1)
        old_history_values = self._speech_history[req_state_indices, safe_positions]
        # The released n-gram processor sees the whole effective speech input
        # sequence, including listening pads and previous responses. Response
        # counters reset independently; the session evidence does not.
        self._speech_history[req_state_indices, safe_positions] = torch.where(
            active, self._last_stoken_tokens[req_state_indices], old_history_values
        )
        self._speech_history_lengths[req_state_indices] = history_positions + active.to(torch.long)
        self._control_modes[req_state_indices] = torch.where(
            active,
            new_modes,
            modes,
        )
        self._speaking_steps[req_state_indices] = torch.where(
            active,
            new_speaking_steps,
            speaking_steps,
        )
        self._control_ticks[req_state_indices] = torch.where(
            active,
            ticks + 1,
            ticks,
        )
        outputs = {
            "lychee_mode_after": self._request_tokens_on_model_rows(
                self._control_modes[req_state_indices],
                input_batch=input_batch,
                num_sampled=num_sampled,
                total_rows=total_rows,
            ),
            "lychee_tail_padding_until": self._request_tokens_on_model_rows(
                self._tail_padding_until[req_state_indices],
                input_batch=input_batch,
                num_sampled=num_sampled,
                total_rows=total_rows,
            ),
            "lychee_next_text_token_ids": self._request_tokens_on_model_rows(
                self._last_text_tokens[req_state_indices],
                input_batch=input_batch,
                num_sampled=num_sampled,
                total_rows=total_rows,
            ),
            "lychee_next_speech_token_ids": self._request_tokens_on_model_rows(
                self._last_stoken_tokens[req_state_indices],
                input_batch=input_batch,
                num_sampled=num_sampled,
                total_rows=total_rows,
            ),
            "lychee_text_token_ids": self._request_tokens_on_model_rows(
                text_tokens,
                input_batch=input_batch,
                num_sampled=num_sampled,
                total_rows=total_rows,
            ),
            "lychee_speech_token_ids": self._request_tokens_on_model_rows(
                speech_tokens,
                input_batch=input_batch,
                num_sampled=num_sampled,
                total_rows=total_rows,
            ),
            "lychee_control_token_ids": self._request_tokens_on_model_rows(
                control_tokens,
                input_batch=input_batch,
                num_sampled=num_sampled,
                total_rows=total_rows,
            ),
            "lychee_tick": self._request_tokens_on_model_rows(
                self._prepared_ticks[req_state_indices],
                input_batch=input_batch,
                num_sampled=num_sampled,
                total_rows=total_rows,
            ),
            "lychee_execution_epoch": self._request_tokens_on_model_rows(
                self._execution_epochs[req_state_indices],
                input_batch=input_batch,
                num_sampled=num_sampled,
                total_rows=total_rows,
            ),
            "lychee_model_position": self._request_tokens_on_model_rows(
                input_batch.positions[logits_rows].to(torch.long),
                input_batch=input_batch,
                num_sampled=num_sampled,
                total_rows=total_rows,
            ),
            "lychee_audio_window_seq": self._request_tokens_on_model_rows(
                self._audio_window_seqs[req_state_indices],
                input_batch=input_batch,
                num_sampled=num_sampled,
                total_rows=total_rows,
            ),
        }
        self._commit_tick(
            req_state_indices=req_state_indices,
            active=active,
        )
        return outputs


__all__ = ["LycheeModelState", "LycheePoisonedRequestError"]
