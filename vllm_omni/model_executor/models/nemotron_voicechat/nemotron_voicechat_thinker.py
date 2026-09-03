# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage-0 thinker for NemotronVoiceChat: 16 kHz speech -> frame-locked text.

``LLM_AR`` stage combining:

* the vendored NeMo FastConformer perception (mel + 24-layer encoder + linear
  projection to the LLM width, 12.5 Hz frames);
* the additive channel fusion from the NeMo duplex STT model
  (``fused[t] = E(text_prev) * w_text + timeline[t] * w_audio +
  E(func_prev) * w_func`` with the checkpoint's AddFusion weights);
* the NemotronH (hybrid Mamba) backbone on vLLM paged execution via
  ``init_vllm_registered_model`` — this is where vLLM earns its keep;
* the separate ``lm_head`` (text channel, sampled by the vLLM engine) and
  ``function_head`` (greedy, maintained internally per request, with client
  tool results injected into the function channel).

Timeline contract (verified against NeMo ``_init_inference``/``_step_zero``/
``_step_inference``; the golden shape test guards it):

* The vLLM prompt covers ``logical_prompt_token_len + 1`` positions: the system
  prompt tokens (``[bos] + tokens + [eos]``) fused with a PAD text channel
  (NeMo's ``_get_bos_embedding`` returns the PAD embedding, so position 0 is
  uniform with the rest of the prompt region), plus acoustic frame 0 as the
  final prefill position. The first engine-sampled token is therefore NeMo's
  ``gen_text[prompt_len]`` — the first acoustic-frame prediction.
* Each decode step t consumes conformer frame t (held in per-request state)
  fused with the previously sampled text token and the previous greedy
  function token. Text EOS never terminates generation (``ignore_eos``); the
  example sets ``max_tokens = acoustic_frame_count``. Never set ``min_tokens``:
  the tokenizer's eos_token is <SPECIAL_12> — the frame-locked PAD/silence
  token — and min_tokens masks "EOS" to -inf, forbidding silence entirely.
* The function channel is load-bearing for numerical correctness (fusion
  weight 2.0), emits tool calls, and accepts client tool results while keeping
  the text channel padded during forced response injection.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.experimental.fullduplex.model_executor import DuplexSamplingRow
from vllm_omni.model_executor.models.nemotron_voicechat.runtime_info import (
    merge_runtime_info,
    require_request_id,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

_LM_ARCHITECTURE = "NemotronHForCausalLM"


def slice_perception_streaming_mel(
    processed_signal: torch.Tensor,
    frame_idx: int,
    streaming_cfg: Any,
) -> tuple[torch.Tensor, int]:
    """Slice one 80 ms mel chunk using NeMo's cache-manager geometry."""

    def _at(value: Any, index: int) -> int:
        return int(value[index] if isinstance(value, list | tuple) else value)

    chunk_size_first = _at(streaming_cfg.chunk_size, 0)
    chunk_size = _at(streaming_cfg.chunk_size, 1)
    shift_size_first = _at(streaming_cfg.shift_size, 0)
    shift_size = _at(streaming_cfg.shift_size, 1)
    pre_encode_first = _at(streaming_cfg.pre_encode_cache_size, 0)
    pre_encode_size = _at(streaming_cfg.pre_encode_cache_size, 1)

    if frame_idx == 0:
        mel_chunk = processed_signal[:, :, :chunk_size_first]
        if pre_encode_first > 0:
            mel_chunk = torch.nn.functional.pad(mel_chunk, (pre_encode_first, 0))
        return mel_chunk, 0

    chunk_start = shift_size_first + (frame_idx - 1) * shift_size
    chunk_end = chunk_start + chunk_size
    # Match NeMo's tail-boundary rule.  With one 80 ms frame per call this
    # normally leaves the expected trailing STFT columns unused.
    offset = chunk_size - shift_size_first
    if chunk_end > processed_signal.shape[-1] - offset:
        chunk_end = processed_signal.shape[-1] - offset
        chunk_start = chunk_end - chunk_size
    main_chunk = processed_signal[:, :, chunk_start:chunk_end]
    cache_start = max(0, chunk_start - pre_encode_size)
    cache_mel = processed_signal[:, :, cache_start:chunk_start]
    if cache_mel.shape[-1] < pre_encode_size:
        cache_mel = torch.nn.functional.pad(
            cache_mel,
            (pre_encode_size - cache_mel.shape[-1], 0),
        )
    mel_chunk = torch.cat([cache_mel, main_chunk], dim=-1)
    return mel_chunk, int(streaming_cfg.drop_extra_pre_encoded)


def validate_timeline_fits_max_model_len(
    logical_prompt_token_len: int,
    acoustic_frame_count: int,
    max_model_len: int,
) -> None:
    """Reject audio whose frame-locked timeline exceeds the stage limit.

    The vLLM request occupies ``logical_prompt_token_len + 1`` prefill
    positions (the +1 is the acoustic-frame-0 placeholder) plus
    ``acoustic_frame_count`` sampled tokens, so the binding sequence size is
    one MORE than NeMo's logical timeline (``prompt + frames``).
    """
    logical_total_len = logical_prompt_token_len + acoustic_frame_count
    vllm_prefill_len = logical_prompt_token_len + 1
    vllm_total_len = vllm_prefill_len + acoustic_frame_count
    if vllm_total_len > max_model_len:
        raise ValueError(
            "NemotronVoiceChat input exceeds the thinker stage's max_model_len: "
            f"vllm_prefill_len={vllm_prefill_len} + "
            f"acoustic_frame_count={acoustic_frame_count} = "
            f"vllm_total_len={vllm_total_len} > max_model_len={max_model_len} "
            f"(logical_prompt_token_len={logical_prompt_token_len}, "
            f"logical_total_len={logical_total_len}). Use shorter audio or raise the "
            "stage's max_model_len in the deploy yaml."
        )


def compute_acoustic_frame_count(nemo_stt_cfg: dict[str, Any], num_samples: int) -> int:
    """Exact 12.5 Hz frame count for ``num_samples`` of 16 kHz audio.

    Replicates the vendored mel featurizer length arithmetic followed by the
    dw-striding conv subsampling length recurrence (NOT ``duration / 0.08``).
    The thinker validates this against the real perception output at prefill,
    and the golden shape test pins it against the NeMo reference.
    """
    from vllm_omni.model_executor.models.nemotron_voicechat.nemo_vendored.asr.subsampling import (
        calc_length,
    )

    perception = nemo_stt_cfg.get("perception") or {}
    pre = perception.get("preprocessor") or {}
    encoder = perception.get("encoder") or {}
    sample_rate = int(pre.get("sample_rate", 16000))
    window_stride = float(pre.get("window_stride", 0.01))
    window_size = float(pre.get("window_size", 0.025))
    n_fft = int(pre.get("n_fft") or 2 ** int(np.ceil(np.log2(window_size * sample_rate))))
    hop = int(window_stride * sample_rate)
    pad_amount = n_fft // 2 * 2  # torch.stft center padding on both sides
    mel_len = (num_samples + pad_amount - n_fft) // hop + 1
    subsampling_factor = int(encoder.get("subsampling", "dw_striding") and encoder.get("subsampling_factor", 8))
    conv_kernel = int(encoder.get("subsampling_conv_kernel_size", 3) or 3)
    stride = 2
    repeat = int(np.log2(subsampling_factor))
    # Mirror the vendored ConvSubsampling padding exactly: causal downsampling
    # (this checkpoint sets encoder.causal_downsampling=true) pads
    # left=kernel-1 plus right=stride-1, i.e. one more column than the
    # non-causal symmetric 2*(kernel//2) — which shifts the output length by
    # one for roughly half of all input durations.
    if bool(encoder.get("causal_downsampling", False)):
        all_paddings = (conv_kernel - 1) + (stride - 1)
    else:
        all_paddings = 2 * (conv_kernel // 2)
    out = calc_length(
        torch.tensor([mel_len]),
        all_paddings=all_paddings,
        kernel_size=conv_kernel,
        stride=stride,
        ceil_mode=False,
        repeat_num=repeat,
    )
    return int(out[0])


class NemotronVoiceChatThinkerForConditionalGeneration(nn.Module, HasInnerState, IsHybrid):
    """vLLM-native thinker (perception + fusion + NemotronH + dual heads).

    Declares the hybrid-Mamba interfaces (``IsHybrid``/``HasInnerState``) so
    vLLM configures the mamba state cache for the wrapped NemotronH backbone;
    the state shape/dtype classmethods mirror ``NemotronHForCausalLM`` but read
    the NemotronH sub-config (``get_text_config()``) instead of the top-level
    checkpoint config.
    """

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig) -> tuple[torch.dtype, torch.dtype]:
        from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM

        return NemotronHForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig) -> tuple[tuple[int, int], tuple[int, int, int]]:
        from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator

        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config.get_text_config()
        intermediate_size = hf_config.mamba_num_heads * hf_config.mamba_head_dim
        return MambaStateShapeCalculator.mamba2_state_shape(
            intermediate_size=intermediate_size,
            tp_world_size=parallel_config.tensor_parallel_size,
            n_groups=hf_config.n_groups,
            num_heads=hf_config.mamba_num_heads,
            head_dim=hf_config.mamba_head_dim,
            state_size=hf_config.ssm_state_size,
            conv_kernel=hf_config.conv_kernel,
            num_spec=vllm_config.num_speculative_tokens,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM

        return NemotronHForCausalLM.get_mamba_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        self.config = config
        stt_cfg: dict[str, Any] = dict(getattr(config, "stt_cfg", {}) or {})
        if not stt_cfg:
            raise ValueError("NemotronVoiceChat checkpoint config lacks 'model.stt.model'; cannot build the thinker.")
        self.stt_cfg = stt_cfg
        text_cfg = config.get_text_config()
        self._hidden = int(getattr(text_cfg, "hidden_size", 4480))
        self._vocab = int(getattr(text_cfg, "vocab_size", 131072))
        self._dtype = getattr(vllm_config.model_config, "dtype", torch.bfloat16)

        # NemotronH backbone (+ lm_head) on vLLM native layers.
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=text_cfg,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=[_LM_ARCHITECTURE],
        )

        # Shared channel embedding + separate function head (both distinct
        # checkpoints subtrees from the backbone).
        self.embed_tokens = nn.Embedding(self._vocab, self._hidden)
        self.function_head = nn.Linear(self._hidden, self._vocab, bias=False)

        # Vendored perception (mel + FastConformer + projection).
        from vllm_omni.model_executor.models.nemotron_voicechat.nemo_vendored.perception import (
            AudioPerceptionModule,
        )

        perception_cfg = stt_cfg.get("perception")
        if not perception_cfg:
            raise ValueError("'model.stt.model.perception' missing from the checkpoint config.")
        from omegaconf import DictConfig

        self.perception = AudioPerceptionModule(DictConfig(perception_cfg))
        # NeMo's streaming wrapper deliberately uses a second, deterministic
        # preprocessor.  The training/offline frontend has dither enabled and
        # pads the mel time axis; both are incompatible with exact incremental
        # chunk boundaries.
        streaming_preprocessor_cfg = copy.deepcopy(self.perception.cfg.preprocessor)
        streaming_preprocessor_cfg.dither = 0.0
        streaming_preprocessor_cfg.pad_to = 0
        self._streaming_preprocessor = self.perception.from_config_dict(streaming_preprocessor_cfg)

        # AddFusion weights (the checkpoint uses fuse_method="add" defaults;
        # the same duplex_*_weight keys double as the fusion weights).
        use_function_head = bool(stt_cfg.get("use_function_head", True))
        self._w_text = float(stt_cfg.get("duplex_text_channel_weight", 1.0))
        self._w_audio = float(stt_cfg.get("duplex_user_channel_weight", 1.0))
        self._w_func = float(stt_cfg.get("duplex_function_channel_weight", 1.0)) if use_function_head else 0.0
        self._use_function_head = use_function_head
        fuse_method = stt_cfg.get("fuse_method", "add") or "add"
        if fuse_method != "add":
            raise NotImplementedError(
                f"NemotronVoiceChat thinker only implements fuse_method='add' "
                f"(this checkpoint uses it); got {fuse_method!r}."
            )
        if bool(stt_cfg.get("predict_user_text", False)):
            raise NotImplementedError(
                "NemotronVoiceChat thinker does not implement the user-ASR channel "
                "(predict_user_text=True); this checkpoint has it disabled."
            )

        self._special_ids: dict[str, int] | None = None

        # Omni AR runner contract.
        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.prefer_model_sampler = True
        self.omni_client_multimodal_output_keys = ("nvc_function_token",)
        self.has_postprocess = True  # per-step greedy function-channel token
        self.requires_full_prefix_cached_hidden_states = False
        # The per-step function token is a scalar; no GPU-resident buffers needed.
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = set()

        # Stateful per-request execution is only validated at batch size 1
        # (the shipped offline scope); fail fast on silent multi-request use.
        max_num_seqs = int(getattr(vllm_config.scheduler_config, "max_num_seqs", 1))
        if max_num_seqs != 1:
            raise NotImplementedError(
                f"NemotronVoiceChat thinker supports max_num_seqs=1 only (got {max_num_seqs}); "
                "multi-request batching of the per-request fusion state is not implemented."
            )

        # request_id -> per-request state (audio frames, prefill embeds, counters).
        self._sessions: dict[str, dict[str, Any]] = {}
        self._duplex_sampling_rows: dict[int, str] = {}
        self._duplex_previous_text_tokens: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Special token ids (from the checkpoint's token strings).
    # ------------------------------------------------------------------
    def _resolve_special_ids(self) -> dict[str, int]:
        if self._special_ids is not None:
            return self._special_ids
        from transformers import AutoTokenizer

        ref = os.environ.get("NEMOTRON_VOICECHAT_LLM_PATH") or self.stt_cfg.get(
            "pretrained_llm", "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
        )
        tok = AutoTokenizer.from_pretrained(ref, trust_remote_code=False)
        ids = {
            "bos": tok.convert_tokens_to_ids(self.stt_cfg.get("bos_token", "<s>")),
            "eos": tok.convert_tokens_to_ids(self.stt_cfg.get("eos_token", "</s>")),
            "pad": tok.convert_tokens_to_ids(self.stt_cfg.get("pad_token", "<SPECIAL_12>")),
        }
        for name, value in ids.items():
            if value is None or value < 0:
                raise ValueError(f"NemotronVoiceChat could not resolve the {name} token id from {ref!r}.")
        self._special_ids = ids
        return ids

    # ------------------------------------------------------------------
    # vLLM model protocol.
    # ------------------------------------------------------------------
    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        # Placeholder; preprocess replaces every position with fused embeds.
        return torch.zeros((input_ids.shape[0], self._hidden), device=input_ids.device, dtype=self._dtype)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        return self.language_model(
            input_ids=None,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> torch.Tensor:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        # vLLM >= 0.25 dropped the sampling_metadata parameter.
        return self.language_model.compute_logits(hidden_states)

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        multimodal_outputs: dict[str, torch.Tensor] = {}
        if self._use_function_head and model_outputs.numel():
            with torch.inference_mode():
                function_token = self.function_head(model_outputs[-1:, :].to(self._dtype)).argmax(dim=-1)
            info_dicts = kwargs.get("model_intermediate_buffer") or kwargs.get("runtime_additional_information")
            info = info_dicts[0] if isinstance(info_dicts, list) and len(info_dicts) == 1 else None
            request_id = info.get("request_id") if isinstance(info, dict) else None
            session = self._sessions.get(request_id) if isinstance(request_id, str) else None
            forced_token = session.get("forced_function_token") if isinstance(session, dict) else None
            if isinstance(forced_token, int):
                function_token = torch.full_like(function_token, forced_token)
            multimodal_outputs["nvc_function_token"] = function_token
        return OmniOutput(
            text_hidden_states=model_outputs,
            multimodal_outputs=multimodal_outputs,
        )

    def prepare_duplex_sampling(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        rows: tuple[DuplexSamplingRow, ...],
    ) -> None:
        del logits, sampling_metadata
        self._duplex_sampling_rows = {row.row_idx: row.request_id for row in rows}

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        del sampling_metadata
        rows = self._duplex_sampling_rows
        if not rows:
            return None
        expected_rows = set(range(int(logits.shape[0])))
        if set(rows) != expected_rows:
            raise RuntimeError(
                "NemotronVoiceChat native-duplex sampling requires every active batch row "
                "to be a duplex request; mixed duplex/non-duplex batches are unsupported."
            )
        sampled = logits.argmax(dim=-1).to(dtype=torch.int32)
        for row_idx, request_id in rows.items():
            session = self._sessions.get(request_id)
            if isinstance(session, dict) and isinstance(session.get("forced_function_token"), int):
                sampled[row_idx] = int(session["nvc_text_pad_id"])
            self._duplex_previous_text_tokens[request_id] = int(sampled[row_idx].item())
        return SamplerOutput(
            sampled_token_ids=sampled.unsqueeze(-1),
            logprobs_tensors=None,
        )

    # ------------------------------------------------------------------
    # Fusion helpers.
    # ------------------------------------------------------------------
    def _fuse(
        self,
        text_ids: torch.Tensor,
        timeline_embeds: torch.Tensor,
        func_ids: torch.Tensor,
    ) -> torch.Tensor:
        """AddFusion over the three channels; shapes [T] ids, [T, H] timeline."""
        text_e = self.embed_tokens(text_ids)
        fused = text_e * self._w_text + timeline_embeds.to(text_e.dtype) * self._w_audio
        if self._use_function_head:
            fused = fused + self.embed_tokens(func_ids) * self._w_func
        return fused

    @staticmethod
    def _duplex_info(info: dict[str, Any]) -> dict[str, Any] | None:
        duplex = info.get("duplex")
        if isinstance(duplex, dict) and duplex.get("data_plane") is True:
            return duplex
        return None

    def _duplex_stable_frame(
        self,
        session: dict[str, Any],
        duplex: dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        try:
            source_input_seq = int(duplex["source_input_seq"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Nemotron VoiceChat duplex input lacks source_input_seq") from exc
        if source_input_seq == session.get("last_input_seq"):
            return session["duplex_frame"]
        last_input_seq = int(session.get("last_input_seq", 0))
        if source_input_seq < last_input_seq:
            raise ValueError(
                f"Nemotron VoiceChat duplex input sequence moved backwards: {source_input_seq} < {last_input_seq}"
            )

        from vllm_omni.experimental.fullduplex.nemotron_voicechat.input import (
            decode_pcm_f32le,
        )

        raw = decode_pcm_f32le(duplex.get("payload"), exact_frame=True)
        frame_audio = torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy()).to(device=device, dtype=torch.float32)
        audio = session.get("duplex_audio")
        if isinstance(audio, torch.Tensor):
            if source_input_seq != last_input_seq + 1:
                raise ValueError(
                    "Nemotron VoiceChat cache-aware perception requires contiguous "
                    f"input frames; got sequence {source_input_seq} after {last_input_seq}"
                )
            audio = torch.cat([audio, frame_audio])
        else:
            if source_input_seq != 1:
                raise ValueError(
                    f"Nemotron VoiceChat cache-aware perception must start at input sequence 1, got {source_input_seq}"
                )
            audio = frame_audio

        encoder = self.perception.encoder
        streaming_cfg = encoder.streaming_cfg
        with torch.inference_mode():
            processed_signal, _ = self._streaming_preprocessor(
                input_signal=audio.unsqueeze(0),
                length=torch.tensor([audio.numel()], device=device, dtype=torch.long),
            )

            frame_idx = source_input_seq - 1
            mel_chunk, drop_extra_pre_encoded = slice_perception_streaming_mel(
                processed_signal,
                frame_idx,
                streaming_cfg,
            )
            if frame_idx == 0:
                caches = encoder.get_initial_cache_state(
                    batch_size=1,
                    dtype=next(encoder.parameters()).dtype,
                    device=device,
                )
            else:
                caches = (
                    session.get("perception_cache_last_channel"),
                    session.get("perception_cache_last_time"),
                    session.get("perception_cache_last_channel_len"),
                )
                if any(cache is None for cache in caches):
                    raise RuntimeError("Nemotron VoiceChat perception cache is missing")

            encoder_dtype = next(encoder.parameters()).dtype
            encoded, encoded_len, cache_channel, cache_time, cache_channel_len = encoder.cache_aware_stream_step(
                processed_signal=mel_chunk.to(encoder_dtype),
                processed_signal_length=torch.tensor(
                    [mel_chunk.shape[-1]],
                    device=device,
                    dtype=torch.long,
                ),
                cache_last_channel=caches[0],
                cache_last_time=caches[1],
                cache_last_channel_len=caches[2],
                keep_all_outputs=True,
                drop_extra_pre_encoded=drop_extra_pre_encoded,
            )
            encoded, encoded_len = self.perception.modality_adapter(
                audio_signal=encoded,
                length=encoded_len,
            )
            stable = self.perception.proj(encoded.transpose(1, 2))

        frame_count = int(encoded_len.reshape(-1)[0])
        if frame_count != 1 or stable.shape[1] != 1:
            raise RuntimeError(
                "Nemotron VoiceChat cache-aware perception must produce exactly "
                f"one frame per 80 ms input; got encoded_len={frame_count}, "
                f"shape={tuple(stable.shape)}"
            )
        stable = stable[0].to(self._dtype)
        session["duplex_audio"] = audio
        session["perception_cache_last_channel"] = cache_channel
        session["perception_cache_last_time"] = cache_time
        session["perception_cache_last_channel_len"] = cache_channel_len
        session["duplex_frame"] = stable
        session["last_input_seq"] = source_input_seq
        return stable

    def _preprocess_duplex(
        self,
        *,
        request_id: str,
        input_ids: torch.Tensor,
        info: dict[str, Any],
        duplex: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        device = input_ids.device
        runtime_config = duplex.get("runtime_config")
        if not isinstance(runtime_config, dict):
            raise ValueError("Nemotron VoiceChat duplex input lacks runtime_config")
        try:
            source_input_seq = int(duplex["source_input_seq"])
            pad_id = int(runtime_config["nvc_text_pad_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Nemotron VoiceChat duplex runtime token metadata is incomplete") from exc

        session = self._sessions.get(request_id)
        if session is None:
            prompt_ids_raw = runtime_config.get("nvc_prompt_token_ids")
            if not isinstance(prompt_ids_raw, list | tuple) or not prompt_ids_raw:
                raise ValueError("Nemotron VoiceChat duplex runtime requires nvc_prompt_token_ids")
            prompt_ids = torch.as_tensor(
                [int(token_id) for token_id in prompt_ids_raw],
                dtype=torch.long,
                device=device,
            )
            session = {
                "prompt_len": int(prompt_ids.numel()),
                "func_token": pad_id,
                "last_input_seq": 0,
                "nvc_text_pad_id": pad_id,
                "function_response_generation": 0,
                "forced_function_tokens": [],
            }
            self._sessions[request_id] = session
            frame = self._duplex_stable_frame(session, duplex, device)
            timeline = torch.cat(
                [self.embed_tokens(prompt_ids).to(self._dtype), frame],
                dim=0,
            )
            pad_ids = torch.full(
                (timeline.shape[0],),
                pad_id,
                dtype=torch.long,
                device=device,
            )
            session["prefill_embeds"] = self._fuse(pad_ids, timeline, pad_ids)
        else:
            frame = self._duplex_stable_frame(session, duplex, device)

        self._sync_forced_function_response(session, runtime_config)

        if source_input_seq <= 1:
            offset = max(0, int(info.get("_omni_num_computed_tokens", 0) or 0))
            prefill = session["prefill_embeds"]
            span = int(input_ids.shape[0])
            if offset + span > int(prefill.shape[0]):
                raise ValueError("Nemotron VoiceChat duplex first append exceeds its fused prompt")
            return (
                input_ids,
                prefill[offset : offset + span].to(self._dtype),
                {
                    "nvc_text_pad_id": pad_id,
                },
            )

        if int(input_ids.shape[0]) != 1:
            raise ValueError("Nemotron VoiceChat duplex continuation must schedule exactly one frame token")
        previous_function = info.get("nvc_prev_function_token")
        if previous_function is not None:
            if isinstance(previous_function, torch.Tensor):
                previous_function = previous_function.reshape(-1)[-1].item()
            session["func_token"] = int(previous_function)
        previous_text = self._duplex_previous_text_tokens.get(request_id)
        if previous_text is None:
            raise RuntimeError("Nemotron VoiceChat duplex continuation has no previous sampled text token")
        text_id = torch.tensor([previous_text], dtype=torch.long, device=device)
        function_id = torch.tensor(
            [int(session["func_token"])],
            dtype=torch.long,
            device=device,
        )
        fused = self._fuse(text_id, frame, function_id)
        return (
            input_ids,
            fused,
            {
                "meta": {"nvc_frame": source_input_seq - 1},
                "nvc_text_pad_id": pad_id,
            },
        )

    @staticmethod
    def _sync_forced_function_response(
        session: dict[str, Any],
        runtime_config: dict[str, Any],
    ) -> None:
        try:
            generation = int(runtime_config.get("nvc_function_response_generation", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("Nemotron VoiceChat function-response generation must be an integer") from exc
        seen = int(session.get("function_response_generation", 0))
        if generation < seen:
            raise ValueError("Nemotron VoiceChat function-response generation moved backwards")
        if generation > seen:
            raw_batches = runtime_config.get("nvc_function_response_batches")
            batches = raw_batches if isinstance(raw_batches, list) else []
            unseen: list[tuple[int, list[int]]] = []
            for raw_batch in batches:
                if not isinstance(raw_batch, dict):
                    raise ValueError("Nemotron VoiceChat function response batch must be an object")
                try:
                    batch_generation = int(raw_batch["generation"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("Nemotron VoiceChat function response batch requires a generation") from exc
                if batch_generation <= seen:
                    continue
                raw_tokens = raw_batch.get("token_ids")
                if not isinstance(raw_tokens, list | tuple) or not raw_tokens:
                    raise ValueError("Nemotron VoiceChat function response requires token ids")
                unseen.append((batch_generation, [int(token_id) for token_id in raw_tokens]))
            if not unseen:
                # Backward-compatible single-response payload used by older
                # serving adapters and serialized test fixtures.
                raw_tokens = runtime_config.get("nvc_function_response_token_ids")
                if not isinstance(raw_tokens, list | tuple) or not raw_tokens:
                    raise ValueError("Nemotron VoiceChat function response requires token ids")
                unseen = [(generation, [int(token_id) for token_id in raw_tokens])]
            expected = seen + 1
            queue = session.setdefault("forced_function_tokens", [])
            if not isinstance(queue, list):
                raise ValueError("Nemotron VoiceChat forced function-token queue is invalid")
            for batch_generation, token_ids in unseen:
                if batch_generation != expected:
                    raise ValueError("Nemotron VoiceChat function-response generations are not contiguous")
                queue.extend(token_ids)
                expected += 1
            if expected - 1 != generation:
                raise ValueError("Nemotron VoiceChat function-response batches are incomplete")
            session["function_response_generation"] = generation
        queue = session.get("forced_function_tokens")
        if session.get("forced_function_token") is None and isinstance(queue, list) and queue:
            session["forced_function_token"] = int(queue[0])

    # ------------------------------------------------------------------
    # Per-request session.
    # ------------------------------------------------------------------
    def _load_audio(self, info: dict[str, Any], device: torch.device) -> torch.Tensor:
        audio = info.get("nvc_audio")
        if audio is None:
            raise ValueError(
                "NemotronVoiceChat thinker request is missing its audio: pass "
                "additional_information={'nvc_audio': <16 kHz mono float waveform>, "
                "'nvc_sr': 16000, 'nvc_prompt_token_ids': [...]}."
            )
        wav = torch.as_tensor(np.asarray(audio), dtype=torch.float32, device=device).reshape(-1)
        if wav.numel() == 0:
            raise ValueError("NemotronVoiceChat thinker received empty audio input.")
        sr = int(info.get("nvc_sr", 16000))
        expected_sr = int(getattr(self.config, "source_sample_rate", 16000))
        if sr != expected_sr:
            raise ValueError(
                f"NemotronVoiceChat thinker expects {expected_sr} Hz audio, got {sr} Hz — "
                "resample client-side (the offline example does this)."
            )
        return wav

    def _init_session(self, request_id: str, info: dict[str, Any], device: torch.device) -> dict[str, Any]:
        pad_id = self._resolve_special_ids()["pad"]
        wav = self._load_audio(info, device)

        prompt_ids = info.get("nvc_prompt_token_ids")
        if prompt_ids is None:
            raise ValueError(
                "NemotronVoiceChat thinker request is missing 'nvc_prompt_token_ids' "
                "(the system prompt token ids, [bos] + tokens + [eos])."
            )
        prompt_ids = torch.as_tensor(list(prompt_ids), dtype=torch.long, device=device)
        if prompt_ids.numel() == 0:
            raise ValueError("NemotronVoiceChat thinker requires a non-empty system prompt.")

        # Perception: mel + FastConformer + projection -> [N, hidden] frames.
        with torch.inference_mode():
            encoded, encoded_len = self.perception(
                input_signal=wav.unsqueeze(0).to(next(self.perception.parameters()).dtype),
                input_signal_length=torch.tensor([wav.numel()], device=device),
            )
        n_frames = int(encoded_len.reshape(-1)[0])
        audio_embeds = encoded[0, :n_frames].to(self._dtype)  # [N, H]
        expected = info.get("nvc_expected_frames")
        if expected is not None and int(expected) != n_frames:
            raise ValueError(
                f"NemotronVoiceChat frame-count mismatch: the request was sized for "
                f"{int(expected)} acoustic frames but perception produced {n_frames}. "
                "The example must size max_tokens with compute_acoustic_frame_count() (never min_tokens)."
            )

        p = int(prompt_ids.numel())
        validate_timeline_fits_max_model_len(
            logical_prompt_token_len=p,
            acoustic_frame_count=n_frames,
            max_model_len=int(self.vllm_config.model_config.max_model_len),
        )

        # Fused prefill embeds for positions 0..P (P prompt slots + frame 0).
        # All prompt positions use the PAD text channel (NeMo's
        # _get_bos_embedding returns the PAD embedding) and PAD function channel.
        timeline = torch.cat([self.embed_tokens(prompt_ids).to(self._dtype), audio_embeds[0:1]], dim=0)  # [P+1, H]
        pad_ids = torch.full((p + 1,), pad_id, dtype=torch.long, device=device)
        prefill = self._fuse(pad_ids, timeline, pad_ids)  # [P+1, H]

        session = {
            "audio_embeds": audio_embeds,
            "prefill_embeds": prefill,
            "prompt_len": p,
            "n_frames": n_frames,
            "func_token": int(pad_id),
            "decode_step": 0,
        }
        self._sessions[request_id] = session
        return session

    # ------------------------------------------------------------------
    # Omni AR hooks.
    # ------------------------------------------------------------------
    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        info = merge_runtime_info(info_dict)
        device = input_ids.device
        span = int(input_ids.shape[0])
        is_prefill_raw = info.get("_omni_is_prefill")
        is_prefill = bool(is_prefill_raw) if isinstance(is_prefill_raw, bool) else span > 1
        request_id = require_request_id(info, "thinker")

        duplex = self._duplex_info(info)
        if duplex is not None:
            return self._preprocess_duplex(
                request_id=request_id,
                input_ids=input_ids,
                info=info,
                duplex=duplex,
            )
        if is_prefill or request_id not in self._sessions:
            session = self._sessions.get(request_id) or self._init_session(request_id, info, device)
            offset = max(0, int(info.get("_omni_num_computed_tokens", 0) or 0))
            prefill: torch.Tensor = session["prefill_embeds"]
            if offset + span > prefill.shape[0]:
                raise ValueError(
                    f"NemotronVoiceChat thinker prefill span [{offset}, {offset + span}) exceeds "
                    f"the fused prompt length {prefill.shape[0]}; the request prompt must be "
                    "the system prompt tokens plus exactly one placeholder position."
                )
            return input_ids, prefill[offset : offset + span].to(self._dtype), {}

        session = self._sessions[request_id]
        pad_id = self._resolve_special_ids()["pad"]

        # Function channel: greedy argmax over the PREVIOUS position's hidden,
        # computed by postprocess right after each forward. At the first decode
        # step this is the prefill's last position (acoustic frame 0), matching
        # NeMo's gen_function[prompt_len]. The cumulative debug timeline
        # (nvc_function_tokens) is owned by postprocess, which also covers the
        # final acoustic frame (no later preprocess exists for it).
        if self._use_function_head:
            prev_func = info.get("nvc_prev_function_token")
            if prev_func is not None:
                token = int(prev_func.reshape(-1)[0]) if isinstance(prev_func, torch.Tensor) else int(prev_func)
                session["func_token"] = token

        # Position-addressed frame index: decode of vLLM position P+k consumes
        # acoustic frame k (frame 0 was the last prefill position). Deriving it
        # from _omni_num_computed_tokens keeps preprocess idempotent — if the
        # runner ever re-enters the same decode position (preemption or
        # recompute), the audio pointer stays aligned with the text channel
        # instead of silently advancing. Falls back to a call counter only if
        # the runner did not inject the key.
        computed = info.get("_omni_num_computed_tokens")
        if computed is not None:
            frame_idx = int(computed) - int(session["prompt_len"])
        else:
            frame_idx = int(session["decode_step"]) + 1
        session["decode_step"] = frame_idx
        n_frames = int(session["n_frames"])
        if frame_idx < 1:
            raise ValueError(
                f"NemotronVoiceChat thinker derived decode frame index {frame_idx} "
                f"(computed_tokens={computed}, prompt_len={session['prompt_len']}); "
                "decode must start at acoustic frame 1."
            )
        if frame_idx >= n_frames:
            raise ValueError(
                f"NemotronVoiceChat thinker was asked for decode step {frame_idx} but only "
                f"{n_frames} acoustic frames exist; max_tokens must equal the frame count."
            )
        text_prev = input_ids.reshape(-1)[:1].to(torch.long)
        func_prev = torch.full_like(text_prev, session["func_token"] if self._use_function_head else pad_id)
        frame = session["audio_embeds"][frame_idx : frame_idx + 1]
        fused = self._fuse(text_prev, frame, func_prev)  # [1, H]

        info_update: dict[str, Any] = {
            "meta": {"nvc_frame": frame_idx},
            "nvc_text_pad_id": pad_id,
        }
        return input_ids, fused, info_update

    def postprocess(self, hidden_states: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
        """Greedy function-channel token from this step's last hidden.

        Runs after EVERY forward (prefill and each decode step), so the final
        acoustic frame's token is covered too — NeMo produces gen_function for
        every non-prompt position. ``nvc_prev_function_token`` is load-bearing
        (it feeds the next step's fusion at weight 2.0). The cumulative debug
        timeline (``nvc_function_tokens``, one entry per acoustic frame) costs
        a host sync plus a growing concat per step, so it is only accumulated
        when ``NEMOTRON_VOICECHAT_DEBUG_FUNCTION_TIMELINE=1``.
        On multi-chunk prefill, only the final chunk (whose forward covers the
        last prefill position, acoustic frame 0) contributes a token.
        """
        if hidden_states is None or hidden_states.numel() == 0 or not self._use_function_head:
            return {}
        if kwargs.get("_omni_is_prefill"):
            computed = int(kwargs.get("_omni_num_computed_tokens", 0) or 0)
            prompt_len = int(kwargs.get("_omni_prompt_len", 0) or 0)
            if prompt_len and computed + int(hidden_states.shape[0]) < prompt_len:
                return {}  # intermediate prefill chunk: frame 0 not reached yet
        multimodal_outputs = kwargs.get("multimodal_outputs")
        token = multimodal_outputs.get("nvc_function_token") if isinstance(multimodal_outputs, dict) else None
        if not isinstance(token, torch.Tensor) or token.numel() == 0:
            with torch.inference_mode():
                func_logits = self.function_head(hidden_states[-1, :].reshape(1, -1).to(self._dtype))
            token = func_logits.argmax(dim=-1)
        token = token.reshape(-1)[-1].reshape(()).detach()
        request_id = kwargs.get("request_id")
        if isinstance(request_id, str):
            session = self._sessions.get(request_id)
            if session is not None:
                # StreamingUpdate replaces the runner payload at each append;
                # retain the function channel in model-owned session state.
                session["func_token"] = int(token.item())
                forced = session.pop("forced_function_token", None)
                if isinstance(forced, int):
                    queue = session.get("forced_function_tokens")
                    if not isinstance(queue, list) or not queue or queue.pop(0) != forced:
                        raise RuntimeError("Nemotron VoiceChat forced function-token queue lost alignment")
        update: dict[str, Any] = {"nvc_prev_function_token": token}
        if os.environ.get("NEMOTRON_VOICECHAT_DEBUG_FUNCTION_TIMELINE", "0") == "1":
            existing = kwargs.get("nvc_function_tokens")
            if isinstance(existing, torch.Tensor) and existing.numel() > 0:
                update["nvc_function_tokens"] = torch.cat([existing.reshape(-1).cpu(), token.reshape(1).cpu()])
            else:
                update["nvc_function_tokens"] = token.reshape(1).cpu()
        return update

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        for request_id in finished_req_ids:
            self._sessions.pop(str(request_id), None)
            self._duplex_previous_text_tokens.pop(str(request_id), None)

    # ------------------------------------------------------------------
    # Weights.
    # ------------------------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Route the unified checkpoint into the thinker's components.

        * ``stt_model.llm.*`` -> vLLM NemotronH (renamed to ``backbone.*``; the
          separate ``stt_model.embed_tokens`` doubles as ``backbone.embeddings``
          since the checkpoint keeps a single shared text embedding).
        * ``stt_model.lm_head.weight`` -> the backbone's ``lm_head``.
        * ``stt_model.embed_tokens/function_head`` -> the fusion embedding /
          function head.
        * ``stt_model.perception.*`` -> the vendored Conformer.
        * everything else (``rnnt_*``, ``tts_model.*``) is skipped lazily.
        """
        loaded: set[str] = set()
        lm_weights: list[tuple[str, torch.Tensor]] = []
        perception_state: dict[str, torch.Tensor] = {}
        embed_weight: torch.Tensor | None = None

        for name, tensor in weights:
            if name.startswith("stt_model.llm."):
                lm_weights.append(("backbone." + name.removeprefix("stt_model.llm."), tensor))
            elif name == "stt_model.lm_head.weight":
                lm_weights.append(("lm_head.weight", tensor))
            elif name == "stt_model.embed_tokens.weight":
                embed_weight = tensor
            elif name == "stt_model.function_head.weight":
                with torch.no_grad():
                    self.function_head.weight.copy_(tensor)
                loaded.add("function_head.weight")
            elif name.startswith("stt_model.perception."):
                perception_state[name.removeprefix("stt_model.perception.")] = tensor

        if embed_weight is None:
            raise ValueError("Checkpoint is missing 'stt_model.embed_tokens.weight'.")
        if not lm_weights:
            raise ValueError("Checkpoint is missing the 'stt_model.llm.*' backbone subtree.")
        if not perception_state:
            raise ValueError("Checkpoint is missing the 'stt_model.perception.*' subtree.")
        if "function_head.weight" not in loaded and self._use_function_head:
            raise ValueError("Checkpoint is missing 'stt_model.function_head.weight'.")

        with torch.no_grad():
            self.embed_tokens.weight.copy_(embed_weight)
        loaded.add("embed_tokens.weight")

        # The vLLM NemotronH model expects an input embedding; the checkpoint's
        # shared embed_tokens fills that slot (inputs_embeds bypass it at runtime).
        lm_weights.append(("backbone.embeddings.weight", embed_weight))
        lm_loaded = self.language_model.load_weights(iter(lm_weights))
        loaded |= {f"language_model.{n}" for n in lm_loaded}

        missing, unexpected = self.perception.load_state_dict(perception_state, strict=False)
        if missing:
            raise ValueError(f"NemotronVoiceChat perception is missing weights: {sorted(missing)[:10]} ...")
        if unexpected:
            logger.warning(
                "NemotronVoiceChat perception ignoring %d unexpected tensors (e.g. %s)",
                len(unexpected),
                sorted(unexpected)[:5],
            )
        device = self.vllm_config.device_config.device
        self.perception.to(device=device, dtype=self._dtype).eval()
        # Keep the deterministic mel frontend in fp32, matching NeMo's
        # PerceptionCacheManager.  Its output is cast only at the encoder edge.
        self._streaming_preprocessor.to(device=device, dtype=torch.float32).eval()
        self.embed_tokens.to(device=device, dtype=self._dtype)
        self.function_head.to(device=device, dtype=self._dtype)
        loaded |= {f"perception.{n}" for n in perception_state}
        logger.info(
            "NemotronVoiceChat thinker loaded: %d backbone tensors, %d perception tensors, "
            "fusion weights (text=%.1f, audio=%.1f, function=%.1f)",
            len(lm_weights),
            len(perception_state),
            self._w_text,
            self._w_audio,
            self._w_func,
        )
        return loaded
