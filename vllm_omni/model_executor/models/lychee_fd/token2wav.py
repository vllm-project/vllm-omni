# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native response-owned Lychee-FD Flow/DiT + HiFT streaming stage."""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from vllm.model_executor.models.interfaces import SupportsPP

from vllm_omni.model_executor.models.output_templates import OmniOutput

_FLOW_CONTRACT = {
    "flow": {
        "input_size": 512,
        "output_size": 80,
        "spk_embed_dim": 192,
        "output_type": "mel",
        "vocab_size": 6561,
        "encoder": {
            "input_size": 512,
            "output_size": 512,
            "input_layer": "linear",
            "pre_lookahead_len": 3,
            "num_blocks": 6,
            "num_up_blocks": 4,
            "up_stride": 2,
            "up_scale_factor": 2,
            "attention_heads": 8,
            "pos_enc_layer_type": "rel_pos_espnet",
            "selfattention_layer_type": "rel_selfattn",
            "key_bias": True,
            "linear_units": 2048,
            "dropout_rate": 0.1,
            "positional_dropout_rate": 0.1,
            "attention_dropout_rate": 0.1,
            "normalize_before": True,
        },
        "decoder": {
            "inference_cfg_rate": 0.7,
            "estimator": {
                "in_channels": 320,
                "out_channels": 80,
                "mlp_ratio": 4.0,
                "depth": 16,
                "num_heads": 8,
                "head_dim": 64,
                "hidden_size": 512,
            },
        },
    },
}
_FLOW_TAGS = {
    "flow": "!new:cosyvoice2.flow.flow.CausalMaskedDiffWithXvec",
    "flow.encoder": "!new:cosyvoice2.transformer.upsample_encoder_v2.UpsampleConformerEncoderV2",
    "flow.decoder": "!new:cosyvoice2.flow.flow_matching.CausalConditionalCFM",
    "flow.decoder.estimator": "!new:cosyvoice2.flow.decoder_dit.DiT",
}


def validate_native_flow_config(path: str | Path) -> None:
    """Parse inert YAML nodes and reject settings the native graph cannot honor."""
    import yaml

    with Path(path).open() as stream:
        node = yaml.compose(stream, Loader=yaml.BaseLoader)

    def validate(value, expected, name):
        if isinstance(expected, dict):
            if not isinstance(value, yaml.MappingNode):
                raise ValueError(f"Native Lychee Token2Wav {name} must be a mapping")
            tag = _FLOW_TAGS.get(name)
            if tag is not None and value.tag != tag:
                raise ValueError(f"Native Lychee Token2Wav {name} constructor must be {tag}")
            fields = {}
            for key, item in value.value:
                if not isinstance(key, yaml.ScalarNode) or key.value in fields:
                    raise ValueError(f"Native Lychee Token2Wav {name} contains an invalid or duplicate setting")
                fields[key.value] = item
            if fields.keys() != expected.keys():
                raise ValueError(
                    f"Native Lychee Token2Wav {name} settings mismatch: "
                    f"missing={sorted(expected.keys() - fields.keys())}, "
                    f"unsupported={sorted(fields.keys() - expected.keys())}"
                )
            for key, item in expected.items():
                validate(fields[key], item, f"{name}.{key}" if name else key)
            return
        if not isinstance(value, yaml.ScalarNode):
            raise ValueError(f"Native Lychee Token2Wav {name} must be scalar")
        text = value.value
        try:
            if isinstance(expected, bool):
                actual = {"true": True, "false": False}[text.lower()]
            elif isinstance(expected, int):
                actual = int(text)
            elif isinstance(expected, float):
                actual = float(text)
            else:
                actual = text
        except (ValueError, KeyError) as exc:
            raise ValueError(f"Native Lychee Token2Wav {name} has an invalid value {text!r}") from exc
        if actual != expected:
            raise ValueError(f"Native Lychee Token2Wav {name}={actual!r} is unsupported; expected {expected!r}")

    validate(node, _FLOW_CONTRACT, "")


@dataclass(frozen=True)
class LycheeSpeechOwner:
    session_id: str
    response_id: str
    execution_epoch: int
    session_epoch: int = 0
    response_number: int = 1

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> LycheeSpeechOwner:
        session = metadata.get("session_id")
        response = metadata.get("response_id")
        epoch = metadata.get("execution_epoch")
        if not isinstance(session, str) or not session or not isinstance(response, str) or not response:
            raise ValueError("Lychee Token2Wav requires session_id and response_id")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("Lychee Token2Wav execution_epoch must be a nonnegative integer")
        session_epoch = metadata.get("session_epoch", 0)
        if isinstance(session_epoch, bool) or not isinstance(session_epoch, int) or session_epoch < 0:
            raise ValueError("Lychee Token2Wav session_epoch must be a nonnegative integer")
        response_number = metadata.get("response_number")
        if isinstance(response_number, bool) or not isinstance(response_number, int) or response_number < 1:
            raise ValueError("Lychee Token2Wav requires positive monotonic response_number")
        return cls(session, response, epoch, session_epoch, response_number)


@dataclass
class LycheeSpeechState:
    owner: LycheeSpeechOwner
    prompt_wav: str
    request_id: str
    pending_tokens: list[int] = field(default_factory=list)
    stream_cache: dict[str, torch.Tensor] | None = None
    hift_cache: dict[str, torch.Tensor] = field(default_factory=dict)
    chunk_seq: int = -1
    finished: bool = False


class LycheeToken2WavCore(nn.Module):
    """The released synthesis graph, with request-owned streaming tensors."""

    sample_rate = 24000
    chunk_size = 25
    pre_lookahead_len = 3
    mel_cache_len = 8
    source_cache_len = 3840
    estimator_cache_keep = 100

    def __init__(self, model_path: str, *, device: str = "cuda", float16: bool = False, chunk_size: int = 25):
        super().__init__()
        if type(chunk_size) is not int or not 1 <= chunk_size <= 25:
            raise ValueError("Token2Wav vocoder hop must be an integer in [1,25]")
        self.chunk_size = chunk_size
        self.model_path = Path(model_path)
        validate_native_flow_config(self.model_path / "flow.yaml")
        self.device = torch.device(device)
        self.float16 = float16
        self.prompt_cache: OrderedDict[str, tuple[torch.Tensor, ...]] = OrderedDict()
        self.flow: nn.Module | None = None
        self.hift: nn.Module | None = None
        self.audio_tokenizer = None
        self.spk_model = None
        self.speech_window = torch.from_numpy(np.hamming(2 * self.source_cache_len)).to(
            self.device, dtype=torch.float32
        )

    def shutdown(self) -> None:
        if self.hift is not None:
            cleanup = getattr(self.hift, "shutdown", None)
            if cleanup is not None:
                cleanup()

    def load_models(self) -> None:
        if self.flow is not None:
            return
        from vllm.utils.torch_utils import set_default_torch_dtype

        # The AR stage uses BF16, but the released synthesis reference is FP32.
        # Build its random-noise buffers and weights in FP32 even inside vLLM loading.
        with set_default_torch_dtype(torch.float32):
            self._load_models_fp32()

    def _load_models_fp32(self) -> None:
        import onnxruntime
        import s3tokenizer

        from .token2wav_modules.cosyvoice2.flow.decoder_dit import DiT
        from .token2wav_modules.cosyvoice2.flow.flow import CausalMaskedDiffWithXvec
        from .token2wav_modules.cosyvoice2.flow.flow_matching import CausalConditionalCFM
        from .token2wav_modules.cosyvoice2.transformer.upsample_encoder_v2 import UpsampleConformerEncoderV2
        from .token2wav_modules.flashcosyvoice.modules.hifigan import HiFTGenerator

        self.audio_tokenizer = (
            s3tokenizer.load_model(str(self.model_path / "speech_tokenizer_v2_25hz.onnx")).to(self.device).eval()
        )
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        self.spk_model = onnxruntime.InferenceSession(
            str(self.model_path / "campplus.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )
        encoder = UpsampleConformerEncoderV2(
            input_size=512,
            output_size=512,
            input_layer="linear",
            pre_lookahead_len=3,
            num_blocks=6,
            num_up_blocks=4,
            up_stride=2,
            up_scale_factor=2,
            attention_heads=8,
            pos_enc_layer_type="rel_pos_espnet",
            selfattention_layer_type="rel_selfattn",
            key_bias=True,
            linear_units=2048,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalize_before=True,
        )
        estimator = DiT(
            in_channels=320, out_channels=80, mlp_ratio=4.0, depth=16, num_heads=8, head_dim=64, hidden_size=512
        )
        decoder = CausalConditionalCFM(estimator=estimator, inference_cfg_rate=0.7)
        flow = CausalMaskedDiffWithXvec(
            input_size=512,
            output_size=80,
            spk_embed_dim=192,
            output_type="mel",
            vocab_size=6561,
            encoder=encoder,
            decoder=decoder,
        )
        if self.float16:
            flow.half()
        flow.load_state_dict(
            torch.load(self.model_path / "flow.pt", map_location="cpu", weights_only=True), strict=True
        )
        flow = flow.to(self.device).eval()
        hift = HiFTGenerator()
        weights = torch.load(self.model_path / "hift.pt", map_location="cpu", weights_only=True)
        hift.load_state_dict({key.removeprefix("generator."): value for key, value in weights.items()}, strict=True)
        self.hift = hift.to(self.device).eval()
        self.flow = flow

    def prepare_prompt(self, prompt_wav: str) -> tuple[torch.Tensor, ...]:
        if prompt_wav in self.prompt_cache:
            self.prompt_cache.move_to_end(prompt_wav)
            return self.prompt_cache[prompt_wav]
        self.load_models()
        import onnxruntime
        import s3tokenizer
        import soundfile as sf
        import torchaudio
        import torchaudio.compliance.kaldi as kaldi

        from .token2wav_modules.flashcosyvoice.utils.audio import mel_spectrogram

        audio = s3tokenizer.load_audio(prompt_wav, sr=16000)
        mels = s3tokenizer.log_mel_spectrogram(audio)
        mels, lens = s3tokenizer.padding([mels])
        tokens, _ = self.audio_tokenizer.quantize(mels.to(self.device), lens.to(self.device))
        features = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
        features -= features.mean(dim=0, keepdim=True)
        if not isinstance(self.spk_model, onnxruntime.InferenceSession):
            raise TypeError("Lychee speaker frontend must be a native ONNX Runtime session")
        speaker = torch.tensor(
            self.spk_model.run(None, {self.spk_model.get_inputs()[0].name: features.unsqueeze(0).cpu().numpy()})[0],
            device=self.device,
        )
        waveform, sr = sf.read(prompt_wav, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(waveform.mean(axis=1)).unsqueeze(0)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        mel = mel_spectrogram(waveform).transpose(1, 2).to(self.device)
        target_len = tokens.shape[1] * self.flow.up_rate
        mel = torch.nn.functional.pad(mel, (0, 0, 0, target_len - mel.shape[1]), mode="replicate")
        self.prompt_cache[prompt_wav] = (tokens, speaker, mel)
        while len(self.prompt_cache) > 4:
            self.prompt_cache.popitem(last=False)
        return tokens, speaker, mel

    @torch.inference_mode()
    def setup(self, state: LycheeSpeechState) -> None:
        tokens, speaker, mel = self.prepare_prompt(state.prompt_wav)
        lookahead = self.pre_lookahead_len
        if tokens.shape[1] == 0:
            raise ValueError("Lychee Token2Wav speaker prompt must contain codec tokens")
        tail = tokens.repeat(1, (lookahead + tokens.shape[1] - 1) // tokens.shape[1])[:, :lookahead]
        state.stream_cache = self.flow.setup_cache(torch.cat((tokens, tail), dim=1), mel, speaker, n_timesteps=10)
        # The reference graph reuses its scratch buffers; persistent state must own slices.
        state.stream_cache = {key: value.clone() for key, value in state.stream_cache.items()}
        state.hift_cache = {
            "mel": torch.zeros(1, 80, 0, device=self.device),
            "source": torch.zeros(1, 1, 0, device=self.device),
            "speech": torch.zeros(1, 0, device=self.device),
        }

    @torch.inference_mode()
    def synthesize(self, tokens: list[int], state: LycheeSpeechState, *, final: bool) -> torch.Tensor:
        if state.stream_cache is None:
            self.setup(state)
        _, speaker, prompt_mel = self.prepare_prompt(state.prompt_wav)
        token_tensor = torch.tensor([tokens], dtype=torch.int32, device=self.device)
        with torch.amp.autocast(self.device.type, dtype=torch.float16, enabled=self.float16):
            chunk_mel, cache = self.flow.inference_chunk(
                token=token_tensor, spk=speaker, cache=state.stream_cache, last_chunk=final, n_timesteps=10
            )
        estimator = cache["estimator_att_cache"]
        prompt_len = prompt_mel.shape[1]
        state.stream_cache = {key: value.clone() for key, value in cache.items() if key != "estimator_att_cache"}
        if estimator.shape[4] > prompt_len + self.estimator_cache_keep:
            # torch.cat already allocates response-owned storage. Cloning this
            # bounded cache again briefly duplicates almost a GiB for the default voice.
            state.stream_cache["estimator_att_cache"] = torch.cat(
                (estimator[..., :prompt_len, :], estimator[..., -self.estimator_cache_keep :, :]), dim=4
            )
        else:
            state.stream_cache["estimator_att_cache"] = estimator.clone()
        previous = state.hift_cache
        mel = torch.cat((previous["mel"], chunk_mel), dim=2)
        speech, source = self.hift(mel, previous["source"])
        overlap = previous["speech"].shape[-1]
        if overlap:
            if speech.shape[-1] < overlap:
                raise RuntimeError("HiFT returned fewer samples than the retained overlap")
            speech = speech.clone()
            speech[..., :overlap] = (
                speech[..., :overlap] * self.speech_window[:overlap]
                + previous["speech"][..., -overlap:]
                * self.speech_window[self.source_cache_len : self.source_cache_len + overlap]
            )
        state.hift_cache = {
            "mel": mel[..., -self.mel_cache_len :].clone(),
            "source": source[..., -self.source_cache_len :].clone(),
            "speech": speech[..., -self.source_cache_len :].clone(),
        }
        if not final:
            speech = speech[..., : -self.source_cache_len]
        return speech.flatten()

    @staticmethod
    def flush_tail(state: LycheeSpeechState) -> torch.Tensor:
        return state.hift_cache.get("speech", torch.empty(0)).flatten().clone()


class LycheeToken2WavSessionStore:
    """Response-owned caches and bounded per-session lifecycle fences."""

    def __init__(self, core: Any, default_prompt_wav: str):
        self.core = core
        self.default_prompt_wav = default_prompt_wav
        self.states: dict[LycheeSpeechOwner, LycheeSpeechState] = {}
        self.active: dict[str, LycheeSpeechOwner] = {}
        self.highwater: dict[str, tuple[int, int, int]] = {}
        self.closed_highwater: dict[str, tuple[int, int, int]] = {}
        self.request_sessions: dict[str, set[str]] = {}
        self.session_requests: dict[str, str] = {}
        # The engine never dispatches retired owners again. Keep a bounded
        # diagnostic fence so direct callers also cannot resurrect a recent abort.
        self.retired_requests: OrderedDict[str, None] = OrderedDict()

    @staticmethod
    def fence(owner: LycheeSpeechOwner) -> tuple[int, int, int]:
        return owner.session_epoch, owner.execution_epoch, owner.response_number

    def cancel(self, owner: LycheeSpeechOwner) -> None:
        self.states.pop(owner, None)
        if self.active.get(owner.session_id) == owner:
            self.active.pop(owner.session_id, None)
        boundary = self.fence(owner)
        self.closed_highwater[owner.session_id] = max(
            boundary, self.closed_highwater.get(owner.session_id, (-1, -1, -1))
        )

    def finish_requests(self, request_ids: Iterable[str]) -> None:
        for request_id in set(request_ids):
            sessions = self.request_sessions.pop(request_id, set())
            for owner, state in list(self.states.items()):
                if state.request_id == request_id:
                    sessions.add(owner.session_id)
                    self.states.pop(owner)
            for session_id in sessions:
                self.active.pop(session_id, None)
                self.highwater.pop(session_id, None)
                self.closed_highwater.pop(session_id, None)
                self.session_requests.pop(session_id, None)
            self.retired_requests[request_id] = None
            self.retired_requests.move_to_end(request_id)
            while len(self.retired_requests) > 128:
                self.retired_requests.popitem(last=False)

    def process(self, tokens: list[int], metadata: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        owner = LycheeSpeechOwner.from_metadata(metadata)
        sequence = metadata.get("chunk_seq")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("Lychee Token2Wav chunk_seq must be a nonnegative integer")
        request_id = metadata.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("Lychee Token2Wav requires its resident engine request_id")
        result = dict(metadata)
        result.update(sample_rate=self.core.sample_rate, discarded=False)
        boundary = self.fence(owner)
        last = self.highwater.get(owner.session_id, (-1, -1, -1))
        closed = self.closed_highwater.get(owner.session_id, (-1, -1, -1))
        previous = self.active.get(owner.session_id)
        if (
            request_id in self.retired_requests
            or boundary < last
            or boundary <= closed
            or (boundary == last and previous is not None and previous != owner)
        ):
            result["discarded"] = True
            return torch.empty(0), result
        if self.states and owner.session_id not in self.active and not metadata.get("cancel", False):
            raise RuntimeError("Lychee Token2Wav correctness profile supports one active response")
        previous_request = self.session_requests.get(owner.session_id)
        if previous_request is not None and previous_request != request_id:
            if owner.session_epoch <= last[0]:
                result["discarded"] = True
                return torch.empty(0), result
            self.request_sessions.get(previous_request, set()).discard(owner.session_id)
        self.session_requests[owner.session_id] = request_id
        self.request_sessions.setdefault(request_id, set()).add(owner.session_id)
        if boundary > last:
            if previous is not None:
                self.cancel(previous)
            self.highwater[owner.session_id] = boundary
        if metadata.get("cancel", False):
            self.cancel(owner)
            return torch.empty(0), result
        state = self.states.get(owner)
        if state is None:
            if self.states:
                raise RuntimeError("Lychee Token2Wav correctness profile supports one active response")
            state = LycheeSpeechState(owner, str(metadata.get("prompt_wav") or self.default_prompt_wav), request_id)
            self.states[owner] = state
            self.active[owner.session_id] = owner
        if sequence <= state.chunk_seq:
            result["discarded"] = True
            return torch.empty(0), result
        if sequence != state.chunk_seq + 1:
            self.cancel(owner)
            raise ValueError("Lychee Token2Wav chunk sequence has a gap")
        if any(isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < 6561 for token in tokens):
            self.cancel(owner)
            raise ValueError("Lychee Token2Wav codec IDs must lie in [0, 6561)")
        state.chunk_seq = sequence
        state.pending_tokens.extend(tokens)
        outputs: list[torch.Tensor] = []
        try:
            if metadata.get("final", False):
                if state.pending_tokens:
                    outputs.append(self.core.synthesize(state.pending_tokens, state, final=True))
                elif state.stream_cache is not None:
                    outputs.append(self.core.flush_tail(state))
                self.cancel(owner)
                state.finished = True
            else:
                required = self.core.chunk_size + self.core.pre_lookahead_len
                while len(state.pending_tokens) >= required:
                    outputs.append(self.core.synthesize(state.pending_tokens[:required], state, final=False))
                    del state.pending_tokens[: self.core.chunk_size]
        except Exception:
            self.cancel(owner)
            raise
        return torch.cat(outputs) if outputs else torch.empty(0), result


class LycheeToken2WavForConditionalGeneration(nn.Module, SupportsPP):
    """MRV2 non-AR model; input metadata carries exact response ownership."""

    requires_exact_input_shape = True
    have_multimodal_outputs = True
    enable_update_additional_information = True

    def __init__(self, *, vllm_config: Any, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        root = Path(vllm_config.model_config.model)
        model_path = os.environ.get("LYCHEEFD_TOKEN2WAV_PATH") or getattr(config, "token2wav_path", None)
        if model_path is None:
            model_path = root.parent / "token2wav"
        prompt = os.environ.get("LYCHEEFD_T2W_PROMPT_WAV") or getattr(config, "token2wav_prompt_wav", None)
        if prompt is None:
            candidates = [
                root.parent.parent / "Lychee-FD/frontend/public/clone_24k_mono/default_male.wav",
                Path(__file__).parents[1] / "step_audio2/assets/default_male.wav",
            ]
            prompt = next((path for path in candidates if path.is_file()), candidates[-1])
        device = str(vllm_config.device_config.device)
        # The released online service drains ten codecs per vocoder hop.
        # Standalone same-codec diagnostics retain their explicit25-codec default.
        hop_setting = os.environ.get("LYCHEEFD_TTS_VOCODER_HOP_SIZE")
        hop_size = int(hop_setting) if hop_setting is not None else getattr(config, "token2wav_vocoder_hop_size", 10)
        self.core = LycheeToken2WavCore(
            str(model_path),
            device=device,
            float16=bool(getattr(config, "token2wav_float16", False)),
            chunk_size=hop_size,
        )
        self.sessions = LycheeToken2WavSessionStore(self.core, str(prompt))
        self.make_empty_intermediate_tensors = lambda: None

    def get_language_model(self) -> nn.Module:
        return self

    def embed_input_ids(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return torch.zeros(input_ids.numel(), self.vllm_config.model_config.get_hidden_size(), device=input_ids.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        information = runtime_additional_information or []
        payloads = [info["lychee_t2w"] for info in information if isinstance(info.get("lychee_t2w"), Mapping)]
        if not payloads:
            # Engine profiling has no response owner. Codec ID zero remains a valid real input.
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [torch.empty(0, device=input_ids.device)],
                    "sr": [torch.tensor(24000)],
                },
            )
        if len(payloads) != 1:
            raise RuntimeError("Lychee Token2Wav supports B1 initially")
        metadata = payloads[0]
        tokens = [] if metadata.get("empty", False) else input_ids.detach().cpu().flatten().tolist()
        waveform, ownership = self.sessions.process(tokens, metadata)
        # The generation runner accepts only tensor leaves. Explicit dotted keys
        # survive its flattening and reconstruct ownership in the output processor.
        numeric_owner = {
            "session_epoch": ownership.get("session_epoch", 0),
            "execution_epoch": ownership["execution_epoch"],
            "response_number": ownership["response_number"],
            "chunk_seq": ownership["chunk_seq"],
            "tick": ownership.get("tick", -1),
            "final": bool(ownership.get("final", False)),
            "discarded": bool(ownership.get("discarded", False)),
            "num_samples": waveform.numel(),
        }
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": [waveform],
                "sr": [torch.tensor(24000)],
                **{f"chunk.lychee_t2w.{key}": torch.tensor(value) for key, value in numeric_owner.items()},
            },
        )

    def shutdown(self) -> None:
        self.core.shutdown()

    def on_requests_finished(self, request_ids: Iterable[str]) -> None:
        self.sessions.finish_requests(request_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> None:
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        self.core.load_models()
        return set(dict(self.named_parameters()))

    def load_weights_without_buffers(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return self.load_weights(weights)
