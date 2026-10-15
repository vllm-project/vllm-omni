# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Base voice-clone reference artifacts: in-context codec codes and the ECAPA x-vector.

The Talker computes them while it builds a prompt, inside the engine step. With
``ref_audio_encode_in_api_server`` the API server computes them before it
submits the request (``Qwen3TTSReferenceEncoder`` behind the shared
``ReferenceEncodingService``) and sends them in ``voice_clone_prompt``, so no
Talker step waits for an encode. Both sides run the functions below with the
same weights and dtypes, so the artifacts are identical.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from vllm.multimodal.audio import AudioResampler

from vllm_omni.utils.shared_reference_encoder import WireFormat

# The Talker runs both reference encoders in BF16.
REFERENCE_DTYPE = torch.bfloat16
# Shorter clips are rejected (``Qwen3TTSPromptEmbedsBuilder.normalize_ref_audio``).
MIN_REFERENCE_SAMPLES = 1024


def encode_reference_codes(
    encoder: nn.Module,
    feature_extractor,
    wavs: list[np.ndarray],
    sr: int,
    *,
    device: torch.device,
    downsample_rate: int,
    num_quantizers: int,
) -> list[torch.Tensor]:
    """Speech-tokenizer codes ``[frames, num_quantizers]`` of each same-rate clip."""
    target_sr = int(feature_extractor.sampling_rate)
    if int(sr) != target_sr:
        resampler = AudioResampler(target_sr=target_sr)
        wavs = [resampler.resample(w.astype(np.float32), orig_sr=int(sr)) for w in wavs]

    inputs = feature_extractor(raw_audio=wavs, sampling_rate=target_sr, return_tensors="pt")
    input_values = inputs["input_values"].squeeze(1).to(device=device, dtype=REFERENCE_DTYPE)
    valid_code_lengths = [(int(wav.size) + downsample_rate - 1) // downsample_rate for wav in wavs]

    with torch.inference_mode():
        encoded = encoder.encode(input_values=input_values.unsqueeze(1), return_dict=True)

    audio_codes = encoded.audio_codes[:, :num_quantizers]
    return [
        code[:, : int(valid_len)].transpose(0, 1).to(dtype=torch.long).contiguous()
        for code, valid_len in zip(audio_codes, valid_code_lengths, strict=True)
    ]


def reference_speaker_embedding(
    speaker_encoder: nn.Module,
    wav: np.ndarray,
    sr: int,
    *,
    target_sr: int,
    resampler: AudioResampler | None,
    device: torch.device,
    dtype: torch.dtype,
    mel_device: torch.device,
) -> torch.Tensor:
    """ECAPA x-vector of one clip (the official 24 kHz, 128-bin mel front end)."""
    from .prompt_embeds_builder import mel_spectrogram

    if int(sr) != int(target_sr):
        assert resampler is not None
        wav = resampler.resample(wav.astype(np.float32), orig_sr=int(sr))
    wav_tensor = torch.from_numpy(wav).to(device=mel_device, dtype=torch.float32).unsqueeze(0)
    mels = mel_spectrogram(
        wav_tensor,
        n_fft=1024,
        num_mels=128,
        sampling_rate=24000,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=12000,
    ).transpose(1, 2)
    spk = speaker_encoder(mels.to(device=device, dtype=dtype))[0]
    return spk.to(dtype=dtype)


@dataclass
class ReferenceArtifacts:
    """What a Base request needs from its reference clip."""

    ref_code: torch.Tensor  # [frames, num_quantizers] int64, CPU
    ref_spk_embedding: torch.Tensor  # [hidden] BF16, CPU

    @property
    def nbytes(self) -> int:
        return self.ref_code.nbytes + self.ref_spk_embedding.nbytes

    def voice_clone_prompt(self, icl_mode: bool) -> dict:
        """Engine-request form (plain lists survive the EngineCore IPC boundary).

        ``ref_code`` is the per-item ``[frames, num_quantizers]`` list that both
        the prompt-length estimate and the Talker read. BF16 values are exact
        in float.
        """
        return {
            "ref_code": self.ref_code.tolist(),
            "ref_spk_embedding": self.ref_spk_embedding.float().tolist(),
            "icl_mode": bool(icl_mode),
        }


def _checkpoint_tensors(directory: str, prefix: str) -> Iterator[tuple[str, torch.Tensor]]:
    from safetensors import safe_open

    files = sorted(glob.glob(os.path.join(directory, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No safetensors checkpoint in {directory}")
    for path in files:
        with safe_open(path, framework="pt") as f:
            for name in f.keys():  # noqa: SIM118 - safe_open is not a mapping
                if name.startswith(prefix):
                    yield name[len(prefix) :], f.get_tensor(name)


def _load_module_state(module: nn.Module, tensors: Iterator[tuple[str, torch.Tensor]]) -> int:
    """Copy checkpoint tensors into parameters and buffers, as the Talker's loader does."""
    targets = dict(module.named_parameters())
    targets.update(dict(module.named_buffers()))
    loaded = 0
    with torch.no_grad():
        for name, tensor in tensors:
            target = targets.get(name)
            if target is not None:
                target.copy_(tensor)
                loaded += 1
    return loaded


class Qwen3TTSReferenceEncoder:
    """The Talker's reference encoders loaded from a checkpoint, outside the engine."""

    def __init__(
        self,
        model_path: str,
        device: torch.device | str,
        construct_dtype: torch.dtype = REFERENCE_DTYPE,
        max_code_batch: int = 16,
    ) -> None:
        """``construct_dtype`` is the Talker's model dtype, the default dtype its modules are built in."""
        from transformers import AutoFeatureExtractor
        from vllm.utils.torch_utils import set_default_torch_dtype

        from vllm_omni.model_executor.models.common.ecapa_tdnn import ECAPATimeDelayNet

        from .configuration_qwen3_tts import Qwen3TTSConfig
        from .tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
        from .tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Encoder

        self.device = torch.device(device)
        config = Qwen3TTSConfig.from_pretrained(model_path)
        self.speaker_sample_rate = int(getattr(config.speaker_encoder_config, "sample_rate", 24000))
        # Same construction and load order as the Talker: build in its model
        # dtype, copy the checkpoint tensors, then cast to BF16.
        with set_default_torch_dtype(construct_dtype):
            self.speaker_encoder = ECAPATimeDelayNet(config.speaker_encoder_config).eval()
            tokenizer_config = Qwen3TTSTokenizerV2Config.from_pretrained(model_path, subfolder="speech_tokenizer")
            self.encoder = Qwen3TTSTokenizerV2Encoder._from_config(tokenizer_config.encoder_config).eval()
        if _load_module_state(self.speaker_encoder, _checkpoint_tensors(model_path, "speaker_encoder.")) == 0:
            raise ValueError(f"{model_path} has no speaker_encoder weights for Base voice cloning")
        self.speaker_encoder.to(device=self.device, dtype=REFERENCE_DTYPE)

        subfolder = os.path.join(model_path, "speech_tokenizer")
        self.encoder.to(dtype=REFERENCE_DTYPE)
        _load_module_state(self.encoder, _checkpoint_tensors(subfolder, "encoder."))
        self.encoder.to(device=self.device, dtype=REFERENCE_DTYPE)
        self.num_quantizers = int(tokenizer_config.encoder_valid_num_quantizers)
        self.downsample_rate = int(tokenizer_config.encode_downsample_rate)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path, subfolder="speech_tokenizer")
        self._resamplers: dict[int, AudioResampler] = {}
        self.namespace = f"qwen3_tts:{os.path.realpath(model_path)}"
        self.max_code_batch = int(max_code_batch)

    def _resampler(self, sr: int) -> AudioResampler | None:
        if int(sr) == self.speaker_sample_rate:
            return None
        return self._resamplers.setdefault(int(sr), AudioResampler(target_sr=self.speaker_sample_rate))

    @torch.inference_mode()
    def encode(self, wav: np.ndarray, sr: int) -> ReferenceArtifacts:
        """Artifacts of one clip, as the Talker's single-request path computes them."""
        (codes,) = encode_reference_codes(
            self.encoder,
            self.feature_extractor,
            [wav],
            sr,
            device=self.device,
            downsample_rate=self.downsample_rate,
            num_quantizers=self.num_quantizers,
        )
        spk = reference_speaker_embedding(
            self.speaker_encoder,
            wav,
            sr,
            target_sr=self.speaker_sample_rate,
            resampler=self._resampler(sr),
            device=self.device,
            dtype=REFERENCE_DTYPE,
            mel_device=self.device,
        )
        return ReferenceArtifacts(codes.cpu(), spk.reshape(-1).cpu())

    def encode_batch(self, items: list[tuple[np.ndarray, int]]) -> Iterator[ReferenceArtifacts | BaseException]:
        """One result per clip, in order, each yielded as soon as it is ready.

        As in the Talker's batched preprocessing, codes of same-rate clips are
        encoded together (a batch's BF16 kernels may round differently from a
        single clip's) and x-vectors one clip at a time (padding would change
        the ECAPA statistics pooling). A failing clip does not fail the others.
        """
        with torch.inference_mode():
            codes: dict[int, torch.Tensor | BaseException] = {}
            by_rate: dict[int, list[int]] = {}
            for index, (_wav, sr) in enumerate(items):
                by_rate.setdefault(int(sr), []).append(index)
            for sr, indices in by_rate.items():
                for start in range(0, len(indices), self.max_code_batch):
                    group = indices[start : start + self.max_code_batch]
                    try:
                        encoded = encode_reference_codes(
                            self.encoder,
                            self.feature_extractor,
                            [items[i][0] for i in group],
                            sr,
                            device=self.device,
                            downsample_rate=self.downsample_rate,
                            num_quantizers=self.num_quantizers,
                        )
                        codes.update(zip(group, (c.cpu() for c in encoded), strict=True))
                    except Exception as exc:  # noqa: BLE001 - reported to these requests only
                        codes.update(dict.fromkeys(group, exc))
            for index, (wav, sr) in enumerate(items):
                ref_code = codes[index]
                if isinstance(ref_code, BaseException):
                    yield ref_code
                    continue
                try:
                    spk = reference_speaker_embedding(
                        self.speaker_encoder,
                        wav,
                        sr,
                        target_sr=self.speaker_sample_rate,
                        resampler=self._resampler(sr),
                        device=self.device,
                        dtype=REFERENCE_DTYPE,
                        mel_device=self.device,
                    )
                except Exception as exc:  # noqa: BLE001
                    yield exc
                    continue
                yield ReferenceArtifacts(ref_code, spk.reshape(-1).cpu())


def _artifact_to_arrays(artifacts: ReferenceArtifacts) -> list[np.ndarray]:
    return [artifacts.ref_code.to(torch.int32).numpy(), artifacts.ref_spk_embedding.to(torch.float32).numpy()]


def _artifact_from_arrays(arrays: list[np.ndarray]) -> ReferenceArtifacts:
    # BF16 values are exact in float32.
    return ReferenceArtifacts(
        torch.from_numpy(arrays[0].copy()).long(), torch.from_numpy(arrays[1].copy()).to(REFERENCE_DTYPE)
    )


# (waveform, sample rate) clips and ReferenceArtifacts across processes.
QWEN3_TTS_REFERENCE_WIRE = WireFormat(
    clip_to_array=lambda clip: (clip[0], int(clip[1])),
    clip_from_array=lambda array, sr: (array.copy(), int(sr)),
    artifact_to_arrays=_artifact_to_arrays,
    artifact_from_arrays=_artifact_from_arrays,
)
