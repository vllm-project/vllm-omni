from __future__ import annotations

import json
import os
import sys
import tempfile
import warnings
import wave
from collections.abc import Callable, Generator, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .voxcpm_loader import (
    _build_prompt_cache_with_soundfile,
    _device_to_string,
    _force_cuda_available_for_npu,
    _import_voxcpm_audio_vae_classes,
    _import_voxcpm_base_model_class,
    _is_torchcodec_load_error,
    _normalize_dtype_name,
    _prepare_runtime_model_dir,
    _resolve_runtime_device,
)
from .voxcpm_runtime_utils import resolve_voxcpm_model_dir
from .voxcpm_stage_wrappers import _DirectVoxCPMAudioVAE, _DirectVoxCPMLatentGenerator

logger = init_logger(__name__)
_VOXCPM_LATENT_MAGIC = 131071


def _make_voxcpm_model_for_omni(base: type[Any]) -> type[Any]:
    """Subclass upstream VoxCPMModel: local ``_inference`` + ``latents_only`` prompt-cache generation."""

    from voxcpm.model.utils import get_dtype

    class VoxCPMModelForOmni(base):
        @torch.inference_mode()
        def build_prompt_cache(self, *args: Any, **kwargs: Any):
            try:
                return super().build_prompt_cache(*args, **kwargs)
            except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
                if not _is_torchcodec_load_error(exc):
                    raise
                return _build_prompt_cache_with_soundfile(self, *args, **kwargs)

        @torch.inference_mode()
        def _inference(
            self,
            text: torch.Tensor,
            text_mask: torch.Tensor,
            feat: torch.Tensor,
            feat_mask: torch.Tensor,
            min_len: int = 2,
            max_len: int = 2000,
            inference_timesteps: int = 10,
            cfg_value: float = 2.0,
            streaming: bool = False,
            streaming_prefix_len: int = 3,
        ) -> Generator[tuple[torch.Tensor, torch.Tensor | list[torch.Tensor]], None, None]:
            B, _, _, _ = feat.shape

            feat_embed = self.feat_encoder(feat)
            feat_embed = self.enc_to_lm_proj(feat_embed)

            scale_emb = self.config.lm_config.scale_emb if self.config.lm_config.use_mup else 1.0
            text_embed = self.base_lm.embed_tokens(text) * scale_emb
            combined_embed = text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed

            prefix_feat_cond = feat[:, -1, ...]
            pred_feat_seq: list[torch.Tensor] = []

            audio_patch_count = int(feat_mask.sum().item())
            if audio_patch_count > 0:
                context_len = min(streaming_prefix_len - 1, audio_patch_count)
                prompt_context_patches = list(feat[:, -context_len:, :, :].split(1, dim=1))
                pred_feat_seq = prompt_context_patches + pred_feat_seq

            enc_outputs, kv_cache_tuple = self.base_lm(
                inputs_embeds=combined_embed,
                is_causal=True,
            )
            self.base_lm.kv_cache.fill_caches(kv_cache_tuple)

            enc_outputs = self.fsq_layer(enc_outputs) * feat_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
            lm_hidden = enc_outputs[:, -1, :]

            residual_enc_outputs, residual_kv_cache_tuple = self.residual_lm(
                inputs_embeds=enc_outputs + feat_mask.unsqueeze(-1) * feat_embed,
                is_causal=True,
            )
            self.residual_lm.kv_cache.fill_caches(residual_kv_cache_tuple)
            residual_hidden = residual_enc_outputs[:, -1, :]

            for step_idx in tqdm(range(max_len)):
                dit_hidden = self.lm_to_dit_proj(lm_hidden) + self.res_to_dit_proj(residual_hidden)
                pred_feat = self.feat_decoder(
                    mu=dit_hidden,
                    patch_size=self.patch_size,
                    cond=prefix_feat_cond.transpose(1, 2).contiguous(),
                    n_timesteps=inference_timesteps,
                    cfg_value=cfg_value,
                ).transpose(1, 2)

                curr_embed = self.enc_to_lm_proj(self.feat_encoder(pred_feat.unsqueeze(1)))
                pred_feat_seq.append(pred_feat.unsqueeze(1))
                prefix_feat_cond = pred_feat

                if streaming:
                    pred_feat_chunk = torch.cat(pred_feat_seq[-streaming_prefix_len:], dim=1)
                    feat_pred = rearrange(pred_feat_chunk, "b t p d -> b d (t p)", b=B, p=self.patch_size)
                    yield feat_pred, pred_feat_seq

                stop_flag = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden))).argmax(dim=-1)[0].cpu().item()
                if step_idx > min_len and stop_flag == 1:
                    break

                lm_hidden = self.base_lm.forward_step(
                    curr_embed[:, 0, :],
                    torch.tensor([self.base_lm.kv_cache.step()], device=curr_embed.device),
                ).clone()
                lm_hidden = self.fsq_layer(lm_hidden)
                residual_hidden = self.residual_lm.forward_step(
                    lm_hidden + curr_embed[:, 0, :],
                    torch.tensor([self.residual_lm.kv_cache.step()], device=curr_embed.device),
                ).clone()

            if not streaming:
                pred_feat_seq_cat = torch.cat(pred_feat_seq, dim=1)
                feat_pred = rearrange(pred_feat_seq_cat, "b t p d -> b d (t p)", b=B, p=self.patch_size)
                yield feat_pred, pred_feat_seq_cat.squeeze(0).cpu()

        @torch.inference_mode()
        def generate_latents_with_prompt_cache(
            self,
            target_text: str,
            prompt_cache: dict,
            min_len: int = 2,
            max_len: int = 2000,
            inference_timesteps: int = 10,
            cfg_value: float = 2.0,
            retry_badcase: bool = False,
            retry_badcase_max_times: int = 3,
            retry_badcase_ratio_threshold: float = 6.0,
            streaming_prefix_len: int = 3,
        ) -> tuple[None, torch.Tensor, torch.Tensor]:
            return next(
                self._generate_with_prompt_cache(
                    target_text=target_text,
                    prompt_cache=prompt_cache,
                    min_len=min_len,
                    max_len=max_len,
                    inference_timesteps=inference_timesteps,
                    cfg_value=cfg_value,
                    retry_badcase=retry_badcase,
                    retry_badcase_max_times=retry_badcase_max_times,
                    retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                    streaming=False,
                    streaming_prefix_len=streaming_prefix_len,
                    latents_only=True,
                )
            )

        @torch.inference_mode()
        def generate_latents_with_prompt_cache_streaming(
            self,
            target_text: str,
            prompt_cache: dict,
            min_len: int = 2,
            max_len: int = 2000,
            inference_timesteps: int = 10,
            cfg_value: float = 2.0,
            retry_badcase: bool = False,
            retry_badcase_max_times: int = 3,
            retry_badcase_ratio_threshold: float = 6.0,
            streaming_prefix_len: int = 3,
        ) -> Generator[tuple[None, torch.Tensor, torch.Tensor], None, None]:
            return self._generate_with_prompt_cache(
                target_text=target_text,
                prompt_cache=prompt_cache,
                min_len=min_len,
                max_len=max_len,
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                retry_badcase=retry_badcase,
                retry_badcase_max_times=retry_badcase_max_times,
                retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                streaming=True,
                streaming_prefix_len=streaming_prefix_len,
                latents_only=True,
            )

        @torch.inference_mode()
        def _generate_with_prompt_cache(
            self,
            target_text: str,
            prompt_cache: dict,
            min_len: int = 2,
            max_len: int = 2000,
            inference_timesteps: int = 10,
            cfg_value: float = 2.0,
            retry_badcase: bool = False,
            retry_badcase_max_times: int = 3,
            retry_badcase_ratio_threshold: float = 6.0,
            streaming: bool = False,
            streaming_prefix_len: int = 3,
            latents_only: bool = False,
        ) -> Generator[tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | list[torch.Tensor]], None, None]:
            if retry_badcase and streaming:
                warnings.warn("Retry on bad cases is not supported in streaming mode, setting retry_badcase=False.")
                retry_badcase = False
            if prompt_cache is None:
                prompt_audio_feat = torch.empty((0, self.patch_size, self.audio_vae.latent_dim), dtype=torch.float32)
                text = target_text
            else:
                prompt_audio_feat = prompt_cache["audio_feat"]
                prompt_text = prompt_cache["prompt_text"]
                text = prompt_text + target_text

            text_token = torch.LongTensor(self.text_tokenizer(text))
            text_token = torch.cat(
                [
                    text_token,
                    torch.tensor([self.audio_start_token], dtype=torch.int32, device=text_token.device),
                ],
                dim=-1,
            )
            target_text_token = torch.LongTensor(self.text_tokenizer(target_text))

            audio_length = prompt_audio_feat.size(0)
            text_length = text_token.shape[0]
            text_pad_token = torch.zeros(audio_length, dtype=torch.int32, device=text_token.device)
            audio_pad_feat = torch.zeros(
                (text_token.shape[0], self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )
            text_token = torch.cat([text_token, text_pad_token])
            audio_feat = torch.cat([audio_pad_feat, prompt_audio_feat], dim=0)
            text_mask = (
                torch.cat([torch.ones(text_length), torch.zeros(audio_length)]).type(torch.int32).to(text_token.device)
            )
            audio_mask = (
                torch.cat([torch.zeros(text_length), torch.ones(audio_length)]).type(torch.int32).to(text_token.device)
            )

            text_token = text_token.unsqueeze(0).to(self.device)
            text_mask = text_mask.unsqueeze(0).to(self.device)
            audio_feat = audio_feat.unsqueeze(0).to(self.device).to(get_dtype(self.config.dtype))
            audio_mask = audio_mask.unsqueeze(0).to(self.device)

            target_text_length = len(self.text_tokenizer(target_text))
            retry_badcase_times = 0
            while retry_badcase_times < retry_badcase_max_times:
                inference_result = self._inference(
                    text_token,
                    text_mask,
                    audio_feat,
                    audio_mask,
                    min_len=min_len,
                    max_len=min(int(target_text_length * retry_badcase_ratio_threshold + 10), max_len),
                    inference_timesteps=inference_timesteps,
                    cfg_value=cfg_value,
                    streaming=streaming,
                    streaming_prefix_len=streaming_prefix_len,
                )
                if streaming:
                    patch_len = self.patch_size * self.chunk_size
                    for latent_pred, pred_audio_feat in inference_result:
                        if latents_only:
                            decode_audio = None
                            yield (decode_audio, target_text_token, latent_pred)
                        else:
                            decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
                            decode_audio = decode_audio[..., -patch_len:].squeeze(1).cpu()
                            yield (decode_audio, target_text_token, pred_audio_feat)
                    break

                latent_pred, pred_audio_feat = next(inference_result)
                if retry_badcase and pred_audio_feat.shape[0] >= target_text_length * retry_badcase_ratio_threshold:
                    ratio = pred_audio_feat.shape[0] / target_text_length
                    print(f"  Badcase detected, audio_text_ratio={ratio}, retrying...", file=sys.stderr)
                    retry_badcase_times += 1
                    continue
                break

            if not streaming:
                if latents_only:
                    decode_audio = None
                else:
                    decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
                    patch_len = self.patch_size * self.chunk_size
                    if audio_mask.sum().item() > 0:
                        decode_audio = decode_audio[..., patch_len * (streaming_prefix_len - 1) :].squeeze(1).cpu()
                    else:
                        decode_audio = decode_audio[..., :].squeeze(1).cpu()
                yield (decode_audio, target_text_token, pred_audio_feat)

    VoxCPMModelForOmni.__name__ = "VoxCPMModelForOmni"
    VoxCPMModelForOmni.__qualname__ = "VoxCPMModelForOmni"
    return VoxCPMModelForOmni


def _import_voxcpm_model_class() -> type[Any]:
    base = _import_voxcpm_base_model_class()
    return _make_voxcpm_model_for_omni(base)


def _load_native_voxcpm_model(
    model_path: str,
    *,
    device: torch.device,
    dtype: str | None,
):
    VoxCPMModel = _import_voxcpm_model_class()
    model_dir = resolve_voxcpm_model_dir(model_path)
    runtime_model_path = _prepare_runtime_model_dir(model_dir, target_device=device, target_dtype=dtype)

    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.set_device(device)

    with _force_cuda_available_for_npu(device):
        return VoxCPMModel.from_local(
            runtime_model_path,
            optimize=device.type == "cuda",
        )


def _load_native_voxcpm_latent_generator(
    model_path: str,
    *,
    device: torch.device,
    dtype: str | None,
) -> _DirectVoxCPMLatentGenerator:
    return _DirectVoxCPMLatentGenerator(_load_native_voxcpm_model(model_path, device=device, dtype=dtype))


def _load_native_voxcpm_audio_vae(
    model_path: str,
    *,
    device: torch.device,
) -> _DirectVoxCPMAudioVAE:
    AudioVAE, AudioVAEConfig = _import_voxcpm_audio_vae_classes()
    model_dir = resolve_voxcpm_model_dir(model_path)
    runtime_model_path = _prepare_runtime_model_dir(model_dir, target_device=device, target_dtype="float32")
    config_dict = json.loads((Path(runtime_model_path) / "config.json").read_text())
    audio_vae_config = config_dict.get("audio_vae_config")
    audio_vae = AudioVAE(config=AudioVAEConfig(**audio_vae_config)) if audio_vae_config is not None else AudioVAE()

    state_dict = torch.load(
        Path(runtime_model_path) / "audiovae.pth",
        map_location="cpu",
        weights_only=True,
    )["state_dict"]
    audio_vae.load_state_dict(state_dict, strict=True)
    audio_vae = audio_vae.to(device=device, dtype=torch.float32).eval()
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.set_device(device)
    patch_size = int(config_dict.get("patch_size", 2))
    return _DirectVoxCPMAudioVAE(audio_vae, patch_size=patch_size)


class VoxCPMForConditionalGeneration(nn.Module):
    input_modalities = "audio"
    _LATENT_STAGES = {"latent_generator", "latent", "ar_dit"}
    _VAE_STAGES = {"vae", "audio_vae"}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        del prefix
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self.model_stage = getattr(vllm_config.model_config, "model_stage", "latent_generator")
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True
        self.inject_omni_request_id_into_runtime_info = True
        self._pipeline = None
        self._latent_stream_gens: dict[str, Any] = {}
        self._latent_stream_terminal_pending: dict[str, int] = {}
        self._latent_stream_completed: set[str] = set()
        self._next_local_stream_key = 0
        self._ar_emit_stop_token = True

    def _runner_hidden_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        device = _resolve_runtime_device(self.vllm_config)
        model_config = getattr(self.vllm_config, "model_config", None)
        dtype = getattr(model_config, "dtype", torch.float32) if model_config is not None else torch.float32
        return device, dtype

    def _ensure_model_loaded(self):
        if self._pipeline is not None:
            return

        target_device = _resolve_runtime_device(self.vllm_config)
        model_dtype = getattr(self.vllm_config.model_config, "dtype", None)
        normalized_dtype = _normalize_dtype_name(model_dtype)
        if self.model_stage in self._LATENT_STAGES:
            self._pipeline = _load_native_voxcpm_latent_generator(
                self.model_path,
                device=target_device,
                dtype=normalized_dtype,
            )
        elif self.model_stage in self._VAE_STAGES:
            self._pipeline = _load_native_voxcpm_audio_vae(
                self.model_path,
                device=target_device,
            )
        else:
            raise ValueError(
                f"Unsupported VoxCPM model_stage: {self.model_stage}. "
                "pure_voxcpm only supports split-stage latent_generator/vae inference."
            )

        logger.info("Loaded VoxCPM stage '%s' on %s", self.model_stage, _device_to_string(target_device))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        del weights
        self._ensure_model_loaded()
        return set()

    @staticmethod
    def _extract_val(info: dict[str, Any], key: str, default: Any) -> Any:
        value = info.get(key, default)
        if isinstance(value, list):
            return value[0] if value else default
        return value

    def _resolve_stream_request_key(self, info: dict[str, Any]) -> str:
        request_key = info.get("__voxcpm_stream_key")
        if request_key is not None:
            return str(request_key)

        request_key = info.get("_omni_req_id")
        if request_key is not None:
            request_key = str(request_key)
            info["__voxcpm_stream_key"] = request_key
            return request_key

        request_key = f"voxcpm-local-{self._next_local_stream_key}"
        self._next_local_stream_key += 1
        info["__voxcpm_stream_key"] = request_key
        return str(request_key)

    def _recover_latent_from_input_ids(self, input_ids: torch.Tensor | None) -> torch.Tensor | None:
        if input_ids is None or input_ids.numel() == 0:
            return None
        flat_ids = input_ids.detach().reshape(-1).to("cpu")
        if flat_ids.numel() < 4 or int(flat_ids[0].item()) != _VOXCPM_LATENT_MAGIC:
            return None
        latent_dim = int(flat_ids[1].item())
        time_dim = int(flat_ids[2].item())
        payload = flat_ids[3:]
        expected = latent_dim * time_dim
        if latent_dim <= 0 or time_dim <= 0:
            raise ValueError(f"Invalid VoxCPM latent header: latent_dim={latent_dim}, time_dim={time_dim}")
        if int(payload.numel()) != expected:
            raise ValueError(
                "Invalid VoxCPM latent payload size: "
                f"expected={expected}, actual={int(payload.numel())}, "
                f"latent_dim={latent_dim}, time_dim={time_dim}"
            )
        packed = payload.to(dtype=torch.int32).to(torch.uint16)
        return packed.view(torch.bfloat16).to(torch.float32).reshape(1, latent_dim, time_dim)

    def _maybe_recover_vae_infos(
        self,
        infos: list[dict[str, Any]],
        input_ids: torch.Tensor | None,
        *,
        async_chunk: bool,
    ) -> list[dict[str, Any]]:
        if not async_chunk:
            return infos
        if any(self._extract_val(info, "latent_audio_feat", None) is not None for info in infos):
            return infos
        recovered = self._recover_latent_from_input_ids(input_ids)
        if recovered is None:
            return infos
        return [{"latent_audio_feat": recovered}]

    @staticmethod
    def _normalize_audio_samples(samples: Any) -> np.ndarray:
        if isinstance(samples, torch.Tensor):
            return samples.detach().cpu().float().reshape(-1).numpy()
        return np.asarray(samples, dtype=np.float32).reshape(-1)

    @classmethod
    def _normalize_ref_audio(cls, ref_audio: Any) -> tuple[np.ndarray, int]:
        if isinstance(ref_audio, str):
            raise TypeError("String ref_audio should be handled as a path before waveform normalization.")

        if isinstance(ref_audio, dict):
            sample_rate = ref_audio.get("sample_rate") or ref_audio.get("sampling_rate") or ref_audio.get("sr")
            samples = None
            for key in ("audio", "wav", "samples", "array", "waveform"):
                if key in ref_audio and ref_audio[key] is not None:
                    samples = ref_audio[key]
                    break
            if sample_rate is None or samples is None:
                raise ValueError("ref_audio dict must contain waveform data and sample rate.")
            return cls._normalize_audio_samples(samples), int(sample_rate)

        if isinstance(ref_audio, (list, tuple)):
            if len(ref_audio) == 1:
                return cls._normalize_ref_audio(ref_audio[0])
            if len(ref_audio) == 2 and np.isscalar(ref_audio[1]):
                return cls._normalize_audio_samples(ref_audio[0]), int(ref_audio[1])

        raise TypeError(f"Unsupported ref_audio format: {type(ref_audio)!r}")

    @staticmethod
    def _write_temp_prompt_wav(waveform: np.ndarray, sample_rate: int) -> str:
        prompt_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        prompt_file.close()

        wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
        wav = np.clip(wav, -1.0, 1.0)
        pcm16 = (wav * 32767.0).astype(np.int16)
        with wave.open(prompt_file.name, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(sample_rate))
            wav_file.writeframes(pcm16.tobytes())

        return prompt_file.name

    @classmethod
    def _resolve_prompt_inputs(cls, info: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
        prompt_text = cls._extract_val(info, "prompt_text", None)
        prompt_wav_path = cls._extract_val(info, "prompt_wav_path", None)
        if prompt_wav_path:
            if prompt_text is None:
                prompt_text = cls._extract_val(info, "ref_text", None)
            return prompt_wav_path, prompt_text, None

        ref_audio = cls._extract_val(info, "ref_audio", None)
        ref_text = cls._extract_val(info, "ref_text", None)
        if ref_audio is None or ref_text is None:
            return None, None, None
        if isinstance(ref_audio, str):
            return ref_audio, ref_text, None

        waveform, sample_rate = cls._normalize_ref_audio(ref_audio)
        temp_prompt_wav = cls._write_temp_prompt_wav(waveform, sample_rate)
        return temp_prompt_wav, ref_text, temp_prompt_wav

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def _get_vocab_size(self) -> int:
        model_config = getattr(self.vllm_config, "model_config", None)
        if model_config is not None:
            getter = getattr(model_config, "get_vocab_size", None)
            if callable(getter):
                try:
                    return int(getter())
                except Exception:
                    pass
            hf_config = getattr(model_config, "hf_text_config", None)
            if hf_config is not None and hasattr(hf_config, "vocab_size"):
                return int(hf_config.vocab_size)
        return 32000

    def _make_empty_output(
        self,
        *,
        output_key: str,
        payload_factory: Callable[[], torch.Tensor],
        infos: list[dict[str, Any]],
        sample_rate: int,
        out_device: torch.device,
        out_dtype: torch.dtype,
        hidden_rows: int | None = None,
    ) -> OmniOutput:
        if hidden_rows is None:
            hidden_rows = len(infos)
        return OmniOutput(
            text_hidden_states=torch.zeros((hidden_rows, 1), device=out_device, dtype=out_dtype),
            multimodal_outputs={
                output_key: [payload_factory() for _ in infos],
                "sr": [torch.tensor(sample_rate, dtype=torch.int32) for _ in infos],
            },
        )

    def _finalize_stage_output(
        self,
        *,
        output_key: str,
        outputs: list[torch.Tensor],
        sample_rates: list[torch.Tensor],
        out_device: torch.device,
        out_dtype: torch.dtype,
        hidden_rows: int | None = None,
    ) -> OmniOutput:
        multimodal_outputs: dict[str, Any] = {output_key: outputs, "sr": sample_rates}
        if hidden_rows is not None:
            text_hidden_states = torch.zeros((hidden_rows, 1), device=out_device, dtype=out_dtype)
        elif outputs:
            outputs_tensor = torch.stack(outputs)
            text_hidden_states = (
                outputs_tensor.unsqueeze(-1)
                if outputs_tensor.ndim == 1
                else outputs_tensor.reshape(-1, outputs_tensor.shape[-1])
            )
        else:
            text_hidden_states = torch.zeros((0, 1), device=out_device, dtype=out_dtype)
        text_hidden_states = text_hidden_states.to(device=out_device, dtype=out_dtype)
        return OmniOutput(
            text_hidden_states=text_hidden_states,
            multimodal_outputs=multimodal_outputs,
        )

    def _forward_vae_stage(
        self,
        infos: list[dict[str, Any]],
        *,
        sample_rate: int,
        async_chunk: bool,
        out_device: torch.device,
        out_dtype: torch.dtype,
    ) -> OmniOutput:
        if all(self._extract_val(info, "latent_audio_feat", None) is None for info in infos):
            self._ar_emit_stop_token = True
            return self._make_empty_output(
                output_key="model_outputs",
                payload_factory=lambda: torch.zeros((0,), dtype=torch.float32),
                infos=infos,
                sample_rate=sample_rate,
                out_device=out_device,
                out_dtype=out_dtype,
            )

        outputs: list[torch.Tensor] = []
        sample_rates: list[torch.Tensor] = []
        for info in infos:
            latent_audio_feat = self._extract_val(info, "latent_audio_feat", None)
            audio_tensor = self._pipeline.decode(latent_audio_feat, trim_streaming_patch=async_chunk)
            outputs.append(audio_tensor.float().cpu())
            sample_rates.append(torch.tensor(sample_rate, dtype=torch.int32))

        self._ar_emit_stop_token = True
        return self._finalize_stage_output(
            output_key="model_outputs",
            outputs=outputs,
            sample_rates=sample_rates,
            out_device=out_device,
            out_dtype=out_dtype,
        )

    def _forward_latent_stage(
        self,
        infos: list[dict[str, Any]],
        *,
        sample_rate: int,
        async_chunk: bool,
        out_device: torch.device,
        out_dtype: torch.dtype,
        hidden_rows: int,
    ) -> OmniOutput:
        texts = [self._extract_val(info, "text", "") for info in infos]
        if all(not text for text in texts):
            self._ar_emit_stop_token = True
            return self._make_empty_output(
                output_key="latent_audio_feat",
                payload_factory=lambda: torch.zeros((0,), dtype=torch.float32),
                infos=infos,
                sample_rate=sample_rate,
                out_device=out_device,
                out_dtype=out_dtype,
                hidden_rows=hidden_rows,
            )

        outputs: list[torch.Tensor] = []
        sample_rates: list[torch.Tensor] = []
        last_chunk_flags: list[bool] | None = [] if async_chunk else None
        payload_finished_flags: list[bool] | None = [] if async_chunk else None
        for info in infos:
            text = self._extract_val(info, "text", "")
            cfg_value = float(self._extract_val(info, "cfg_value", 2.0))
            inference_timesteps = int(self._extract_val(info, "inference_timesteps", 10))
            min_len = int(self._extract_val(info, "min_len", 2))
            max_len = int(self._extract_val(info, "max_len", self._extract_val(info, "max_new_tokens", 4096)))
            retry_badcase = bool(self._extract_val(info, "retry_badcase", True))
            retry_badcase_max_times = int(self._extract_val(info, "retry_badcase_max_times", 3))
            retry_badcase_ratio_threshold = float(self._extract_val(info, "retry_badcase_ratio_threshold", 6.0))
            streaming_prefix_len = int(self._extract_val(info, "streaming_prefix_len", 3))

            request_key = self._resolve_stream_request_key(info)
            created_temp: str | None = None

            if async_chunk:
                terminal_pending = self._latent_stream_terminal_pending.get(request_key, 0)
                if terminal_pending > 0:
                    outputs.append(torch.zeros((0,), dtype=torch.float32))
                    assert last_chunk_flags is not None
                    last_chunk_flags.append(True)
                    assert payload_finished_flags is not None
                    payload_finished_flags.append(terminal_pending == 1)
                    if terminal_pending == 1:
                        self._latent_stream_terminal_pending.pop(request_key, None)
                    else:
                        self._latent_stream_terminal_pending[request_key] = terminal_pending - 1
                    sample_rates.append(torch.tensor(sample_rate, dtype=torch.int32))
                    continue

                if request_key in self._latent_stream_completed:
                    outputs.append(torch.zeros((0,), dtype=torch.float32))
                    assert last_chunk_flags is not None
                    last_chunk_flags.append(True)
                    assert payload_finished_flags is not None
                    payload_finished_flags.append(False)
                    sample_rates.append(torch.tensor(sample_rate, dtype=torch.int32))
                    continue

                if request_key not in self._latent_stream_gens:
                    prompt_wav_path, prompt_text, temp_prompt_wav = self._resolve_prompt_inputs(info)
                    created_temp = temp_prompt_wav
                    self._latent_stream_gens[request_key] = self._pipeline.iter_latent_chunks_streaming(
                        text=text,
                        prompt_wav_path=prompt_wav_path,
                        prompt_text=prompt_text,
                        cfg_value=cfg_value,
                        inference_timesteps=inference_timesteps,
                        min_len=min_len,
                        max_len=max_len,
                        streaming_prefix_len=streaming_prefix_len,
                        retry_badcase=False,
                        retry_badcase_max_times=retry_badcase_max_times,
                        retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                    )
                generator = self._latent_stream_gens[request_key]
                try:
                    chunk_latent, is_last = next(generator)
                except StopIteration:
                    self._latent_stream_gens.pop(request_key, None)
                    self._latent_stream_terminal_pending[request_key] = 1
                    self._latent_stream_completed.add(request_key)
                    outputs.append(torch.zeros((0,), dtype=torch.float32))
                    assert last_chunk_flags is not None
                    last_chunk_flags.append(True)
                    assert payload_finished_flags is not None
                    payload_finished_flags.append(True)
                else:
                    if is_last:
                        self._latent_stream_gens.pop(request_key, None)
                        self._latent_stream_terminal_pending[request_key] = 1
                        self._latent_stream_completed.add(request_key)
                    outputs.append(chunk_latent.detach().float().cpu())
                    assert last_chunk_flags is not None
                    last_chunk_flags.append(bool(is_last))
                    assert payload_finished_flags is not None
                    payload_finished_flags.append(False)
                finally:
                    if created_temp is not None and os.path.exists(created_temp):
                        os.unlink(created_temp)
                sample_rates.append(torch.tensor(sample_rate, dtype=torch.int32))
                continue

            prompt_wav_path, prompt_text, temp_prompt_wav = self._resolve_prompt_inputs(info)
            try:
                latent_audio_feat = self._pipeline.generate_latents(
                    text=text,
                    prompt_wav_path=prompt_wav_path,
                    prompt_text=prompt_text,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    min_len=min_len,
                    max_len=max_len,
                    retry_badcase=retry_badcase,
                    retry_badcase_max_times=retry_badcase_max_times,
                    retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                )
                outputs.append(latent_audio_feat.float().cpu())
            finally:
                if temp_prompt_wav is not None and os.path.exists(temp_prompt_wav):
                    os.unlink(temp_prompt_wav)

            sample_rates.append(torch.tensor(sample_rate, dtype=torch.int32))

        self._ar_emit_stop_token = all(last_chunk_flags) if async_chunk and last_chunk_flags else True
        output = self._finalize_stage_output(
            output_key="latent_audio_feat",
            outputs=outputs,
            sample_rates=sample_rates,
            out_device=out_device,
            out_dtype=out_dtype,
            hidden_rows=hidden_rows,
        )
        if async_chunk and payload_finished_flags is not None:
            output.multimodal_outputs["finished"] = [
                torch.tensor(flag, dtype=torch.bool) for flag in payload_finished_flags
            ]
        return output

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> torch.Tensor:
        del sampling_metadata
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            device, dtype = self._runner_hidden_device_dtype()
            hidden_states = torch.zeros((0, 1), device=device, dtype=dtype)
        if hidden_states.ndim == 1:
            hidden_states = hidden_states.unsqueeze(-1)
        elif hidden_states.ndim > 2:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        vocab_size = self._get_vocab_size()
        num_rows = int(hidden_states.shape[0])
        logits = torch.zeros((num_rows, vocab_size), dtype=torch.float32, device=hidden_states.device)
        eos_id = 2 if vocab_size > 2 else 0
        safe_id = 1 if vocab_size > 1 and 1 != eos_id else 0
        emit_stop = getattr(self, "_ar_emit_stop_token", True)
        if num_rows > 0:
            if emit_stop:
                logits[:, eos_id] = 1.0e6
            else:
                logits[:, eos_id] = -1.0e9
                logits[:, safe_id] = 1.0e6
        return logits

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        model_intermediate_buffer: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        del positions, intermediate_tensors, inputs_embeds, kwargs
        self._ensure_model_loaded()
        out_device, out_dtype = self._runner_hidden_device_dtype()
        if input_ids is not None and input_ids.device.type == out_device.type:
            out_device = input_ids.device

        infos = model_intermediate_buffer or runtime_additional_information or [{}]
        hidden_rows = len(infos)
        if input_ids is not None and len(input_ids.shape) > 0:
            hidden_rows = max(hidden_rows, int(input_ids.shape[0]))
        sample_rate = int(getattr(self._pipeline, "sample_rate", 24000))
        async_chunk = bool(getattr(self.vllm_config.model_config, "async_chunk", False))
        if self.model_stage in self._VAE_STAGES:
            infos = self._maybe_recover_vae_infos(infos, input_ids, async_chunk=async_chunk)
            return self._forward_vae_stage(
                infos,
                sample_rate=sample_rate,
                async_chunk=async_chunk,
                out_device=out_device,
                out_dtype=out_dtype,
            )
        if self.model_stage in self._LATENT_STAGES:
            return self._forward_latent_stage(
                infos,
                sample_rate=sample_rate,
                async_chunk=async_chunk,
                out_device=out_device,
                out_dtype=out_dtype,
                hidden_rows=hidden_rows,
            )
        raise ValueError(f"Unsupported VoxCPM model_stage at runtime: {self.model_stage}")

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        del batch_size, dtype, device
        return {}


__all__ = ["VoxCPMForConditionalGeneration"]
