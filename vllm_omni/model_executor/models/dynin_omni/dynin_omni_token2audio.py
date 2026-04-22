from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .dynin_omni import DyninOmniStageBase
from .dynin_omni_common import (
    DetokTarget,
    _looks_like_hf_repo_id,
    coerce_token_ids_1d,
    normalize_runtime_info,
    resolve_dynin_infer_sources,
    resolve_hidden_size,
    unwrap_first_value,
)

logger = init_logger(__name__)


def _get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _ensure_remote_s2u_vendor_root(
    *,
    repo_id: str,
    local_files_only: bool,
) -> str | None:
    if local_files_only or not _looks_like_hf_repo_id(repo_id):
        return None

    existing = os.environ.get("DYNIN_S2U_VENDOR_ROOT")
    if existing:
        existing_path = Path(existing).expanduser().resolve()
        if existing_path.is_dir():
            return str(existing_path)

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        logger.warning("huggingface_hub unavailable; cannot fetch s2u_vendor from %s: %s", repo_id, e)
        return None

    token = _get_hf_token()
    last_error: Exception | None = None
    revisions: list[str | None] = [None]

    for revision in revisions:
        try:
            snapshot_dir = snapshot_download(
                repo_id=repo_id,
                revision=revision,
                allow_patterns=["s2u_vendor/**"],
                token=token,
            )
        except TypeError:
            try:
                snapshot_dir = snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    allow_patterns=["s2u_vendor/**"],
                )
            except Exception as e:
                last_error = e
                continue
        except Exception as e:
            last_error = e
            continue

        vendor_root = (Path(snapshot_dir) / "s2u_vendor").resolve()
        if vendor_root.is_dir():
            os.environ["DYNIN_S2U_VENDOR_ROOT"] = str(vendor_root)
            logger.info("Using remote S2U vendor root: %s", vendor_root)
            return str(vendor_root)

    if last_error is not None:
        logger.warning("Failed to download remote s2u_vendor from %s: %s", repo_id, last_error)
    return None


class DyninOmniToken2Audio(DyninOmniStageBase):
    """Stage-3: token detokenization to speech (or pass-through)."""

    stage_name = "Dynin token2audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        del prefix
        super().__init__()
        self.vllm_config = vllm_config
        self.have_multimodal_outputs = True
        self.requires_raw_input_tokens = True
        self.hidden_size = resolve_hidden_size(vllm_config=vllm_config)
        self._vq_audio = None
        self._vq_audio_path: str | None = None
        self._vq_audio_local_files_only: bool | None = None

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        del positions, intermediate_tensors, inputs_embeds
        if input_ids is None:
            raise ValueError("token2audio stage requires input_ids")

        runtime_info = normalize_runtime_info(kwargs.get("runtime_additional_information"))
        detok_id = int(unwrap_first_value(runtime_info.get("detok_id"), 0))
        tokens = coerce_token_ids_1d(input_ids)

        if detok_id != DetokTarget.AUDIO:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "token_ids": tokens,
                    "detok_id": torch.tensor([detok_id], dtype=torch.long, device=tokens.device),
                },
            )

        audio, sample_rate = self._decode_audio_tokens(tokens, runtime_info=runtime_info)
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "speech": audio,
                "audio": audio,
                "sr": torch.tensor([sample_rate], dtype=torch.int, device=audio.device),
                "detok_id": torch.tensor([detok_id], dtype=torch.long, device=audio.device),
            },
        )

    def _decode_audio_tokens(self, tokens: torch.Tensor, runtime_info: dict[str, Any]) -> tuple[torch.Tensor, int]:
        # Follow DYNIN validation path:
        #   token list -> "<|speech_x|>" string -> vq_model_audio.decode(...).
        vq_audio = self._ensure_vq_audio(runtime_info=runtime_info, ref_device=tokens.device)

        audio_codebook_size = int(unwrap_first_value(runtime_info.get("audio_codebook_size"), 4096))
        audio_vocab_offset = unwrap_first_value(
            runtime_info.get("audio_vocab_offset"),
            unwrap_first_value(runtime_info.get("t2s_vocab_start"), None),
        )

        token_ids = tokens.to(torch.long)
        if audio_vocab_offset is not None:
            off = int(audio_vocab_offset)
            token_ids = torch.where(token_ids >= off, token_ids - off, token_ids)
        token_ids = token_ids[(token_ids >= 0) & (token_ids < audio_codebook_size)]
        if token_ids.numel() == 0:
            raise RuntimeError("Audio detokenizer got no valid audio token ids.")

        speech_unit_str = " ".join(map(str, token_ids.detach().cpu().tolist()))
        speech_unit_for_decode = "".join(f"<|speech_{unit}|>" for unit in speech_unit_str.split(" ") if unit != "")

        condition = unwrap_first_value(
            runtime_info.get("condition"),
            unwrap_first_value(runtime_info.get("t2s_condition"), None),
        )
        output_wav_file = unwrap_first_value(runtime_info.get("output_wav_file"), None)
        created_tmp = False
        if output_wav_file is None:
            fd, tmp_wav = tempfile.mkstemp(prefix="dynin_t2s_", suffix=".wav")
            os.close(fd)
            output_wav_file = tmp_wav
            created_tmp = True

        audio_array = vq_audio.decode(speech_unit_for_decode, condition=condition, output_wav_file=output_wav_file)
        if created_tmp:
            try:
                os.remove(output_wav_file)
            except Exception:
                pass
        if not isinstance(audio_array, torch.Tensor):
            audio_array = torch.as_tensor(audio_array, dtype=torch.float32, device=tokens.device)
        else:
            audio_array = audio_array.to(device=tokens.device, dtype=torch.float32)

        if audio_array.ndim > 1:
            audio_array = audio_array.reshape(-1)
        audio_array = audio_array.contiguous()

        sample_rate = int(
            unwrap_first_value(
                runtime_info.get("sr"),
                unwrap_first_value(runtime_info.get("sample_rate"), 24000),
            )
        )
        try:
            cfg = getattr(vq_audio, "u2s_config", None)
            cfg_sr = getattr(cfg, "sampling_rate", None)
            if cfg_sr is None:
                cfg_sr = getattr(getattr(cfg, "data", None), "sampling_rate", None)
            if cfg_sr is not None:
                sample_rate = int(cfg_sr)
        except Exception:
            pass
        return audio_array, sample_rate

    def _ensure_vq_audio(self, runtime_info: dict[str, Any], ref_device: torch.device) -> Any:
        sources = resolve_dynin_infer_sources(vllm_config=self.vllm_config, runtime_info=runtime_info)
        model_path = str(sources.vq_audio_source)
        local_files_only = bool(sources.vq_audio_local_files_only)

        _ensure_remote_s2u_vendor_root(
            repo_id=model_path,
            local_files_only=local_files_only,
        )

        if (
            self._vq_audio is None
            or self._vq_audio_path != model_path
            or self._vq_audio_local_files_only != local_files_only
        ):
            logger.info(
                "Loading DYNIN audio detokenizer from %s (local_files_only=%s)",
                model_path,
                local_files_only,
            )
            try:
                from transformers import AutoModel
            except Exception as e:
                raise RuntimeError(
                    "transformers is required to load EMOVASpeechTokenizer remote code from Hugging Face."
                ) from e

            try:
                self._vq_audio = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=local_files_only,
                    low_cpu_mem_usage=False,
                )
            except TypeError:
                try:
                    self._vq_audio = AutoModel.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        local_files_only=local_files_only,
                    )
                except TypeError:
                    self._vq_audio = AutoModel.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load EMOVASpeechTokenizer from Hugging Face remote code for model path '{model_path}'."
                ) from e

            if not hasattr(self._vq_audio, "decode"):
                raise RuntimeError(
                    "Loaded audio tokenizer does not expose decode(). "
                    "Check HF config.json auto_map/model_type and ensure trust_remote_code=True."
                )
            self._vq_audio.eval()
            self._vq_audio.requires_grad_(False)
            self._vq_audio_path = model_path
            self._vq_audio_local_files_only = local_files_only
        if hasattr(self._vq_audio, "to"):
            self._vq_audio = self._vq_audio.to(ref_device)
        return self._vq_audio

    def embed_multimodal(self, **kwargs: Any) -> Any:
        del kwargs
        return None
