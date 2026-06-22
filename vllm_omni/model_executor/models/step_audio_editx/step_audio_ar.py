import logging
import os
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.models.step1 import Step1ForCausalLM

from vllm_omni.data_entry_keys import OmniPayload
from vllm_omni.model_executor.models.output_templates import OmniOutput

from .step_audio_tokenizer import StepAudioTokenizer

logger = logging.getLogger(__name__)


class StepAudioAR(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self.tokenizer_path = os.environ["STEP_AUDIO_TOKENIZER_PATH"]
        self.model = Step1ForCausalLM(vllm_config=vllm_config, prefix=prefix)
        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = False
        self.tokenizer = StepAudioTokenizer(
            tokenizer_path=self.tokenizer_path,
            config_path=self.model_path,
        )

    def _build_prompt_embeds(
        self,
        *,
        info_dict: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def _first(x, default=None):
            if isinstance(x, list):
                return x[0] if x else default
            return x if x is not None else default

        audio = _first(info_dict.get("ref_audio"), None)
        sample_rate = _first(info_dict.get("sr"), 16000)
        ref_audio, sr = self.tokenizer._load_audio(audio, sample_rate)
        ref_text = _first(info_dict.get("ref_text"), "")
        text = _first(info_dict.get("text"), "")
        edit_type = _first(info_dict.get("edit_type"), "clone")
        if edit_type == "clone":
            prompt = (ref_text, text)
        else:
            edit_type = _first(info_dict.get("edit_type"), None)
            edit_info = _first(info_dict.get("edit_info"), None)
            prompt = (ref_text, edit_type, edit_info, text)

        prompt_token, codec_token = self.tokenizer.encode(edit_type, audio=ref_audio, prompt=prompt, sr=sr)
        input_ids = torch.tensor(prompt_token.input_ids, dtype=torch.long).to(next(self.model.parameters()).device)
        input_ids = self.embed_input_ids(input_ids)
        tts_pad_id = self.tokenizer.text_tokenizer.pad_token_id
        tts_pad_embed = self.embed_input_ids(torch.tensor([tts_pad_id]).to(input_ids.device))
        return input_ids, codec_token, tts_pad_embed

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        hidden = model_outputs
        info_dicts = kwargs.get("model_intermediate_buffer")
        if info_dicts is None:
            info_dicts = kwargs.get("runtime_additional_information") or []
        ref_code_list: list[torch.Tensor] = []
        audio_code_list: list[torch.Tensor] = []
        has_ref_code = False
        for info in info_dicts:
            if not isinstance(info, dict):
                ref_code_list.append(torch.empty(0, dtype=torch.long))
                continue
            codes = info.get("codes", {})
            ref_code = codes.get("ref")
            audio_code = codes.get("audio")
            if isinstance(audio_code, torch.Tensor) and audio_code.numel() > 0:
                audio_code_list.append(audio_code)
            if isinstance(ref_code, torch.Tensor) and ref_code.numel() > 0:
                ref_code_list.append(ref_code)
                has_ref_code = True
            else:
                ref_code_list.append(torch.empty(0, dtype=torch.long))

        if not audio_code_list:
            if has_ref_code:
                mm: OmniPayload = {"codes": {"ref": ref_code_list}}
                return OmniOutput(text_hidden_states=hidden, multimodal_outputs=mm)
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

        mm: OmniPayload = {"codes": {}}
        # Batch-aligned passthrough data. The runner selects each request's
        # entry before serializing the multimodal payload.
        mm["codes"]["ref"] = ref_code_list
        mm["codes"]["audio"] = audio_code_list
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs=mm)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        additional_information = info_dict.get("additional_information")
        if isinstance(additional_information, dict):
            merged: dict[str, Any] = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in additional_information.items():
                merged.setdefault(k, v)
            info_dict = merged

        payload: OmniPayload = info_dict
        embed = payload.get("embed", {})
        meta = payload.get("meta", {})

        span_len = int(input_ids.shape[0])
        if span_len <= 0:
            embeds = input_embeds
            if embeds is None:
                embeds = self.embed_input_ids(input_ids.to(torch.long))
            return input_ids, embeds, {}

        prompt_embeds_cpu = embed.get("prefill")
        tts_pad_embed_buf = embed.get("tts_pad")
        tts_pad_embed = None
        if isinstance(tts_pad_embed_buf, torch.Tensor) and tts_pad_embed_buf.numel() > 0:
            tts_pad_embed = tts_pad_embed_buf.to(
                device=input_ids.device,
                dtype=next(self.model.parameters()).dtype,
            ).reshape(1, -1)

        is_first_prefill = not isinstance(prompt_embeds_cpu, torch.Tensor) or prompt_embeds_cpu.ndim != 2

        if is_first_prefill:
            full_prompt_embeds, ref_code, tts_pad_embed = self._build_prompt_embeds(
                info_dict=info_dict,
            )
            prompt_embeds_cpu = full_prompt_embeds.detach().to("cpu").contiguous()
            total_prefill_len = int(prompt_embeds_cpu.shape[0])

            take = prompt_embeds_cpu[:span_len]
            if int(take.shape[0]) < span_len:
                pad_n = span_len - int(take.shape[0])
                pad_rows = tts_pad_embed.reshape(1, -1).to("cpu").expand(pad_n, -1)
                take = torch.cat([take, pad_rows], dim=0)

            prompt_embeds = take.to(
                device=input_ids.device,
                dtype=next(self.model.parameters()).dtype,
            )
            info_update: OmniPayload = {
                "embed": {
                    "prefill": prompt_embeds_cpu,
                    "tts_pad": tts_pad_embed.detach(),
                },
                "meta": {
                    "talker_prefill_offset": min(span_len, total_prefill_len),
                },
            }

            if isinstance(ref_code, torch.Tensor) and ref_code.numel() > 0:
                info_update.setdefault("codes", {})["ref"] = ref_code.detach().to("cpu").contiguous()

            input_ids_out = input_ids.clone()
            input_ids_out[:] = 0
            return input_ids_out, prompt_embeds, info_update

        total_prefill_len = int(prompt_embeds_cpu.shape[0])
        offset = int(meta.get("talker_prefill_offset", 0) or 0)
        if offset < 0:
            offset = 0

        # Subsequent prefill chunk.
        if offset < total_prefill_len:
            if tts_pad_embed is None:
                raise RuntimeError(
                    "Missing `embed.tts_pad` in additional_information; first prefill must initialize it."
                )

            s = max(0, min(offset, total_prefill_len))
            e = max(0, min(offset + span_len, total_prefill_len))
            take = prompt_embeds_cpu[s:e]

            if int(take.shape[0]) < span_len:
                pad_n = span_len - int(take.shape[0])
                pad_rows = tts_pad_embed.reshape(1, -1).to("cpu").expand(pad_n, -1)
                take = torch.cat([take, pad_rows], dim=0)

            prompt_embeds = take.to(
                device=input_ids.device,
                dtype=next(self.model.parameters()).dtype,
            )
            info_update: OmniPayload = {
                "meta": {
                    "talker_prefill_offset": min(offset + span_len, total_prefill_len),
                }
            }
            input_ids_out = input_ids.clone()
            input_ids_out[:] = 0
            return input_ids_out, prompt_embeds, info_update

        # Decode stage.
        # The prompt prefill is finished. Feed the sampled token itself back into the AR model.
        input_ids_out = input_ids.to(torch.long)
        prompt_embeds = self.embed_input_ids(input_ids_out).to(
            device=input_ids.device,
            dtype=next(self.model.parameters()).dtype,
        )
        audio_ids = input_ids_out.detach().to("cpu").reshape(-1)
        if audio_ids.numel() > 0:
            info_update = {"codes": {"audio": audio_ids.reshape(-1, 1)}}
        else:
            info_update = {}
        return input_ids_out, prompt_embeds, info_update

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        out = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return out

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        return self.model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        inner_loaded = self.model.load_weights(weights)  # set[str]
        fixed = set()

        suffixes = (".input_layernorm.weight", ".post_attention_layernorm.weight", ".norm.weight")

        for name in inner_loaded:
            if name.startswith("model.") and name.endswith(suffixes):
                # model.layers... -> model.model.layers...
                fixed.add("model." + name)
            else:
                fixed.add(name)

        return fixed
