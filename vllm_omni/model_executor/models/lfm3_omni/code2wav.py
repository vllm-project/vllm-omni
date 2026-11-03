# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LiquidCodec decoder, with a separate overlap-add/cache per request."""

import torch
from torch import nn
from vllm.config import VllmConfig

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.stage_input_processors.lfm3_omni import CODEC_CHUNK_FRAMES

from .reference import reference_module


class Lfm3OmniCode2Wav(nn.Module):
    have_multimodal_outputs = True
    has_preprocess = True
    has_postprocess = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.model_config = vllm_config.model_config
        ref = reference_module(self.model_config, "modeling_liquid_codec", "LiquidCodecModel")
        self._cache_class = ref.LiquidCodecStreamingCache
        device = torch.empty(0).device
        # Match AutoProcessor's CPU construction, including non-persistent
        # rotary buffers, before moving the codec to the worker device.
        with torch.device("cpu"):
            self.codec = ref.LiquidCodecModel.from_pretrained(
                self.model_config.model,
                subfolder="audio_tokenizer",
                revision=self.model_config.revision,
                dtype="auto",
            ).eval()
        self.codec.to(device)
        self._streams = {}

    def embed_input_ids(self, input_ids, **kwargs):
        return torch.zeros(
            (len(input_ids), self.model_config.hf_config.text_config.hidden_size),
            device=input_ids.device,
            dtype=self.model_config.dtype,
        )

    def preprocess(self, input_ids, input_embeds, **info):
        return input_ids, self.embed_input_ids(input_ids), {"request_id": info["request_id"]}

    def make_omni_output(self, model_outputs, **kwargs):
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        return OmniOutput(*model_outputs)

    def forward(self, input_ids=None, positions=None, **kwargs):
        infos = kwargs.get("runtime_additional_information") or []
        device = self.codec.from_quantizer.weight.device
        audios = []
        for info in infos:
            codes = info.get("codes", {}).get("audio")
            if codes is None:
                codes = info.get("codes.audio")
            if codes is None:
                raise ValueError("LiquidCodec requires the codes.audio stage payload")
            codes = torch.as_tensor(codes, device=device, dtype=torch.long).reshape(1, self.codec.config.n_q, -1)
            meta = info.get("meta", {})
            finished = bool(meta.get("stream_finished", meta.get("finished", not self.model_config.async_chunk)))
            req_id = info.get("request_id") or info.get("req_id")
            if not isinstance(req_id, str):
                raise ValueError("LiquidCodec streaming requires a scheduler request id")
            streaming = self.model_config.async_chunk
            cache = None
            if streaming:
                if req_id not in self._streams:
                    self._streams[req_id] = self._cache_class(self.codec.config)
                cache = self._streams[req_id]
            if streaming:
                # Transport can coalesce several producer chunks. Keep the
                # reference decode shape independent of arrival timing.
                chunks = codes.split(CODEC_CHUNK_FRAMES, dim=-1)
                parts = [
                    self.codec.decode(
                        chunk, streaming_cache=cache, finalize=finished and index == len(chunks) - 1
                    ).audio_values[0]
                    for index, chunk in enumerate(chunks)
                ]
                audio = torch.cat(parts)
            else:
                audio = self.codec.decode(codes, finalize=finished).audio_values[0]
            audios.append(audio)
            if finished:
                self._streams.pop(req_id, None)
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": [torch.tensor(24000)] * len(audios)},
        )

    def compute_logits(self, hidden_states):
        return None

    def on_requests_finished(self, finished_req_ids):
        for req_id in finished_req_ids:
            self._streams.pop(req_id, None)

    def load_weights(self, weights):
        # This stage loads the separately versioned subfolder above. It never
        # instantiates the temporal model or loads its weights into this process.
        return set(dict(self.named_parameters()))
