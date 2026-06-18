# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adopted from https://github.com/inclusionAI/Ming-omni-tts/blob/main/spkemb_extractor.py
import os

import torch
import torchaudio
from vllm.multimodal.media.audio import load_audio

from vllm_omni.model_executor.models.ming_flash_omni.spk_embedding import SpkembExtractor


def resolve_model_to_local_path(model):
    if os.path.isdir(model):
        return model

    from huggingface_hub import snapshot_download

    return snapshot_download(model)


class MingSpeakerEmbeddingExtractor:
    def __init__(self, model, target_sr=16000):
        local_model_path = resolve_model_to_local_path(model)
        campplus_path = os.path.join(local_model_path, "campplus.onnx")
        if not os.path.exists(campplus_path):
            raise RuntimeError(f"Missing Ming speaker extractor model: {campplus_path}")

        self.target_sr = int(target_sr)
        self._core = SpkembExtractor(campplus_path, target_sr=self.target_sr)

    def extract_from_waveform(self, waveform, sample_rate):
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.as_tensor(waveform)

        tensor = waveform.detach().to(torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if int(sample_rate) != self.target_sr:
            tensor = torchaudio.transforms.Resample(orig_freq=int(sample_rate), new_freq=self.target_sr)(tensor)

        embedding = self._core._extract_spk_embedding(tensor)
        return embedding.squeeze(0).to(dtype=torch.float32)

    def extract_from_file(self, audio_path):
        audio, sample_rate = load_audio(audio_path, sr=None, mono=False)
        waveform = torch.from_numpy(audio).contiguous()
        return self.extract_from_waveform(waveform, sample_rate)

    def extract_many(self, audio_paths):
        return [self.extract_from_file(path) for path in audio_paths]
