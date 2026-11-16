# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright (c) Ant Group. All rights reserved.
# Ported from:
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/modeling_bailing_talker.py

from __future__ import annotations

import torch


class SpkembExtractor:
    """CAMPPlus ONNX-based speaker embedding extractor (runs on CPU)."""

    def __init__(self, campplus_model: str, target_sr: int = 16000):
        import onnxruntime
        import torchaudio.compliance.kaldi as kaldi

        self.kaldi = kaldi
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 2
        self.campplus_session = onnxruntime.InferenceSession(
            campplus_model, sess_options=option, providers=["CPUExecutionProvider"]
        )
        self.target_sr = target_sr

    def _extract_spk_embedding(self, speech):
        feat = self.kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = (
            self.campplus_session.run(
                None,
                {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()},
            )[0]
            .flatten()
            .tolist()
        )
        embedding = torch.tensor([embedding])
        return embedding

    def __call__(self, waveform, **kwargs) -> torch.Tensor | None:
        spk_emb = self._extract_spk_embedding(waveform)
        return spk_emb
