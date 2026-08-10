# ruff: noqa
# Vendored from THUDM/GLM-4-Voice flow_inference.py at commit
# 54d667262bc1b92a62211aca44c7fcfe290e5df9.
#
# vLLM-Omni modification: the upstream decoder loads a shared config that
# constructs its language model, flow model, and HiFT vocoder. This integration
# only uses the flow model and HiFT vocoder to decode Omni-Diffusion audio
# tokens, so it overrides the unused top-level ``llm`` entry during loading.
# This avoids vendoring and initializing GLM-4-Voice's unrelated LLM runtime.
import uuid
from collections import defaultdict

import numpy as np
import torch
from hyperpyyaml import load_hyperpyyaml


def fade_in_out(fade_in_mel, fade_out_mel, window):
    device = fade_in_mel.device
    fade_in_mel, fade_out_mel = fade_in_mel.cpu(), fade_out_mel.cpu()
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel[..., :mel_overlap_len] = (
        fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len]
        + fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    )
    return fade_in_mel.to(device)


class AudioDecoder:
    def __init__(self, config_path, flow_ckpt_path, hift_ckpt_path, device="cuda"):
        self.device = device

        with open(config_path) as f:
            # HyperPyYAML applies overrides before constructing ``!new``
            # objects. Flow and HiFT remain identical to the upstream config;
            # only the unused language model is skipped.
            self.scratch_configs = load_hyperpyyaml(f, overrides={"llm": None})

        # Load models
        self.flow = self.scratch_configs["flow"]
        self.flow.load_state_dict(torch.load(flow_ckpt_path, map_location=self.device))
        self.hift = self.scratch_configs["hift"]
        self.hift.load_state_dict(torch.load(hift_ckpt_path, map_location=self.device))

        # Move models to the appropriate device
        self.flow.to(self.device)
        self.hift.to(self.device)
        self.mel_overlap_dict = defaultdict(lambda: None)
        self.hift_cache_dict = defaultdict(lambda: None)
        self.token_min_hop_len = 2 * self.flow.input_frame_rate
        self.token_max_hop_len = 4 * self.flow.input_frame_rate
        self.token_overlap_len = 5
        self.mel_overlap_len = int(self.token_overlap_len / self.flow.input_frame_rate * 22050 / 256)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        # hift cache
        self.mel_cache_len = 1
        self.source_cache_len = int(self.mel_cache_len * 256)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)

    def token2wav(
        self,
        token,
        uuid,
        prompt_token=torch.zeros(1, 0, dtype=torch.int32),
        prompt_feat=torch.zeros(1, 0, 80),
        embedding=torch.zeros(1, 192),
        finalize=False,
        option_steps=10,
    ):
        tts_mel = self.flow.inference(
            token=token.to(self.device),
            token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
            prompt_token=prompt_token.to(self.device),
            prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
            prompt_feat=prompt_feat.to(self.device),
            prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
            embedding=embedding.to(self.device),
            option_steps=option_steps,
        )

        # mel overlap fade in out
        if self.mel_overlap_dict[uuid] is not None:
            tts_mel = fade_in_out(tts_mel, self.mel_overlap_dict[uuid], self.mel_window)
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]["mel"], self.hift_cache_dict[uuid]["source"]
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)

        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # _tts_mel=tts_mel.contiguous()
        # keep overlap mel and hift cache
        if finalize is False:
            self.mel_overlap_dict[uuid] = tts_mel[:, :, -self.mel_overlap_len :]
            tts_mel = tts_mel[:, :, : -self.mel_overlap_len]
            tts_speech, tts_source = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)

            self.hift_cache_dict[uuid] = {
                "mel": tts_mel[:, :, -self.mel_cache_len :],
                "source": tts_source[:, :, -self.source_cache_len :],
                "speech": tts_speech[:, -self.source_cache_len :],
            }
            # if self.hift_cache_dict[uuid] is not None:
            #     tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            tts_speech = tts_speech[:, : -self.source_cache_len]

        else:
            tts_speech, tts_source = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)
            del self.hift_cache_dict[uuid]
            del self.mel_overlap_dict[uuid]
            # if uuid in self.hift_cache_dict.keys() and self.hift_cache_dict[uuid] is not None:
            #     tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech, tts_mel

    def offline_inference(self, token):
        this_uuid = str(uuid.uuid1())
        tts_speech, tts_mel = self.token2wav(token, uuid=this_uuid, finalize=True)
        return tts_speech.cpu()

    def stream_inference(self, token):
        token.to(self.device)
        this_uuid = str(uuid.uuid1())

        # Prepare other necessary input tensors
        llm_embedding = torch.zeros(1, 192).to(self.device)
        prompt_speech_feat = torch.zeros(1, 0, 80).to(self.device)
        flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32).to(self.device)

        tts_speeches = []
        tts_mels = []

        block_size = self.flow.encoder.block_size
        prev_mel = None

        for idx in range(0, token.size(1), block_size):
            # if idx>block_size: break
            tts_token = token[:, idx : idx + block_size]

            print(tts_token.size())

            if prev_mel is not None:
                prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)
                flow_prompt_speech_token = token[:, :idx]

            if idx + block_size >= token.size(-1):
                is_finalize = True
            else:
                is_finalize = False

            tts_speech, tts_mel = self.token2wav(
                tts_token,
                uuid=this_uuid,
                prompt_token=flow_prompt_speech_token.to(self.device),
                prompt_feat=prompt_speech_feat.to(self.device),
                finalize=is_finalize,
            )

            prev_mel = tts_mel
            prev_speech = tts_speech
            print(tts_mel.size())

            tts_speeches.append(tts_speech)
            tts_mels.append(tts_mel)

        # Convert Mel spectrogram to audio using HiFi-GAN
        tts_speech = torch.cat(tts_speeches, dim=-1).cpu()

        return tts_speech.cpu()
