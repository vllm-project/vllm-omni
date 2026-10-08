# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Test-only lifecycle observers; no logits, weights or sampler substitution."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def install_trace() -> None:
    import torch

    from vllm_omni.model_executor.models.zonos2.zonos2_dac_decoder import (
        Zonos2Code2WavForConditionalGeneration as Decoder,
    )
    from vllm_omni.model_executor.models.zonos2.zonos2_talker import Zonos2TalkerForConditionalGeneration as Talker

    if getattr(Talker, "_e2e_observed", False):
        return
    Talker._e2e_observed = True
    directory = Path(os.environ["ZONOS2_E2E_TRACE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)

    def write(row):
        with (directory / f"trace-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    def digest(tensor):
        return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    preprocess = Talker._managed_preprocess

    def observe_preprocess(self, input_ids, frames, info):
        result = preprocess(self, input_ids, frames, info)
        if int(info["_omni_num_computed_tokens"]) == 0:
            speaker = info.get("zonos2_speaker_embedding")
            write(
                {
                    "kind": "conditioning",
                    "request": info["_omni_req_id"],
                    "prompt_sha": digest(frames),
                    "speaker_sha": digest(speaker) if speaker is not None else None,
                }
            )
        return result

    Talker._managed_preprocess = observe_preprocess
    sample = Talker.sample

    def observe_sample(self, logits, metadata):
        for index, (key, eligible, info) in enumerate(self._sampling_plan):
            if eligible:
                state = self._request_states[key]
                if state.params.temperature == 0 and len(state.history) < 16:
                    from vllm_omni.model_executor.models.zonos2.zonos2_sampler import repetition_logits

                    adjusted = repetition_logits(self._last_fused_logits[index], state.history, state.params)
                    torch.save(
                        {"logits": adjusted.detach().cpu(), "history": state.history.detach().cpu()},
                        directory / f"{key}_step{len(state.history):02}.pt",
                    )
        result = sample(self, logits, metadata)
        for index, (key, eligible, info) in enumerate(self._sampling_plan):
            if not eligible or int(result.sampled_token_ids[index, 0]) != 1:
                continue
            state = self._request_states[key]
            torch.save(state.history.detach().cpu(), directory / f"{key}.pt")
            write(
                {
                    "kind": "finish",
                    "request": key,
                    "seed": state.params.seed,
                    "temperature": state.params.temperature,
                    "frames": len(state.history),
                    "eos_frame": int(state.eos_frame),
                    "countdown": int(state.countdown),
                    "reached_cap": len(state.history) >= state.params.max_tokens,
                }
            )
        return result

    Talker.sample = observe_sample
    talker_cleanup = Talker.on_requests_finished

    def observe_talker_cleanup(self, ids):
        ids = list(ids)
        talker_cleanup(self, ids)
        assert not any(str(key) in self._request_states for key in ids)
        write({"kind": "talker_cleanup", "ids": ids, "remaining": list(self._request_states)})

    Talker.on_requests_finished = observe_talker_cleanup
    forward = Decoder.forward

    def observe_decode(self, *args, **kwargs):
        result = forward(self, *args, **kwargs)
        for index, key in enumerate(kwargs.get("request_ids", [])):
            wav = result.multimodal_outputs["model_outputs"][index]
            info = (kwargs.get("runtime_additional_information") or [{}])[index]
            codes = info.get("codes", {}).get("audio")
            raw_frames = codes.shape[-1] if codes is not None else 0
            raw_sha = digest(codes.T.to(torch.int64)) if codes is not None else None
            write(
                {
                    "kind": "decode",
                    "request": key,
                    "samples": wav.numel(),
                    "finite": bool(torch.isfinite(wav).all()),
                    "dtype": str(wav.dtype),
                    "raw_frames": raw_frames,
                    "raw_sha": raw_sha,
                    "last_chunk": bool(
                        info.get("meta", {}).get("last_chunk", info.get("meta", {}).get("finished", True))
                    ),
                    "target": info.get("zonos2_target", info.get("meta", {}).get("num_processed_tokens")),
                    "codec_loaded": self._dac.codec is not None,
                }
            )
        return result

    Decoder.forward = observe_decode
    dac_cleanup = Decoder.on_requests_finished

    def observe_dac_cleanup(self, ids):
        ids = list(ids)
        dac_cleanup(self, ids)
        assert not any(str(key) in self._stream.states or str(key) in self._stream.closed for key in ids)
        write({"kind": "dac_cleanup", "ids": ids, "remaining": list(self._stream.states)})

    Decoder.on_requests_finished = observe_dac_cleanup
