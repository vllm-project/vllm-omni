# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Flash talker (stage 1 of Sophon-Realtime-32B-Flash): Qwen3-TTS with streaming
text input.

The stock Qwen3-TTS talker gets the whole text up front. Here the thinker's text
arrives while it is being written (cumulative token ids in ``ids.output``, see
``stage_input_processors.sophon_realtime``):

* prefill uses the official streaming layout (role tokens, codec prefix with the
  voice, and only the first text token);
* every decode step appends the newly arrived text to ``trailing_text``, which
  the stock decode path reads one token per frame; ``tts_eos`` is appended once
  the thinker has finished;
* while the thinker is still writing, codec EOS is masked (running out of text is
  not the end of the reply); a runaway reply is ended after
  ``FRAMES_PER_TEXT_TOKEN`` frames per text token.

Voices are ``<checkpoint>/voices/<name>.pt`` (a Qwen3-TTS x-vector) or a 1-15 s
reference clip ``<name>.wav`` / ``<name>.flac`` (zero-shot cloning: its x-vector
is computed by the same speaker encoder on first use), looked up as
``<voice>_<lang>`` then ``<voice>``.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import regex as re
import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import Qwen3TTSTalkerForConditionalGeneration

logger = init_logger(__name__)

LANGUAGES = {
    "zh": "chinese",
    "en": "english",
    "ja": "japanese",
    "ko": "korean",
    "de": "german",
    "fr": "french",
    "es": "spanish",
    "it": "italian",
    "pt": "portuguese",
    "ru": "russian",
}
ASSISTANT_ROLE_IDS = (151644, 77091, 198)  # "<|im_start|>assistant\n"
PROMPT_LEN = 10  # 3 role + 6 codec prefix + 1 (first text token)
DEFAULT_VOICE = "chelsie"
VOICE_NAME = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")
MAX_REFERENCE_S = 15.0
FRAMES_PER_TEXT_TOKEN = 12
MAX_TRACKED_REQUESTS = 512


def _first(x, default=None):
    if isinstance(x, (list, tuple)):
        return x[0] if x else default
    return default if x is None else x


def _flag(x) -> bool:
    if isinstance(x, torch.Tensor):
        return bool(x.reshape(-1)[0].item()) if x.numel() else False
    if isinstance(x, (list, tuple)):
        return bool(x and _flag(x[0]))
    return bool(x)


def _text_ids(info: dict) -> list[int]:
    ids = info.get("ids")
    ids = ids.get("output") if isinstance(ids, dict) else None
    if isinstance(ids, torch.Tensor):
        ids = ids.reshape(-1).tolist()
    return [int(x) for x in (ids or [])]


def _thinker_finished(info: dict) -> bool:
    return _flag((info.get("meta") or {}).get("finished", False))


class SophonFlashTalker(Qwen3TTSTalkerForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        root = os.path.dirname(os.path.abspath(str(self.model_path).rstrip("/")))
        self._voice_dir = os.path.join(root, "voices")
        self._voices: dict[str, tuple[torch.Tensor, tuple[str, float]]] = {}
        self._bad_voice_files: set[tuple[str, float]] = set()
        self._reported_missing: set[str] = set()
        self._force_eos: set[str] = set()
        self._text_open: set[str] = set()  # tts_eos not fed yet: codec EOS would cut the reply
        # Model Runner V2 names the batch's requests (mrv2_sampling_context) only for models asking for output ids.
        self.logitsprocs_need_output_token_ids = True

    # ------------------------------------------------------------------ embeddings
    def _text_embed(self, ids, device) -> torch.Tensor:
        t = torch.tensor([list(ids)], dtype=torch.long, device=device)
        return self.text_projection(self.text_embedding(t)).reshape(len(ids), -1).to(self._embedding_dtype)

    def _codec_embed(self, ids, device) -> torch.Tensor:
        t = torch.tensor(list(ids), dtype=torch.long, device=device).reshape(-1, 1)
        return self.embed_input_ids(t).reshape(len(ids), -1).to(self._embedding_dtype)

    # ------------------------------------------------------------------ voices
    def _voice(self, voice: str, lang: str, device) -> torch.Tensor:
        voice = (voice or DEFAULT_VOICE).lower().strip()
        if not VOICE_NAME.fullmatch(voice):  # names come from clients: no paths
            voice = DEFAULT_VOICE
        names = (f"{voice}_{lang}", voice, f"{DEFAULT_VOICE}_{lang}", f"{DEFAULT_VOICE}_zh")
        for i, name in enumerate(names):
            for ext in (".pt", ".wav", ".flac"):
                path = os.path.join(self._voice_dir, name + ext)
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    continue
                hit = self._voices.get(name)
                if hit is None or hit[1] != (path, mtime):
                    if (path, mtime) in self._bad_voice_files:
                        continue
                    try:
                        hit = self._voices[name] = (self._load_voice(path, ext), (path, mtime))
                    except Exception as e:  # a bad voice file must not take the stage down
                        self._bad_voice_files.add((path, mtime))
                        logger.error("Sophon-Realtime: cannot load voice %s: %s", path, e)
                        continue
                if i >= 2 and voice not in (DEFAULT_VOICE, f"{DEFAULT_VOICE}_{lang}"):
                    if voice not in self._reported_missing:
                        self._reported_missing.add(voice)
                        logger.warning(
                            "Sophon-Realtime: voice %r not found in %s, using %s", voice, self._voice_dir, name
                        )
                return hit[0].to(device)
        raise ValueError(f"Sophon-Realtime: voice {voice!r} not found in {self._voice_dir}")

    def _load_voice(self, path: str, ext: str) -> torch.Tensor:
        if ext == ".pt":
            vec = torch.load(path, map_location="cpu", weights_only=True)
        else:
            import soundfile as sf

            wav, sr = sf.read(path, dtype="float32", always_2d=True)
            wav = np.ascontiguousarray(wav.mean(1)[: int(sr * MAX_REFERENCE_S)])
            if len(wav) < sr:
                raise ValueError(f"reference clip {path} is shorter than 1 s")
            with torch.no_grad():
                vec = self._prompt_builder.extract_speaker_embedding(wav, int(sr))
        return vec.detach().reshape(1, -1).to(self._embedding_dtype).cpu()

    # ------------------------------------------------------------------ prefill
    def preprocess_batch(self, **_: Any) -> None:
        return None  # the prompt is built per request in preprocess()

    def preprocess_batch_mrv2(self, **_: Any) -> None:
        return None

    def _track(self, rid: str | None, *, text_open: bool) -> None:
        if rid is None:
            return
        (self._text_open.add if text_open else self._text_open.discard)(rid)
        if len(self._text_open) > MAX_TRACKED_REQUESTS:
            for stale in list(self._text_open)[: len(self._text_open) - MAX_TRACKED_REQUESTS]:
                self._text_open.discard(stale)

    def _prefill(self, input_ids: torch.Tensor, info: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
        dev = input_ids.device
        span = int(input_ids.shape[0])
        cfg, tc = self.config, self.talker_config
        ids, finished = _text_ids(info), _thinker_finished(info)
        rid = info.get("req_id") or info.get("request_id")  # Model Runner V2 names it req_id
        lang = str(_first(info.get("language"), "zh") or "zh").lower()[:2]
        voice = str(_first(info.get("speaker"), DEFAULT_VOICE) or DEFAULT_VOICE)
        lang_id = tc.codec_language_id.get(LANGUAGES.get(lang, "english"), tc.codec_language_id["english"])
        bos, eos, pad = self._text_embed([cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id], dev).split(
            1, 0
        )
        codec = torch.cat(
            [
                self._codec_embed([tc.codec_think_id, tc.codec_think_bos_id, lang_id, tc.codec_think_eos_id], dev),
                self._voice(voice, lang, dev),
                self._codec_embed([tc.codec_pad_id, tc.codec_bos_id], dev),
            ],
            0,
        )  # (7, H)
        first = self._text_embed(ids[:1], dev) if ids else pad
        prompt = torch.cat(
            [
                self._text_embed(ASSISTANT_ROLE_IDS, dev),
                torch.cat([pad.expand(codec.shape[0] - 2, -1), bos], 0) + codec[:-1],
                first + codec[-1:],
            ],
            0,
        )
        tail = [self._text_embed(ids[1:], dev)] if len(ids) > 1 else []
        if finished:
            tail.append(eos)
        self._track(rid, text_open=not finished)
        trailing = torch.cat(tail, 0) if tail else prompt.new_zeros((0, prompt.shape[1]))
        if prompt.shape[0] != span:
            logger.warning_once(
                "Sophon-Realtime Flash talker: prefill span %d != prompt length %d "
                "(set hf_overrides.async_chunk_prewarm_prompt_len=%d)",
                span,
                prompt.shape[0],
                PROMPT_LEN,
            )
            prompt = torch.cat([prompt, pad.expand(max(0, span - prompt.shape[0]), -1)], 0)[:span]
        ids_out = input_ids.clone()
        ids_out[:] = int(tc.codec_pad_id)
        if rid is not None:
            self._force_eos.discard(rid)
            if finished and not ids:
                self._force_eos.add(rid)  # nothing to say: finish at once
            else:
                self._req_x_vector_only[rid] = True  # no silence in the first frames (stock x-vector guard)
        update = {
            "hidden_states": {"trailing_text": trailing.detach()},
            "meta": {
                "talker_text_offset": 0,
                "codec_streaming": True,
                "talker_prefill_offset": span,
                "codec_frame_valid": torch.zeros((), dtype=torch.bool, device=dev),
                "sophon_fed": len(ids),
                "sophon_eos": finished,
                "sophon_steps": 0,
            },
            "codes": {"audio": torch.zeros((span, int(tc.num_code_groups)), dtype=torch.long, device=dev)},
        }
        return ids_out, prompt, update

    # ------------------------------------------------------------------ decode: append new text
    def _extend(self, info: dict, device) -> dict | None:
        meta = info.get("meta")
        if not isinstance(meta, dict):
            meta = info["meta"] = {}
        ids, finished = _text_ids(info), _thinker_finished(info)
        fed = int(meta.get("sophon_fed", 0) or 0)
        eos_done = bool(meta.get("sophon_eos", False))
        steps = int(meta.get("sophon_steps", 0) or 0) + 1
        meta["sophon_steps"] = steps
        rid = info.get("req_id") or info.get("request_id")  # Model Runner V2 names it req_id
        if rid is not None and eos_done and steps > FRAMES_PER_TEXT_TOKEN * max(1, len(ids)) + 60:
            if rid not in self._force_eos:
                logger.warning("Sophon-Realtime Flash talker: ending runaway reply %s after %d frames", rid, steps)
            self._force_eos.add(rid)
        new: list[torch.Tensor] = []
        if len(ids) > fed:
            new.append(self._text_embed(ids[fed:], device))
            fed = len(ids)
        if finished and not eos_done:
            new.append(self._text_embed([self.config.tts_eos_token_id], device))
            eos_done = True
            self._track(rid, text_open=False)
        if not new:
            return {"sophon_steps": steps}
        hs = info.get("hidden_states")
        if not isinstance(hs, dict):
            hs = info["hidden_states"] = {}
        tail = hs.get("trailing_text")
        offset = int(meta.get("talker_text_offset", 0) or 0)
        rest = (
            tail[offset:].to(device=device, dtype=self._embedding_dtype)
            if isinstance(tail, torch.Tensor) and tail.ndim == 2
            else None
        )
        new_tail = torch.cat(([rest] if rest is not None and rest.shape[0] else []) + new, 0)
        hs["trailing_text"] = new_tail
        meta["talker_text_offset"] = 0
        meta["sophon_fed"] = fed
        meta["sophon_eos"] = eos_done
        return {"trailing_text": new_tail, "sophon_fed": fed, "sophon_eos": eos_done, "sophon_steps": steps}

    @staticmethod
    def _merge(update: dict, ext: dict | None) -> dict:
        if ext is None:
            return update
        update.setdefault("meta", {}).update({k: v for k, v in ext.items() if k != "trailing_text"})
        if "trailing_text" in ext:
            hs = update.setdefault("hidden_states", {})
            hs.setdefault("trailing_text", ext["trailing_text"].detach())
        return update

    def preprocess(self, input_ids: torch.Tensor, input_embeds: torch.Tensor | None, **info: Any):
        extra = info.get("additional_information")
        if isinstance(extra, dict):
            merged = {k: v for k, v in info.items() if k != "additional_information"}
            for k, v in extra.items():
                merged.setdefault(k, v)
            info = merged
        span = int(input_ids.shape[0])
        is_prefill = info.get("_omni_is_prefill")
        if not isinstance(is_prefill, bool):
            is_prefill = span > 1
        if is_prefill and span > 0:
            return self._prefill(input_ids, info)
        ext = self._extend(info, input_ids.device)
        ids_out, embeds_out, update = super().preprocess(input_ids, input_embeds, **info)
        return ids_out, embeds_out, self._merge(update, ext)

    def preprocess_decode_batch(
        self,
        *,
        input_ids: torch.Tensor,
        req_infos: list[dict[str, Any]],
        input_embeds: torch.Tensor | None = None,
    ):
        exts = [self._extend(info, input_ids.device) for info in req_infos]
        out = super().preprocess_decode_batch(input_ids=input_ids, req_infos=req_infos, input_embeds=input_embeds)
        updates = out[-1]
        for i, ext in enumerate(exts):
            updates[i] = self._merge(updates[i], ext)
        return out

    # ------------------------------------------------------------------ codec EOS control
    def set_batch_req_ids(self, req_ids) -> None:
        super().set_batch_req_ids(req_ids)
        self.logitsprocs_need_output_token_ids = True  # the parent recomputes it from its own needs
        if self._force_eos:
            self._force_eos &= set(self._batch_req_ids)

    def compute_logits(self, hidden_states, sampling_metadata: Any = None):
        logits = super().compute_logits(hidden_states, sampling_metadata)
        rids = self._batch_req_ids
        if logits is None or not (self._force_eos or self._text_open) or len(rids) != logits.shape[0]:
            return logits
        eos = self._codec_eos_token_id
        hold = [i for i, r in enumerate(rids) if r in self._text_open and r not in self._force_eos]
        if hold:  # the thinker is still writing
            logits[torch.tensor(hold, device=logits.device), eos] = -1e4
        stop = [i for i, r in enumerate(rids) if r in self._force_eos]
        if stop:
            idx = torch.tensor(stop, device=logits.device)
            logits[idx] = logits[idx] - 1e4
            logits[idx, eos] = 0.0  # not -inf: min_tokens may mask EOS
        return logits
