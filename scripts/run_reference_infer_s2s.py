#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Reference smoke run: UO4.

Runs the Fun-Audio-Chat reference S2S pipeline on a user-supplied audio file,
saves the synthesised WAV under `saves/`, and (optionally) transcribes via
Whisper for a sanity check.

Works around transformers-5.5 incompatibilities in the reference repo by
monkey-patching the two API shims that changed (get_text_config, the
processor subcomponent loader).
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

THIS = Path(__file__).resolve()
REPO = THIS.parents[1]                         # /home/jovyan/ye/vllm-omni-fa-impl
REF = Path("/home/jovyan/ye/vllm-omni/src/funaudiochat")
CKPT_DEFAULT = Path("/home/jovyan/ye/vllm-omni/pretrained_models/Fun-Audio-Chat-8B")
AUDIO_DEFAULT = REF / "examples" / "ck7vv9ag.wav"
SAVES = REPO / "saves" / "reference"


def patch_ref_for_tf5() -> None:
    """Make funaudiochat code work under transformers 5.5.4.

    Two incompatibilities found empirically:
      1) Config.get_text_config() is called with `decoder=True` by tf5's
         strict-dataclass validator.
      2) ProcessorMixin._get_arguments_from_pretrained now passes a
         processor_dict positional in addition to the path.
    """
    sys.path.insert(0, str(REF))
    from funaudiochat.configuration_funaudiochat import FunAudioChatConfig
    from funaudiochat.modeling_funaudiochat import FunAudioChatForConditionalGeneration
    from funaudiochat.processing_funaudiochat import FunAudioChatProcessor

    def _patched_get_text_config(self, *args, **kwargs):
        return self.text_config

    FunAudioChatConfig.get_text_config = _patched_get_text_config

    # transformers 5.x removed `_tie_or_clone_weights` and passes
    # `recompute_mapping=` to tie_weights. Reference's tie_weights expected
    # the old API — replace with a direct Parameter share.
    def _patched_tie(self, *_args, **_kwargs):
        if getattr(self, "audio_invert_tower", None) is not None \
                and getattr(self, "audio_tower", None) is not None:
            self.audio_invert_tower.lm_head.weight = self.audio_tower.embed_tokens.weight
    FunAudioChatForConditionalGeneration.tie_weights = _patched_tie

    # transformers 5.x passes `cache_position` into forward; reference forward
    # doesn't accept it. Wrap forward to drop unknown kwargs.
    _orig_fwd = FunAudioChatForConditionalGeneration.forward
    def _fwd_drop_unknowns(self, *args, cache_position=None, **kwargs):
        return _orig_fwd(self, *args, **kwargs)
    FunAudioChatForConditionalGeneration.forward = _fwd_drop_unknowns

    # tf5 strictly validates model_kwargs against forward signature. Bypass.
    FunAudioChatForConditionalGeneration._validate_model_kwargs = (
        lambda self, *_a, **_kw: None
    )

    # transformers 5.x removed _get_initial_cache_position. Provide a shim.
    import torch as _torch
    def _shim_get_initial_cache_position(self, cur_len, device, model_kwargs):
        model_kwargs["cache_position"] = _torch.arange(cur_len, device=device)
        return model_kwargs
    FunAudioChatForConditionalGeneration._get_initial_cache_position = (
        _shim_get_initial_cache_position
    )

    # transformers 5.x doesn't always pass `streamer` positionally; make the
    # reference _sample accept it as a kwarg with a None default.
    _orig_sample = FunAudioChatForConditionalGeneration._sample
    def _patched_sample(
        self, input_ids, logits_processor, stopping_criteria, generation_config,
        synced_gpus=False, streamer=None, **model_kwargs,
    ):
        return _orig_sample(
            self, input_ids, logits_processor, stopping_criteria,
            generation_config, synced_gpus, streamer, **model_kwargs,
        )
    FunAudioChatForConditionalGeneration._sample = _patched_sample

    # Original descriptor (classmethod); call its underlying function with cls
    # as the first arg, dropping the extra processor_dict positional.
    _orig_cm = FunAudioChatProcessor.__dict__["_get_arguments_from_pretrained"]
    _orig_func = _orig_cm.__func__

    @classmethod
    def _patched_get_args(cls, pretrained_model_name_or_path, *_ignored, **kwargs):
        return _orig_func(cls, pretrained_model_name_or_path, **kwargs)

    FunAudioChatProcessor._get_arguments_from_pretrained = _patched_get_args


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=str(CKPT_DEFAULT))
    ap.add_argument("--audio", default=str(AUDIO_DEFAULT))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--whisper-model", default="small")
    ap.add_argument("--skip-whisper", action="store_true")
    args = ap.parse_args()

    patch_ref_for_tf5()
    from funaudiochat.register import register_funaudiochat
    register_funaudiochat()

    # Deterministic mode.
    import torch
    torch.manual_seed(args.seed)

    os.chdir(REF)  # utils/cosyvoice_detokenizer relies on cwd to find spk2info.pt
    from utils.cosyvoice_detokenizer import get_audio_detokenizer, token2wav
    from utils.constant import (
        DEFAULT_S2M_GEN_KWARGS, DEFAULT_SP_GEN_KWARGS, SPOKEN_S2M_PROMPT,
        AUDIO_TEMPLATE,
    )
    import librosa
    import torchaudio
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[ref] device={device} model={args.model_path}")

    config = AutoConfig.from_pretrained(args.model_path)
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_path, config=config, torch_dtype=torch.bfloat16, device_map=device,
    )

    # Match infer_example defaults.
    sp_gen_kwargs = DEFAULT_SP_GEN_KWARGS.copy()
    sp_gen_kwargs["text_greedy"] = True
    gen_kwargs = DEFAULT_S2M_GEN_KWARGS.copy()
    gen_kwargs["max_new_tokens"] = args.max_new_tokens
    model.sp_gen_kwargs.update(sp_gen_kwargs)

    audio = [librosa.load(args.audio, sr=16000)[0]]
    conversation = [
        {"role": "system", "content": SPOKEN_S2M_PROMPT},
        {"role": "user", "content": AUDIO_TEMPLATE},
    ]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False,
    )
    inputs = processor(text=text, audio=audio, return_tensors="pt",
                       return_token_type_ids=False).to(model.device)
    generate_ids, audio_ids = model.generate(**inputs, **gen_kwargs)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]
    generate_text = processor.decode(generate_ids[0], skip_special_tokens=True)
    print(f"[ref] text: {generate_text}")
    print(f"[ref] audio ids (len {audio_ids[0].numel()}): "
          f"{audio_ids[0].tolist()[:16]}...")

    token_for_cosyvoice = [int(x) for x in audio_ids[0].tolist() if 0 <= x < 6561]
    cosy = get_audio_detokenizer()
    speech = token2wav(
        cosy, token_for_cosyvoice, embedding=None,
        token_hop_len=25 * 30, pre_lookahead_len=3,
    )

    SAVES.mkdir(parents=True, exist_ok=True)
    out = SAVES / f"ref_{Path(args.audio).stem}.wav"
    torchaudio.save(str(out), speech.cpu(), cosy.sample_rate)
    print(f"[ref] wav -> {out}  sr={cosy.sample_rate}  len={speech.shape[-1]}")

    if not args.skip_whisper:
        try:
            import whisper
            model_w = whisper.load_model(args.whisper_model)
            res = model_w.transcribe(str(out), language="zh")
            print(f"[whisper {args.whisper_model}] {res['text']}")
        except Exception as exc:  # noqa: BLE001
            print(f"[whisper] skipped: {exc}")

    # Dump a minimal tokens.json for later parity checks.
    import json
    tokens_out = SAVES / f"ref_{Path(args.audio).stem}.tokens.json"
    tokens_out.write_text(
        json.dumps(
            {
                "text_tokens": generate_ids[0].tolist(),
                "crq_tokens": audio_ids[0].tolist(),
                "text": generate_text,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"[ref] tokens -> {tokens_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
