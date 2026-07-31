"""Diagnose LongCat-Next generation-quality: raw token/code-stream evidence.

The 2-stage pipeline e2e "passes" but produces nonsense: audio stops at ~1
code frame (60ms) right after <longcat_audiogen_start>, and image overruns the
37x37 grid (1994 codes instead of 1369, then truncates). The two failures go
in OPPOSITE directions, so a stop-token misconfiguration cannot explain both:
the coherent-forward-pass hypothesis is that the codebook logits are garbage
(MLA attention or the gen-time talker_mtp path), not the state machine.

This script runs three prompts through the SAME wired pipeline and prints the
evidence that discriminates between the hypotheses:

  text  - plain text, no multimodal trigger. If this decodes to coherent
          Chinese, the backbone+MLA decode path is fine and the bug is in the
          gen-time multimodal machinery; if it is also garbage, suspect MLA/
          attention correctness directly.
  image - dumps the visible token stream (pad/newline/end markers) plus the
          real [T, 8] visual codes: T, unique level-0 codes, sentinel (16384)
          appearance, img_start/img_end/newline counts, finish reason.
  audio - dumps the visible stream with positions of <longcat_audiogen_start>
          (131123) and <longcat_audiogen_end> (131124), the [T, 8] audio codes,
          level-0 sentinel (8192) rows, unique level-0 values, finish reason.

Run on the pod (optionally LONGCAT_AUDIO_DEBUG=1 for per-step gen-state logs):

  python longcat_next_debug_quality.py <model_path> <deploy_yaml> <out_dir> \
      [--runs text,image,audio]

Reads the client-side payload key directly (stage engine_output_type), matching
the other longcat_next_wired_* scripts.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from vllm import SamplingParams
from vllm.multimodal.media.audio import load_audio

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniTextPrompt

AUDIO_START, AUDIO_END, AUDIO_PAD = 131103, 131104, 131105
IMG_START, IMG_END, IMG_PAD, IMG_NEWLINE = 131106, 131107, 131108, 131109
AUDIOTEXT_START, AUDIOTEXT_END, AUDIOTEXT_PAD = 131120, 131121, 131122
AUDIOGEN_START, AUDIOGEN_END = 131123, 131124

NAME = {
    AUDIO_START: "audio_start",
    AUDIO_END: "audio_end",
    AUDIO_PAD: "audio_pad",
    IMG_START: "img_start",
    IMG_END: "img_end",
    IMG_PAD: "img_pad",
    IMG_NEWLINE: "img_newline",
    AUDIOTEXT_START: "audiotext_start",
    AUDIOTEXT_END: "audiotext_end",
    AUDIOTEXT_PAD: "audiotext_pad",
    AUDIOGEN_START: "audiogen_start",
    AUDIOGEN_END: "audiogen_end",
}

def _thinker_output(outputs) -> object | None:
    for o in outputs:
        if getattr(o, "stage_id", None) == 0:
            return o
    return None


def _eos_id(model_path: str) -> int:
    """Resolve the checkpoint's eos_token_id at runtime."""
    try:
        import json

        with open(os.path.join(model_path, "config.json")) as f:
            return int(json.load(f).get("eos_token_id", 2))
    except Exception:
        return 2


def _dump_visible(token_ids: list[int], out: dict, label: str, eos_id: int) -> None:
    """Positions of every special token in the generated visible stream."""
    spec: dict[str, list[int]] = defaultdict(list)
    for i, tid in enumerate(token_ids):
        name = NAME.get(tid)
        if name is not None:
            spec[name].append(i)
    out[f"{label}_visible_len"] = len(token_ids)
    out[f"{label}_special_positions"] = {k: v for k, v in spec.items()}
    if eos_id in token_ids:
        out[f"{label}_eos_at"] = token_ids.index(eos_id)
    print(f"[debug] {label} visible tokens: {len(token_ids)}; special: "
          f"{ {k: len(v) for k, v in spec.items()} }")


def _dump_codes(codes: torch.Tensor, out: dict, label: str, sentinel: int) -> None:
    codes = codes.cpu()
    level0 = codes[:, 0].tolist()
    real = [c for c in level0 if c != sentinel]
    out[f"{label}_frames"] = int(codes.shape[0])
    out[f"{label}_real_frames"] = len(real)
    out[f"{label}_sentinel_rows"] = int(codes.shape[0]) - len(real)
    out[f"{label}_level0_unique"] = sorted(set(level0))
    out[f"{label}_level0_counter"] = Counter(level0).most_common(8)
    out[f"{label}_all_same"] = len(set(level0)) == 1
    out[f"{label}_min_max"] = (min(level0), max(level0))
    print(f"[debug] {label} codes: {codes.shape[0]} rows, {len(real)} real "
          f"(sentinel {sentinel}: {out[f'{label}_sentinel_rows']}), "
          f"all_same={out[f'{label}_all_same']}, min_max={out[f'{label}_min_max']}, "
          f"top={out[f'{label}_level0_counter']}")


def run_text(llm: Omni, model_path: str, out: dict) -> None:
    prompt = OmniTextPrompt(
        prompt=(
            "<longcat_system>You are a helpful assistant. "
            "<longcat_user>请简单介绍一下你自己。 "
            "<longcat_assistant>"
        )
    )
    params = SamplingParams(max_tokens=256, temperature=0.2, top_k=20, top_p=0.85, detokenize=True)
    outputs = llm.generate([prompt], params)
    out["num_outputs"] = len(outputs)
    thinker = _thinker_output(outputs)
    if thinker is None:
        out["text_verdict"] = "no stage-0 output"
        print("[debug] text: no stage-0 output!")
        return
    tok_ids = thinker.request_output.outputs[0].token_ids
    out["text_finish_reason"] = str(thinker.request_output.outputs[0].finish_reason)
    _dump_visible(tok_ids, out, "text", _eos_id(model_path))
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        text = tok.decode(tok_ids)
    except Exception:
        text = "(decode failed)"
    out["text_generated"] = text
    print(f"[debug] text finish={out['text_finish_reason']}")
    print(f"[debug] text generated: {text!r}")


def run_image(llm: Omni, model_path: str, out: dict) -> None:
    prompt = OmniTextPrompt(
        prompt=(
            "<longcat_system>You are a helpful assistant. "
            "<longcat_user>请生成一张图片，内容是一只猫。 "
            "<longcat_assistant><longcat_img_start>"
        )
    )
    params = SamplingParams(max_tokens=2048, temperature=0.4, top_p=0.9, detokenize=True)
    outputs = llm.generate([prompt], params)
    out["num_outputs"] = len(outputs)
    thinker = _thinker_output(outputs)
    if thinker is None:
        out["image_verdict"] = "no stage-0 output"
        print("[debug] image: no stage-0 output!")
        return
    out["image_finish_reason"] = str(thinker.request_output.outputs[0].finish_reason)
    _dump_visible(thinker.request_output.outputs[0].token_ids, out, "image", _eos_id(model_path))

    mm = thinker.multimodal_output
    codes = mm.get("codes", {}) if isinstance(mm, Mapping) else {}
    img = codes.get("visual") or codes.get("image") or codes.get("model_outputs")
    if img is not None:
        _dump_codes(img, out, "image", sentinel=16384)
        out["image_vs_expected"] = f"{out['image_frames']} vs 1369 (37x37 grid)"
    else:
        print("[debug] image: no visual codes in thinker multimodal_output")
    out["image_verdict"] = (
        "PASS" if out.get("image_frames", 0) == 1369 and out.get("image_finish_reason") != "length"
        else "SUSPECT"
    )
    print(f"[debug] image verdict: {out['image_verdict']} (finish={out['image_finish_reason']})")


def run_audio(llm: Omni, model_path: str, out: dict) -> None:
    ref_voice = os.path.join(model_path, "assets", "vc_zh3.wav")
    audio_signal, sr = load_audio(ref_voice, sr=16000)
    placeholder = "<longcat_audio_start><longcat_audio_pad><longcat_audio_end>"
    prompt_text = (
        "<longcat_system>Replicate the voice in the audio clip to formulate an answer. "
        f"{placeholder} "
        "<longcat_user>用这个声音合成以下内容：明天的meeting在三楼的Conference Room举行。 "
        "<longcat_assistant><longcat_audiogen_start>"
    )
    prompt = OmniTextPrompt(prompt=prompt_text, multi_modal_data={"audio": (audio_signal, sr)})
    params = SamplingParams(
        max_tokens=2048,
        temperature=0.2,
        top_k=20,
        top_p=0.85,
        repetition_penalty=1.1,
        detokenize=True,
    )
    outputs = llm.generate([prompt], params)
    out["num_outputs"] = len(outputs)
    thinker = _thinker_output(outputs)
    if thinker is None:
        out["audio_verdict"] = "no stage-0 output"
        print("[debug] audio: no stage-0 output!")
        return
    out["audio_finish_reason"] = str(thinker.request_output.outputs[0].finish_reason)
    tok_ids = thinker.request_output.outputs[0].token_ids
    _dump_visible(tok_ids, out, "audio", _eos_id(model_path))

    mm = thinker.multimodal_output
    codes = mm.get("codes", {}) if isinstance(mm, Mapping) else {}
    aud = codes.get("audio")
    if aud is not None:
        _dump_codes(aud, out, "audio", sentinel=8192)
    else:
        print("[debug] audio: no audio codes in thinker multimodal_output")

    spec = out.get("audio_special_positions", {})
    starts = spec.get("audiogen_start", [])
    ends = spec.get("audiogen_end", [])
    out["audio_rounds"] = len(starts)
    if starts and ends:
        out["audio_tokens_between_first_start_end"] = ends[0] - starts[0]
        print(f"[debug] audio: first round spans visible tokens "
              f"[{starts[0]} .. {ends[0]}] ({(ends[0] - starts[0])} tokens, "
              f"{out.get('audio_real_frames')} frames)")
    out["audio_verdict"] = (
        "PASS" if out.get("audio_real_frames", 0) > 10 else "SUSPECT"
    )
    print(f"[debug] audio verdict: {out['audio_verdict']} (rounds={out['audio_rounds']})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("deploy_yaml")
    parser.add_argument("out_dir")
    parser.add_argument("--runs", default="text,image,audio")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    llm = Omni(model=args.model_path, deploy_config=args.deploy_yaml, trust_remote_code=True)

    results: dict = {"pipeline": args.deploy_yaml}
    for run in args.runs.split(","):
        run = run.strip()
        if run == "text":
            run_text(llm, args.model_path, results)
        elif run == "image":
            run_image(llm, args.model_path, results)
        elif run == "audio":
            run_audio(llm, args.model_path, results)
        else:
            print(f"[debug] unknown run: {run}")

    out_json = os.path.join(args.out_dir, "quality_diagnostics.json")
    with open(out_json, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"[debug] wrote diagnostics to {out_json}")


if __name__ == "__main__":
    main()
