"""Single-node, wired 2-stage LongCat-Next audio generation test.

Orchestrated pipeline: thinker (TP=4 fp8) + audio decoder colocated on GPU 3,
via longcat_next_4gpu_audio_fp8.yaml (pipeline: longcat_next_thinker_audio).
The orchestrator routes stage 0's finished codes through
thinker2audio_decoder_token_only into stage 1 -- no manual JSON handoff.

Using the official vllm-omni multi-stage API: Omni.generate() returns one
OmniRequestOutput per stage. Stage 0 (thinker) produces text + audio code
frames; stage 1 (audio decoder) produces the final waveform.

The multimodal_output property automatically resolves the nested structure
(MultimodalPayload) -- no need to manually traverse request_output.

Hardware assumptions: 4x A100-40GB, fp8 quantization, TP=4, audio decoder
colocated on GPU 3 alongside TP rank 3.

Writes <out_wav> (24 kHz waveform) and <out_json> (metadata + verdict).

Run with: python longcat_next_wired_audio.py <model_path> <deploy_yaml> <out_wav> <out_json>
"""

import json
import os
import sys
from collections.abc import Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from vllm import SamplingParams
from vllm.multimodal.media.audio import load_audio

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniTextPrompt


def main() -> None:
    model_path = sys.argv[1]
    deploy_yaml = sys.argv[2]
    out_wav = sys.argv[3]
    out_json = sys.argv[4]

    llm = Omni(
        model=model_path,
        deploy_config=deploy_yaml,
        trust_remote_code=True,
    )

    # Identical prompt to the proven thinker-only run (job 15025030, PASS:
    # 2 audio frames). Keeping it byte-for-byte the same isolates "does the
    # decoder/wiring work" from "does the model still enter audio-gen mode",
    # which is already answered.
    ref_voice_path = os.path.join(model_path, "assets", "vc_zh3.wav")
    audio_signal, sr = load_audio(ref_voice_path, sr=16000)
    audio_placeholder = "<longcat_audio_start><longcat_audio_pad><longcat_audio_end>"
    prompt_text = (
        "<longcat_system>Replicate the voice in the audio clip to formulate an answer. "
        f"{audio_placeholder} "
        "<longcat_user>用这个声音合成以下内容：明天的meeting在三楼的Conference Room举行。 "
        "<longcat_assistant><longcat_audiogen_start>"
    )
    prompt = OmniTextPrompt(
        prompt=prompt_text,
        multi_modal_data={"audio": (audio_signal, sr)},
    )

    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=0.2,
        top_k=20,
        top_p=0.85,
        repetition_penalty=1.1,
        detokenize=True,
    )

    outputs = llm.generate([prompt], sampling_params)
    print(f"[wired] generate() returned {len(outputs)} OmniRequestOutput message(s)")

    thinker_out = None
    decoder_out = None
    for i, o in enumerate(outputs):
        stage_id = getattr(o, "stage_id", None)
        final_output_type = getattr(o, "final_output_type", None)
        mm = o.multimodal_output
        print(
            f"[wired] outputs[{i}]: stage_id={stage_id} final_output_type={final_output_type} "
            f"finished={getattr(o, 'finished', None)} multimodal_output_type={type(mm).__name__}"
        )
        if stage_id == 0:
            thinker_out = o
        elif stage_id == 1:
            decoder_out = o

    if thinker_out is not None:
        thinker_mm = thinker_out.multimodal_output
        if isinstance(thinker_mm, Mapping):
            codes = thinker_mm.get("codes", {})
            if isinstance(codes, Mapping):
                audio = codes.get("audio")
                if audio is not None:
                    n_frames = audio.shape[0] if hasattr(audio, "shape") else len(audio)
                    print(f"[wired] thinker produced {n_frames} audio code frames")
        token_ids = getattr(thinker_out.request_output.outputs[0], "token_ids", []) if thinker_out.request_output else []
        print(f"[wired] thinker generated {len(token_ids)} visible tokens")
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            print(f"[wired] thinker decoded text: {tok.decode(token_ids)!r}")
        except Exception as e:
            print(f"[wired] (could not decode thinker tokens: {e})")
    else:
        print("[wired] WARNING: no stage_id=0 (thinker) output found in the returned list")

    result: dict[str, object] = {"stage0_seen": thinker_out is not None, "stage1_seen": decoder_out is not None}

    if decoder_out is None:
        verdict = "FAIL: no stage_id=1 (audio decoder) output in generate()'s return list at all"
        result["verdict"] = verdict
        print(f"[wired] VERDICT {verdict}")
        with open(out_json, "w") as f:
            json.dump(result, f)
        return

    mm = decoder_out.multimodal_output
    print(f"[wired] decoder multimodal_output type={type(mm).__name__} keys={list(mm.keys()) if mm else None}")
    # The client-side payload key is the stage's engine_output_type ("audio"),
    # not the raw "model_outputs" producer key.
    waveform = None
    if mm is not None:
        for key in ("audio", "model_outputs"):
            waveform = mm.get(key)
            if waveform is not None:
                break
    sr_tensor = mm.get("sr") if mm is not None else None

    if waveform is None:
        verdict = "FAIL: audio decoder stage ran but produced no waveform"
        result["verdict"] = verdict
        result["decoder_mm_keys"] = list(mm.keys()) if mm is not None else None
        print(f"[wired] VERDICT {verdict}")
        print(f"[wired] decoder multimodal_output keys: {result['decoder_mm_keys']}")
        with open(out_json, "w") as f:
            json.dump(result, f)
        return

    sample_rate = int(sr_tensor.item()) if hasattr(sr_tensor, "item") else 24000
    wave_1d = waveform.reshape(-1).to(dtype=torch.float32).cpu()
    num_samples = int(wave_1d.shape[0])
    duration_s = num_samples / sample_rate if sample_rate else 0.0
    print(f"[wired] waveform shape={list(waveform.shape)} sample_rate={sample_rate} duration={duration_s:.2f}s")

    try:
        import soundfile as sf

        sf.write(out_wav, wave_1d.numpy(), sample_rate)
        print(f"[wired] wrote waveform to {out_wav}")
        wav_written = True
    except Exception as e:
        print(f"[wired] WARNING: failed to write wav file: {e}")
        wav_written = False

    verdict = f"PASS: audio decoder produced a {duration_s:.2f}s waveform at {sample_rate}Hz"
    result.update(
        {
            "verdict": verdict,
            "num_samples": num_samples,
            "sample_rate": sample_rate,
            "duration_s": duration_s,
            "wav_written": wav_written,
            "out_wav": out_wav if wav_written else None,
        }
    )
    print(f"[wired] VERDICT {verdict}")

    with open(out_json, "w") as f:
        json.dump(result, f)
    print(f"[wired] wrote result metadata to {out_json}")


if __name__ == "__main__":
    main()
