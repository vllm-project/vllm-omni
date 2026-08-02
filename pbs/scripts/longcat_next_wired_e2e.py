"""Single-node, wired multi-stage LongCat-Next e2e test (one modality per run).

Orchestrated pipeline via the official vllm-omni multi-stage API: Omni.generate()
returns one OmniRequestOutput per stage.

Usage:
  python longcat_next_wired_e2e.py <model_path> <deploy_yaml> <out_dir> [--modality audio|image]

No `modalities` are passed on the prompt: LongCat-Next is a native multimodal
model, so the modality is inferred from whatever the thinker actually generates
(the decoders no-op when their codes are absent).

Decoder stages are selected by payload CONTENT, not by the static
final_output_type label, because the combined multi-decoder stage is typed
"audio" yet still emits images for image-gen prompts. The client-side payload
key is the stage's engine_output_type ("audio"/"image"), never the raw
"model_outputs" producer key.

Examples:
  # Audio-only test (use audio-decoder yaml; image decoder not even loaded)
  python longcat_next_wired_e2e.py /model longcat_next_4gpu_80gb_audio.yaml /results --modality audio

  # Image-only test (use image+audio yaml; audio decoder runs but no-ops)
  python longcat_next_wired_e2e.py /model longcat_next_4gpu_80gb_image_audio.yaml /results --modality image

  # Either modality via the combined 2-stage multi-decoder yaml
  python longcat_next_wired_e2e.py /model longcat_next_5gpu_a40_multi_decoder.yaml /results --modality audio
  python longcat_next_wired_e2e.py /model longcat_next_5gpu_a40_multi_decoder.yaml /results --modality image
"""

import argparse
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


def test_audio(model_path: str, llm: Omni, out_dir: str, num_stages: int) -> dict:
    result: dict = {"modality": "audio"}

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
        temperature=0.4,
        top_k=20,
        top_p=0.85,
        repetition_penalty=1.1,
        detokenize=True,
    )

    outputs = llm.generate([prompt], [sampling_params] + [None] * (num_stages - 1))
    result["num_outputs"] = len(outputs)

    audio_decoder_out = None
    for o in outputs:
        stage_id = getattr(o, "stage_id", None)
        final_output_type = getattr(o, "final_output_type", None)
        mm = o.multimodal_output
        print(
            f"[wired-audio] outputs: stage_id={stage_id} final_output_type={final_output_type} "
            f"multimodal_output_type={type(mm).__name__}"
        )
        if stage_id == 0:
            thinker_mm = mm
            if isinstance(thinker_mm, Mapping):
                codes = thinker_mm.get("codes", {})
                if isinstance(codes, Mapping):
                    audio = codes.get("audio")
                    if audio is not None:
                        n_frames = audio.shape[0] if hasattr(audio, "shape") else len(audio)
                        result["thinker_audio_frames"] = n_frames
                        print(f"[wired-audio] thinker produced {n_frames} audio code frames")
        elif stage_id >= 1 and isinstance(mm, Mapping) and any(
            k in mm for k in ("audio", "model_outputs")
        ):
            audio_decoder_out = o

    if audio_decoder_out is None:
        result["verdict"] = "FAIL: no decoder output found"
        return result

    # The client-side payload key is the stage's engine_output_type, never
    # the raw "model_outputs" producer key: an "audio"-typed stage (both the
    # standalone audio decoder and the combined multi-decoder) surfaces its
    # waveform under "audio". Select by content.
    mm = audio_decoder_out.multimodal_output
    waveform = None
    if mm is not None:
        for key in ("audio", "model_outputs"):
            waveform = mm.get(key)
            if waveform is not None:
                break
    sr_tensor = mm.get("sr") if mm is not None else None

    if waveform is None:
        result["verdict"] = "FAIL: audio decoder produced no waveform"
        return result

    sample_rate = int(sr_tensor.item()) if hasattr(sr_tensor, "item") else 24000
    wave_1d = waveform.reshape(-1).to(dtype=torch.float32).cpu()
    num_samples = int(wave_1d.shape[0])
    duration_s = num_samples / sample_rate if sample_rate else 0.0
    result.update({"sample_rate": sample_rate, "num_samples": num_samples, "duration_s": duration_s, "verdict": "PASS"})

    try:
        import soundfile as sf
        out_wav = os.path.join(out_dir, "audio.wav")
        sf.write(out_wav, wave_1d.numpy(), sample_rate)
        result["wav_written"] = True
        print(f"[wired-audio] wrote {out_wav}")
    except Exception as e:
        print(f"[wired-audio] warning: failed to write wav: {e}")
        result["wav_written"] = False

    return result


def test_image(model_path: str, llm: Omni, out_dir: str, num_stages: int,
               token_w: int = 37, token_h: int = 37) -> dict:
    result: dict = {"modality": "image"}

    # Any-resolution prefix: the reference's visual_generation_config inserts
    # "<longcat_img_token_size>{h} {w}</longcat_img_token_size>" right before
    # <longcat_img_start> at visual entry (its `anyres_prefix` custom param),
    # and the checkpoint's own img_gen test cases carry it in the prompt too.
    # Without it the model never learns the requested canvas and the image
    # content drifts from the description. token_w/token_h are ALSO threaded
    # per-request via additional_information (the reference's per-request
    # input_extra_infos[0]["token_w"]), since image_generation_config is None.
    prompt_text = (
        "<longcat_system>You are a helpful assistant. "
        "<longcat_user>请生成一张图片，内容是一只猫。 "
        "<longcat_assistant>"
        f"<longcat_img_token_size>{token_h} {token_w}</longcat_img_token_size>"
        "<longcat_img_start>"
    )
    # Overridable via LONGCAT_CFG_SCALE for A/B testing (e.g. 1.0 collapses
    # cfg_scale * (cond - uncond) + uncond to plain cond -- identical to
    # having no CFG at all -- vs. the checkpoint default 3.0) without editing
    # this file between runs.
    cfg_scale = float(os.environ.get("LONGCAT_CFG_SCALE", 3.0))
    prompt = OmniTextPrompt(
        prompt=prompt_text,
        multi_modal_data=None,
        additional_information={
            "token_w": token_w,
            "token_h": token_h,
            # Visual CFG scale (generation_config.json custom_params.cfg_scale
            # = 3.0). The thinker combines the conditional and its uncond twin
            # streams' depth-head logits with
            # cfg_scale * (cond - uncond) + uncond.
            "cfg_scale": cfg_scale,
        },
    )

    # Fixed seed so cfg_scale A/B comparisons only vary in cfg_scale, not in
    # sampling randomness.
    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=0.4,
        top_k=20,
        top_p=0.85,
        repetition_penalty=1.1,
        detokenize=True,
        seed=42,
    )
    result["token_w"], result["token_h"] = token_w, token_h
    result["cfg_scale"] = cfg_scale

    outputs = llm.generate([prompt], [sampling_params] + [None] * (num_stages - 1))
    result["num_outputs"] = len(outputs)

    image_decoder_out = None
    for o in outputs:
        stage_id = getattr(o, "stage_id", None)
        final_output_type = getattr(o, "final_output_type", None)
        mm = o.multimodal_output
        print(
            f"[wired-image] outputs: stage_id={stage_id} final_output_type={final_output_type} "
            f"multimodal_output_type={type(o.multimodal_output).__name__}"
        )
        # Select by content, not the static label: the combined multi-decoder
        # stage is statically final_output_type="audio" yet still emits an
        # image for image-gen prompts, keyed "audio" client-side.
        if stage_id >= 1 and isinstance(mm, Mapping) and any(
            k in mm for k in ("image", "audio", "model_outputs")
        ):
            image_decoder_out = o

    if image_decoder_out is None:
        result["verdict"] = "FAIL: no decoder output found"
        return result

    mm = image_decoder_out.multimodal_output
    image_tensor = None
    if mm is not None:
        for key in ("image", "audio", "model_outputs"):
            image_tensor = mm.get(key)
            if image_tensor is not None:
                break

    if image_tensor is None:
        result["verdict"] = "FAIL: image decoder produced no output"
        return result

    img_np = image_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).to(dtype=torch.float32).cpu().numpy()
    result.update({"image_shape": list(image_tensor.shape), "verdict": "PASS"})

    try:
        from PIL import Image
        pil = Image.fromarray((img_np * 255).astype("uint8")).convert("RGB")
        out_png = os.path.join(out_dir, f"image_cfg{cfg_scale:g}.png")
        pil.save(out_png)
        result["png_written"] = True
        print(f"[wired-image] wrote {out_png}")
    except Exception as e:
        print(f"[wired-image] warning: failed to write png: {e}")
        result["png_written"] = False

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="LongCat-Next wired e2e test")
    parser.add_argument("model_path", help="Path to model")
    parser.add_argument("deploy_yaml", help="Path to deploy config YAML")
    parser.add_argument("out_dir", help="Output directory")
    parser.add_argument("--modality", choices=["audio", "image"], default="audio",
                        help="Modality to test (default: audio)")
    parser.add_argument("--img-token-w", type=int, default=37,
                        help="Image grid width in tokens (default: 37)")
    parser.add_argument("--img-token-h", type=int, default=37,
                        help="Image grid height in tokens (default: 37)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    llm = Omni(
        model=args.model_path,
        deploy_config=args.deploy_yaml,
        trust_remote_code=True,
    )
    num_stages = llm.num_stages

    print("\n" + "=" * 60)
    print(f"[wired] === {args.modality.title()} modality test ===")
    print("=" * 60 + "\n")

    if args.modality == "audio":
        result = test_audio(args.model_path, llm, args.out_dir, num_stages)
    else:
        result = test_image(
            args.model_path, llm, args.out_dir, num_stages,
            token_w=args.img_token_w, token_h=args.img_token_h,
        )

    print(f"\n[wired-{args.modality}] result: {json.dumps(result, indent=2)}\n")

    # Summary
    print("=" * 60)
    print(f"[wired] === E2E Summary ===")
    label = f"{result.get('duration_s', 0):.2f}s" if result.get("duration_s") else ""
    print(f"[wired] {args.modality.title()}: {result.get('verdict', 'UNKNOWN')} {label}")

    out_json = os.path.join(args.out_dir, "result.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[wired] wrote result metadata to {out_json}")


if __name__ == "__main__":
    main()
