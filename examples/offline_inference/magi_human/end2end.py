import argparse

from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end inference script for MagiHuman.")
    parser.add_argument("--model", type=str, required=True, help="Path or ID of the MagiHuman model.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Text prompt containing visual description, dialogue, and background sound.",
    )
    parser.add_argument(
        "--tensor-parallel-size", "-tp", type=int, default=4, help="Tensor parallel size (number of GPUs)."
    )
    parser.add_argument(
        "--output", type=str, default="output_magihuman.mp4", help="Path to save the generated mp4 file."
    )
    parser.add_argument("--height", type=int, default=256, help="Video height.")
    parser.add_argument("--width", type=int, default=448, help="Video width.")
    parser.add_argument("--num-inference-steps", type=int, default=8, help="Number of denoising steps.")
    parser.add_argument("--seed", type=int, default=52, help="Random seed for generation.")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Initializing MagiHuman pipeline with TP={args.tensor_parallel_size}...")
    omni = Omni(
        model=args.model,
        init_timeout=1200,
        tensor_parallel_size=args.tensor_parallel_size,
        devices=list(range(args.tensor_parallel_size)),
    )

    prompt = args.prompt
    if not prompt:
        prompt = (
            "A young woman with long, wavy golden blonde hair and bright blue eyes, "
            "wearing a fitted ivory silk blouse with a delicate lace collar, sits "
            "stationary in front of a softly lit, blurred warm-toned interior. Her "
            "overall disposition is warm, composed, and gently confident. The camera "
            "holds a static medium close-up, framing her from the shoulders up, "
            "with shallow depth of field keeping her face in sharp focus. Soft "
            "directional key light falls from the upper left, casting a gentle "
            "highlight along her cheekbone and nose bridge. She draws a quiet breath, "
            "the levator labii superiors relaxing as her lips part. She speaks in "
            "clear, warm, unhurried American English: "
            "\"The most beautiful things in life aren't things at all — "
            "they're moments, feelings, and the people who make you feel truly alive.\" "
            "Her jaw descends smoothly on each stressed syllable; the orbicularis oris "
            "shapes each vowel with precision. A faint, genuine smile engages the "
            "zygomaticus major, lifting her lip corners fractionally. Her brows rest "
            "in a soft, neutral arch throughout. She maintains steady, forward-facing "
            "eye contact. Head position remains level; no torso displacement occurs.\n\n"
            "Dialogue:\n"
            "<Young blonde woman, American English>: "
            "\"The most beautiful things in life aren't things at all — "
            "they're moments, feelings, and the people who make you feel truly alive.\"\n\n"
            "Background Sound:\n"
            "<Soft, warm indoor ambience with a faint distant piano melody>"
        )

    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        extra_args={
            "seconds": 5,
            "sr_height": 1080,
            "sr_width": 1920,
            "sr_num_inference_steps": 5,
        },
    )

    print(f"Generating with prompt: {prompt[:80]}...")
    outputs = omni.generate(
        prompts=[prompt],
        sampling_params_list=[sampling_params],
    )

    print(f"Generation complete. Output type: {type(outputs)}")
    if outputs:
        first = outputs[0]

        if hasattr(first, "images") and first.images:
            video_frames = first.images[0]
            print(f"Video frames: shape={video_frames.shape}, dtype={video_frames.dtype}")

            audio_waveform = None
            if hasattr(first, "multimodal_output") and first.multimodal_output:
                audio_waveform = first.multimodal_output.get("audio")
                if audio_waveform is not None:
                    print(f"Audio waveform: shape={audio_waveform.shape}, dtype={audio_waveform.dtype}")

            video_bytes = mux_video_audio_bytes(
                video_frames,
                audio_waveform,
                fps=25.0,
                audio_sample_rate=44100,
            )
            with open(args.output, "wb") as f:
                f.write(video_bytes)
            print(f"Saved MP4 ({len(video_bytes)} bytes) to {args.output}")
        print("SUCCESS: MagiHuman pipeline generation completed.")
    else:
        print("WARNING: No outputs returned.")


if __name__ == "__main__":
    main()
