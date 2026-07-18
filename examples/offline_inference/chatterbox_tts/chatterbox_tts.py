"""Offline inference demo for Chatterbox TTS (Turbo) via vLLM Omni.

Generates speech from text using Chatterbox Turbo with an optional reference
audio for zero-shot voice cloning.  Outputs 24 kHz WAV files.

Requirements:
    pip install chatterbox-tts   # required for vocoder and speaker encoder

Usage:
    python chatterbox_tts.py --text "Hello world" --ref-audio ref.wav
    python chatterbox_tts.py --text "Hello world"  # uses default ref audio
"""

import importlib
import os
import sys


def _check_dependencies():
    """Check that required packages are installed before doing anything else."""
    missing = []
    for pkg in ["chatterbox", "soundfile", "torchaudio"]:
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)
    if missing:
        pip_names = {"chatterbox": "chatterbox-tts", "soundfile": "soundfile", "torchaudio": "torchaudio"}
        install_cmd = " ".join(pip_names.get(p, p) for p in missing)
        print(
            f"ERROR: Missing required packages: {', '.join(missing)}\n"
            f"\n"
            f"Chatterbox TTS requires the upstream chatterbox-tts package for its\n"
            f"vocoder (S3Gen), speaker encoder (VoiceEncoder), and audio tokenizer.\n"
            f"vllm-omni handles the autoregressive T3 stage; everything else comes\n"
            f"from chatterbox-tts.\n"
            f"\n"
            f"Install with:\n"
            f"    pip install {install_cmd}\n",
            file=sys.stderr,
        )
        sys.exit(1)


_check_dependencies()

import soundfile as sf  # noqa: E402
import torch  # noqa: E402

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402

from vllm_omni import Omni  # noqa: E402

DEFAULT_MODEL = "ResembleAI/chatterbox-turbo"
DEFAULT_STAGE_CONFIGS = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "vllm_omni",
    "model_executor",
    "stage_configs",
    "chatterbox_turbo.yaml",
)


def compute_prompt_len(
    text: str, ref_audio_path: str | None, model_path: str, speech_cond_prompt_len: int = 375
) -> int:
    """Compute exact prompt token count to avoid zero-padding artefacts.

    The prompt layout is: [speaker(1) | cond_speech(N) | text_tokens(T) | start_speech(1)]
    Passing an over-estimated placeholder causes the model to decode from a position
    far past the real conditioning, producing degenerate (short) outputs.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    text_len = len(tokenizer.encode(text, add_special_tokens=False))

    if ref_audio_path is not None:
        from chatterbox.models.s3tokenizer import S3Tokenizer
        from torchaudio.functional import resample as _resample

        s3tok = S3Tokenizer("speech_tokenizer_v2_25hz")
        wav_np, _sr = sf.read(ref_audio_path, dtype="float32", always_2d=False)
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=-1)
        if _sr != 16000:
            wav_np = _resample(torch.from_numpy(wav_np).unsqueeze(0), _sr, 16000).squeeze(0).numpy()
        tokens, _ = s3tok([wav_np], max_len=speech_cond_prompt_len)
        cond_len = int(tokens.shape[1])
    else:
        # No ref audio — T3 loads default conditioning from conds.pt which
        # contains pre-computed cond_prompt_speech_tokens.  Read the actual
        # token count so the prompt length matches what _build_prompt_embeds
        # will produce.
        from transformers.utils.hub import cached_file

        conds_path = cached_file(model_path, "conds.pt")
        if conds_path is not None:
            conds = torch.load(conds_path, map_location="cpu", weights_only=True)
            t3_cond = conds.get("t3", {})
            default_tokens = t3_cond.get("cond_prompt_speech_tokens")
            if default_tokens is not None:
                cond_len = min(int(default_tokens.reshape(1, -1).shape[1]), speech_cond_prompt_len)
            else:
                cond_len = 0
        else:
            cond_len = 0

    return 1 + cond_len + text_len + 1


def main(args):
    """Run offline Chatterbox TTS inference."""
    texts = [args.text]
    if args.txt_prompts:
        with open(args.txt_prompts) as f:
            texts = [line.strip() for line in f if line.strip()]
        if not texts:
            raise ValueError(f"No valid prompts found in {args.txt_prompts}")

    ref_audio = args.ref_audio

    # Compute the exact prompt length per prompt before creating the engine.
    # The length depends on the text token count, so it must be recomputed for
    # every line of a --txt-prompts batch: reusing one length would zero-pad the
    # shorter prompts (292+ padded positions), degrading conditioning and causing
    # early EOS -- exactly what compute_prompt_len exists to prevent.
    print(f"Computing prompt lengths for model {args.model}...")

    inputs = []
    for text in texts:
        prompt_len = compute_prompt_len(text, ref_audio, args.model)

        additional_information = {
            "text": [text],
        }
        if ref_audio:
            additional_information["ref_audio"] = [ref_audio]

        inputs.append(
            {
                "prompt_token_ids": [0] * prompt_len,
                "additional_information": additional_information,
            }
        )

    stage_configs_path = args.stage_configs_path or DEFAULT_STAGE_CONFIGS
    omni = Omni(
        model=args.model,
        stage_configs_path=stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    outputs = list(omni.generate(inputs, sampling_params_list=None))
    num_saved = 0
    for i, omni_output in enumerate(outputs):
        ro = omni_output.request_output
        if ro is None:
            print(f"Request {i}: no request_output")
            continue

        mm = getattr(ro, "multimodal_output", None)
        if not mm and ro.outputs:
            mm = getattr(ro.outputs[0], "multimodal_output", None)

        if not mm or "audio" not in mm:
            print(f"Request {i}: no audio in multimodal_output (keys={list(mm.keys()) if mm else None})")
            continue

        audio_data = mm["audio"]
        if isinstance(audio_data, list):
            audio_tensor = torch.cat(audio_data, dim=-1)
        else:
            audio_tensor = audio_data

        sr_val = mm.get("sr")
        if sr_val is None:
            sample_rate = 24000
        elif hasattr(sr_val, "item"):
            sample_rate = sr_val.item()
        elif isinstance(sr_val, list):
            sample_rate = int(sr_val[-1])
        else:
            sample_rate = int(sr_val)

        audio_numpy = audio_tensor.float().detach().cpu().numpy()
        if audio_numpy.ndim > 1:
            audio_numpy = audio_numpy.flatten()

        request_id = getattr(ro, "request_id", str(i))
        output_wav = os.path.join(output_dir, f"chatterbox_{request_id}.wav")
        sf.write(output_wav, audio_numpy, samplerate=sample_rate, format="WAV")
        print(f"Request ID: {request_id}, Saved audio to {output_wav}")
        num_saved += 1

    if num_saved < len(outputs):
        # Fail loudly: a run that produced no (or partial) audio is a broken
        # run even when the engine itself reported success.
        raise SystemExit(f"Expected {len(outputs)} audio outputs, saved {num_saved}")


def parse_args():
    parser = FlexibleArgumentParser(description="Chatterbox TTS offline inference via vLLM Omni")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name or path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello! This is a test of the Chatterbox text to speech system.",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Path to reference audio for voice cloning (optional).",
    )
    parser.add_argument(
        "--txt-prompts",
        type=str,
        default=None,
        help="Path to a .txt file with one prompt per line.",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=None,
        help="Path to stage configs YAML (default: chatterbox_turbo.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        default="output_audio",
        help="Output directory for generated wav files.",
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        default=False,
        help="Enable writing detailed statistics.",
    )
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=300,
        help="Timeout for initializing a single stage in seconds.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
