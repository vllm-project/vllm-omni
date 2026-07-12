"""Offline inference demo for Chatterbox TTS (Original) via vLLM Omni.

Generates speech from text using Chatterbox Original (520M, LLaMA backbone)
with an optional reference audio for zero-shot voice cloning, plus
exaggeration control.  Outputs 24 kHz WAV files.

NOTE: this is a PREVIEW. Native Chatterbox Original runs with AR-stage
classifier-free guidance (cfg_weight=0.5); that is NOT yet implemented here,
so output is muffled/over-amplified relative to native. CFG is a tracked
follow-up. The script prints this warning on startup.

Requirements:
    pip install chatterbox-tts   # required for vocoder and speaker encoder

Usage:
    python chatterbox_tts_original.py --text "Hello world" --ref-audio ref.wav
    python chatterbox_tts_original.py --text "Hello world" --exaggeration 0.5
"""

import importlib
import logging
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

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "ResembleAI/chatterbox"
DEFAULT_STAGE_CONFIGS = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "vllm_omni",
    "model_executor",
    "stage_configs",
    "chatterbox.yaml",
)


def estimate_prompt_len(text: str, speech_cond_prompt_len: int = 150) -> int:
    """Rough estimate of prompt token count for placeholder allocation."""
    text_len = max(1, len(text) // 4 + 10)
    return 1 + speech_cond_prompt_len + text_len + 1


def main(args):
    """Run offline Chatterbox TTS (Original) inference."""
    print(
        "[WARNING] Chatterbox Original is a PREVIEW: AR-stage classifier-free "
        "guidance (CFG) is not yet implemented, so audio will sound muffled / "
        "over-amplified relative to native Chatterbox. This is a known, tracked "
        "follow-up.",
        flush=True,
    )
    texts = [args.text]
    if args.txt_prompts:
        with open(args.txt_prompts) as f:
            texts = [line.strip() for line in f if line.strip()]
        if not texts:
            raise ValueError(f"No valid prompts found in {args.txt_prompts}")

    ref_audio = args.ref_audio

    inputs = []
    for text in texts:
        additional_information = {
            "text": [text],
        }
        if ref_audio:
            additional_information["ref_audio"] = [ref_audio]
        if args.exaggeration is not None:
            additional_information["exaggeration"] = [args.exaggeration]

        inputs.append(
            {
                "prompt_token_ids": [0] * estimate_prompt_len(text),
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

    for i, stage_outputs in enumerate(omni.generate(inputs, sampling_params_list=None)):
        ro = stage_outputs.request_output
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
        output_wav = os.path.join(output_dir, f"chatterbox_original_{request_id}.wav")
        sf.write(output_wav, audio_numpy, samplerate=sample_rate, format="WAV")
        print(f"Request ID: {request_id}, Saved audio to {output_wav}")


def parse_args():
    parser = FlexibleArgumentParser(description="Chatterbox TTS (Original) offline inference via vLLM Omni")
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
        "--exaggeration",
        type=float,
        default=0.5,
        help="Exaggeration factor for emotion (0.0 to 1.0+, default: 0.5).",
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
        help="Path to stage configs YAML (default: chatterbox.yaml).",
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
