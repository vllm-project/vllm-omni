"""Offline e2e example for Ming-flash-omni-2.0 standalone talker (TTS)."""

import os
from typing import Any

import soundfile as sf
import torch
from vllm.utils.argparse_utils import FlexibleArgumentParser

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.ming_flash_omni.prompt_utils import (
    DEFAULT_PROMPT,
    create_instruction,
)

MODEL_NAME = "Jonathan1909/Ming-flash-omni-2.0"
DEFAULT_STAGE_CONFIG = "vllm_omni/model_executor/stage_configs/ming_flash_omni_tts.yaml"


def get_messages(case: str, text_override: str | None) -> dict[str, Any]:
    if case == "style":
        text = text_override or "我会一直在这里陪着你，直到你慢慢、慢慢地沉入那个最温柔的梦里……好吗？"
        instruction = create_instruction(
            {
                "风格": "这是一种ASMR耳语，属于一种旨在引发特殊感官体验的创意风格。这个女性使用轻柔的普通话进行耳语，声音气音成分重。音量极低，紧贴麦克风，语速极慢，旨在制造触发听者颅内快感的声学刺激。",
            }
        )
        return {
            "prompt": DEFAULT_PROMPT,
            "text": text,
            "instruction": instruction,
            "use_zero_spk_emb": True,
        }
    if case == "ip":
        text = text_override or "这款产品的名字，叫变态坑爹牛肉丸。"
        return {
            "prompt": DEFAULT_PROMPT,
            "text": text,
            "instruction": create_instruction({"IP": "灵小甄"}),
            "use_zero_spk_emb": True,
        }
    if case == "basic":
        text = text_override or "我们当迎着阳光辛勤耕作，去摘取，去制作，去品尝，去馈赠。"
        return {
            "prompt": DEFAULT_PROMPT,
            "text": text,
            "instruction": create_instruction({"语速": "快速", "基频": "中", "音量": "中"}),
            "use_zero_spk_emb": True,
        }
    raise ValueError(f"Unknown case: {case}")


def save_audio(mm: dict[str, Any], output_path: str) -> None:
    if not mm or "audio" not in mm:
        raise RuntimeError("No audio found in model output")
    audio = mm["audio"]
    sr_raw = mm.get("sr", 44100)
    if isinstance(sr_raw, torch.Tensor):
        sample_rate = int(sr_raw.item())
    else:
        sample_rate = int(sr_raw)
    waveform = audio.squeeze().float().cpu().numpy()
    sf.write(output_path, waveform, sample_rate)
    print(f"Saved {output_path} ({len(waveform) / sample_rate:.2f}s, {sample_rate}Hz)")


def parse_args():
    parser = FlexibleArgumentParser(description="Ming-flash-omni standalone talker offline e2e example")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name or local path.")
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=DEFAULT_STAGE_CONFIG,
        help="Path to stage configs yaml for standalone talker deployment.",
    )
    parser.add_argument(
        "--case",
        type=str,
        default="style",
        choices=["style", "ip", "basic"],
        help="Example case.",
    )
    parser.add_argument("--text", type=str, default=None, help="Override default text for the selected case.")
    parser.add_argument("--output", type=str, default=None, help="Output wav path.")
    parser.add_argument("--log-stats", action="store_true", default=False, help="Enable stats logging.")
    parser.add_argument("--init-timeout", type=int, default=600, help="Engine init timeout in seconds.")
    parser.add_argument("--stage-init-timeout", type=int, default=300, help="Single stage init timeout in seconds.")
    return parser.parse_args()


def main():
    args = parse_args()

    omni = Omni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        init_timeout=args.init_timeout,
        stage_init_timeout=args.stage_init_timeout,
    )

    messages = get_messages(args.case, args.text)
    decode_args = {
        # Standalone TTS deployment
        "ming_task": "instruct",
        "max_decode_steps": 200,
        "cfg": 2.0,
        "sigma": 0.25,
        "temperature": 0.0,
    }
    req = OmniTokensPrompt(
        prompt_token_ids=[0],
        additional_information={**messages, **decode_args},
    )

    outputs = omni.generate(req)
    mm = outputs[0].outputs[0].multimodal_output

    output_path = args.output or f"output_{args.case}.wav"
    save_audio(mm, output_path)
    omni.close()


if __name__ == "__main__":
    main()
