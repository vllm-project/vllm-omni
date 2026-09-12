# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run one Kimi-Audio text/audio request and save its complete outputs."""

import copy
from pathlib import Path

import soundfile as sf
import torch
from vllm.multimodal.audio import resample_audio_pyav
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.model_executor.models.kimi_audio.audio_processing import prepare_kimi_audio_inputs


def main():
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--model", default="moonshotai/Kimi-Audio-7B-Instruct", help="Model ID or local checkpoint directory")
    parser.add_argument("--glm-tokenizer-path", help="Optional local glm-4-voice-tokenizer snapshot")
    parser.add_argument(
        "--deploy-config",
        default=str(Path(__file__).resolve().parents[3] / "vllm_omni/deploy/kimi_audio.yaml"),
        help="Omni deployment YAML; defaults to the bundled non-streaming configuration",
    )
    parser.add_argument("--text", help="Text prompt or an instruction accompanying the input audio")
    parser.add_argument("--audio-path", type=Path, help="Optional local mono audio file")
    parser.add_argument("--output-type", choices=("text", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/kimi_audio"))
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum AR generation steps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")

    text = args.text
    if text is None and args.audio_path is None:
        text = "你好，请用一句话介绍你自己。"
    messages, audio_inputs = [], {}
    if text:
        messages.append({"role": "user", "message_type": "text", "content": text})
    if args.audio_path is not None:
        waveform, sample_rate = sf.read(args.audio_path, dtype="float32")
        if waveform.ndim != 1:
            parser.error("--audio-path must contain mono audio")
        if sample_rate != 16000:
            waveform = resample_audio_pyav(waveform, orig_sr=sample_rate, target_sr=16000)
        audio_inputs[len(messages)] = waveform
        messages.append({"role": "user", "message_type": "audio", "content": str(args.audio_path)})
    if not messages:
        parser.error("Provide nonempty --text or --audio-path")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    additional_config = {}
    if args.glm_tokenizer_path:
        additional_config["kimi_audio"] = {"glm_tokenizer_path": args.glm_tokenizer_path}
    omni = Omni(
        model=args.model,
        deploy_config=args.deploy_config,
        trust_remote_code=True,
        additional_config=additional_config,
    )
    try:
        # Reuse the engine's tokenizer/config and the model's shared input path.
        prompt = prepare_kimi_audio_inputs(
            messages,
            omni.engine.input_processor.renderer.prompt_builder,
            audio_inputs=audio_inputs,
            output_type=args.output_type,
        )
        params = copy.deepcopy(omni.default_sampling_params_list)
        params[0].max_tokens = args.max_tokens
        params[0].seed = args.seed
        # Keep the deployment's separate text/audio sampling settings.
        texts, audio_index = [], 0
        for result in omni.generate([prompt], sampling_params_list=params, use_tqdm=False):
            for completion in result.outputs:
                if result.final_output_type == "text":
                    texts.append(completion.text)
                    print(f"Text: {completion.text}\nAR finish reason: {completion.finish_reason}")
                elif result.final_output_type == "audio":
                    mm = completion.multimodal_output
                    if not mm or "audio" not in mm:
                        continue
                    chunks = mm["audio"]
                    audio = torch.cat(chunks, dim=-1) if isinstance(chunks, list) else chunks
                    rate = mm["sr"]
                    rate = rate[-1] if isinstance(rate, list) else rate
                    sample_rate = int(torch.as_tensor(rate).item())
                    output_path = args.output_dir / f"audio-{audio_index}.wav"
                    sf.write(output_path, audio.detach().float().cpu().reshape(-1).numpy(), sample_rate, subtype="PCM_16")
                    audio_index += 1
                    print(f"Audio: {output_path} ({sample_rate} Hz)")
        text_path = args.output_dir / "text.txt"
        text_path.write_text("\n".join(texts), encoding="utf-8")
        print(f"Text saved to: {text_path}")
    finally:
        omni.close()


if __name__ == "__main__":
    main()
