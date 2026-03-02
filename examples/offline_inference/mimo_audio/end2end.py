# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use vLLM for running offline inference
with the correct prompt format on MiMo-Audio-Omni.
"""

import copy
import json
import os
from typing import NamedTuple

import soundfile as sf
from message_convert import (
    get_audio_data,
    get_audio_understanding_sft_prompt,
    get_s2t_dialogue_sft_multiturn_prompt,
    get_spoken_dialogue_sft_multiturn_prompt,
    get_text_dialogue_sft_multiturn_prompt,
    get_tts_sft_prompt,
    to_prompt,
)
from vllm import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniTokensPrompt

SEED = 42

MAX_CODE2WAV_TOKENS = 18192  # Maximum tokens supported by code2wav model


class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


def get_codes_query_from_json(codes_path: str) -> QueryResult:
    with open(codes_path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        code_final = data
    elif isinstance(data, dict) and "code_final" in data:
        code_final = data["code_final"]
    else:
        raise ValueError(
            f"Unsupported codes json format in {codes_path}.\n"
            "Expect a JSON list[int] or {{'code_final': list[int]}}."
        )

    if not isinstance(code_final, list) or not all(isinstance(x, int) for x in code_final):
        raise ValueError("code_final must be a list[int].")

    if len(code_final) > MAX_CODE2WAV_TOKENS:
        print(f"[Warn] code_final len={len(code_final)} > {MAX_CODE2WAV_TOKENS}, truncating.")
        code_final = code_final[:MAX_CODE2WAV_TOKENS]

    return QueryResult(
        inputs=OmniTokensPrompt(
            prompt_token_ids=code_final,
            multi_modal_data=None,
            mm_processor_kwargs=None,
        ),
        limit_mm_per_prompt={},
    )


def get_tts_sft(
    text="The weather is so nice today.",
    instruct=None,
    read_text_only=True,
    prompt_speech=None,
    audio_list=None,
):
    res = get_tts_sft_prompt(
        text,
        instruct=instruct,
        read_text_only=read_text_only,
        prompt_speech=prompt_speech,
    )

    prompt = to_prompt(res)
    final_prompt = {
        "prompt": prompt,
    }
    if audio_list is not None:
        final_prompt.update(
            {
                "multi_modal_data": {
                    "audio": audio_list,
                },
            }
        )
    return final_prompt


def get_audio_understanding_sft(audio_path, text="", thinking=False, use_sostm=False):
    audio_list = []
    audio_list.append(get_audio_data(audio_path))
    res = get_audio_understanding_sft_prompt(
        input_speech=audio_path, input_text=text, thinking=thinking, use_sostm=use_sostm
    )
    prompt = to_prompt(res)
    final_prompt = {
        "prompt": prompt,
        "multi_modal_data": {
            "audio": audio_list,
        },
    }
    return final_prompt


def get_spoken_dialogue_sft_multiturn(message_list, system_prompt=None, ref_audio_path=None, audio_list=None):
    res = get_spoken_dialogue_sft_multiturn_prompt(
        message_list, system_prompt=system_prompt, prompt_speech=ref_audio_path
    )
    prompt = to_prompt(res)
    final_prompt = {
        "prompt": prompt,
        "multi_modal_data": {
            "audio": audio_list,
        },
    }
    return final_prompt


def get_speech2text_dialogue_sft_multiturn(message_list, thinking=False, audio_list=None):
    res = get_s2t_dialogue_sft_multiturn_prompt(
        message_list,
        thinking=thinking,
    )
    prompt = to_prompt(res)
    final_prompt = {
        "prompt": prompt,
        "multi_modal_data": {
            "audio": audio_list,
        },
    }
    return final_prompt


def get_text_dialogue_sft_multiturn(
    message_list,
):
    res = get_text_dialogue_sft_multiturn_prompt(
        message_list,
    )
    prompt = to_prompt(res)
    final_prompt = {
        "prompt": prompt,
    }
    return final_prompt


query_map = {
    "tts_sft": get_tts_sft,
    "tts_sft_with_instruct": get_tts_sft,
    "tts_sft_with_audio": get_tts_sft,
    "tts_sft_with_natural_instruction": get_tts_sft,
    "audio_trancribing_sft": get_audio_understanding_sft,
    "audio_understanding_sft": get_audio_understanding_sft,
    "audio_understanding_sft_with_thinking": get_audio_understanding_sft,
    "spoken_dialogue_sft_multiturn": get_spoken_dialogue_sft_multiturn,
    "speech2text_dialogue_sft_multiturn": get_speech2text_dialogue_sft_multiturn,
    "text_dialogue_sft_multiturn": get_text_dialogue_sft_multiturn,
}


def main(args):
    model_name = args.model_name

    # Get paths from args
    text = getattr(args, "text", None)
    audio_path = getattr(args, "audio_path", None)

    instruct = getattr(args, "instruct", None)

    # Get the query function and call it with appropriate parameters
    query_func = query_map[args.query_type]

    omni_llm = Omni(
        model=model_name,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.enable_stats,
        log_file=("omni_llm_pipeline.log" if args.enable_stats else None),
        init_sleep_seconds=args.init_sleep_seconds,
        batch_timeout=args.batch_timeout,
        init_timeout=args.init_timeout,
        shm_threshold_bytes=args.shm_threshold_bytes,
    )

    thinker_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=2048,
        seed=SEED,
        logit_bias={},
        repetition_penalty=1.1,
    )

    code2wav_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=4096 * 16,
        seed=SEED,
        detokenize=True,
        repetition_penalty=1.1,
    )

    sampling_params_list = [
        thinker_sampling_params,
        code2wav_sampling_params,
    ]

    # Build query result based on query type
    # Notice: The audio files used in this example are available at: https://github.com/XiaomiMiMo/MiMo-Audio/tree/main/examples
    if args.query_type == "tts_sft":
        # python3 -u end2end.py --stage-configs-path ${config_file} --model ${MODEL_PATH}  --query-type tts_sft
        query_result = query_func(text=text, read_text_only=True)
    elif args.query_type == "tts_sft_with_instruct":
        # python3 -u end2end.py --stage-configs-path ${config_file} --model ${MODEL_PATH}  --query-type tts_sft_with_instruct --instruct "Speak happily in a child's voice"
        query_result = query_func(text=text, instruct=instruct, read_text_only=True)
    elif args.query_type == "tts_sft_with_audio":
        # python3 -u end2end.py --stage-configs-path ${config_file} --model ${MODEL_PATH}  --query-type tts_sft_with_audio --audio_path "./spoken_dialogue_assistant_turn_1.wav"
        audio_list = [get_audio_data(audio_path)]
        query_result = query_func(text=text, read_text_only=True, prompt_speech=audio_path, audio_list=audio_list)
    elif args.query_type == "tts_sft_with_natural_instruction":
        # python3 -u end2end.py --stage-configs-path ${config_file} --model ${MODEL_PATH}  --query-type tts_sft_with_natural_instruction --text "In a panting young male voice, he said: I can't run anymore, wait for me!"
        query_result = query_func(text=text, read_text_only=False)
    elif args.query_type == "audio_trancribing_sft":
        # python3 -u end2end.py --stage-configs-path ${config_file} --model ${MODEL_PATH}  --query-type audio_trancribing_sft --audio_path "./spoken_dialogue_assistant_turn_1.wav"
        audio_path = "spoken_dialogue_assistant_turn_1.wav"
        text = "Please transcribe this audio and repeat it once."
        query_result = query_func(text=text, audio_path=audio_path, use_sostm=True)
    elif args.query_type == "audio_understanding_sft":
        # python3 -u end2end.py --stage-configs-path ${config_file} --model ${MODEL_PATH}  --query-type audio_understanding_sft --text "Summarize the audio." --audio_path "./spoken_dialogue_assistant_turn_1.wav"
        query_result = query_func(text=text, audio_path=audio_path)
    elif args.query_type == "audio_understanding_sft_with_thinking":
        # python3 -u end2end.py --stage-configs-path ${config_file} --model ${MODEL_PATH}  --query-type audio_understanding_sft_with_thinking --text "Summarize the audio." --audio_path "./spoken_dialogue_assistant_turn_1.wav"
        query_result = query_func(text=text, audio_path=audio_path, thinking=True)
    elif args.query_type == "spoken_dialogue_sft_multiturn":
        # python3 -u end2end.py --stage-configs-path ${config_file} --model ${MODEL_PATH}  --query-type spoken_dialogue_sft_multiturn  --audio_path "./prompt_speech_zh_m.wav"
        first_turn_text_response = "我没办法获取实时的天气信息。不过呢，你可以试试几个方法来查看今天的天气。首先，你可以用手机自带的天气功能，比如苹果手机的天气应用，或者直接在系统设置里查看。其次，你也可以用一些专业的天气服务，像是国外的AccuWeather、Weather.com，或者国内的中国天气网、墨迹天气等等。再有就是，你还可以在谷歌或者百度里直接搜索你所在的城市加上天气这两个字。如果你能告诉我你所在的城市，我也可以帮你分析一下历史天气趋势，不过最新的数据还是需要你通过官方渠道去获取哦。"
        audio_list = []
        s1_audio_path = "weather_of_today.mp3"
        s2_audio_path = "spoken_dialogue_assistant_turn_1.wav"
        s3_audio_path = "beijing.mp3"
        audio_list.append(get_audio_data(audio_path))
        audio_list.append(get_audio_data(s1_audio_path))
        audio_list.append(get_audio_data(s2_audio_path))
        audio_list.append(get_audio_data(s3_audio_path))

        message_list = [
            {"role": "user", "content": s1_audio_path},
            {"role": "assistant", "content": {"text": first_turn_text_response, "audio": s2_audio_path}},
            {"role": "user", "content": s3_audio_path},
        ]
        query_result = query_func(message_list, system_prompt=None, ref_audio_path=audio_path, audio_list=audio_list)
    elif args.query_type == "speech2text_dialogue_sft_multiturn":
        # python3 -u end2end.py --stage-configs-path ${config_file} --model ${MODEL_PATH}  --query-type speech2text_dialogue_sft_multiturn
        s1_audio_path = "weather_of_today.mp3"
        s2_audio_path = "beijing.mp3"
        audio_list = []
        audio_list.append(get_audio_data(s1_audio_path))
        audio_list.append(get_audio_data(s2_audio_path))
        message_list = [
            {"role": "user", "content": s1_audio_path},
            {
                "role": "assistant",
                "content": "Hello, I can't get real-time weather information. If you can tell me your city, I can help you analyze historical weather trends, but for the latest data, you'll need to get it through official channels.",
            },
            {"role": "user", "content": s2_audio_path},
        ]
        query_result = query_func(message_list, thinking=True, audio_list=audio_list)
    elif args.query_type == "text_dialogue_sft_multiturn":
        # python3 -u end2end.py --stage-configs-path ${config_file} --model ${MODEL_PATH}  --query-type text_dialogue_sft_multiturn
        message_list = [
            {"role": "user", "content": "Could you recommend some tourist attractions in China?"},
            {"role": "assistant", "content": "Hello, which city would you like to travel to?"},
            {"role": "user", "content": "Beijing"},
        ]
        query_result = query_func(message_list=message_list)
    else:
        raise ValueError(f"Invalid query type: {args.query_type}")

    prompts = [copy.deepcopy(query_result) for _ in range(args.num_prompts)]

    print("prompts", prompts)
    omni_outputs = omni_llm.generate(prompts, sampling_params_list)

    output_dir = args.output_dir if getattr(args, "output_dir", None) else args.output_wav
    if args.query_type is not None:
        output_dir = os.path.join(output_dir, args.query_type)
    os.makedirs(output_dir, exist_ok=True)

    for stage_outputs in omni_outputs:
        if stage_outputs.final_output_type == "text":
            for output in stage_outputs.request_output:
                request_id = output.request_id
                text_output = output.outputs[0].text
                # Save aligned text file per request
                prompt_text = output.prompt
                out_txt = os.path.join(output_dir, f"{request_id}.txt")
                lines = []
                lines.append("Prompt:\n")
                lines.append(str(prompt_text) + "\n")
                lines.append("vllm_text_output:\n")
                lines.append(str(text_output).strip() + "\n")
                try:
                    with open(out_txt, "w", encoding="utf-8") as f:
                        print("lines", lines)
                        f.writelines(lines)
                except Exception as e:
                    print(f"[Warn] Failed writing text file {out_txt}: {e}")
                print(f"Request ID: {request_id}, Text saved to {out_txt}\n")
        elif stage_outputs.final_output_type == "audio":
            for output in stage_outputs.request_output:
                request_id = output.request_id
                audio_tensor = output.outputs[0].multimodal_output.get("audio")

                if audio_tensor is None:
                    continue

                output_wav = os.path.join(output_dir, f"{request_id}.wav")

                # Convert to numpy array and ensure correct format
                audio_numpy = audio_tensor.float().detach().cpu().numpy()

                # Ensure audio is 1D (flatten if needed)
                if audio_numpy.ndim > 1:
                    audio_numpy = audio_numpy.flatten()

                # Save audio file with explicit WAV format
                sf.write(output_wav, audio_numpy, samplerate=24000, format="WAV")
                print(f"Request ID: {request_id}, Audio saved to {output_wav}")


def parse_args():
    parser = FlexibleArgumentParser(description="Demo on using vLLM for offline inference with audio language models")
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="XiaomiMiMo/MiMo-Audio-7B-Instruct",
        help="Backbone LLM path.",
    )
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        default="The weather is so nice today.",
        help="input text",
    )
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="tts_sft",
        choices=query_map.keys(),
        help="Query type.",
    )
    parser.add_argument(
        "--audio-path",
        "-a",
        type=str,
        default=None,
        help="Path to local audio file. If not provided, uses default audio asset.",
    )
    parser.add_argument(
        "--instruct",
        type=str,
        default=None,
        help="instruct",
    )
    parser.add_argument(
        "--enable-stats",
        action="store_true",
        help="Enable writing detailed statistics (default: disabled)",
    )
    parser.add_argument(
        "--init-sleep-seconds",
        type=int,
        default=20,
        help="Sleep seconds after starting each stage process to allow initialization (default: 20)",
    )
    parser.add_argument(
        "--batch-timeout",
        type=int,
        default=5,
        help="Timeout for batching in seconds (default: 5)",
    )
    parser.add_argument(
        "--init-timeout",
        type=int,
        default=5000,
        help="Timeout for initializing stages in seconds (default: 5000)",
    )
    parser.add_argument(
        "--shm-threshold-bytes",
        type=int,
        default=65536,
        help="Threshold for using shared memory in bytes (default: 65536)",
    )
    parser.add_argument(
        "--output-dir",
        default="./output_audio",
        help="Output audio wav directory.",
    )
    parser.add_argument(
        "--output-wav",
        default="output_audio",
        help="[Deprecated] Output wav directory (use --output-dir).",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Number of prompts to generate.",
    )

    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=24000,
        help="Sampling rate for audio.",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default="../../../model_executor/stage_configs/mimo_audio.yaml",
        help="Path to a stage configs file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
