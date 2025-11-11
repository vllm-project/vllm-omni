import argparse
import os
import os as _os_env_toggle
import random
from types import SimpleNamespace

import gradio as gr
import numpy as np
import torch

from utils import make_omni_prompt

from vllm.sampling_params import SamplingParams
from vllm_omni.entrypoints.omni_llm import OmniLLM

_os_env_toggle.environ["VLLM_USE_V1"] = "1"

SEED = 42

# Ensure deterministic behavior across runs.
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gradio demo for Qwen2.5-Omni offline inference."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to merged model directory (will be created if downloading).",
    )
    parser.add_argument(
        "--tokenize",
        action="store_true",
        help="Return tokenized prompts instead of raw text prompts.",
    )
    parser.add_argument(
        "--use-torchvision",
        action="store_true",
        help="Use torchvision to decode videos when applicable.",
    )
    parser.add_argument(
        "--prompt-type",
        default="text",
        choices=["text"],
        help="Prompt type to build with the demo interface.",
    )
    parser.add_argument(
        "--server-name",
        default="127.0.0.1",
        help="Host/IP for gradio `launch`.",
    )
    parser.add_argument(
        "--server-port", type=int, default=7860, help="Port for gradio `launch`."
    )
    parser.add_argument(
        "--share", action="store_true", help="Share the Gradio demo publicly."
    )
    return parser.parse_args()


def build_sampling_params(seed: int) -> list[SamplingParams]:
    thinker_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=2048,
        seed=seed,
        detokenize=True,
        repetition_penalty=1.1,
    )
    talker_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=2048,
        seed=seed,
        detokenize=True,
        repetition_penalty=1.1,
        stop_token_ids=[8294],
    )
    code2wav_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=2048,
        seed=seed,
        detokenize=True,
        repetition_penalty=1.1,
    )
    return [
        thinker_sampling_params,
        talker_sampling_params,
        code2wav_sampling_params,
    ]


def create_prompt_args(base_args: argparse.Namespace) -> SimpleNamespace:
    # The prompt builder expects a minimal namespace with these attributes.
    return SimpleNamespace(
        model=base_args.model,
        prompt_type=base_args.prompt_type,
        tokenize=base_args.tokenize,
        use_torchvision=base_args.use_torchvision,
        legacy_omni_video=False,
    )


def build_interface(
    omni_llm: OmniLLM,
    sampling_params: list[SamplingParams],
    prompt_args_template: SimpleNamespace,
):
    def run_inference(user_prompt: str):
        if not user_prompt.strip():
            return "Please provide a valid text prompt.", None

        prompt_args = SimpleNamespace(**prompt_args_template.__dict__)
        omni_prompt = make_omni_prompt(prompt_args, user_prompt)

        try:
            omni_outputs = omni_llm.generate([omni_prompt], sampling_params)
        except Exception as exc:  # pylint: disable=broad-except
            return f"Inference failed: {exc}", None

        text_outputs: list[str] = []
        audio_output = None

        for stage_outputs in omni_outputs:
            if stage_outputs.final_output_type == "text":
                for output in stage_outputs.request_output:
                    if output.outputs:
                        text_outputs.append(output.outputs[0].text)
            elif stage_outputs.final_output_type == "audio":
                for output in stage_outputs.request_output:
                    audio_tensor = output.multimodal_output["audio"]
                    audio_np = audio_tensor.detach().cpu().numpy()
                    audio_output = (
                        24000,
                        audio_np,
                    )

        text_response = "\n\n".join(text_outputs) if text_outputs else "No text output."
        return text_response, audio_output

    with gr.Blocks() as demo:
        gr.Markdown("# Qwen2.5-Omni Offline Inference Gradio Demo")
        with gr.Row():
            input_box = gr.Textbox(
                label="Input Prompt",
                placeholder="For example: Please introduce Qwen2.5-Omni.",
                lines=4,
            )
        with gr.Row():
            generate_btn = gr.Button("Generate", variant="primary")
        with gr.Row():
            text_output = gr.Textbox(label="Text Output", lines=6)
            audio_output = gr.Audio(label="Audio Output", interactive=False)

        generate_btn.click(
            fn=run_inference,
            inputs=[input_box],
            outputs=[text_output, audio_output],
        )
        demo.queue(concurrency_count=1)
    return demo


def main():
    args = parse_args()
    sampling_params = build_sampling_params(SEED)
    omni_llm = OmniLLM(model=args.model)
    prompt_args_template = create_prompt_args(args)

    demo = build_interface(omni_llm, sampling_params, prompt_args_template)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()

