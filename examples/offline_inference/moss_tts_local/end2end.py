"""
Offline TTS inference with MOSS-TTS-Local on vllm-omni.

Usage:
  cd /workspace/vllm-omni/examples/offline_inference/moss_tts_local
  MOSS_AUDIO_TOKENIZER_PATH=/workspace/vllm-omni/weights/moss-audio-tokenizer \
  python end2end.py \
    --model /workspace/vllm-omni/weights/moss-tts-local \
    --stage-configs-path ../../../vllm_omni/model_executor/stage_configs/moss_tts_local.yaml \
    --text "The weather is so nice today." \
    --output-dir ./output_audio
"""
import copy
import os
import soundfile as sf
import torch
from vllm import SamplingParams
from vllm_omni.entrypoints.omni import Omni

SEED = 42

# <user_inst> template — mirrors UserMessage.__post_init__ in processing_moss_tts.py
_USER_INST_TEMPLATE = """\
<user_inst>
- Reference(s):
{reference}
- Instruction:
{instruction}
- Tokens:
{tokens}
- Quality:
{quality}
- Sound Event:
{sound_event}
- Ambient Sound:
{ambient_sound}
- Language:
{language}
- Text:
{text}
</user_inst>"""


def build_tts_prompt(
    text: str,
    model_path: str,
    instruction=None,
    tokens=None,
    quality=None,
    sound_event=None,
    ambient_sound=None,
    language=None,
) -> str:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        os.path.abspath(model_path),
        trust_remote_code=True,
    )

    content = (
        _USER_INST_TEMPLATE
        .replace("{reference}", "None")
        .replace("{instruction}", str(instruction))
        .replace("{tokens}", str(tokens))
        .replace("{quality}", str(quality))
        .replace("{sound_event}", str(sound_event))
        .replace("{ambient_sound}", str(ambient_sound))
        .replace("{language}", str(language))
        .replace("{text}", str(text))
    )

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt = prompt + "<|audio_start|>"
    return prompt


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/vllm-omni/weights/moss-tts-local")
    parser.add_argument("--stage-configs-path", default="../../../vllm_omni/model_executor/stage_configs/moss_tts_local.yaml")
    parser.add_argument("--text", default="The weather is so nice today.")
    parser.add_argument("--output-dir", default="./output_audio")
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument("--init-sleep-seconds", type=int, default=20)
    parser.add_argument("--batch-timeout", type=int, default=5)
    parser.add_argument("--init-timeout", type=int, default=5000)
    parser.add_argument("--shm-threshold-bytes", type=int, default=65536)
    args = parser.parse_args()

    if not os.environ.get("MOSS_AUDIO_TOKENIZER_PATH"):
        sibling = os.path.join(
            os.path.dirname(os.path.abspath(args.model)),
            "moss-audio-tokenizer",
        )
        if os.path.exists(sibling):
            os.environ["MOSS_AUDIO_TOKENIZER_PATH"] = sibling
            print(f"[Info] MOSS_AUDIO_TOKENIZER_PATH auto-set → {sibling}")
        else:
            raise SystemExit(
                "[Error] MOSS_AUDIO_TOKENIZER_PATH must point to a local "
                "MOSS-Audio-Tokenizer snapshot. Download "
                "OpenMOSS-Team/MOSS-Audio-Tokenizer first, or place it beside "
                "the MOSS-TTS-Local model as 'moss-audio-tokenizer'."
            )

    os.makedirs(args.output_dir, exist_ok=True)

    prompt = build_tts_prompt(args.text, args.model)
    print(f"[Info] Prompt ({len(prompt)} chars):\n{prompt}\n")

    from transformers import AutoConfig
    _cfg = AutoConfig.from_pretrained(
        os.path.abspath(args.model), trust_remote_code=True
    )
    _audio_end_id = getattr(_cfg, "audio_end_token_id", None)
    _eos_id       = getattr(_cfg, "eos_token_id", None)

    if isinstance(_eos_id, list):
        _eos_ids = _eos_id
    elif _eos_id is not None:
        _eos_ids = [_eos_id]
    else:
        _eos_ids = []
    ar_stop_ids = list(dict.fromkeys(
        ([_audio_end_id] if _audio_end_id is not None else []) + _eos_ids
    ))
    print(f"[Info] AR stop token IDs: {ar_stop_ids}  "
          f"(audio_end={_audio_end_id}, eos={_eos_ids})")


    ar_params = SamplingParams(
        temperature=0.6, 
        top_p=0.95, 
        top_k=50,
        max_tokens=500, 
        seed=SEED,
        stop_token_ids=ar_stop_ids if ar_stop_ids else None,
    )
    decoder_params = SamplingParams(
        temperature=0.0, 
        top_p=1.0, 
        top_k=-1,
        max_tokens=18192, 
        seed=SEED, 
        detokenize=False,
    )

    omni = Omni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
        init_sleep_seconds=args.init_sleep_seconds,
        batch_timeout=args.batch_timeout,
        init_timeout=args.init_timeout,
        shm_threshold_bytes=args.shm_threshold_bytes,
    )

    prompts = [copy.deepcopy({"prompt": prompt}) for _ in range(args.num_prompts)]
    print(f"[Info] Running {len(prompts)} prompt(s)...")
    omni_outputs = omni.generate(prompts, [ar_params, decoder_params])

    for stage_outputs in omni_outputs:
        output = stage_outputs.request_output
        rid    = output.request_id

        if stage_outputs.final_output_type == "text":
            text_out = output.outputs[0].text
            print(f"[{rid}] AR text: {text_out[:300]!r}")

        elif stage_outputs.final_output_type == "audio":
            audio_tensor = output.outputs[0].multimodal_output.get("audio")
            if audio_tensor is None:
                print(f"[{rid}] No audio in output — check AR stage generated gen_slot tokens.")
                continue

            if isinstance(audio_tensor, list):
                chunks = [
                    t for t in audio_tensor
                    if isinstance(t, torch.Tensor) and t.numel() > 0
                ]
                if not chunks:
                    print(f"[{rid}] No audio chunks in output.")
                    continue
                audio_tensor = torch.cat(chunks, dim=0)

            audio_np = audio_tensor.float().detach().cpu().numpy()
            if audio_np.ndim > 1:
                audio_np = audio_np.flatten()

            wav_path = os.path.join(args.output_dir, f"{rid}.wav")
            sf.write(wav_path, audio_np, samplerate=24000, format="WAV")
            dur = len(audio_np) / 24000
            print(f"[{rid}] Saved → {wav_path}  ({dur:.2f}s)")


if __name__ == "__main__":
    main()
