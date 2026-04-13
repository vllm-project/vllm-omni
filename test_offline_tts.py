"""Offline TTS test for MiniCPM-o 2.6 via vllm-omni."""
import numpy as np
import soundfile as sf


def main():
    from vllm_omni.entrypoints.omni import Omni

    MODEL = "/cache/caitianchi/model/MiniCPM-o-2_6"
    STAGE_CFG = "vllm_omni/model_executor/stage_configs/minicpmo_8x4090.yaml"
    TTS_SUFFIX = "<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>"

    print("=" * 60)
    print("Creating Omni engine (3-stage: thinker -> talker -> code2wav)")
    print("=" * 60)
    omni = Omni(
        model=MODEL,
        stage_configs_path=STAGE_CFG,
        trust_remote_code=True,
    )

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n你好，用一句话介绍北京。<|im_end|>\n"
        "<|im_start|>assistant\n" + TTS_SUFFIX
    )

    print("\n" + "=" * 60)
    print("Generating...")
    print("=" * 60)
    outputs = omni.generate(prompt)

    print(f"\nGot {len(outputs)} output(s)\n")
    for i, out in enumerate(outputs):
        print(f"--- Output {i}: stage={out.stage_id}, type={out.final_output_type} ---")

        if out.multimodal_output:
            for k, v in out.multimodal_output.items():
                if hasattr(v, "shape"):
                    print(f"  mm['{k}']: shape={v.shape}, dtype={v.dtype}")
                elif hasattr(v, "__len__"):
                    print(f"  mm['{k}']: len={len(v)}")

            if "audio" in out.multimodal_output:
                aud = out.multimodal_output["audio"]
                if hasattr(aud, "numpy"):
                    aud = aud.numpy()
                if isinstance(aud, np.ndarray) and aud.size > 0:
                    sf.write("/tmp/minicpm_tts.wav", aud, 24000)
                    duration = aud.size / 24000
                    print(f"  >>> REAL AUDIO: /tmp/minicpm_tts.wav ({aud.size} samples, {duration:.2f}s)")
                else:
                    print(f"  audio empty (size={getattr(aud, 'size', '?')})")

        for ro in out.request_output:
            for o in ro.outputs:
                if o.text:
                    print(f"  text: {o.text}")
                if hasattr(o, "multimodal_output") and o.multimodal_output:
                    for k, v in o.multimodal_output.items():
                        info = f"shape={v.shape}" if hasattr(v, "shape") else f"len={len(v)}"
                        print(f"  output_mm['{k}']: {info}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
