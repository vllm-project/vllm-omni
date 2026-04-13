"""Save 4.5 TTS audio to file."""
import numpy as np
import soundfile as sf


def main():
    from vllm_omni.entrypoints.omni import Omni

    MODEL = "/cache/caitianchi/model/MiniCPM-o-4_5"
    STAGE_CFG = "vllm_omni/model_executor/stage_configs/minicpmo45_8x4090.yaml"
    TTS_SUFFIX = "<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>"

    omni = Omni(model=MODEL, stage_configs_path=STAGE_CFG, trust_remote_code=True)
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n你好，用一句话介绍北京<|im_end|>\n"
        "<|im_start|>assistant\n" + TTS_SUFFIX
    )
    outputs = omni.generate(prompt)
    for out in outputs:
        print(f"stage={out.stage_id} type={out.final_output_type}")
        if out.multimodal_output and "audio" in out.multimodal_output:
            aud = out.multimodal_output["audio"]
            if hasattr(aud, "numpy"):
                aud = aud.numpy()
            if isinstance(aud, np.ndarray) and aud.size > 100:
                path = "/cache/caitianchi/code/o45/test/offline_tts_45_北京.wav"
                sf.write(path, aud, 24000)
                dur = aud.size / 24000
                print(f"AUDIO SAVED: {path} ({aud.size} samples, {dur:.1f}s)")
        for ro in out.request_output:
            for o in ro.outputs:
                if o.text:
                    print(f"text: {o.text[:200]}")
    print("DONE")


if __name__ == "__main__":
    main()
