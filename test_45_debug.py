"""Debug MiniCPM-o 4.5 TTS data flow."""
import torch


def main():
    from vllm_omni.entrypoints.omni import Omni

    MODEL = "/cache/caitianchi/model/MiniCPM-o-4_5"
    STAGE_CFG = "vllm_omni/model_executor/stage_configs/minicpmo45_8x4090.yaml"
    TTS_SUFFIX = "<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>"

    print("=== Creating Omni ===")
    omni = Omni(
        model=MODEL,
        stage_configs_path=STAGE_CFG,
        trust_remote_code=True,
    )
    print("=== Omni ready ===")

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n你好<|im_end|>\n"
        "<|im_start|>assistant\n" + TTS_SUFFIX
    )
    outputs = omni.generate(prompt)

    print(f"\nGot {len(outputs)} output(s)")
    for i, out in enumerate(outputs):
        print(f"\n--- Output {i}: stage={out.stage_id} type={out.final_output_type} ---")
        if out.multimodal_output:
            for k, v in out.multimodal_output.items():
                if hasattr(v, "shape"):
                    print(f"  mm['{k}']: shape={v.shape} dtype={v.dtype}")
                elif hasattr(v, "__len__"):
                    print(f"  mm['{k}']: len={len(v)}")
        for ro in out.request_output:
            for o in ro.outputs:
                if o.text:
                    print(f"  text: {o.text[:200]}")
                if hasattr(o, "multimodal_output") and o.multimodal_output:
                    for k, v in o.multimodal_output.items():
                        info = f"shape={v.shape}" if hasattr(v, "shape") else f"len={len(v)}"
                        print(f"  output_mm['{k}']: {info}")
    print("\nDONE")


if __name__ == "__main__":
    main()
