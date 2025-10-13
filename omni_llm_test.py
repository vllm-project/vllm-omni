from vllm_omni.entrypoints.omni_llm import OmniLM

def main():
    model_name = "Qwen/Qwen2.5-Omni-7B"
    omni_lm = OmniLM(model=model_name)
    print("omni_lm.stage_configs: ", omni_lm.stage_configs)
    print("omni_lm.stage_list: ", omni_lm.stage_list)


if __name__ == "__main__":
    main()