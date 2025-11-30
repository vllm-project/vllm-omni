# Offline Example of vLLM-Omni for Qwen3-omni

Source <https://github.com/vllm-project/vllm/tree/main/examples/offline_inference/qwen3_omni>.


## 🛠️ Installation

Please refer to [README.md](https://github.com/vllm-project/vllm/tree/main/README.md)

## Run examples (Qwen3-omni)
### Multiple Prompts
Download dataset from [seed_tts](https://drive.google.com/file/d/1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP/edit). For processing dataset please refer to [Qwen2.5-omni README.md](https://github.com/vllm-project/vllm/tree/main/examples/offline_inference/qwen2_5_omni/README.md)
Get into the example folder
```bash
cd examples/offline_inference/qwen3_omni
```
Then run the command below.
```bash
bash run_multiple_prompts.sh
```
### Single Prompt
Get into the example folder
```bash
cd examples/offline_inference/qwen3_omni
```
Then run the command below.
```bash
bash run_single_prompt.sh
```
If you have not enough memory, you can set thinker with tensor parallel. Just run the command below.
```bash
bash run_single_prompt_tp.sh
```

### FAQ

If you encounter error about backend of librosa, try to install ffmpeg with command below.
```
sudo apt update
sudo apt install ffmpeg
```

## Example materials

??? abstract "end2end.py"
    ``````py
    --8<-- "/mnt/vllm_open_release/vllm-omni-cursor/vllm-omni/examples/offline_inference/qwen3_omni/end2end.py"
    ``````
??? abstract "qwen3_omni_moe_tp.yaml"
    ``````yaml
    --8<-- "/mnt/vllm_open_release/vllm-omni-cursor/vllm-omni/examples/offline_inference/qwen3_omni/qwen3_omni_moe_tp.yaml"
    ``````
??? abstract "run_multiple_prompts.sh"
    ``````sh
    --8<-- "/mnt/vllm_open_release/vllm-omni-cursor/vllm-omni/examples/offline_inference/qwen3_omni/run_multiple_prompts.sh"
    ``````
??? abstract "run_single_prompt.sh"
    ``````sh
    --8<-- "/mnt/vllm_open_release/vllm-omni-cursor/vllm-omni/examples/offline_inference/qwen3_omni/run_single_prompt.sh"
    ``````
??? abstract "run_single_prompt_tp.sh"
    ``````sh
    --8<-- "/mnt/vllm_open_release/vllm-omni-cursor/vllm-omni/examples/offline_inference/qwen3_omni/run_single_prompt_tp.sh"
    ``````
