# Offline Example of vLLM-Omni for Qwen3-Omni

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen3_omni>.


## 🛠️ Installation

Please refer to [README.md](https://github.com/vllm-project/vllm-omni/tree/main/README.md)

## Run examples (Qwen3-Omni)
### Multiple Prompts
Download dataset from [seed_tts](https://drive.google.com/file/d/1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP/edit). For processing dataset please refer to [Qwen2.5-Omni README.md](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen2_5_omni/README.md)
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
    --8<-- "examples/offline_inference/qwen3_omni/end2end.py"
    ``````
??? abstract "run_multiple_prompts.sh"
    ``````sh
    --8<-- "examples/offline_inference/qwen3_omni/run_multiple_prompts.sh"
    ``````
??? abstract "run_single_prompt.sh"
    ``````sh
    --8<-- "examples/offline_inference/qwen3_omni/run_single_prompt.sh"
    ``````
??? abstract "run_single_prompt_tp.sh"
    ``````sh
    --8<-- "examples/offline_inference/qwen3_omni/run_single_prompt_tp.sh"
    ``````
