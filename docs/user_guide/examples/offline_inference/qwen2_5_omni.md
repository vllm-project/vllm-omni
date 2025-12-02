# Offline Example of vLLM-Omni for Qwen2.5-Omni

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen2_5_omni>.


## 🛠️ Installation

Please refer to [README.md](https://github.com/vllm-project/vllm-omni/tree/main/README.md)

## Run examples (Qwen2.5-Omni)
### Multiple Prompts
Download dataset from [seed_tts](https://drive.google.com/file/d/1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP/edit). To get the prompt, you can:
```bash
tar -xf <Your Download Path>/seedtts_testset.tar
cp seedtts_testset/en/meta.lst examples/offline_inference/qwen2_5_omni/meta.lst
python3 examples/offline_inference/qwen2_5_omni/extract_prompts.py \
  --input examples/offline_inference/qwen2_5_omni/meta.lst \
  --output examples/offline_inference/qwen2_5_omni/top100.txt \
  --topk 100
```
Get into the example folder
```bash
cd examples/offline_inference/qwen2_5_omni
```
Then run the command below.
```bash
bash run_multiple_prompts.sh
```
### Single Prompt
Get into the example folder
```bash
cd examples/offline_inference/qwen2_5_omni
```
Then run the command below.
```bash
bash run_single_prompt.sh
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
    --8<-- "examples/offline_inference/qwen2_5_omni/end2end.py"
    ``````
??? abstract "extract_prompts.py"
    ``````py
    --8<-- "examples/offline_inference/qwen2_5_omni/extract_prompts.py"
    ``````
??? abstract "run_multiple_prompts.sh"
    ``````sh
    --8<-- "examples/offline_inference/qwen2_5_omni/run_multiple_prompts.sh"
    ``````
??? abstract "run_single_prompt.sh"
    ``````sh
    --8<-- "examples/offline_inference/qwen2_5_omni/run_single_prompt.sh"
    ``````
