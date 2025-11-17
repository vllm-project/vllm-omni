# Offline Example of vLLM-omni for Qwen2.5-omni

## 🛠️ Installation

Please refer to [README.md](../../../README.md)

## Run examples (Qwen2.5-omni)
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
### Single Prompts
Get into the example folder
```bash
cd examples/offline_inference/qwen2_5_omni
```
Then run the command below.
```bash
bash run_single_prompt.sh
```
