# Offline Example of vLLM-Omni for Qwen3-omni

## 🛠️ Installation

Please refer to [README.md](../../../README.md)

## Run examples (Qwen3-omni)
### Multiple Prompts
Download dataset from [seed_tts](https://drive.google.com/file/d/1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP/edit). For processing dataset please refer to [Qwen2.5-omni README.md](../qwen2_5_omni/README.md)
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
