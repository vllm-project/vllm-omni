# Frequently Asked Questions

> Q: How many chips do I need to infer a model in vLLM-Omni?

A: Now, we support natively disaggregated deployment for different model stages within a model. There is a restriction that one chip can only has one AutoRegressive model stage. This is because the unified KV cache management of vLLM. Stages of other types can coexist with in a chip. The restriction will be resolved in later version.

> Q: When trying to run examples, I encounter error about backend of librosa or soundfile. How to solve it?

A: If you encounter error about backend of librosa, try to install ffmpeg with command below.
```
sudo apt update
sudo apt install ffmpeg
```
