"""Run HF eager forward up to the first audio frame and dump logits at codebook 0.

Compare with our vLLM output at sample#2 (first decode after audio_bos) to find
the source of model divergence (mode collapse on token 0).
"""

import sys
import torch
from transformers import AutoProcessor, HiggsAudioV2ForConditionalGeneration

MODEL_ID = "bosonai/higgs-audio-v2-generation-3B-base"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = HiggsAudioV2ForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()

conv = [
    {"role": "system", "content": [{"type": "text", "text": "Generate audio following instruction."}]},
    {"role": "user", "content": [{"type": "text", "text": "The quick brown fox jumps over the lazy dog."}]},
]
inputs = processor.apply_chat_template(
    conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
)
input_ids = inputs["input_ids"].cuda()
print(f"[hf_probe] input_ids shape={tuple(input_ids.shape)}, last 5={input_ids[0, -5:].tolist()}", flush=True)

# Build audio_token_mask: positions with audio_token_id or audio_delay_token_id
ATID = int(model.config.audio_token_id)
ADID = int(model.config.audio_delay_token_id)
ABOS = int(model.config.audio_bos_token_id)
ATBOS = int(model.config.audio_stream_bos_id)
ATEOS = int(model.config.audio_stream_eos_id)
NUM_CB = int(model.config.num_codebooks)
CB_SIZE = int(model.config.codebook_size)
print(f"[hf_probe] ATID={ATID} ADID={ADID} ABOS={ABOS} ATBOS={ATBOS} ATEOS={ATEOS} NUM_CB={NUM_CB} CB_SIZE={CB_SIZE}", flush=True)

# Forward prefill in eager mode to get the hidden state at audio_bos position
with torch.no_grad():
    # The HF generation path. Use generate with max_new_tokens=3 to capture
    # the first few audio frames and the hidden states.
    output = model.generate(
        input_ids,
        max_new_tokens=3,
        do_sample=False,  # greedy
        output_hidden_states=True,
        output_logits=True,
        return_dict_in_generate=True,
    )

print(f"[hf_probe] output keys: {list(output.keys()) if hasattr(output, 'keys') else type(output)}", flush=True)
print(f"[hf_probe] sequences shape: {tuple(output.sequences.shape)}", flush=True)
print(f"[hf_probe] new tokens (LM): {output.sequences[0, input_ids.shape[1]:].tolist()}", flush=True)
if hasattr(output, 'audio_sequences') and output.audio_sequences is not None:
    print(f"[hf_probe] audio_sequences shape: {tuple(output.audio_sequences.shape)}", flush=True)
    print(f"[hf_probe] audio_sequences (3 frames, 8 codebooks):", flush=True)
    print(output.audio_sequences[0].cpu().tolist(), flush=True)
if hasattr(output, 'logits') and output.logits:
    print(f"[hf_probe] num logit steps: {len(output.logits)}", flush=True)
    for i, lg in enumerate(output.logits[:3]):
        print(f"  step {i}: logits shape {tuple(lg.shape)}", flush=True)
if hasattr(output, 'hidden_states') and output.hidden_states:
    print(f"[hf_probe] num hidden_states steps: {len(output.hidden_states)}", flush=True)
    for i, hs in enumerate(output.hidden_states[:3]):
        if hs is not None:
            last = hs[-1]
            print(f"  step {i}: last layer hidden shape {tuple(last.shape)} norm[-1]={float(last[0, -1].float().norm()):.3f}", flush=True)

# Probe the audio_lm_head output for the FIRST audio frame
# In HF the audio frame logits are computed at audio positions.
# Look at .audio_logits if available.
if hasattr(output, 'audio_logits') and output.audio_logits:
    print(f"[hf_probe] audio_logits available, len={len(output.audio_logits)}", flush=True)
    for i, lg in enumerate(output.audio_logits[:3]):
        if lg is None:
            print(f"  step {i}: None", flush=True); continue
        print(f"  step {i}: audio logits shape {tuple(lg.shape)}", flush=True)
        # If shape is [1, num_codebooks, codebook_size], dump top5 per codebook
        if lg.dim() == 3 and lg.shape[1] == NUM_CB:
            for q in range(NUM_CB):
                top = lg[0, q].float().topk(5)
                idx = top.indices.tolist()
                val = [f'{v:.3f}' for v in top.values.tolist()]
                print(f"    q={q} top5_idx={idx} top5_val={val}", flush=True)
