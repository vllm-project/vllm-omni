from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/cache/caitianchi/model/MiniCPM-o-2_6', trust_remote_code=True)
msgs = [{'role': 'user', 'content': '你好'}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print('Prompt:', repr(prompt))
ids = tok.encode(prompt)
print('Last 10 token IDs:', ids[-10:])
print('Total tokens:', len(ids))
print('Has 151688 (spk_bos):', 151688 in ids)
print('Has 151691 (tts_bos):', 151691 in ids)
