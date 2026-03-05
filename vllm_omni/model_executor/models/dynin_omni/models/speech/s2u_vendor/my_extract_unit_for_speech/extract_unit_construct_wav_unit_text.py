from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
import librosa

try:
    import torch_npu
    from torch_npu.npu import amp
    from torch_npu.contrib import transfer_to_npu
    print('Successful import torch_npu')
except Exception as e:
    print(e)

default_sample_rate = 16000

def read_audio(path):
    # SPIRAL model can only consume audio with sample_rate of 16000, resample may be carried out
    wav, origin_sample_rate = librosa.load(path, sr=None)
    if origin_sample_rate != default_sample_rate:
        wav = librosa.resample(wav, orig_sr=origin_sample_rate, target_sr=default_sample_rate)
    assert wav.ndim == 1, wav.ndim
    return wav

def extract_feature(model, audio, audio_length, enable_vq=True):
    encoded, encoded_len = model.encoder(audio, audio_length, None, None, mask=False, features_only=True)  # [B, T, D]
    if enable_vq:
        # For finte sclar quantizer
        encoded = model.pre_quant(encoded)
        q_feat = model.quantizer(encoded)
        q_feat_ids = model.quantizer.codes_to_indexes(q_feat)
        q_feat_ids = q_feat_ids.to(torch.int)
        return q_feat_ids, encoded_len
    return encoded, encoded_len

# reduced_unit_sequence surpasses duplicate patterns
def sample_extract_unit(wav_path, model, show_result=False):
    # load audio
    audio = read_audio(wav_path)
    x = torch.from_numpy(audio)
    max_chunk = 1600000
    x_len = [x.shape[0]]
    device = next(model.parameters()).device # we assume speech tokenizer is stored in a single device
    with torch.no_grad():
        x = x.float().to(device)
        x = x.view(1, -1)
        x_len = torch.from_numpy(np.array(x_len)).int().to(device)
        feat = []
        for start in range(0, x.size(1), max_chunk):
            x_chunk = x[:, start: start + max_chunk]
            feat_chunk, _ = extract_feature(model, x_chunk, x_len)
            feat.append(feat_chunk)
    # return T x C
    unit_list = torch.cat(feat, 1).squeeze(0).cpu().numpy().tolist()
    reduced_unit_list = []
    prev_unit = None
    for unit in unit_list:
        if unit != prev_unit:
            reduced_unit_list.append(unit)
            prev_unit = unit
    unit_sequence = ' '.join([str(x) for x in unit_list])
    reduced_unit_sequence = ' '.join([str(x) for x in reduced_unit_list])
    if show_result:
        print('unit_sequence:')
        print(unit_sequence)
        print('reduced_unit_sequence:')
        print(reduced_unit_sequence)
    return unit_sequence, reduced_unit_sequence

def batch_extract_unit(wav_file_list, model, show_result=False, max_chunk = 480000 ):
    # in this setting, we dont chunck the wav, but to process by a batch
    #  max_chunk = 480000 limit to 30 s to avoid out-of-memory issue
    # load audio
    audio_list = []
    audio_len_list = []
    skip_audio_num = 0
    extracted_wav_file_list = []
    skipped_wav_file_list = []
    for wav_file in wav_file_list:
        audio = read_audio(wav_file)
        audio_len = audio.shape[0]
        if audio_len > max_chunk:
            print(
                f'x is too long, x_len {audio_len} is longer than max_chunk {max_chunk}, skip it and extract later')
            skip_audio_num += 1
            skipped_wav_file_list.append(wav_file)
            continue  # not to stop the extraction
        extracted_wav_file_list.append(wav_file)
        audio_list.append(audio)
        audio_len_list.append(audio_len)
    actual_batch_size = len(extracted_wav_file_list)  # Actual batch size after removal of too-long audio
    if actual_batch_size == 0:
        return [], skipped_wav_file_list, [], []
    
    device = next(model.parameters()).device # we assume speech tokenizer is stored in a single device
    max_len = max(audio_len_list)
    batch_audio = torch.zeros([actual_batch_size, max_len]).float().to(device)
    for i in range(actual_batch_size):
        batch_audio[i, :audio_len_list[i]] = torch.from_numpy(audio_list[i])  # stack audio of different length
    batch_audio_len = torch.from_numpy(np.array(audio_len_list)).int().to(device)
    with torch.no_grad():
        batch_feat, batch_feat_len = extract_feature(model, batch_audio, batch_audio_len)
        # return B x T x C
    unit_sequence_list, reduced_unit_sequence_list = [], []
    for i in range(len(batch_feat_len)):
        feat_len = batch_feat_len[i]
        feat = batch_feat[i][:feat_len]  # Todo: should check!
        unit_list = feat.cpu().numpy().tolist()  # Todo: squeeze 0 should be removed, because we have a batch
        reduced_unit_list = []
        prev_unit = None
        for unit in unit_list:
            if unit != prev_unit:
                reduced_unit_list.append(unit)
                prev_unit = unit
        unit_sequence = ' '.join([str(x) for x in unit_list])
        reduced_unit_sequence = ' '.join([str(x) for x in reduced_unit_list])
        unit_sequence_list.append(unit_sequence)
        reduced_unit_sequence_list.append(reduced_unit_sequence)
    if show_result:
        print('unit_sequence:')
        print(unit_sequence)
        print('reduced_unit_sequence:')
        print(reduced_unit_sequence)
    return extracted_wav_file_list, skipped_wav_file_list, unit_sequence_list, reduced_unit_sequence_list

def get_S2U_ckpt_config_path(unit_type, language='English'):
    assert language in ['English', 'Chinese']
    assert unit_type == '40ms_multilingual_8888'
    # English and Chinese using the same SPIRAL model and config!!
    ckpt_path = "./speech_tokenization/SPIRAL_L2_BN_FSQ_CTC/my_S2U_model/SPIRAL2_base_mutilingual_wenet_lv13k960_pretrain_aishell1_ls100_finetune_FSQ_8888_CTC_4ConvDec_phone_40ms_1n8g_20240522/checkpoints/vq_ctc_finetune--val_wer=0.0293-epoch=316.ckpt"
    config_path = "my_conf/st2vec_multi_lingual_wenetspeech_13k960_pretrained_aishell1_ls100_finetune_FSQ_8888_CTC_4ConvDec_phone_40ms_20240522"
    return ckpt_path, config_path