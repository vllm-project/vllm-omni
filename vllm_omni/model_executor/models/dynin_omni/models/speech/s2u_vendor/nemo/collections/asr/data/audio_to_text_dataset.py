# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Optional

import torch
from omegaconf import DictConfig

from nemo.collections.asr.data import audio_to_text, audio_to_text_dali

def get_phone_dataset(config: dict, augmentor: Optional['AudioAugmentor'] = None) -> audio_to_text.AudioToCharDataset:
    """
    Instantiates a Character Encoding based AudioToCharDataset.

    Args:
        config: Config of the AudioToCharDataset.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.

    Returns:
        An instance of AudioToCharDataset.
    """
    manifest_dir = config.get('manifest_dir', '')
    manifest_filepath = config.get('manifest_filepath', '')
    if manifest_dir:
        manifest_filepath = [os.path.join(manifest_dir, fp_i) for fp_i in manifest_filepath.split(',')]
        manifest_filepath = ','.join(manifest_filepath)

    dataset = audio_to_text.AudioToPhoneDataset(
        manifest_filepath=manifest_filepath,
        labels=config['labels'],
        sample_rate=config['sample_rate'],
        int_values=config.get('int_values', False),
        augmentor=augmentor,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        blank_index=config.get('blank_index', -1),
        unk_index=config.get('unk_index', -1),
        normalize=config.get('normalize_transcripts', False),
        trim=config.get('trim_silence', False),
        load_audio=config.get('load_audio', True),
        parser=config.get('parser', 'en'),
        parser_add_end_space=config.get('parser_add_end_space', False),
        add_misc=config.get('add_misc', False),
        data_dir=config.get('data_dir', ''),
        dup_factor=config.get('dup_factor', 1)
    )
    return dataset


def get_char_dataset(config: dict, augmentor: Optional['AudioAugmentor'] = None) -> audio_to_text.AudioToCharDataset:
    """
    Instantiates a Character Encoding based AudioToCharDataset.

    Args:
        config: Config of the AudioToCharDataset.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.

    Returns:
        An instance of AudioToCharDataset.
    """
    manifest_dir = config.get('manifest_dir', '')
    manifest_filepath = config.get('manifest_filepath', '')
    if manifest_dir:
        manifest_filepath = [os.path.join(manifest_dir, fp_i) for fp_i in manifest_filepath.split(',')]
        manifest_filepath = ','.join(manifest_filepath)

    dataset = audio_to_text.AudioToCharDataset(
        manifest_filepath=manifest_filepath,
        labels=config['labels'],
        sample_rate=config['sample_rate'],
        int_values=config.get('int_values', False),
        augmentor=augmentor,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        blank_index=config.get('blank_index', -1),
        unk_index=config.get('unk_index', -1),
        normalize=config.get('normalize_transcripts', False),
        trim=config.get('trim_silence', False),
        load_audio=config.get('load_audio', True),
        parser=config.get('parser', 'en'),
        parser_add_end_space=config.get('parser_add_end_space', False),
        add_misc=config.get('add_misc', False),
        data_dir=config.get('data_dir', ''),
        dup_factor=config.get('dup_factor', 1)
    )
    return dataset


def get_audio_dataset(config: dict, augmentor: Optional['AudioAugmentor'] = None, return_both=False) -> audio_to_text.AudioToCharDataset:
    manifest_dir = config.get('manifest_dir', '')
    manifest_filepath = config.get('manifest_filepath', '')
    if manifest_dir:
        manifest_filepath = [os.path.join(manifest_dir, fp_i) for fp_i in manifest_filepath.split(',')]
        manifest_filepath = ','.join(manifest_filepath)

    dataset = audio_to_text.AudioDataset(
        manifest_filepath=manifest_filepath,
        sample_rate=config['sample_rate'],
        int_values=config.get('int_values', False),
        augmentor=augmentor,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        crop_size=config.get('crop_size', None),
        max_utts=config.get('max_utts', 0),
        trim=config.get('trim_silence', False),
        data_dir=config.get('data_dir', ''),
        return_both=return_both
    )
    return dataset


def get_bpe_dataset(
    config: dict, tokenizer: 'TokenizerSpec', augmentor: Optional['AudioAugmentor'] = None, *, aux_tokenizer=None, las_format=False
) -> audio_to_text.AudioToBPEDataset:
    """
    Instantiates a Byte Pair Encoding / Word Piece Encoding based AudioToBPEDataset.

    Args:
        config: Config of the AudioToBPEDataset.
        tokenizer: An instance of a TokenizerSpec object.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.

    Returns:
        An instance of AudioToBPEDataset.
    """
    manifest_dir = config.get('manifest_dir', '')
    manifest_filepath = config.get('manifest_filepath', '')
    if manifest_dir:
        manifest_filepath = [os.path.join(manifest_dir, fp_i) for fp_i in manifest_filepath.split(',')]
        manifest_filepath = ','.join(manifest_filepath)

    dataset = audio_to_text.AudioToBPEDataset(
        manifest_filepath=manifest_filepath,
        tokenizer=tokenizer,
        sample_rate=config['sample_rate'],
        int_values=config.get('int_values', False),
        augmentor=augmentor,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        trim=config.get('trim_silence', False),
        load_audio=config.get('load_audio', True),
        add_misc=config.get('add_misc', False),
        use_start_end_token=config.get('use_start_end_token', True),
        data_dir=config.get('data_dir', ''),
        dup_factor=config.get('dup_factor', 1),
        sampling_nbest_size=config['subword_sampling_nbest_size'],
        sampling_alpha=config['subword_sampling_alpha'],
        aux_tokenizer=aux_tokenizer,
        las_format=las_format,
    )
    return dataset


def get_tarred_char_dataset(
    config: dict, shuffle_n: int, global_rank: int, world_size: int, augmentor: Optional['AudioAugmentor'] = None
) -> audio_to_text.TarredAudioToCharDataset:
    """
    Instantiates a Character Encoding based TarredAudioToCharDataset.

    Args:
        config: Config of the TarredAudioToCharDataset.
        shuffle_n: How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
        global_rank: Global rank of this device.
        world_size: Global world size in the training method.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.

    Returns:
        An instance of TarredAudioToCharDataset.
    """
    dataset = audio_to_text.TarredAudioToCharDataset(
        audio_tar_filepaths=config['tarred_audio_filepaths'],
        manifest_filepath=config['manifest_filepath'],
        labels=config['labels'],
        sample_rate=config['sample_rate'],
        int_values=config.get('int_values', False),
        augmentor=augmentor,
        shuffle_n=shuffle_n,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        blank_index=config.get('blank_index', -1),
        unk_index=config.get('unk_index', -1),
        normalize=config.get('normalize_transcripts', False),
        trim=config.get('trim_silence', False),
        parser=config.get('parser', 'en'),
        add_misc=config.get('add_misc', False),
        shard_strategy=config.get('tarred_shard_strategy', 'scatter'),
        global_rank=global_rank,
        world_size=world_size,
    )
    return dataset


def get_tarred_bpe_dataset(
    config: dict,
    tokenizer: 'TokenizerSpec',
    shuffle_n: int,
    global_rank: int,
    world_size: int,
    augmentor: Optional['AudioAugmentor'] = None,
) -> audio_to_text.TarredAudioToBPEDataset:
    """
    Instantiates a Byte Pair Encoding / Word Piece Encoding based TarredAudioToBPEDataset.

    Args:
        config: Config of the TarredAudioToBPEDataset.
        tokenizer: An instance of a TokenizerSpec object.
        shuffle_n: How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
        global_rank: Global rank of this device.
        world_size: Global world size in the training method.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.

    Returns:
        An instance of TarredAudioToBPEDataset.
    """
    dataset = audio_to_text.TarredAudioToBPEDataset(
        audio_tar_filepaths=config['tarred_audio_filepaths'],
        manifest_filepath=config['manifest_filepath'],
        tokenizer=tokenizer,
        sample_rate=config['sample_rate'],
        int_values=config.get('int_values', False),
        augmentor=augmentor,
        shuffle_n=shuffle_n,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        trim=config.get('trim_silence', False),
        add_misc=config.get('add_misc', False),
        use_start_end_token=config.get('use_start_end_token', True),
        shard_strategy=config.get('tarred_shard_strategy', 'scatter'),
        global_rank=global_rank,
        world_size=world_size,
    )
    return dataset


def get_dali_char_dataset(
    config: dict,
    shuffle: bool,
    device_id: int,
    global_rank: int,
    world_size: int,
    preprocessor_cfg: Optional[DictConfig] = None,
) -> audio_to_text_dali.AudioToCharDALIDataset:
    """
    Instantiates a Character Encoding based AudioToCharDALIDataset.

    Args:
        config: Config of the AudioToCharDALIDataset.
        shuffle: Bool flag whether to shuffle the dataset.
        device_id: Index of the GPU to be used (local_rank). Only applicable when device == 'gpu'. Defaults to 0.
        global_rank: Global rank of this device.
        world_size: Global world size in the training method.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.

    Returns:
        An instance of AudioToCharDALIDataset.
    """
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    dataset = audio_to_text_dali.AudioToCharDALIDataset(
        manifest_filepath=config['manifest_filepath'],
        device=device,
        batch_size=config['batch_size'],
        labels=config['labels'],
        sample_rate=config['sample_rate'],
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        blank_index=config.get('blank_index', -1),
        unk_index=config.get('unk_index', -1),
        normalize=config.get('normalize_transcripts', False),
        trim=config.get('trim_silence', False),
        parser=config.get('parser', 'en'),
        shuffle=shuffle,
        device_id=device_id,
        global_rank=global_rank,
        world_size=world_size,
        preprocessor_cfg=preprocessor_cfg,
    )
    return dataset
