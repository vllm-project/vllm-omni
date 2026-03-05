import gzip
import json
import os
import random

import numpy as np
import torch

from nemo.core import Dataset
from nemo.utils import logging


class Text2TextDataset(Dataset):
    def __init__(self, file_paths, parser, *, parse_online, mask_id, mask_prob, word_mask,
                 space_id, space_num_min, space_num_max, in_word_space_num_min, in_word_space_num_max,
                 min_text_len, max_text_len, max_crop_words, max_crop_chars, pad_id=0, encoding='utf-8',
                 input_parser=None, auxiliary_parser=None, replace_prob=0.0, replace_ids=None):
        self.parser = parser
        self.auxiliary_parser = auxiliary_parser
        if input_parser is None:
            input_parser = parser
        else:
            assert parse_online
        self.input_parser = input_parser
        self.mask_id = mask_id
        self.mask_prob = mask_prob
        self.word_mask = word_mask
        self.replace_prob = replace_prob
        if self.replace_prob > 0:
            self.replace_ids = list(replace_ids)
            assert len(self.replace_ids) > 0
        else:
            self.replace_ids = None
        self.space_id = space_id
        self.space_num_min = space_num_min
        self.space_num_max = space_num_max
        self.in_word_space_num_min = in_word_space_num_min
        self.in_word_space_num_max = in_word_space_num_max
        assert self.space_num_max >= self.space_num_min
        assert self.in_word_space_num_max >= self.in_word_space_num_min
        if self.in_word_space_num_max:
            assert parse_online
        self.max_crop_words = max_crop_words
        self.max_crop_chars = max_crop_chars
        self.pad_id = pad_id
        self.parse_online = parse_online
        print('parse online: ', self.parse_online)

        self.text_data = []
        filter_lines = 0
        for fp_i in file_paths:
            text_lines, filter_lines_i = load_data(fp_i, None if parse_online else self.parser,
                                                   min_text_len=min_text_len, max_text_len=max_text_len, encoding=encoding)
            self.text_data.extend(text_lines)
            filter_lines += filter_lines_i

        logging.info("Text Dataset filtered {} lines".format(filter_lines))
        logging.info("Text Dataset loaded with {} lines".format(len(self.text_data)))

    def _offline_get_item(self, idx):
        text = self.text_data[idx]
        masked_text = random_mask_add_space(text, self.space_id, self.space_num_min, self.space_num_max,
                                            self.mask_id, self.mask_prob)
        assert isinstance(text, list)
        return masked_text, text.copy()

    def _online_get_item(self, idx):
        text = self.text_data[idx]

        words = text.split()

        if self.max_crop_words:
            words = random_crop_words(words, self.max_crop_words, self.max_crop_chars)

        # todo: numpy seed in dataset init?
        masked_tokens, unmasked_tokens = words_random_mask_add_space(words, self.input_parser, space_id=self.space_id,
                                                                     space_num_min=self.space_num_min, space_num_max=self.space_num_max,
                                                                     in_word_space_num_min=self.in_word_space_num_min,
                                                                     in_word_space_num_max=self.in_word_space_num_max,
                                                                     mask_id=self.mask_id, mask_prob=self.mask_prob, word_mask=self.word_mask,
                                                                     replace_prob=self.replace_prob, replace_ids=self.replace_ids)
        target_text = ' '.join(words)
        target_tokens = self.parser(target_text)
        assert isinstance(target_tokens, list)
        if self.auxiliary_parser:
            assert self.auxiliary_parser == 'reuse_input_parser'
            if self.auxiliary_parser == 'reuse_input_parser':
                aux_target_tokens = unmasked_tokens
            else:
                aux_target_tokens = self.auxiliary_parser(target_text)
            return masked_tokens, target_tokens, aux_target_tokens
        return masked_tokens, target_tokens

    def __getitem__(self, idx):
        if self.parse_online:
            return self._online_get_item(idx)
        else:
            return self._offline_get_item(idx)

    def __len__(self):
        return len(self.text_data)

    def _collate_fn(self, batch):
        if self.auxiliary_parser:
            return auxiliary_collate_fn(batch, self.pad_id)
        else:
            return collate_fn(batch, self.pad_id)


def get_text2text_dataset(config, target_tokenizer, input_tokenizer, auxiliary_target_tokenizer=None):
    manifest_filepath = config['manifest_filepath']
    manifest_dir = config.get('manifest_dir', None)
    if manifest_dir:
        manifest_filepath = [os.path.join(manifest_dir, fp_i) for fp_i in manifest_filepath.split(',')]

    parser = target_tokenizer.text_to_ids

    input_parser = input_tokenizer.text_to_ids if input_tokenizer else None

    if isinstance(auxiliary_target_tokenizer, str):
        auxiliary_parser = auxiliary_target_tokenizer
        print('use auxiliary_parser:', auxiliary_parser)
    else:
        auxiliary_parser = auxiliary_target_tokenizer.text_to_ids if auxiliary_target_tokenizer else None

    dataset = Text2TextDataset(manifest_filepath, parser, parse_online=config['parse_online'],
                               mask_id=config['mask_id'], mask_prob=config['mask_prob'], word_mask=config['word_mask'],
                               replace_prob=config['replace_prob'], replace_ids=config['replace_ids'],
                               space_id=config['space_id'], space_num_min=config['space_num_min'], space_num_max=config['space_num_max'],
                               in_word_space_num_min=config['in_word_space_num_min'], in_word_space_num_max=config['in_word_space_num_max'],
                               min_text_len=config['min_text_len'], max_text_len=config['max_text_len'],
                               max_crop_words=config['max_crop_words'], max_crop_chars=config['max_crop_chars'],
                               encoding=config['encoding'], input_parser=input_parser, auxiliary_parser=auxiliary_parser)
    return dataset


def random_mask_add_space(input, space_id, space_num_min, space_num_max, mask_id, mask_prob):
    spaced_masked_text = []

    space_num = random.randint(space_num_min, space_num_max)
    spaced_masked_text.extend([space_id] * space_num)
    for i in range(len(input)):
        if random.random() < mask_prob:
            spaced_masked_text.append(mask_id)
        else:
            spaced_masked_text.append(input[i])

        space_num = random.randint(space_num_min, space_num_max)
        spaced_masked_text.extend([space_id] * space_num)
    return spaced_masked_text


def words_random_mask_add_space(words, parser, *, space_id, space_num_min, space_num_max,
                                in_word_space_num_min, in_word_space_num_max, mask_id, mask_prob, word_mask,
                                replace_prob, replace_ids):
    spaced_masked_token_ids = []
    unmasked_tokens_ids = []

    space_num = random.randint(space_num_min, space_num_max)
    spaced_masked_token_ids.extend([space_id] * space_num)
    for word_i in words:
        token_ids = parser(word_i)
        unmasked_tokens_ids.extend(token_ids)

        mask_word = word_mask and random.random() < mask_prob
        for i, token in enumerate(token_ids):
            if random.random() < replace_prob:
                spaced_masked_token_ids.append(random.choice(replace_ids))
            else:
                if word_mask:
                    if mask_word:
                        spaced_masked_token_ids.append(mask_id)
                    else:
                        spaced_masked_token_ids.append(token)
                else:
                    if random.random() < mask_prob:
                        spaced_masked_token_ids.append(mask_id)
                    else:
                        spaced_masked_token_ids.append(token)
            if i != len(token_ids) - 1:
                space_num = random.randint(in_word_space_num_min, in_word_space_num_max)
                spaced_masked_token_ids.extend([space_id] * space_num)

        space_num = random.randint(space_num_min, space_num_max)
        spaced_masked_token_ids.extend([space_id] * space_num)

    return spaced_masked_token_ids, unmasked_tokens_ids


def test_words_random_mask_add_space():
    words = ['a', 'good', 'day']
    parser = lambda w: list(w)
    masked_tokens = words_random_mask_add_space(words, parser, space_id='<sil>', space_num_min=2, space_num_max=2,
                                in_word_space_num_min=1, in_word_space_num_max=1,
                                mask_id='<mask>', mask_prob=0.0, word_mask=True)
    assert masked_tokens == ['<sil>', '<sil>', 'a',
                            '<sil>', '<sil>', 'g', '<sil>', 'o', '<sil>', 'o', '<sil>', 'd',
                            '<sil>', '<sil>', 'd', '<sil>', 'a', '<sil>', 'y',
                            '<sil>', '<sil>']


def load_data(fp, parser, *, min_text_len, max_text_len, encoding):
    text_lines = []

    is_json = fp.endswith('.json')

    file_format = 'gzip' if fp.endswith('.gz') else 'txt'

    filter_lines = 0
    with open_text_file(fp, file_format, encoding=encoding) as f:
        for line_i in f:
            if is_json:
                line_i = json.loads(line_i)['text']
            line_i = line_i.strip()
            if parser:
                token_ids = parser(line_i)
            else:
                token_ids = line_i
            if len(token_ids) > max_text_len or len(token_ids) < min_text_len:
                filter_lines += 1
                continue
            text_lines.append(token_ids)

    return text_lines, filter_lines


def open_text_file(fp, file_format, encoding=None):
    if file_format == 'txt':
        print('open txt file: {}'.format(fp))
        return open(fp, encoding=encoding)
    else:
        print('open gzip file: {}'.format(fp))
        assert file_format == 'gzip'
        return gzip.open(fp, mode='rt', encoding=encoding)


def collate_fn(batch, pad_id):
    masked_text_lens = [len(masked_text_i) for masked_text_i, _ in batch]
    max_masked_text_len = max(masked_text_lens)
    text_lens = [len(text_i) for _, text_i in batch]
    max_text_len = max(text_lens)

    padded_masked_text = []
    padded_text = []
    for masked_text_ids, text_ids in batch:
        if len(masked_text_ids) < max_masked_text_len:
            pad_num = max_masked_text_len - len(masked_text_ids)
            # masked_text_ids = F.pad(masked_text_ids, (0, pad_num), value=pad_id)
            masked_text_ids.extend([pad_id] * pad_num)
        padded_masked_text.append(masked_text_ids)

        if len(text_ids) < max_text_len:
            pad_num = max_text_len - len(text_ids)
            # text_ids = F.pad(text_ids, (0, pad_num), value=pad_id)
            text_ids.extend([pad_id] * pad_num)
        padded_text.append(text_ids)

    return torch.tensor(padded_masked_text, dtype=torch.int64), torch.tensor(masked_text_lens, dtype=torch.int64), \
           torch.tensor(padded_text, dtype=torch.int64), torch.tensor(text_lens, dtype=torch.int64)


def auxiliary_collate_fn(batch, pad_id):
    masked_text_lens = [len(masked_text_i) for masked_text_i, _, _ in batch]
    max_masked_text_len = max(masked_text_lens)
    text_lens = [len(text_i) for _, text_i, _ in batch]
    max_text_len = max(text_lens)
    aux_text_lens = [len(aux_text_i) for _, _, aux_text_i in batch]
    max_aux_text_len = max(aux_text_lens)

    padded_masked_text = []
    padded_text = []
    padded_aux_text = []
    for masked_text_ids, text_ids, aux_text_ids in batch:
        if len(masked_text_ids) < max_masked_text_len:
            pad_num = max_masked_text_len - len(masked_text_ids)
            # masked_text_ids = F.pad(masked_text_ids, (0, pad_num), value=pad_id)
            masked_text_ids.extend([pad_id] * pad_num)
        padded_masked_text.append(masked_text_ids)

        if len(text_ids) < max_text_len:
            pad_num = max_text_len - len(text_ids)
            # text_ids = F.pad(text_ids, (0, pad_num), value=pad_id)
            text_ids.extend([pad_id] * pad_num)
        padded_text.append(text_ids)
        
        if len(aux_text_ids) < max_aux_text_len:
            pad_num = max_aux_text_len - len(aux_text_ids)
            # aux_text_ids = F.pad(aux_text_ids, (0, pad_num), value=pad_id)
            aux_text_ids.extend([pad_id] * pad_num)
        padded_aux_text.append(aux_text_ids)

    return torch.tensor(padded_masked_text, dtype=torch.int64), torch.tensor(masked_text_lens, dtype=torch.int64), \
           torch.tensor(padded_text, dtype=torch.int64), torch.tensor(text_lens, dtype=torch.int64), \
           torch.tensor(padded_aux_text, dtype=torch.int64), torch.tensor(aux_text_lens, dtype=torch.int64)


def random_crop_words(words, max_words, max_chars=0):
    diff = len(words) - max_words
    if diff > 0:
        start = np.random.randint(0, diff + 1)
        end = len(words) - diff + start
        words = words[start:end]

    if max_chars:
        cropped_words = []
        cropped_chars = 0
        for word in words:
            cropped_chars += len(word)
            if cropped_chars > max_chars:
                break
            else:
                cropped_words.append(word)
    else:
        cropped_words = words

    return cropped_words


def random_replace(inputs: torch.Tensor, rep_prob, rep_id):
    mask = torch.bernoulli(torch.full(inputs.size(), rep_prob, device=inputs.device)).type(inputs.dtype)
    return mask * rep_id + (1 - mask) * inputs
