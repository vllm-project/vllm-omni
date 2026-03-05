import random
from typing import List

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


class G2PTableTokenizer(TokenizerSpec):
    def __init__(self, vocab, g2p_table, unk_label='<unk>'):
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        self.unk_label = unk_label
        self.unk_id = self.vocab[unk_label]

        self.g2p_table = g2p_table

        self.g2p_id_table = {}
        for word, phone_seqs in self.g2p_table.items():
            phone_id_seqs = []
            for phone_seq_i in phone_seqs:
                phone_seq_id_i = tuple(self.vocab[phone] for phone in phone_seq_i)
                assert phone_seq_id_i not in phone_id_seqs
                phone_id_seqs.append(phone_seq_id_i)
            assert len(phone_id_seqs) > 0
            self.g2p_id_table[word] = phone_id_seqs

    @classmethod
    def load(cls, vocab_fp, lexicon_fp):
        vocab = load_vocab(vocab_fp)

        g2p_table = {}
        with open(lexicon_fp, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                word, phone_seq = line.split('\t')
                if word not in g2p_table:
                    g2p_table[word] = []
                phone_seq = tuple(phone_seq.split(' '))
                if phone_seq in g2p_table[word]:
                    print('found duplicated mapping: ', line)
                else:
                    assert len(phone_seq) > 0
                    g2p_table[word].append(phone_seq)
        return cls(vocab, g2p_table)

    def text_to_tokens(self, text: str) -> List[str]:
        text = text.strip()
        words = text.split()
        tokens = []
        for word_i in words:
            phones_list = self.g2p_table.get(word_i)
            if phones_list:
                if len(phones_list) == 1:
                    tokens.extend(phones_list[0])
                else:
                    tokens.extend(random.choice(phones_list))
            else:
                tokens.append(self.unk_label)
        return tokens

    def tokens_to_text(self, tokens: List[str]) -> str:
        return ' '.join(tokens)

    def text_to_ids(self, text: str) -> List[int]:
        words = text.split()
        ids = []
        for word_i in words:
            phone_ids_list = self.g2p_id_table.get(word_i)
            if phone_ids_list:
                if len(phone_ids_list) == 1:
                    ids.extend(phone_ids_list[0])
                else:
                    ids.extend(random.choice(phone_ids_list))
            else:
                print('found unk: ', text)
                ids.append(self.unk_id)
        return ids

    def ids_to_text(self, ids: List[int]) -> str:
        ids_ = [id_ for id_ in ids]
        return ' '.join(self.ids_to_tokens(ids_))

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.vocab[token] for token in tokens]

    def token_to_id(self, token: str) -> int:
        return self.vocab[token]

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.inv_vocab[id] for id in ids]


def load_vocab(path, encoding='utf-8'):
    vocab_dict = {}
    with open(path, encoding=encoding) as f:
        idx = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            token = line.split('\t')[0]
            assert token not in vocab_dict
            vocab_dict[token] = idx
            idx += 1
    return vocab_dict
