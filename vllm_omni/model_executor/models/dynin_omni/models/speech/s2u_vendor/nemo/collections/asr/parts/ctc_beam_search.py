import time

import torch

import numpy as np


class CTCBeamSearchDecoder:
    def __init__(
        self,
        beam_size: int,
        blank_index: int = 0,
        temperature: float = 1.0,
        combine_path: bool = True
    ):

        self.blank_id = blank_index
        # self.vocab_size = decoder_model.vocab_size

        if beam_size < 1:
            raise ValueError("Beam search size cannot be less than 1!")

        self.beam_size = beam_size

        self.beam_stepwise_ln_alpha = 0.
        self.beam_word_reward_ratio = 0.
        if self.beam_word_reward_ratio > 0:
            assert self.beam_stepwise_ln_alpha == 0
        self.beam_combine_path = combine_path
        self.beam_temperature = temperature

        self.search_algorithm = self.beam_search

    def decode(self, logits, logits_len):
        assert len(logits) == len(logits_len)
        all_best_hyps = []
        all_nbest_hyps = []
        decode_st = time.perf_counter()
        for i, (logits_i, logits_i_len) in enumerate(zip(logits, logits_len)):
            st = time.perf_counter()
            best_hyps = []
            nbest_hyps = []
            for batch_idx in range(logits_i.shape[0]):
                inseq = logits_i[batch_idx:batch_idx + 1]
                logitlen = logits_i_len[batch_idx]

                nbest_hyps_i = self.search_algorithm(inseq, logitlen)  # sorted list of hypothesis

                best_hyps.append(nbest_hyps_i[0])
                nbest_hyps.append(nbest_hyps_i)
            all_best_hyps.append(best_hyps)
            all_nbest_hyps.append(nbest_hyps)

            et = time.perf_counter()
            print('decoding {}/{}, took {:.2f}s, all {:.1f}s, avg {:.2f}s/it'.format(i + 1, len(logits), et - st, et - decode_st,
                                                                         (et - decode_st) / (i + 1)), flush=True)

        return all_best_hyps, all_nbest_hyps

    def beam_search(self, logits, logits_len):
        assert logits.shape[0] == 1
        vocab_size = logits.shape[-1]

        hyps = [Hyp(score=1.0, labels=tuple(), last_label=None)]

        blank_label_id = self.blank_id

        logits = torch.from_numpy(logits)
        if self.beam_temperature != 1.0:
            logits = logits / self.beam_temperature
        prob = logits.softmax(dim=-1)
        prob = prob.cpu().numpy().astype(np.float64)

        for t in range(int(logits_len)):
            prob_t = prob[:, t:t+1, np.newaxis, :]
            hyps_score = np.array([hyp_i.score for hyp_i in hyps])
            hyps_score = hyps_score[:, np.newaxis]
            new_hyp_score = hyps_score * prob_t
            # [B, T, beam_size, V] -> [B, 1, beam_size * V]
            new_hyp_score = np.reshape(new_hyp_score, [new_hyp_score.shape[0], new_hyp_score.shape[1], -1])

            # prob_t_topk_idx = np.argsort(new_hyp_score)
            prob_t_topk_idx = np.argpartition(new_hyp_score, -self.beam_size)
            prob_t_topk_idx = prob_t_topk_idx[0, 0, -self.beam_size:]

            unique_hyps = {}
            for path_i in prob_t_topk_idx:
                hyp_num = path_i // vocab_size
                hyp_i = hyps[hyp_num]
                label_i = path_i % vocab_size

                if label_i == hyp_i.last_label or label_i == blank_label_id:
                    hyp_i_labels = hyp_i.labels
                else:
                    hyp_i_labels = hyp_i.labels + (label_i,)
                new_hyp = Hyp(score=new_hyp_score[0, 0, path_i],
                              labels=hyp_i_labels,
                              last_label=label_i)

                new_hyp_hash = (new_hyp.labels, new_hyp.last_label)
                if new_hyp_hash in unique_hyps:
                    if self.beam_combine_path:
                        unique_hyps[new_hyp_hash].score += new_hyp.score
                    else:
                        if new_hyp.score > unique_hyps[new_hyp_hash].score:
                            unique_hyps[new_hyp_hash] = new_hyp
                else:
                    unique_hyps[new_hyp_hash] = new_hyp

            hyps = list(unique_hyps.values())
            assert len(hyps) <= self.beam_size

        return sorted(hyps, key=lambda x: x.score, reverse=True)


class Hyp:
  __slots__ = ('score', 'labels', 'pred_net_state', 'pred_net_output', 'last_label')

  def __init__(self, score, labels=None, last_label=None):
    self.score = score
    self.labels = labels
    self.last_label = last_label

  def length_norm_score(self, alpha):
    return length_norm(np.log(self.score), len(self.labels), alpha)

  def word_reward_score(self, reward_ratio):
    label_len = len(self.labels)
    # if not self.blank:
    #   label_len -= 1
    return np.log(self.score) + label_len * reward_ratio


def length_norm(log_prob, label_len, alpha):
  len_norm = (label_len + 5.) / 6.
  if alpha != 1:
    len_norm = len_norm ** alpha
  return log_prob / len_norm


def get_length_normed_best(hyps):
  best_hyp = None
  best_hyp_score = None
  score_len_normed = []
  for hyp_i in hyps:
    length_normed_score = np.log(hyp_i.score) / (len(hyp_i.labels) + 1e-16)
    score_len_normed.append(length_normed_score)
    if best_hyp_score is None or length_normed_score > best_hyp_score:
      best_hyp_score = length_normed_score
      best_hyp = hyp_i
  assert best_hyp is not None
  return best_hyp, score_len_normed
