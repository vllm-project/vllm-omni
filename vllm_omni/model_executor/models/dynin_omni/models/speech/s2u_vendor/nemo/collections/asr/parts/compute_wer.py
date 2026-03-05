import sys
from collections import OrderedDict

import pandas as pd

from nemo.collections.asr.metrics.wer_bpe import add_delim_space
from nemo.collections.asr.parts.simple_wer_v2 import SimpleWER


def analyze(texts, output_html_path=None, add_zh_space=False):
  # def remove_sep(inp_text):
  #   if SEP in inp_text:
  #     inp_text = ' '.join(sp for sp in inp_text.split() if sp != SEP)
  #   return inp_text
  wer_obj = SimpleWER(
    preprocess_handler=None)

  total_empty_lev = 0
  for fname, true_text, pred_text in texts:
    if add_zh_space:
      true_text = add_delim_space(true_text)
      pred_text = add_delim_space(pred_text)
    # if ignore_sep:
    #   assert SEP not in pred_text
    wer_obj.AddHypRef(pred_text, true_text)

    if not pred_text:
      print('found empty pred: {}, {}'.format(fname, true_text))
      total_empty_lev += len(true_text.split())

  str_summary, str_details, _, (wer, total_error, nref) = wer_obj.GetSummaries()

  print('empty token num: {}, empty error ratio: {}'.format(total_empty_lev, total_empty_lev/total_error if total_error > 0 else 0))

  if output_html_path:
    wer_obj.write_html(output_html_path)
  return (str_summary, str_details), (wer, total_error, nref)


def calc_wer(true_fp, pred_fp):
  true_df = pd.read_csv(true_fp, index_col='wav_filename', usecols=['wav_filename', 'transcript'], encoding='utf-8')

  true_dic = true_df['transcript'].to_dict()

  pred_dic = pd.read_csv(pred_fp, index_col='wav_filename', encoding='utf-8', keep_default_na=False)['predicted_transcript'].to_dict(into=OrderedDict)

  assert true_dic.keys() == pred_dic.keys()

  texts = []
  for fname, pred_text in pred_dic.items():
    texts.append([fname, true_dic[fname], pred_text])

  (str_summary, str_details), (total_wer, total_word_lev, total_word_count) = analyze(texts,
                                                                                        output_html_path=pred_fp + '_diagnosis.html')
  print(str_details)
  print(str_summary)
  return total_wer, (total_word_lev, total_word_count)


if __name__ == '__main__':
  true_fp, pred_fp = sys.argv[1:]

  calc_wer(true_fp=true_fp, pred_fp=pred_fp)