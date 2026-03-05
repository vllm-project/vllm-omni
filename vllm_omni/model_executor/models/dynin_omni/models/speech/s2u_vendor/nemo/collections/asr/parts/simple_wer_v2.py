# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""The new version script to evalute the word error rate (WER) for ASR tasks.

Tensorflow and Lingvo are not required to run this script.

Example of Usage:

a) `python simple_wer_v2.py file_hypothesis file_reference`
b) `python simple_wer_v2.py file_hypothesis file_reference file_keyphrases`

where `file_hypothesis` is the filename for hypothesis text,
`file_reference` is the filename for reference text, and
`file_keyphrases` is the optional filename for important phrases
(one phrase per line).

Note that the program will also generate a html to diagnose the errors,
and the html filename is `{$file_hypothesis}_diagnois.html`.

Another way is to use this file as a stand-alone library, by calling class
SimpleWER with the following member functions:

- AddHypRef(hyp, ref): Updates the evaluation for each (hyp,ref) pair.
- GetWER(): Computes word error rate (WER) for all the added hyp-ref pairs.
- GetSummaries(): Generates strings to summarize word and key phrase errors.
- GetKeyPhraseStats(): Measures stats for key phrases.
    Stats include:
    (1) Jaccard similarity: https://en.wikipedia.org/wiki/Jaccard_index.
    (2) F1 score: https://en.wikipedia.org/wiki/Precision_and_recall.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys
from six.moves import range


def TxtPreprocess(txt):
  """Preprocess text before WER caculation."""

  # Lowercase, remove \t and new line.
  txt = re.sub(r'[\t\n]', ' ', txt.lower())

  # Remove punctuation before space.
  txt = re.sub(r'[,.\?!]+ ', ' ', txt)

  # Remove punctuation before end.
  txt = re.sub(r'[,.\?!]+$', ' ', txt)

  # Remove punctuation after space.
  txt = re.sub(r' [,.\?!]+', ' ', txt)

  # Remove quotes, [, ], ( and ).
  txt = re.sub(r'["\(\)\[\]]', '', txt)

  # Remove extra space.
  txt = re.sub(' +', ' ', txt.strip())

  return txt


def RemoveCommentTxtPreprocess(txt):
  """Preprocess text and remove comments in the brancket, such as [comments]."""

  # Remove comments surrounded by box brackets:
  txt = re.sub(r'\[\w+\]', '', txt)

  return TxtPreprocess(txt)


def HighlightAlignedHtml(hyp, ref, err_type):
  """Generate a html element to highlight the difference between hyp and ref.

  Args:
    hyp: Hypothesis string.
    ref: Reference string.
    err_type: one of 'none', 'sub', 'del', 'ins'.

  Returns:
    a html string where disagreements are highlighted.
      Note `hyp` is highlighted in green, and marked with <del> </del>
      `ref` is highlighted in yellow. If you want html with nother styles,
      consider to write your own function.

  Raises:
    ValueError: if err_type is not among ['none', 'sub', 'del', 'ins'].
      or if when err_type == 'none', hyp != ref
  """

  highlighted_html = ''
  if err_type == 'none':
    if hyp != ref:
      raise ValueError('hyp (%s) does not match ref (%s) for none error' %
                       (hyp, ref))
    highlighted_html += '%s ' % hyp

  elif err_type == 'sub':
    highlighted_html += """<span style="background-color: greenyellow">
        <del>%s</del></span><span style="background-color: yellow">
        %s </span> """ % (hyp, ref)

  elif err_type == 'del':
    highlighted_html += """<span style="background-color: yellow">
        %s</span> """ % (
            ref)

  elif err_type == 'ins':
    highlighted_html += """<span style="background-color: greenyellow">
        <del>%s</del> </span> """ % (
            hyp)

  else:
    raise ValueError('unknown err_type ' + err_type)

  return highlighted_html


def ComputeEditDistanceMatrix(hyp_words, ref_words):
  """Compute edit distance between two list of strings.

  Args:
    hyp_words: the list of words in the hypothesis sentence
    ref_words: the list of words in the reference sentence

  Returns:
    Edit distance matrix (in the format of list of lists), where the first
    index is the reference and the second index is the hypothesis.
  """
  reference_length_plus = len(ref_words) + 1
  hypothesis_length_plus = len(hyp_words) + 1
  edit_dist_mat = [[]] * reference_length_plus

  # Initialization.
  for i in range(reference_length_plus):
    edit_dist_mat[i] = [0] * hypothesis_length_plus
    for j in range(hypothesis_length_plus):
      if i == 0:
        edit_dist_mat[0][j] = j
      elif j == 0:
        edit_dist_mat[i][0] = i

  # Do dynamic programming.
  for i in range(1, reference_length_plus):
    for j in range(1, hypothesis_length_plus):
      if ref_words[i - 1] == hyp_words[j - 1]:
        edit_dist_mat[i][j] = edit_dist_mat[i - 1][j - 1]
      else:
        tmp0 = edit_dist_mat[i - 1][j - 1] + 1
        tmp1 = edit_dist_mat[i][j - 1] + 1
        tmp2 = edit_dist_mat[i - 1][j] + 1
        edit_dist_mat[i][j] = min(tmp0, tmp1, tmp2)

  return edit_dist_mat


class SimpleWER(object):
  """Compute word error rates after the alignment.

  Attributes:
    key_phrases: list of important phrases.
    aligned_htmls: list of diagnois htmls, each of which corresponding to a pair
      of hypothesis and reference.
    hyp_keyphrase_counts: dict. `hyp_keyphrase_counts[w]` counts how often a key
      phrases `w` appear in the hypotheses.
    ref_keyphrase_counts: dict. `ref_keyphrase_counts[w]` counts how often a key
      phrases `w` appear in the references.
    matched_keyphrase_counts: dict. `matched_keyphrase_counts[w]` counts how
      often a key phrase `w` appear in the aligned transcripts when the
      reference and hyp_keyphrase match.
    wer_info: dict with four keys: 'sub' (substitution error), 'ins' (insersion
      error), 'del' (deletion error), 'nw' (number of words). We can use
      wer_info to compute word error rate (WER) as
      (wer_info['sub']+wer_info['ins']+wer_info['del'])*100.0/wer_info['nw']
  """

  def __init__(self,
               key_phrases=None,
               html_handler=HighlightAlignedHtml,
               preprocess_handler=RemoveCommentTxtPreprocess):
    """Initialize SimpleWER object.

    Args:
      key_phrases:  list of strings as important phrases. If key_phrases is
        None, no key_phrases related metric will be computed.
      html_handler: function to generate a string with html tags.
      preprocess_handler: function to my_preprocess text before computing WER.
    """
    self._preprocess_handler = preprocess_handler
    self._html_handler = html_handler
    self.key_phrases = key_phrases
    self.aligned_htmls = []
    self.wer_info = {'sub': 0, 'ins': 0, 'del': 0, 'nw': 0}
    if key_phrases:
      # Pre-process key_phrase list
      if self._preprocess_handler:
        self.key_phrases = \
            [self._preprocess_handler(k) for k in self.key_phrases]

      # Init keyphrase_counts for every key phrase
      self.ref_keyphrase_counts = {}
      self.hyp_keyphrase_counts = {}
      self.matched_keyphrase_counts = {}
      for k in self.key_phrases:
        self.ref_keyphrase_counts[k] = 0
        self.hyp_keyphrase_counts[k] = 0
        self.matched_keyphrase_counts[k] = 0
    else:
      self.ref_keyphrase_counts = None
      self.hyp_keyphrase_counts = None
      self.matched_keyphrase_counts = None

  def AddHypRef(self, hypothesis, reference):
    """Update WER when adding one pair of strings: (hypothesis, reference).

    Args:
      hypothesis: Hypothesis string.
      reference: Reference string.

    Raises:
      ValueError: when the program fails to parse edit distance matrix.
    """
    if self._preprocess_handler:
      hypothesis = self._preprocess_handler(hypothesis)
      reference = self._preprocess_handler(reference)

    # Compute edit distance.
    hyp_words = hypothesis.split()
    ref_words = reference.split()
    distmat = ComputeEditDistanceMatrix(hyp_words, ref_words)

    # Back trace, to distinguish different erroref_words: ins, del, sub.
    pos_hyp, pos_ref = len(hyp_words), len(ref_words)
    wer_info = {'sub': 0, 'ins': 0, 'del': 0, 'nw': len(ref_words)}
    aligned_html = ''
    matched_ref = ''
    while pos_hyp > 0 or pos_ref > 0:
      err_type = ''

      # Distinguish error type by back tracking
      if pos_ref == 0:
        err_type = 'ins'
      elif pos_hyp == 0:
        err_type = 'del'
      else:
        if hyp_words[pos_hyp - 1] == ref_words[pos_ref - 1]:
          err_type = 'none'  # correct error
        elif distmat[pos_ref][pos_hyp] == distmat[pos_ref - 1][pos_hyp - 1] + 1:
          err_type = 'sub'  # substitute error
        elif distmat[pos_ref][pos_hyp] == distmat[pos_ref - 1][pos_hyp] + 1:
          err_type = 'del'  # deletion error
        elif distmat[pos_ref][pos_hyp] == distmat[pos_ref][pos_hyp - 1] + 1:
          err_type = 'ins'  # insersion error
        else:
          raise ValueError('fail to parse edit distance matrix.')

      # Generate aligned_html
      if self._html_handler:
        if pos_hyp == 0 or not hyp_words:
          tmph = ' '
        else:
          tmph = hyp_words[pos_hyp - 1]
        if pos_ref == 0 or not ref_words:
          tmpr = ' '
        else:
          tmpr = ref_words[pos_ref - 1]
        aligned_html = self._html_handler(tmph, tmpr, err_type) + aligned_html

      # If no error, go to previous ref and hyp.
      if err_type == 'none':
        matched_ref = hyp_words[pos_hyp - 1] + ' ' + matched_ref
        pos_hyp, pos_ref = pos_hyp - 1, pos_ref - 1
        continue

      # Update error.
      wer_info[err_type] += 1

      # Adjust position of ref and hyp.
      if err_type == 'del':
        pos_ref = pos_ref - 1
      elif err_type == 'ins':
        pos_hyp = pos_hyp - 1
      else:  # err_type == 'sub'
        pos_hyp, pos_ref = pos_hyp - 1, pos_ref - 1

    # Verify the computation of edit distance finishes
    assert distmat[-1][-1] == wer_info['ins'] + \
        wer_info['del'] + wer_info['sub']

    # Accumulate err_info before the next (hyp, ref).
    for k in wer_info:
      self.wer_info[k] += wer_info[k]

    # Collect aligned_htmls.
    if self._html_handler:
      self.aligned_htmls += [aligned_html]

    # Update key phrase info.
    if self.key_phrases:
      for w in self.key_phrases:
        self.ref_keyphrase_counts[w] += reference.count(w)
        self.hyp_keyphrase_counts[w] += hypothesis.count(w)
        self.matched_keyphrase_counts[w] += matched_ref.count(w)

  def GetWER(self):
    """Compute Word Error Rate (WER) to summarize word erroref_words.

    Note WER can be larger than 100.0, esp when there are many insertion errors.

    Returns:
      WER as percentage number, usually between 0.0 to 100.0
    """
    nref = self.wer_info['nw']
    nref = max(1, nref)  # non_zero value for division
    total_error = self.wer_info['ins'] \
        + self.wer_info['del'] + self.wer_info['sub']
    return total_error * 100.0 / nref

  def GetKeyPhraseStats(self):
    """Measure the Jaccard similarity of key phrases between hyps and refs.

    Returns:
      jaccard_similarity: jaccard similarity, between 0.0 and 1.0
      F1_keyphrase:  F1 score (=2/(1/prec + 1/recall)), between 0.0 and 1.0
      matched_keyphrases: num of matched key phrases.
      ref_keyphrases:  num of key phrases in the reference strings.
      hyp_keyphrases:  num of key phrases in the hypothesis strings.
    """

    matched_k = sum(self.matched_keyphrase_counts.values())
    ref_k = sum(self.ref_keyphrase_counts.values())
    hyp_k = sum(self.hyp_keyphrase_counts.values())
    joined_k = ref_k + hyp_k - matched_k
    joined_k = max(1, joined_k)  # non_zero value for division
    jaccard_similarity = matched_k * 1.0 / joined_k

    f1_k = 2.0 * matched_k / max(ref_k + hyp_k, 1.0)
    return (jaccard_similarity, f1_k, matched_k, ref_k, hyp_k)

  def GetSummaries(self):
    """Generate strings to summarize word errors and key phrase errors.

    Returns:
      str_sum: string summarizing total error, total word and WER.
      str_details: string breaking down three error types: del, ins, sub.
      str_str_keyphrases_info: string summarizing kerphrase information.
    """
    wer = self.GetWER()
    nref = self.wer_info['nw']
    total_error = self.wer_info['ins'] \
        + self.wer_info['del'] + self.wer_info['sub']
    str_sum = 'WER = %.2f%% (%.4f%%), total error = %d, total word = %d' % (
      wer, wer, total_error, nref)

    str_details = 'Error breakdown: del = %.2f%%, ins=%.2f%%, sub=%.2f%%' % (
        self.wer_info['del'] * 100.0 / nref, self.wer_info['ins'] * 100.0 /
        nref, self.wer_info['sub'] * 100.0 / nref)

    str_keyphrases_info = ''
    if self.key_phrases:
      jaccard_p, f1_p, matched_p, ref_p, hyp_p = self.GetKeyPhraseStats()
      str_keyphrases_info = ('matched %d key phrases (%d in ref, %d in hyp), '
                             'jaccard similarity=%.2f, F1=%.2f') % \
                              (matched_p, ref_p, hyp_p, jaccard_p, f1_p)

    return str_sum, str_details, str_keyphrases_info, (wer, total_error, nref)

  def write_html(self, fn_output):
    try:
      aligned_html = '\n'.join('<li>{}</li>'.format(html) for html in self.aligned_htmls)
      aligned_html = '<ol>{}</ol>'.format(aligned_html)
      with open(fn_output, 'wt') as fp:
        fp.write('<head><meta charset="utf8"></head>')
        fp.write('<body><html>')
        fp.write('<div>%s</div>' % aligned_html)
        fp.write('</body></html>')
    except IOError:
      print('failed to write diagnosis html')


def main(argv):
  hypothesis = open(argv[1], 'r').read()
  reference = open(argv[2], 'r').read()

  if len(argv) == 4:
    phrase_lines = open(argv[3]).readlines()
    keyphrases = [line.strip() for line in phrase_lines]
  else:
    keyphrases = None

  wer_obj = SimpleWER(
      key_phrases=keyphrases,
      html_handler=HighlightAlignedHtml,
      preprocess_handler=RemoveCommentTxtPreprocess)

  wer_obj.AddHypRef(hypothesis, reference)

  str_summary, str_details, str_keyphrases_info = wer_obj.GetSummaries()
  print(str_summary)
  print(str_details)
  print(str_keyphrases_info)

  try:
    fn_output = argv[1] + '_diagnosis.html'
    aligned_html = '<br>'.join(wer_obj.aligned_htmls)
    with open(fn_output, 'wt') as fp:
      fp.write('<body><html>')
      fp.write('<div>%s</div>' % aligned_html)
      fp.write('</body></html>')
  except IOError:
    print('failed to write diagnosis html')


if __name__ == '__main__':
  if len(sys.argv) < 3 or len(sys.argv) > 4:
    print("""
Example of Usage:

  python simple_wer_v2.py file_hypothesis file_reference
or
  python simple_wer_v2.py file_hypothesis file_reference file_keyphrases

  where file_hypothesis is the file name for hypothesis text
        file_reference  is the file name for reference text.
        file_keyphrases (optional) is the filename of key phrases over which
           you want to measure accuracy.

Or you can use this file as a library, and call class SimpleWER
  .AddHypRef(hyp, ref): add one pair of hypothesis/reference. You can call this
    function multiple times.
  .GetWER(): get the Word Error Rate (WER).
  .GetKeyPhraseStats():   get stats for key phrases. The first value is Jaccard
    Similarity of key phrases.
  .GetSummaries(): generate strings to summarize word error and
    key phrase errors.
""")
    sys.exit(1)

  main(sys.argv)
