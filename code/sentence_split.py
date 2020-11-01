import nltk
import nltk.data

from collections import defaultdict

nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')

def sentence_split_helper(paragraph, gold):
  gold_set = set(gold)
  # split the paragraph into sentences
  sentence_spans = tokenizer.span_tokenize(paragraph)
  # slices the sentences and its respective label
  sentences, labels, spans = [], [], []
  for i, j in sentence_spans:
    # get the sentence
    sentences.append(paragraph[i:j])
    # check whether this sentence is toxic (1) or not (0)
    label = 0
    for k in range(i, j):
      if k in gold_set:
        # sentence contains toxic word / phrase according to ground truth
        label = 1
        break
    
    labels.append(label)
    spans.append((i, j))
  
  return sentences, labels, spans


def split_into_setences(texts, spans):
  examples, gold, sentence_spans, post_to_sentence_num = [], [], [], defaultdict(list)
  for i, (text, span) in enumerate(zip(texts, spans)):
    sentences, labels, sentence_span = sentence_split_helper(text, span)
    start = len(examples)
    examples.extend(sentences)
    end = len(examples) - 1
    post_to_sentence_num[i] = [k for k in range(start, end + 1)]

    gold.extend(labels)
    sentence_spans.extend(sentence_span)

  return examples, gold, sentence_spans, post_to_sentence_num


