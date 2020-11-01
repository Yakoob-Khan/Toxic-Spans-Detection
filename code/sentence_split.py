import nltk.data

from load_dataset import load_dataset

tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')

def sentence_split_helper(paragraph, gold):
  gold_set = set(gold)
  # split the paragraph into sentences
  sentence_spans = tokenizer.span_tokenize(paragraph)
  # slices the sentences and its respective label
  sentences, labels = [], []
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
  
  return sentences, labels


def create_sentence_classification_dataset(dataset_path):
  texts, spans = load_dataset(dataset_path)
  examples, gold = [], []
  for text, span in zip(texts, spans):
    sentences, labels = sentence_split_helper(text, span)
    examples.extend(sentences)
    gold.extend(labels)

  return examples, gold


