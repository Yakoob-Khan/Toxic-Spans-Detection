import json
import pandas as pd 
import nltk
import ast
import argparse

from nltk.corpus import stopwords
from utils.helper import _contiguous_ranges
from collections import defaultdict
from pre_process.load_dataset import load_dataset

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", type=str, default='../data/tsd_train.csv', help="file path to train dataset")
parser.add_argument("--hateXplain", type=str, default='../data/hateXplain.json', help="File path for training dataset")
parser.add_argument("--output", type=str, default='./augmented_data/tsd_hateXplain.csv', help="File path for augmented dataset")

args = parser.parse_args()

training_texts, training_spans = load_dataset(args.train_dir)

# Create spans, text data format
with open(args.hateXplain) as json_file:
  data = json.load(json_file)
  toxic_spans_data = {'spans': [span for span in training_spans], 'text': [text for text in training_texts]}

  for num, post_key in enumerate(data):
    # extract the post
    post = data[post_key]
    # get the post tokens
    post_tokens = post['post_tokens']
    # get the three annotations if available and pick one at random
    labels = post['rationales']

    if len(labels) == 0:
      # neutral post
      continue
      # text = " ".join(post_tokens)
      # toxic_spans_data['spans'].append([])
      # toxic_spans_data['text'].append(text.strip())
    
    else:
      # combine the multiple annotations to get a unified gold annotation
      combined_labels = []
      rows, cols = len(labels), len(labels[0])
      for c in range(cols):
        label_freq = defaultdict(int)
        for r in range(rows):
          label_freq[labels[r][c]] += 1
      
        label = max(label_freq, key=lambda l: label_freq[l])
        combined_labels.append(label)

      # create the span level annotations
      text, span = '', []
      for token, label in zip(post_tokens, combined_labels):
        start = len(text) 
        text += token 

        if label == 1 and token.lower() not in stop_words:
          # include space if toxic phrase
          if len(span) > 0 and span[-1] == start - 2:
            start = start - 1
          # add toxic span
          span.extend([k for k in range(start, len(text))])
        
        text += " "

      # add the toxic post and its span
      toxic_spans_data['spans'].append(span)
      toxic_spans_data['text'].append(text.strip())


  df = pd.DataFrame(toxic_spans_data)
  df.to_csv(args.output, index=False)

  print(f'> Written {df.shape[0]} posts to {args.output} file!')

