import time
import pandas as pd
import nltk
import nltk.data
import json
import argparse

from nltk.tokenize import WhitespaceTokenizer
from pre_process.load_dataset import load_dataset, training_validation_split
from pre_process.tokenize_data import preserve_labels
from pre_process.sentence_split import split_into_setences
from utils.helper import _contiguous_ranges
from transformers import BertTokenizerFast

nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default='bert-base-cased', help="variant of bert model to use")
parser.add_argument("--train_dir", type=str, default='../data/tsd_train.csv', help="file path to train dataset")
parser.add_argument("--dev_dir", type=str, default='../data/tsd_trial.csv', help="file path to dev dataset")
parser.add_argument("--test_dir", type=str, default='../data/tsd_test.csv', help="file path to test dataset")
args = parser.parse_args()

# Load the tokenizer
bert_tokenizer = BertTokenizerFast.from_pretrained(args.model_type)

# Load the train, dev and test datasets
training_texts, training_spans = load_dataset(args.train_dir)
val_texts, val_spans = load_dataset(args.dev_dir)
test_texts, test_spans = load_dataset(args.test_dir)

# Split the datasets into sentences for classification task
training_sentences, training_labels, training_sentence_spans, tr_post_to_sentence_num = split_into_setences(training_texts, training_spans)
val_sentences, val_labels, val_sentence_spans, val_post_to_sentence_num = split_into_setences(val_texts, val_spans)
test_sentences, test_labels, test_sentence_spans, test_post_to_sentence_num = split_into_setences(test_texts, test_spans)

def write_bert_tokenized_json_ner(filename, texts, spans):
  data = []
  offset_mappings = []
  texts = [text for text in texts]
  spans = [span for span in spans]
  text_encodings = bert_tokenizer(texts, return_offsets_mapping=True, padding=False, truncation=True)
  labels = [preserve_labels(text_encodings[i], span) for i, span in enumerate(spans)]
  for i, label in enumerate(labels):
    # update the CLS and SEP label ids
    label[0], label[-1] = 2, 3
    # retrieve the token ids
    token_id = text_encodings[i].ids
    # retrieve the type ids
    type_id = text_encodings[i].type_ids
    # add tokenized post to data
    data.append({'uid': i, 'label': label, 'token_id': token_id, 'type_id': type_id})
    # save the offsets mapping for computing scores later
    offset_mappings.append(text_encodings[i].offsets)
    
  # Write the JSON dataset to ./ensemble_modeling directory 
  with open(f'./ensemble_modeling/multi_task_learning/{filename}.json', 'w') as json_file:
    for line in data:
      json.dump(line, json_file)
      json_file.write('\n')
  
  # Write the JSON dataset to mt-dnn canonical_data directory
  with open(f'../mt-dnn/canonical_data/{filename}.json', 'w') as json_file:
    for line in data:
      json.dump(line, json_file)
      json_file.write('\n')
  
  # Write the token offset mappings
  with open(f'./ensemble_modeling/multi_task_learning/{filename}_offsets.txt', 'w') as json_file:
    for line in offset_mappings:
      json.dump(line, json_file)
      json_file.write('\n')

  # Write the gold span labels
  with open(f'./ensemble_modeling/multi_task_learning/{filename}_spans.txt', 'w') as json_file:
    for span in spans:
      json.dump(span, json_file)
      json_file.write('\n')


def write_bert_tokenized_json_classification(filename, sentences, labels):
  data = []
  sentence_encodings = bert_tokenizer(sentences, return_offsets_mapping=False, padding=False, truncation=True)
  for i, label in enumerate(labels):
    token_id = sentence_encodings[i].ids
    type_id = sentence_encodings[i].type_ids
    data.append({'uid': str(i), 'label': label, 'token_id': token_id, 'type_id': type_id})
  
  # Write the JSON dataset to ./ensemble_modeling directory 
  with open(f'./ensemble_modeling/multi_task_learning/{filename}.json', 'w') as json_file:
    for line in data:
      json.dump(line, json_file)
      json_file.write('\n')

  # Write the JSON dataset to mt-dnn canonical data directory
  with open(f'../mt-dnn/canonical_data/{filename}.json', 'w') as json_file:
    for line in data:
      json.dump(line, json_file)
      json_file.write('\n')

# Create train, dev and test JSON datasets for NER task
print(f'> Writing data for MT-DNN NER task\n')
write_bert_tokenized_json_ner(f'ner_train', training_texts, training_spans)
write_bert_tokenized_json_ner(f'ner_dev', val_texts, val_spans)
write_bert_tokenized_json_ner(f'ner_test', test_texts, test_spans)

# Create train, dev and test JSON for classification task
print(f'> Writing data for MT-DNN classification task\n')
write_bert_tokenized_json_classification(f'sentenceclassification_train', training_sentences, training_labels)
write_bert_tokenized_json_classification(f'sentenceclassification_dev', val_sentences, val_labels)
write_bert_tokenized_json_classification(f'sentenceclassification_test', test_sentences, test_labels)
