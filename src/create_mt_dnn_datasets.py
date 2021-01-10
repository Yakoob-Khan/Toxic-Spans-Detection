import time
import pandas as pd
import nltk
import nltk.data
import json
from nltk.tokenize import WhitespaceTokenizer

from pre_process.load_dataset import load_dataset, training_validation_split
from pre_process.tokenize_data import preserve_labels
from pre_process.sentence_split import split_into_setences
from utils.helper import _contiguous_ranges
from transformers import BertTokenizerFast

nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')
bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

# Load the train and test datasets
texts, spans = load_dataset('../data/tsd_train.csv')
test_texts, test_spans = load_dataset('../data/tsd_trial.csv')

# Split the train dataset into training / validation sets
training_texts, val_texts, training_spans, val_spans = training_validation_split(texts, spans, test_size=0.2)

# Split the train and validation sets into sentences for classification task
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
    
  # Write the JSON dataset
  with open(f'{filename}.json', 'w') as json_file:
    for line in data:
      json.dump(line, json_file)
      json_file.write('\n')
  
  # Write the token offset mappings
  with open(f'{filename}_offsets.txt', 'w') as json_file:
    for line in offset_mappings:
      json.dump(line, json_file)
      json_file.write('\n')

  # Write the gold span labels
  with open(f'{filename}_spans.txt', 'w') as json_file:
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
  
  # Write the JSON dataset
  with open(f'{filename}.json', 'w') as json_file:
    for line in data:
      json.dump(line, json_file)
      json_file.write('\n')


# Create train, dev and test JSON datasets for NER task
write_bert_tokenized_json_ner('./bert_base_cased_lower/ner_train', training_texts, training_spans)
write_bert_tokenized_json_ner('./bert_base_cased_lower/ner_dev', val_texts, val_spans)
write_bert_tokenized_json_ner('./bert_base_cased_lower/ner_test', test_texts, test_spans)

# Create train, dev and test JSON for classification task
write_bert_tokenized_json_classification('./bert_base_cased_lower/sentenceclassification_train', training_sentences, training_labels)
write_bert_tokenized_json_classification('./bert_base_cased_lower/sentenceclassification_dev', val_sentences, val_labels)
write_bert_tokenized_json_classification('./bert_base_cased_lower/sentenceclassification_test', test_sentences, test_labels)



