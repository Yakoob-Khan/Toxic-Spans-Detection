import json

from ast import literal_eval
from post_process.character_offsets import mt_dnn_character_offsets
from utils.compute_metrics import system_precision_recall_f1
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def load_tokens(filepath):
  tokens = []
  f = open(filepath, 'r')
  for line in f:
    data = literal_eval(line.strip())
    token_ids = data['token_id']
    token_list = tokenizer.convert_ids_to_tokens(token_ids)
    tokens.append(token_list)
  
  return tokens

def load_model_predictions(filepath):
  with open(filepath) as f:
    output = json.load(f)
    return output['predictions']

def load_data(filepath):
  data = []
  f = open(filepath, 'r')
  for line in f:
    data.append(literal_eval(line.strip()))
  return data

# Load the tokens
dev_tokens = load_tokens('./ensemble_modeling/multi_task_learning/ner_dev.json')
test_tokens = load_tokens('./ensemble_modeling/multi_task_learning/ner_test.json')

# Load the offset mappings
dev_offsets = load_data('./ensemble_modeling/multi_task_learning/ner_dev_offsets.txt')
test_offsets = load_data('./ensemble_modeling/multi_task_learning/ner_test_offsets.txt')

# Load the gold labels
dev_gold_spans = load_data('./ensemble_modeling/multi_task_learning/ner_dev_spans.txt')
test_gold_spans = load_data('./ensemble_modeling/multi_task_learning/ner_test_spans.txt')

# Get the performance metrics for each epoch
for epoch in range(5):
  # Load the predictions
  dev_predictions = load_model_predictions(f'./ensemble_modeling/multi_task_learning/ner_dev_scores_epoch_{epoch}.json')
  test_predictions = load_model_predictions(f'./ensemble_modeling/multi_task_learning/ner_test_scores_epoch_{epoch}.json')

  # Get the predicted toxic spans
  dev_toxic_char_preds = mt_dnn_character_offsets(dev_tokens, dev_predictions, dev_offsets)
  test_toxic_char_preds = mt_dnn_character_offsets(test_tokens, test_predictions, test_offsets)

  # Compute the system performance metrics
  dev_precision, dev_recall, dev_f1 = system_precision_recall_f1(dev_toxic_char_preds, dev_gold_spans)
  test_precision, test_recall, test_f1 = system_precision_recall_f1(test_toxic_char_preds, test_gold_spans)

  print(f'Epoch {epoch+1}:')
  print(f"Dev - Precision: {dev_precision}, Recall: {dev_recall}, F1: {dev_f1}")
  print(f"Test - Precision: {test_precision}, Recall: {test_recall}, F1: {test_f1} \n")
