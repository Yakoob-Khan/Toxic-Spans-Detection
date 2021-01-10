import json

from ast import literal_eval
from post_process.character_offsets import mt_dnn_character_offsets
from utils.compute_metrics import system_precision_recall_f1


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


# Load the predictions
dev_predictions = load_model_predictions('./multitask_result/ner_dev_scores_epoch_2.json')
test_predictions = load_model_predictions('./multitask_result/ner_test_scores_epoch_2.json')

# Load the offset mappings
dev_offsets = load_data('./ner_dev_offsets.txt')
test_offsets = load_data('./ner_test_offsets.txt')

# Load the gold labels
dev_gold_spans = load_data('./ner_dev_offsets.txt')
test_gold_spans = load_data('./ner_test_offsets.txt')

# Get the predicted toxic spans
dev_toxic_char_preds = mt_dnn_character_offsets(dev_predictions, dev_offsets)
test_toxic_char_preds = mt_dnn_character_offsets(test_predictions, test_offsets)

# Compute the system performance metrics
dev_precision, dev_recall, dev_f1 = system_precision_recall_f1(dev_toxic_char_preds, dev_gold_spans)
test_precision, test_recall, test_f1 = system_precision_recall_f1(test_toxic_char_preds, test_gold_spans)

print(f"Dev - Precision: {dev_precision}, Recall: {dev_recall}, F1: {dev_f1}")
print(f"Test - Precision: {test_precision}, Recall: {test_recall}, F1: {test_f1}")
