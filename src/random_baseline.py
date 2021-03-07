import pandas as pd
import random
import numpy as np

from pre_process.load_dataset import load_dataset
from utils.compute_metrics import system_precision_recall_f1

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)

# Build a random baseline (yields offsets at random)
# Credits: https://github.com/ipavlopoulos/toxic_spans/blob/master/ToxicSpans_SemEval21.ipynb
random_baseline = lambda text: [i for i, char in enumerate(text) if random.random() > 0.5]

# Load the train, dev and test sets
train_texts, train_spans = load_dataset('../data/tsd_train.csv')
val_texts, val_spans = load_dataset('../data/tsd_trial.csv')
test_texts, test_spans = load_dataset('../data/tsd_test.csv')

# Make random predictions 
train_preds = [random_baseline(text) for text in train_texts]
val_preds = [random_baseline(text) for text in val_texts]
test_preds = [random_baseline(text) for text in test_texts]

# Compute performance metrics
train_scores = system_precision_recall_f1(train_preds, train_spans)
dev_scores = system_precision_recall_f1(val_preds, val_spans)
test_scores = system_precision_recall_f1(test_preds, test_spans)

# Print the results
print(f'\n> Train Scores: Precision: {train_scores[0]}, Recall: {train_scores[1]}, F1: {train_scores[2]}')
print(f'\n> Dev Scores: Precision: {dev_scores[0]}, Recall: {dev_scores[1]}, F1: {dev_scores[2]}')
print(f'\n> Test Scores: Precision: {test_scores[0]}, Recall: {test_scores[1]}, F1: {test_scores[2]}')
