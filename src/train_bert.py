import time
import torch
import numpy as np
import pandas as pd 
import json
import random
import time
import ast
import argparse

from collections import defaultdict
from pre_process.load_dataset import load_dataset, training_validation_split, load_testset
from pre_process.tokenize_data import tokenize_data, tokenize_testset
from pre_process.create_tensor_dataset import ToxicSpansDataset
from post_process.character_offsets import character_offsets_with_thresholding
from visualize.plot import plot
from utils.compute_metrics import compute_metrics
from utils.write_predictions import write_toxic_strings, create_submission_file, write_test_strings, write_toxic_strings_with_prob
from utils.write_bert_confidence_scores import write_bert_confidence_scores
from utils.helper import _contiguous_ranges
from utils.helper import fix_spans
from visualize.confusion_matrix import create_confusion_matrix
from transformers import BertForTokenClassification, Trainer, TrainingArguments, BertTokenizerFast 

start = time.time()

# set seed for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default='bert-base-cased', help="variant of bert model to use")
parser.add_argument("--train_dir", type=str, default='../data/tsd_train.csv', help="file path to train dataset")
parser.add_argument("--dev_dir", type=str, default='../data/tsd_trial.csv', help="file path to dev dataset")
parser.add_argument("--test_dir", type=str, default='../data/tsd_test.csv', help="file path to test dataset")
parser.add_argument("--epochs", type=float, default=2, help="number of epochs to fine-tune")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--warm_up_steps", type=int, default=500, help="number of steps for linear warm up")
parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
parser.add_argument("--logging_dir", type=str, default='./logs', help="file path to tensorflow logging directory")
parser.add_argument("--logging_steps", type=int, default=5, help="logging steps")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="learning rate")
parser.add_argument("--output_dir", type=str, default='./checkpoints', help="directory where check points are saved during training")
parser.add_argument("--threshold", type=int, default=-float('inf'), help="threshold value for predicting toxic tokens during post-processing")
parser.add_argument("--plot_loss", type=str, default='False', help="plot training and dev loss")
parser.add_argument("--plot_confusion_matrix", type=str, default='False', help="plot confusion matrix")
parser.add_argument("--train_dev_split", type=str, default='False', help="split train set into train/dev")
parser.add_argument("--dev_size", type=float, default=0.1, help="dev set size")
parser.add_argument("--confidence_scores", type=str, default='False', help="write confidence scores of test set to text file for plotting precision-recall curve")
parser.add_argument("--experiment_name", type=str, default='BERT', help="name to help identify among various experiments")


args = parser.parse_args()

# Load the BERT base cased tokenizer and pre-trained model
tokenizer = BertTokenizerFast.from_pretrained(args.model_type)
model = BertForTokenClassification.from_pretrained(args.model_type, num_labels=2)

# Load the train, val and test csv files
training_texts, training_spans = load_dataset(args.train_dir)
val_texts, val_spans = load_dataset(args.dev_dir)
test_texts, test_spans = load_dataset(args.test_dir)

# Split the dataset into training / validation sets
if ast.literal_eval(args.train_dev_split):
  training_texts, val_texts, training_spans, val_spans = training_validation_split(texts, spans, test_size=args.dev_size)

val_sentences_info = {}

print('\n> Tokenizing text and generating word embeddings.. \n')
train_text_encodings, train_labels_encodings = tokenize_data(tokenizer, training_texts, training_spans)
val_text_encodings, val_labels_encodings = tokenize_data(tokenizer, val_texts, val_spans)
test_text_encodings, test_labels_encodings = tokenize_data(tokenizer, test_texts, test_spans)

# Create Torch Dataset Objects for train / valid sets
print('> Creating Tensor Datasets.. \n')
train_dataset = ToxicSpansDataset(train_text_encodings, train_labels_encodings)
val_dataset = ToxicSpansDataset(val_text_encodings, val_labels_encodings)
test_dataset = ToxicSpansDataset(test_text_encodings, test_labels_encodings)

print(f"> Training examples: {len(train_dataset)}")
print(f"> Validation examples: {len(val_dataset)}")
print(f"> Test examples: {len(test_dataset)}\n")

# We don't want to pass offset mappings to the model
train_offset_mapping = train_text_encodings.pop("offset_mapping") 
val_offset_mapping = val_text_encodings.pop("offset_mapping")
test_offset_mapping = test_text_encodings.pop("offset_mapping")

metrics = defaultdict(list)
# custom metric wrapper function
def custom_metrics(pred):
  # compute the precision, recall and f1 of the system at evaluation step
  ret = compute_metrics(pred, val_spans, val_offset_mapping, val_text_encodings, val_sentences_info, threshold=args.threshold)
  # store the metrics for visualization later
  metrics['precision'].append(ret['precision'])
  metrics['recall'].append(ret['recall'])
  metrics['f1'].append(ret['f1'])
 
  return ret

# Training Argument Object with hyper-parameter configuration.
training_args = TrainingArguments(
  output_dir=args.output_dir,                   # output directory
  num_train_epochs=args.epochs,                 # total number of training epochs
  per_device_train_batch_size=args.batch_size,  # batch size per device during training
  per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
  warmup_steps=args.warm_up_steps,              # number of warmup steps for learning rate scheduler
  weight_decay=args.weight_decay,               # strength of weight decay
  logging_dir=args.logging_dir,                 # directory for storing logs
  logging_steps=args.logging_steps,             # log after every x steps
  do_eval=True,                                 # whether to run evaluation on the val set
  evaluation_strategy="steps",                  # evaluation is done (and logged) every logging_steps 
  learning_rate=args.learning_rate,             # 5e-5 is default learning rate
  disable_tqdm=True,                            # remove tqdm statements to reduce clutter
)

# Trainer Object
trainer = Trainer(
  model=model,                 # the instantiated 🤗 Transformers model to be trained
  args=training_args,          # training arguments, defined above
  train_dataset=train_dataset,       
  eval_dataset=val_dataset,         
  compute_metrics=custom_metrics
)

print('> Started Toxic Spans Detection training! \n')
trainer.train()

# use trained model to make predictions on dev and test datasets
val_pred = trainer.predict(val_dataset)
test_pred = trainer.predict(test_dataset)

print("> Making Predictions on Test Set")

# Print final metrics
val_results = compute_metrics(val_pred, val_spans, val_offset_mapping, val_text_encodings, val_sentences_info, threshold=args.threshold)
test_results = compute_metrics(test_pred, test_spans, test_offset_mapping, test_text_encodings, val_sentences_info, threshold=args.threshold)
print(f'\n> Dev Scores: {val_results}')
print(f'\n> Test Scores: {test_results}')

# retrieve the predictions
val_predictions = val_pred.predictions.argmax(-1)
test_predictions = test_pred.predictions.argmax(-1)
val_prediction_scores = val_pred.predictions
test_prediction_scores = test_pred.predictions

# get the predicted character offsets with their confidence scores, applying thresholding
val_toxic_char_preds = character_offsets_with_thresholding(val_text_encodings, val_offset_mapping, val_predictions, val_sentences_info, val_prediction_scores, threshold=args.threshold)
test_toxic_char_preds = character_offsets_with_thresholding(test_text_encodings, test_offset_mapping, test_predictions, val_sentences_info,  test_prediction_scores, threshold=args.threshold)

# Write model predictions with confidence scores
write_toxic_strings_with_prob('./output/val_predictions_scores.txt', val_texts, val_toxic_char_preds, val_spans)
write_toxic_strings_with_prob('./output/test_predictions_scores.txt', test_texts, test_toxic_char_preds, test_spans)

# Create the submission file
print("\n> Creating Submission File\n")
toxic_char_offsets = [span[0] for span in test_toxic_char_preds]
create_submission_file("./output/spans-pred.txt", toxic_char_offsets)

if ast.literal_eval(args.plot_loss):
  print('\n> Plotting Toxic Spans Detection training metrics. \n')
  plot(metrics, f'output/toxic_spans_training.pdf')

if ast.literal_eval(args.plot_confusion_matrix):
  print('\n> Plotting Confusion Matrix. \n')
  create_confusion_matrix(test_text_encodings, test_predictions, test_labels_encodings)

if ast.literal_eval(args.confidence_scores):
  print('\n> Writing Confidence Scores for test set \n')
  write_bert_confidence_scores(test_labels_encodings, test_prediction_scores, f"output/scores_{args.experiment_name}.txt")
  
# Write the metric values to a text file
with open('metrics.txt', 'w') as file:
  file.write(json.dumps(metrics))

end = time.time()
print(f"Time: {(end-start)/60} mins")
