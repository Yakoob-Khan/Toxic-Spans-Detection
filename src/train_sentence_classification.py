import time
import torch
import numpy as np
import json
import random
import argparse
import ast

from collections import defaultdict
from pre_process.sentence_split import split_into_setences
from pre_process.load_dataset import load_dataset, training_validation_split
from pre_process.tokenize_data import tokenize_sentences
from pre_process.create_tensor_dataset import ToxicSpansDataset
from visualize.plot import plot
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizerFast 
from sklearn.metrics import precision_recall_fscore_support

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

start = time.time()

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

args = parser.parse_args()

# Load the BERT base cased tokenizer and pre-trained model
tokenizer = BertTokenizerFast.from_pretrained(args.model_type)
seq_model = BertForSequenceClassification.from_pretrained(args.model_type, num_labels=2)

# Load the dataset
training_texts, training_spans = load_dataset(args.train_dir)
val_texts, val_spans = load_dataset(args.dev_dir)
test_texts, test_spans = load_dataset(args.test_dir)

# Split each post into sentences
training_sentences, training_labels, training_sentence_spans, tr_post_to_sentence_num = split_into_setences(training_texts, training_spans)
val_sentences, val_labels, val_sentence_spans, val_post_to_sentence_num = split_into_setences(val_texts, val_spans)
test_sentences, test_labels, test_sentence_spans, test_post_to_sentence_num = split_into_setences(test_texts, test_spans)

# Tokenize the sentences
train_sentence_encodings = tokenize_sentences(tokenizer, training_sentences)
val_sentence_encodings = tokenize_sentences(tokenizer, val_sentences)
test_sentence_encodings = tokenize_sentences(tokenizer, test_sentences)

# Create Torch Dataset Objects for train / valid / test sets
seq_train_dataset = ToxicSpansDataset(train_sentence_encodings, training_labels)
seq_val_dataset = ToxicSpansDataset(val_sentence_encodings, val_labels)
seq_test_dataset = ToxicSpansDataset(test_sentence_encodings, test_labels)

print(f"Training sentences: {len(seq_train_dataset)}")
print(f"Validation sentences: {len(seq_val_dataset)}")
print(f"Test sentences: {len(seq_test_dataset)}\n")

# We don't want to pass offset mappings to the model
train_offset_mapping = train_sentence_encodings.pop("offset_mapping") 
val_offset_mapping = val_sentence_encodings.pop("offset_mapping")
test_offset_mapping = test_sentence_encodings.pop("offset_mapping")

# Training Argument Object with hyper-parameter configuration.
seq_training_args = TrainingArguments(
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
  disable_tqdm=True,                            # remove print statements to reduce clutter
)

# Trainer Object
seq_trainer = Trainer(
  model=seq_model,                              # the instantiated 🤗 Transformers model to be trained
  args=seq_training_args,                       # training arguments, defined above
  train_dataset=seq_train_dataset,              # training dataset
  eval_dataset=seq_val_dataset,                 # evaluation dataset
)

print('> Started training toxic sentence binary classifier!\n')
seq_trainer.train()

print('> Making Predictions on test set!\n')
pred = seq_trainer.predict(seq_test_dataset)
test_predictions = pred.predictions.argmax(-1)

# whether to write test sentence predictions to text file for manual inspection
print("\n> Writing Test Set Sentence Binary Classifications to ./ensemble_modeling/test_seq_classifications.txt file \n")
f = open('./ensemble_modeling/test_seq_classifications.txt', "w")
for prediction, val_label, val_sentence in zip(test_predictions, test_labels, test_sentences):
  pred_label = 'Toxic' if prediction == 1 else 'Non-toxic'
  gold_label = 'Toxic' if val_label == 1 else 'Non-toxic'
  f.write(f"Text: {val_sentence} \n")
  f.write(f'Gold: {gold_label}, Pred: {pred_label} \n')
  f.write(f'\n')
f.close()

# Print performance metrics
val_predictions = seq_trainer.predict(seq_val_dataset).predictions.argmax(-1)
val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(val_labels, val_predictions, average='macro')
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_labels, test_predictions, average='macro')
print(f'> Dev Scores: Precision: {val_precision}, Recall: {val_recall}, F1: {val_f1}')
print(f'\n> Test Scores: Precision: {test_precision}, Recall: {test_recall}, F1: {test_f1}')

# Create a dictionary object to store the binary classification dataset used in late fusion.
post_to_sentence_classifications = defaultdict(dict)

# helper function to add each dataset type to binary classification dataset
def add_sequence_classifications(post_type_to_sentence_num, text_type, type_sentence_spans, text_labels, predictions):
  for post_num, sentence_list in post_type_to_sentence_num.items():
    post_content = text_type[post_num]
    post_to_sentence_classifications[post_content] = dict()
    for sen_num in sentence_list:
      span = str(type_sentence_spans[sen_num])
      post_to_sentence_classifications[post_content][span] = dict()
      post_to_sentence_classifications[post_content][span]['Gold'] = int(text_labels[sen_num])
      post_to_sentence_classifications[post_content][span]['Pred'] = int(predictions[sen_num])

# Add the test sentence sequence classifications to dictionary
add_sequence_classifications(tr_post_to_sentence_num, training_texts, training_sentence_spans, training_labels, training_labels)
add_sequence_classifications(val_post_to_sentence_num, val_texts, val_sentence_spans, val_labels, val_predictions)
add_sequence_classifications(test_post_to_sentence_num, test_texts, test_sentence_spans, test_labels, test_predictions)

# Write the dictionary content to a JSON encoded file
print("\n> Writing binary sentence classification datset to ./ensemble_modeling/binary_sentence_classifications.json\n")
with open('./ensemble_modeling/binary_sentence_classifications.json', 'w') as file:
  file.write(json.dumps(post_to_sentence_classifications))

end = time.time()
print(f"Time: {(end-start)/60} mins")
