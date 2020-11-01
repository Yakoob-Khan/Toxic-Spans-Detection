import time
import torch
import numpy as np
import json

from load_dataset import load_dataset, training_validation_split
from tokenize_data import tokenize_data
from create_tensor_dataset import ToxicSpansDataset
from compute_metrics import compute_metrics
from character_offsets import character_offsets
from plot import plot
from write_predictions import write_toxic_strings
from collections import defaultdict

from transformers import BertForTokenClassification, Trainer, TrainingArguments, BertTokenizerFast 

start = time.time()

# Load the BERT base cased tokenizer and pre-trained model

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=2)

# Load the dataset
texts, spans = load_dataset('../data/tsd_train.csv')
# texts, spans = load_dataset('../data/tsd_trial.csv')

# Split the dataset into training / validation sets
training_texts, val_texts, training_spans, val_spans = training_validation_split(texts, spans, test_size=0.2)

print('> Tokenizing text and generating word embeddings.. \n')
train_text_encodings, train_labels_encodings = tokenize_data(tokenizer, training_texts, training_spans)
val_text_encodings, val_labels_encodings = tokenize_data(tokenizer, val_texts, val_spans)

# Create Torch Dataset Objects for train / valid sets
print('> Create Tensor Datasets.. \n')
train_dataset = ToxicSpansDataset(train_text_encodings, train_labels_encodings)
val_dataset = ToxicSpansDataset(val_text_encodings, val_labels_encodings)

print(f"Training examples: {len(train_dataset)}")
print(f"Validation examples: {len(val_dataset)}\n")

# We don't want to pass offset mappings to the model
train_offset_mapping = train_text_encodings.pop("offset_mapping") 
val_offset_mapping = val_text_encodings.pop("offset_mapping")

metrics = defaultdict(list)
# custom metric wrapper function
def custom_metrics(pred):
  # comute the precision, recall and f1 of the system at evaluation step
  ret = compute_metrics(pred, val_spans, val_offset_mapping, val_text_encodings)

  # store the metrics for visualization later
  metrics['precision'].append(ret['precision'])
  metrics['recall'].append(ret['recall'])
  metrics['f1'].append(ret['f1'])
  
  return ret
  

# Training Argument Object with hyper-parameter configuration.
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=2,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,                # log after every x steps
    do_eval=True,                    # whether to run evaluation on the val set
    evaluation_strategy="steps",     # evaluation is done (and logged) every logging_steps 
    learning_rate=5e-5,              # 5e-5 is default learning rate
    disable_tqdm=True,               # remove print statements to reduce clutter
    # do_predict=True,               # whether to run predictions on the test set
)

# Trainer Object
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,            # evaluation dataset
    compute_metrics=custom_metrics
)

print('> Started training..\n')
trainer.train()

print("\n> ---- Making Predictions ---- \n")
# use trained model to make predictions
pred = trainer.predict(val_dataset)

# Print final metrics
ret = compute_metrics(pred, val_spans, val_offset_mapping, val_text_encodings)

print(f'System performance: {ret}')

# retrieve the predictions
predictions = pred.predictions.argmax(-1)
# get the predicted character offsets
toxic_char_preds = character_offsets(val_text_encodings, val_offset_mapping, predictions)
# write the predicted and ground truth toxic strings to examine model outputs
write_toxic_strings('predictions.txt', val_texts, val_spans, toxic_char_preds)

print('\n> Plotting results.. \n')
plot(metrics)

# Write the metric values to a text file
with open('metrics.txt', 'w') as file:
  file.write(json.dumps(metrics))

end = time.time()
print(f"Time: {(end-start)/60} mins")

