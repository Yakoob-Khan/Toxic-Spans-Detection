import time
import torch
import numpy as np

from load_dataset import load_dataset
from tokenize_data import tokenize_data
from generate_embeddings import generate_embeddings
from create_tensor_dataset import ToxicSpansDataset
from compute_metrics import compute_metrics, f1_system_score
from character_offsets import character_offsets

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from transformers import BertForTokenClassification, Trainer, TrainingArguments, BertTokenizerFast

start = time.time()

# Load the dataset
texts, spans = load_dataset('../data/tsd_trial.csv')

# Split the dataset into training / validation sets using sklearn function
training_texts, val_texts, training_spans, val_spans = train_test_split(texts, spans, test_size=0.2)

# Tokenize the text into tokens and get the corresponding labels
tokenized_training_texts, tokenized_training_labels = tokenize_data(training_texts, training_spans)
tokenized_val_texts, tokenized_val_labels = tokenize_data(val_texts, val_spans)

# Generate the word embeddings 
train_text_encodings, train_labels_encodings = generate_embeddings(tokenized_training_texts, tokenized_training_labels)
val_text_encodings, val_labels_encodings = generate_embeddings(tokenized_val_texts, tokenized_val_labels)

# Create Torch Dataset Objects for train / valid sets
train_dataset = ToxicSpansDataset(train_text_encodings, train_labels_encodings)
val_dataset = ToxicSpansDataset(val_text_encodings, val_labels_encodings)

print(f"Training examples: {len(train_dataset)}")
print(f"Validation examples: {len(val_dataset)}")

# We don't want to pass offset mappings to the model
train_offset_mapping = train_text_encodings.pop("offset_mapping") 
val_offset_mapping = val_text_encodings.pop("offset_mapping")

# Load the BERT base cased tokenizer and pre-trained model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=2)

# Training Argument Object with hyper-parameter configuration.
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=5,              # total number of training epochs
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
    compute_metrics=compute_metrics
)

print('> Started training..\n')
trainer.train()
print('\n -- Finished training! \n')

print("> ---- Computing Predictions ---- \n")
# use trained model to make predictions
output = trainer.predict(val_dataset)

# get the gold labels and model predictions
labels, predictions = output.label_ids, output.predictions.argmax(-1)

# get the sub-tokens created using BERT tokenizer 
input_ids = [row['input_ids'] for row in val_dataset]
tokenized_docs = [tokenizer.convert_ids_to_tokens(input_id) for input_id in input_ids]

# retrieve the gold toxic character offsets
gold_char_offsets = [span for span in val_spans]

# compute the toxic character offsets predicted by the model
toxic_char_preds = character_offsets(tokenizer, input_ids, val_offset_mapping, predictions)

# Compute the System F1 score using the method described by the task
f1_score = f1_system_score(toxic_char_preds, gold_char_offsets)

print(f'System F1 Score: {f1_score}')

end = time.time()
print(f"Time: {end-start}s")