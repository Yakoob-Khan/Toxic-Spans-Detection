import time
import torch
import numpy as np
import json
import random
import time

from collections import defaultdict
from pre_process.load_dataset import load_dataset, training_validation_split
from pre_process.tokenize_data import tokenize_data
from pre_process.create_tensor_dataset import ToxicSpansDataset
from post_process.character_offsets import character_offsets
from visualize.plot import plot
from utils.compute_metrics import compute_metrics
from utils.write_predictions import write_toxic_strings
# from train_sentence_classification import sentence_classifier
from utils.helper import fix_spans

from transformers import BertForTokenClassification, Trainer, TrainingArguments, BertTokenizerFast 

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

start = time.time()

# Load the BERT base cased tokenizer and pre-trained model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=2)

# Load the datasets
texts, spans = load_dataset('../data/tsd_train.csv')
test_texts, test_spans = load_dataset('../data/tsd_trial.csv')

test_texts = [test_text for test_text in test_texts]
test_spans = [fix_spans(test_span, test_texts[i]) for i, test_span in enumerate(test_spans)]

# Split the dataset into training / validation sets
training_texts, val_texts, training_spans, val_spans = training_validation_split(texts, spans, test_size=0.1)

# Load the pre-trained toxic sentence classifications first
# with open("binary_sentence_classifications.json", "r") as read_file:
#   post_to_sentences_classification = json.load(read_file)
post_to_sentences_classification = dict()

# val_sentence_pred, val_post_to_sentence_num, val_sentence_spans = sentence_classifier(training_texts, val_texts, training_spans, val_spans)
# val_sentence_pred, val_post_to_sentence_num, val_sentence_spans = [], [], []
val_sentences_info = {
  'post_to_sentences_classification' : post_to_sentences_classification,
  'val_texts': val_texts,
  # 'val_sentence_pred': val_sentence_pred,
  # 'val_post_to_sentence_num': val_post_to_sentence_num,
  # 'val_sentence_spans': val_sentence_spans
}

print('\n> Tokenizing text and generating word embeddings.. \n')
train_text_encodings, train_labels_encodings = tokenize_data(tokenizer, training_texts, training_spans)
val_text_encodings, val_labels_encodings = tokenize_data(tokenizer, val_texts, val_spans)
test_text_encodings, test_labels_encodings = tokenize_data(tokenizer, test_texts, test_spans)

# Create Torch Dataset Objects for train / valid sets
print('> Creating Tensor Datasets.. \n')
train_dataset = ToxicSpansDataset(train_text_encodings, train_labels_encodings)
val_dataset = ToxicSpansDataset(val_text_encodings, val_labels_encodings)
test_dataset = ToxicSpansDataset(test_text_encodings, test_labels_encodings)

print(f"Training examples: {len(train_dataset)}")
print(f"Validation examples: {len(val_dataset)}\n")
print(f"Test examples: {len(test_dataset)}\n")

# We don't want to pass offset mappings to the model
train_offset_mapping = train_text_encodings.pop("offset_mapping") 
val_offset_mapping = val_text_encodings.pop("offset_mapping")
test_offset_mapping = test_text_encodings.pop("offset_mapping")

metrics = defaultdict(list)
# custom metric wrapper function
def custom_metrics(pred):
  # comute the precision, recall and f1 of the system at evaluation step
  ret = compute_metrics(pred, val_spans, val_offset_mapping, val_text_encodings, val_sentences_info)
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
    per_device_eval_batch_size=32,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,                # log after every x steps
    do_eval=True,                    # whether to run evaluation on the val set
    evaluation_strategy="steps",     # evaluation is done (and logged) every logging_steps 
    learning_rate=5e-5,              # 5e-5 is default learning rate
    disable_tqdm=True,               # remove print statements to reduce clutter
    # do_predict=True,                 # whether to run predictions on the test set
)

# Trainer Object
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,            # evaluation dataset
    compute_metrics=custom_metrics
)

print('> Started Toxic Spans Detection training! \n')
trainer.train()

print("\n> Making Predictions on Test Set\n")
# use trained model to make predictions on test dataset
# pred = trainer.predict(val_dataset)
pred = trainer.predict(test_dataset)

# Print final metrics
# ret = compute_metrics(pred, val_spans, val_offset_mapping, val_text_encodings, val_sentences_info)
ret = compute_metrics(pred, test_spans, test_offset_mapping, test_text_encodings, val_sentences_info)

print(f'System performance: {ret}')

# retrieve the predictions
predictions = pred.predictions.argmax(-1)

# get the predicted character offsets
# toxic_char_preds = character_offsets(val_text_encodings, val_offset_mapping, predictions, val_sentences_info)
toxic_char_preds = character_offsets(test_text_encodings, test_offset_mapping, predictions, val_sentences_info)

# write the predicted and ground truth toxic strings to examine model outputs
# write_toxic_strings('predictions.txt', val_texts, val_spans, toxic_char_preds)
write_toxic_strings('predictions.txt', test_texts, test_spans, toxic_char_preds)

print('\n> Plotting Toxic Spans Detection training metrics. \n')
plot(metrics, 'toxic_spans_training.png')

# # Write the metric values to a text file
# with open('metrics.txt', 'w') as file:
#   file.write(json.dumps(metrics))

end = time.time()
print(f"Time: {(end-start)/60} mins")

