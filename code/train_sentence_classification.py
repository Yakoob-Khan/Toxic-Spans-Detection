import time
import torch
import numpy as np
import json
import random
from collections import defaultdict

from sentence_split import split_into_setences
from load_dataset import load_dataset, training_validation_split
from tokenize_data import tokenize_sentences
from create_tensor_dataset import ToxicSpansDataset
from plot import plot

from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizerFast 
from sklearn.metrics import precision_recall_fscore_support

# Load the BERT base cased tokenizer and pre-trained model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
seq_model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# def sentence_classifier(training_texts, val_texts, training_spans, val_spans):
start = time.time()

# Load the dataset
texts, spans = load_dataset('../data/tsd_train.csv')
# texts, spans = load_dataset('../data/tsd_trial.csv')

# Split the dataset into training / validation sets
training_texts, val_texts, training_spans, val_spans = training_validation_split(texts, spans, test_size=0.2)

# Split each post into sentences
training_sentences, training_labels, training_sentence_spans, tr_post_to_sentence_num = split_into_setences(training_texts, training_spans)
val_sentences, val_labels, val_sentence_spans, val_post_to_sentence_num = split_into_setences(val_texts, val_spans)

# Tokenize the sentences
train_sentence_encodings = tokenize_sentences(tokenizer, training_sentences)
val_sentence_encodings = tokenize_sentences(tokenizer, val_sentences)

# Create Torch Dataset Objects for train / valid sets
seq_train_dataset = ToxicSpansDataset(train_sentence_encodings, training_labels)
seq_val_dataset = ToxicSpansDataset(val_sentence_encodings, val_labels)

print(f"Training sentences: {len(seq_train_dataset)}")
print(f"Validation sentences: {len(seq_val_dataset)}\n")

# We don't want to pass offset mappings to the model
train_offset_mapping = train_sentence_encodings.pop("offset_mapping") 
val_offset_mapping = val_sentence_encodings.pop("offset_mapping")

seq_metrics = defaultdict(list)
def custom_metrics(pred):
  y_pred = pred.predictions.argmax(-1)
  precision, recall, f1, _ = precision_recall_fscore_support(val_labels, y_pred, average='macro')
  # store the metrics for visualization later
  seq_metrics['precision'].append(precision)
  seq_metrics['recall'].append(recall)
  seq_metrics['f1'].append(f1)

  return {
    'precision': precision,
    'recall': recall,
    'f1': f1
  }

# Training Argument Object with hyper-parameter configuration.
seq_training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=1,              # total number of training epochs
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
    # do_predict=True,               # whether to run predictions on the test set
)

# Trainer Object
seq_trainer = Trainer(
    model=seq_model,                         # the instantiated 🤗 Transformers model to be trained
    args=seq_training_args,                  # training arguments, defined above
    train_dataset=seq_train_dataset,         # training dataset
    eval_dataset=seq_val_dataset,            # evaluation dataset
    # compute_metrics=custom_metrics
)

print('> Started training toxic sentence binary classifier!\n')
seq_trainer.train()

print("\n> Writing Validation Set Sentence Binary Classifications to seq_classifications.txt file \n")
# use trained model to make predictions on validation set

pred = seq_trainer.predict(seq_val_dataset)
# retrieve the predictions
predictions = pred.predictions.argmax(-1)

f = open('seq_classifications.txt', "w")
for prediction, val_label, val_sentence in zip(predictions, val_labels, val_sentences):
  pred_label = 'Toxic' if prediction == 1 else 'Non-toxic'
  gold_label = 'Toxic' if val_label == 1 else 'Non-toxic'
  f.write(f"Text: {val_sentence} \n")
  f.write(f'Gold: {gold_label}, Pred: {pred_label} \n')
  f.write(f'\n')
f.close()


# Create a dictionary object to create the binary classification dataset.
post_to_sentence_classifications = defaultdict(dict)
for post_num, sentence_list in val_post_to_sentence_num.items():
  post_content = val_texts[post_num]
  post_to_sentence_classifications[post_content] = dict()
  for sen_num in sentence_list:
    # sentence = val_sentences[sen_num]
    span = str(val_sentence_spans[sen_num])
    post_to_sentence_classifications[post_content][span] = dict()
    post_to_sentence_classifications[post_content][span]['Gold'] = int(val_labels[sen_num])
    post_to_sentence_classifications[post_content][span]['Pred'] = int(predictions[sen_num])


# use trained model to make predictions on training set as well
pred = seq_trainer.predict(seq_train_dataset)
# retrieve the predictions
predictions = pred.predictions.argmax(-1)

for post_num, sentence_list in tr_post_to_sentence_num.items():
  post_content = training_texts[post_num]
  post_to_sentence_classifications[post_content] = dict()
  for sen_num in sentence_list:
    # sentence = training_sentences[sen_num]
    span = str(training_sentence_spans[sen_num])
    post_to_sentence_classifications[post_content][span] = dict()
    post_to_sentence_classifications[post_content][span]['Gold'] = int(training_labels[sen_num])
    post_to_sentence_classifications[post_content][span]['Pred'] = int(predictions[sen_num])

# Write the dictionary content to a JSON encoded file
with open('binary_sentence_classifications.json', 'w') as file:
  file.write(json.dumps(post_to_sentence_classifications))

# with open("binary_sentence_classifications.json", "r") as read_file:
#   data = json.load(read_file)


# print('\n> Plotting binary classification training mertrics.. \n')
# plot(seq_metrics, 'binary_classification_training.png')

end = time.time()
print(f"Time: {(end-start)/60} mins")

  # return pred.predictions.argmax(-1), val_post_to_sentence_num, val_sentence_spans

