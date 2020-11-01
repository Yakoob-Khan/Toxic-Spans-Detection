import time
from collections import defaultdict

from sentence_split import create_sentence_classification_dataset
from load_dataset import training_validation_split
from tokenize_data import tokenize_sentences
from create_tensor_dataset import ToxicSpansDataset
from plot import plot

from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizerFast 
from sklearn.metrics import precision_recall_fscore_support

start = time.time()

# Load the BERT base cased tokenizer and pre-trained model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

# Load and convert the dataset into a binary classification task
sentences, labels = create_sentence_classification_dataset('../data/tsd_train.csv')

# Split the dataset into training / validation sets
training_sentences, val_sentences, training_labels, val_labels = training_validation_split(sentences, labels, test_size=0.2)

print('> Tokenizing text and generating word embeddings.. \n')
train_sentence_encodings = tokenize_sentences(tokenizer, training_sentences)
val_sentence_encodings = tokenize_sentences(tokenizer, val_sentences)

# Create Torch Dataset Objects for train / valid sets
print('> Create Tensor Datasets.. \n')
train_dataset = ToxicSpansDataset(train_sentence_encodings, training_labels)
val_dataset = ToxicSpansDataset(val_sentence_encodings, val_labels)

print(f"Training examples: {len(train_dataset)}")
print(f"Validation examples: {len(val_dataset)}\n")

# We don't want to pass offset mappings to the model
train_offset_mapping = train_sentence_encodings.pop("offset_mapping") 
val_offset_mapping = val_sentence_encodings.pop("offset_mapping")

metrics = defaultdict(list)
def custom_metrics(pred):
  y_pred = pred.predictions.argmax(-1)
  precision, recall, f1, _ = precision_recall_fscore_support(val_labels, y_pred, average='macro')
  # store the metrics for visualization later
  metrics['precision'].append(precision)
  metrics['recall'].append(recall)
  metrics['f1'].append(f1)

  return {
    'precision': precision,
    'recall': recall,
    'f1': f1
  }

# Training Argument Object with hyper-parameter configuration.
# Training Argument Object with hyper-parameter configuration.
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=1,              # total number of training epochs
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

print('> Started training!\n')
trainer.train()

print("\n> ---- Making Predictions ---- \n")
# use trained model to make predictions
pred = trainer.predict(val_dataset)
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

print('\n> Plotting results.. \n')
plot(metrics)

end = time.time()
print(f"Time: {(end-start)/60} mins")

