import time

from load_dataset import load_dataset
from tokenize_data import tokenize_data
from train_val_split import train_val_split

from transformers import BertForTokenClassification, Trainer, TrainingArguments

start = time.time()

texts, spans = load_dataset('../data/tsd_train.csv')

tokenized_texts, labels = tokenize_data(texts, spans)

train_dataset, val_dataset = train_val_split(tokenized_texts, labels, 0.2)


training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=1000,
)

model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=3)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

print('Beginning training..\n')
trainer.train()


end = time.time()
print(f"Time: {end-start}s")