import pandas as pd
from ast import literal_eval

from sklearn.model_selection import train_test_split

def load_dataset(dataset_path):
    dataset = pd.read_csv(dataset_path)
    print(f"\n> Loading {dataset.shape[0]} examples located at '{dataset_path}'\n")

    dataset["spans"] = dataset.spans.apply(literal_eval)
    text = dataset["text"]
    spans = dataset["spans"]
   
    return text, spans

def training_validation_split(texts, spans, test_size):
  # Use sklearn function to split the dataset
  training_texts, val_texts, training_spans, val_spans = train_test_split(texts, spans, test_size=test_size)
  # Create list of lists
  training_texts = [train_text for train_text in training_texts]
  training_spans = [training_span for training_span in training_spans]
  val_texts = [val_text for val_text in val_texts]
  val_spans = [val_span for val_span in val_spans]

  return training_texts, val_texts, training_spans, val_spans
