import pandas as pd
from ast import literal_eval
from sklearn.model_selection import train_test_split

from utils.helper import fix_spans

def load_dataset(dataset_path):
    dataset = pd.read_csv(dataset_path)
    print(f"\n> Loading {dataset.shape[0]} examples located at '{dataset_path}'\n")

    dataset["spans"] = dataset.spans.apply(literal_eval)
    texts, spans = dataset["text"], dataset["spans"]
    texts = [text for text in texts]
    spans = [fix_spans(span, texts[i]) for i, span in enumerate(spans)]
   
    return texts, spans

def load_testset(dataset_path):
  dataset = pd.read_csv(dataset_path)
  print(f"\n> Loading {dataset.shape[0]} test examples located at '{dataset_path}'\n")
  texts = dataset["text"]
  texts = [text for text in texts]
  return texts



def training_validation_split(texts, spans, test_size):
  # Use sklearn function to split the dataset
  training_texts, val_texts, training_spans, val_spans = train_test_split(texts, spans, test_size=test_size)
  # Create list of lists
  training_texts = [train_text for train_text in training_texts]
  training_spans = [fix_spans(training_span, training_texts[i]) for i, training_span in enumerate(training_spans)]
  val_texts = [val_text for val_text in val_texts]
  val_spans = [fix_spans(val_span, val_texts[i]) for i, val_span in enumerate(val_spans)]

  return training_texts, val_texts, training_spans, val_spans
