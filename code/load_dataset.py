import pandas as pd
from ast import literal_eval

def load_dataset(dataset_path):
    dataset = pd.read_csv(dataset_path)
    print(f"> Loading {dataset.shape[0]} examples located at '{dataset_path}'")

    dataset["spans"] = dataset.spans.apply(literal_eval)
    text = dataset["text"]
    spans = dataset["spans"]
   
    return text, spans
