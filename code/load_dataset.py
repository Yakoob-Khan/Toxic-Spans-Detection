import pandas as pd
from ast import literal_eval

def load_dataset(dataset_path):
    print(f"> Loading the dataset located at '{dataset_path}'")

    dataset = pd.read_csv(dataset_path)
    dataset["spans"] = dataset.spans.apply(literal_eval)
    text = dataset["text"]
    spans = dataset["spans"]

    print(f"-- Done loading {dataset.shape[0]} examples! \n")
   
    return text, spans
