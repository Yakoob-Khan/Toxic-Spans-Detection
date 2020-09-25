import pandas as pd
from ast import literal_eval

def load_dataset(dataset_path):
    print(f"> Loading the dataset located at '{dataset_path}'")

    dataset = pd.read_csv(dataset_path)
    dataset["spans"] = dataset.spans.apply(literal_eval)
    text = dataset["text"]
    spans = dataset["spans"]

    print('-- Done!\n')
   
    return text, spans

if __name__ == '__main__':
    load_dataset('../data/tsd_trial.csv')