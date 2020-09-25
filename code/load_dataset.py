import pandas as pd
from ast import literal_eval

def load_dataset(dataset_path):
    """
    Load the respective csv dataset file
    Arguments
    ---------
    dataset_path (string): representing the file path of the dataset

    Returns
    -------
    text (pd Series): 1D array of texts of size (n, 1), n = number of examples
    spans (pd Series): 1D array of respective spans (n, 1)
    """
    print(f"Loading the dataset located at '{dataset_path}'")

    dataset = pd.read_csv(dataset_path)
    dataset["spans"] = dataset.spans.apply(literal_eval)
    text = dataset["text"]
    spans = dataset["spans"]

    print('Done!')
   
    return text, spans

if __name__ == '__main__':
    load_dataset('../data/tsd_trial.csv')