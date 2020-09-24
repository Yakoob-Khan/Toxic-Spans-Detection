import pandas as pd

def load_dataset(dataset_path):
    """
    Load the respective csv dataset file
    Arguments
    ---------
    dataset_path: (string) representing the file path of the dataset

    Returns
    -------
    dataset (pandas dataframe): The text and the corresponding toxic spans
    
    """
    dataset = pd.read_csv(dataset_path)
    return dataset

