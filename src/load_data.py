import json
import pandas as pd

def load_clinc150_data(json_path):
    """
    Load the CLINC150 dataset from a JSON file.

    Parameters:
        json_path (str): Path to the data_full.json file

    Returns:
        tuple: train_df, val_df, test_df, oos_df (all as pandas DataFrames)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    def convert(split):
        texts, labels = [], []
        for text, label in data[split]:
            texts.append(text)
            labels.append(label)
        return pd.DataFrame({'text': texts, 'intent': labels})
    
    train_df = convert('train')
    val_df = convert('val')
    test_df = convert('test')
    oos_df = convert('oos_test')

    return train_df, val_df, test_df, oos_df
