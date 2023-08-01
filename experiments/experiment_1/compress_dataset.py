import pickle
import pandas

def compress_dataset() -> pandas.DataFrame:
    """
    Converts dataset into .pkl format
    which makes dataset more light
    """
    dataset = pandas.read_json("data/raw_data/category_dataset.json")
    pickle.dump(dataset, file=open('data/compressed_data/category_dataset.pkl'))