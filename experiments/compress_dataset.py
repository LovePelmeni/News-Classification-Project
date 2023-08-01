import pandas 
import pickle

dataset = pandas.read_json("../data/raw_data/category_dataset.json", lines=True)
pickle.dump(dataset, file=open('../data/compressed_data/category_dataset.pkl', mode='wb'))
    