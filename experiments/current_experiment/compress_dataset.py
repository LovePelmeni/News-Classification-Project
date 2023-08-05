import pandas 
import pickle

data_path = str(input("Enter path of the data to compress (allowed types: json): ")).strip()

def set_datatypes(dataset):
    dataset['category'] = dataset['category'].astype('category')
    dataset['date'] = dataset['date'].astype('datetime64[ns]')

try:
    dataset = pandas.read_json(data_path, lines=True)
    set_datatypes(dataset)

    pickle.dump(
        dataset, 
        file=open('./data/compressed_data/category_dataset.pkl', mode='wb')
    )

except(FileNotFoundError):
    raise SystemExit("file does not exist")

