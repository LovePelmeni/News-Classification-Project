from ....src.encoders import encoders 
import pandas

class DatasetEncoder(encoders.BaseDatasetEncoder):
    
    def encode_dataset(self, dataset: pandas.DataFrame) -> pandas.DataFrame:
        pass