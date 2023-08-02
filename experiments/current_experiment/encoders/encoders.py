from ....src.encoders import encoders 
import pandas 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, TargetEncoder

class DatasetEncoder(encoders.BaseDatasetEncoder):
    
    def encode_dataset(self, dataset: pandas.DataFrame) -> pandas.DataFrame:
        pass