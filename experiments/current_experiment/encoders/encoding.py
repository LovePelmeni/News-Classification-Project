import pandas
from src.encoders import encoders
from experiments.current_experiment.text_classification import text_encoding
from sklearn.preprocessing import LabelEncoder

class DatasetEncoder(encoders.BaseDatasetEncoder):

    def encode_dataset(self, dataset: pandas.DataFrame) -> pandas.DataFrame:
        """
        Function encodes each part of the dataset

        Args:
            dataset (pandas.DataFrame) -> dataset to encode

        Returns:
            encoded dataset (pandas.DataFrame)
        """
        rest_fields = dataset.select_dtypes(exclude='object')
        target_encodings = self.encode_target_variable(dataset)
        word_encodings = self.encode_words(dataset)
        result = pandas.concat([rest_fields, target_encodings, word_encodings], axis=1)
        return result

    def encode_target_variable(self, dataset: pandas.DataFrame) -> pandas.DataFrame:
        """
        Function encodes target variable using combination of 
        One-Hot and Target Encodings

        Args: 
            dataset (pandas.DataFrame) - dataset, that contains target variable 

        Returns:
            Pandas Dataframe with encoded target variable
        """
        encoder = LabelEncoder()
        encoded_labels = encoder.fit_transform(
            y=dataset['category'].to_numpy().reshape(-1, 1)
        )
        return encoded_labels

    def encode_words(self, dataset: pandas.DataFrame) -> pandas.DataFrame:
        """
        Function encodes text using TF/IDF Vectorization Technique

        Args:
            dataset (pandas.DataFrame) - dataset, containing text features to encode
        """
        encoded_dataset = text_encoding.TFIDFVectorizedDataset(dataset)
        categories = dataset['category'].unique()
        for category in categories:
            encoded_dataset.encode_categorical_documents(
                category=category
            )
        return encoded_dataset.get_dataframe()