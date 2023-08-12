import pandas
from src.encoders import encoders
from experiments.current_experiment.text_classification import text_encoding
from experiments.current_experiment.text_classification import constants
import numpy

class DatasetEncoder(encoders.BaseDatasetEncoder):

    def encode_dataset(self, dataset: pandas.DataFrame) -> pandas.DataFrame:
        """
        Function encodes each part of the dataset

        Args:
            dataset (pandas.DataFrame) -> dataset to encode

        Returns:
            encoded dataset (pandas.DataFrame)
        """
        if dataset.shape[0] == 0:
            raise ValueError("Dataset does not have any records")

        if len(dataset.columns) == 0:
            raise ValueError("Dataset does not have any columns")

        rest_fields = dataset.select_dtypes(exclude='object').drop(columns=['category']) # filtering rest fields
        target_encodings = self.encode_target_variable(dataset) # encoding target variable
        word_encodings = self.encode_words(dataset) # encoding text data using TF/IDF Vectorization
        result = pandas.concat([rest_fields, target_encodings, word_encodings], axis=1) # merging everything back into new dataframe
        return result

    def encode_target_variable(self, dataset: pandas.DataFrame) -> pandas.DataFrame:
        """
        Function encodes target variable using standard Label Encoding
        technique

        Args:
            dataset (pandas.DataFrame) -> dataset, that contains target variable

        Returns:
            pandas.DataFrame object with encoded target variable
        """
        if 'category' not in dataset.columns:
            raise ValueError("Target variable is not presented in the dataset")

        enc_data = pandas.DataFrame(
            {
                'category': dataset['category'].copy()
            }
        )
        enc_data['category'] = enc_data['category'].map(
            constants.TARGET_VAR_CATEGORIES
        ).astype(numpy.int8)
        return enc_data

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

