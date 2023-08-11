import pandas
from src.encoders import encoders
import category_encoders as ct
from experiments.current_experiment.text_classification import text_encoding


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
        encoded_hot = self.__get_one_hot_encoded(dataset)
        categories = dataset['category'].unique()

        X = pandas.DataFrame()
        X_data = encoded_hot.select_dtypes(include='object')

        for category in categories:
            encoder = ct.TargetEncoder(smoothing=0)
            data = encoder.fit_transform(X_data, encoded_hot[category])
            X = pandas.concat([X, data], axis=1)
        return X

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

    @staticmethod
    def __get_one_hot_encoded(dataset: pandas.DataFrame) -> pandas.DataFrame:
        """
        Function returns one-hot encoded representation of the dataset
        """
        enc_data = pandas.get_dummies(dataset['category'])
        return pandas.concat([dataset, enc_data], axis=1)


