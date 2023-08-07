import pandas
from sklearn.feature_extraction import text
import warnings 
import numpy

warnings.filterwarnings('ignore')

class TFIDFVectorizedDataset(dict):

    """
    Class used for vectorizing key words for provided category 
    using TF/IDF

    text_set: pandas.Dataframe()
    """

    def __init__(self, text_set: pandas.DataFrame):
        self.text_data = text_set

    def get_dataframe(self):
        return self.text_data

    def encode_categorical_documents(self, category):
        """
        Function vectorizes text data using 
        TF / IDF vectors and then add new columns to dataframe
        with their respective frequencies for each given document
        """
        if category not in self.text_data['category'].unique(): 
            return 

        vectorizer = text.TfidfVectorizer(
            stop_words='english', 
            max_features=200,
            min_df=0.01,
            lowercase=True
        )

        category_data = self.text_data.loc[self.text_data['category'] == category, :]
        vectorized = vectorizer.fit_transform(
            raw_documents=category_data["label"]
        )

        for idx, field in enumerate(vectorizer.get_feature_names_out()):
            feature_df = vectorized[:, idx].toarray().flatten()
            if feature_df.shape[0] < self.text_data.shape[0]:
                feature_df = pandas.DataFrame(
                    {
                        field: numpy.concatenate(
                            [
                                feature_df, 
                                numpy.zeros_like(
                                    (self.text_data.shape[0] - feature_df.shape[0], 1)
                                )
                            ]
                        )
                    }
                )
            self.text_data = pandas.concat([self.text_data, feature_df], axis=1)
