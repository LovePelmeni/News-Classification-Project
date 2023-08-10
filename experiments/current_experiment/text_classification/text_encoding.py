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

        category_data = self.text_data.loc[
            self.text_data['category'] == category, :
        ]

        vectorized = vectorizer.fit_transform(
            raw_documents=category_data["label"]
        ).toarray()

        feature_word_names = vectorizer.get_feature_names_out()
        words_freqs = numpy.column_stack(tup=vectorized)

        feature_word_set = pandas.DataFrame(
            dict(zip(feature_word_names, words_freqs))
        )

        self.text_data = pandas.concat([
            self.text_data,
            feature_word_set
        ], axis=1)
