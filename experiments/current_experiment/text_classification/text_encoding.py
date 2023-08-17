import pandas
from sklearn.feature_extraction import text

import warnings 
import numpy

from sklearn.manifold import TSNE
from text_classification import constants

warnings.filterwarnings('ignore')
class TFIDFVectorizedDataset(object):

    """
    Class used for vectorizing key words for provided category 
    using TF/IDF

    text_set: pandas.Dataframe()
    """

    def __init__(self, text_set: pandas.DataFrame):
        self.text_data = text_set
        self.output_data: pandas.DataFrame = pandas.DataFrame(index=text_set.index)

    def get_dataframe(self):
        self.output_data.fillna(value=0, inplace=True)
        self.output_data = self.__reduce_dims(dataset=self.output_data)
        return self.output_data

    def __reduce_dims(self, dataset: pandas.DataFrame):
        """
        Function reduces number of dimensions in the text data 
        Args:
            dataset (pandas.DataFrame) - dataset, containing text data
        """
        tsne = TSNE(n_components=constants.N_TSNE_COMPONENTS, random_state=True)
        reduced = pandas.DataFrame(
            tsne.fit_transform(dataset), columns=['x', 'y']
        )
        return reduced

    def encode_categorical_documents(self, category):
        """
        Function vectorizes text data using 
        TF / IDF vectors and then add new columns to dataframe
        with their respective frequencies for each given document
        """
        if category not in self.text_data['category'].unique(): 
            raise ValueError("Invalid Category %s" % category)

        vectorizer = text.TfidfVectorizer(
            stop_words='english', 
            max_features=200,
            min_df=0.07,
            lowercase=True,
            ngram_range=(1, 3),
            use_idf=True
        )

        category_data = self.text_data[self.text_data['category'] == category]
        vectorized = vectorizer.fit_transform(
            raw_documents=category_data["label"]
        ).toarray()

        labels = vectorizer.get_feature_names_out()
        feature_df = numpy.column_stack(vectorized)
        new_text_data = pandas.DataFrame(dict(zip(labels, feature_df)))
        self.output_data = pandas.concat([self.output_data, new_text_data], axis=1)
