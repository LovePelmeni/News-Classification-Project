import pandas
from sklearn.feature_extraction import text
import warnings 

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
        vectorizer = text.TfidfVectorizer(
            stop_words='english', 
            max_features=200,
            min_df=0.2,
            lowercase=True
        )

        category_data = self.text_data.loc[self.text_data['category'] == category, :]
        vectorized = vectorizer.fit_transform(
            raw_documents=category_data["label"]
        )
        for idx, field in enumerate(vectorizer.get_feature_names_out()):
            self.text_data[field] = vectorized[:, idx].toarray()