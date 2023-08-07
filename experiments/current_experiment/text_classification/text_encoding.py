import pandas
from sklearn.feature_extraction import text
import warnings 

warnings.filterwarnings('ignore')

class TFIDFVectorizedDataset(dict):

    def __init__(self, text_data: pandas.DataFrame):
        self.text_data = text_data 
        self.vectorizer = text.TfidfVectorizer(
            stop_words='english', 
            max_features=200,
            min_df=0.05,
            lowercase=True
        )

    def get_vectorized_df(self):
        """
        Function vectorizes text data using 
        TF / IDF vectors and then add new columns to dataframe
        with their respective frequencies for each given document
        """
        vectorized = self.vectorizer.fit_transform(
            raw_documents=self.text_data["label"]
        )
        for idx, field in enumerate(self.vectorizer.get_feature_names_out()):
            self.text_data[field] = vectorized[:, idx].toarray()
        return self.text_data