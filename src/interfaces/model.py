import pandas
import sklearn
from src.forms import feature_form
import pickle

MODEL = pickle.load() 

class PredictionModel(object):
    """
    Class represents basic ML Algorithm for predicting 
    news article related tags

    :methods: 
        predict_news_topic - main method, that returns predicted output
    """
    __model = MODEL
    
    def predict_news_topic(self, news_form: feature_form.NewsFeatureForm) -> feature_form.NewsCategory:
        """
        Function analyzes information given at 'news_form' and predicts news article category 
        using ML Algorithm provided via well-protected object variable
        """
        pass
        