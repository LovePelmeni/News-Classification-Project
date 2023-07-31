import logging 
from ..feature_form import feature_form
from ..exceptions import PredictionFailed
import sys
import pickle
from ...constants import constants

Logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(filename=sys.path[1] + "logs/models.log")
model = pickle.load(open(constants.MODEL_URL, mode='rb'))


class NewsPredictionModel(object):
    """
    Class for predicting article topics using ML Algorithm
    """
    def __init__(self, model):
        self.__model = model 

    def predict_article_tags(self, feature_form: feature_form.BaseFeatureForm):
        """
        Method predicts relevant article topics, based on the form data passed
        """
        try:
            relevant_tags = self.__model.predict(feature_form.get_dataframe())
            return relevant_tags
        except Exception as err:
            Logger.error({'msg': err})
            raise PredictionFailed() 

predictor = NewsPredictionModel(model=model)