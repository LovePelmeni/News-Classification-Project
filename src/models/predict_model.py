import logging 
from src.feature_form import feature_form
from src.exceptions import PredictionFailed

import pickle
from constants import constants
import definitions

Logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(filename=definitions.ROOT_DIR + "/logs/models.log")

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

model = pickle.load(open(constants.MODEL_URL, mode='rb'))
predictor = NewsPredictionModel(model=model)