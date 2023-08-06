import logging
import typing
import pandas
import numpy
from src.feature_form import feature_form
from current_experiment.features import build_features

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(filename="../experiment_logs/form.log")
logger.addHandler(file_handler)
class NewsClassificationForm(feature_form.BaseFeatureForm):

    """
    Dataset for Model Training
    """

    headline: str
    short_description: str 
    authors: typing.List[str]

    def get_dataframe(self):

        # initializing dataframe 
        df = pandas.DataFrame()
        df['headline'] = df['headline'].astype(numpy.object_)
        df['short_description'] = df['short_description'].astype(numpy.object_)

        # building features
        df = self.get_featured_dataset(df)
        return df

    @staticmethod
    def get_featured_dataset(dataset):
        return build_features.build_features(dataset)