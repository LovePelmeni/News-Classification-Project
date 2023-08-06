import abc
import pydantic
import pandas


class BaseFeatureForm(pydantic.BaseModel, abc.ABC):
    """
    Abtract Class represents basic feature form, that should be passed 
    to ML Model Algorithm for making predictions. 

    You should manually add features, based on model state 
    and optionally add some validation techniques
    """
    @abc.abstractmethod
    def get_dataframe(self, *args, **kwargs) -> pandas.DataFrame:
        """
        Method should return pandas.DataFrame object with features sorted in the 
        order, compatible with ML Model
        """
