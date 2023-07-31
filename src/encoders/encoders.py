import abc
import pandas 

class BaseDatasetEncoder(abc.ABC):

    """
    Base class for encoding feature dataset
    """

    @abc.abstractmethod
    def encode_dataset(self, *args, **kwargs) -> pandas.DataFrame:
        """
        Main method for encoding dataset. All other encoding methods for 
        specific features should provided separately inside `BaseDatasetEncoder` class
        and triggered inside this method. 
        """