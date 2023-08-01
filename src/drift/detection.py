import pandas 
from statsmodels.api import stats
import scipy


def compare_categorical_features(old_f: pandas.Series, new_f: pandas.Series) -> bool:
    """
    Function compares categorical features
    and detects drift, once they have certain distinctions

    Args:
        old_f (pandas.Series) old feature distribution
        new_f (pandas.Series) new feature distribution

    Returns:
        bool True whether no drift detected else False
    """


def compare_numerical_features(old_f: pandas.Series, new_f: pandas.Series) -> bool:
    """
    Function compares numerical features 
    
    Args:
        old_f (pandas.Series) old feature distribution
        new_f (pandas.Series) new feature distribution

    Returns:
        bool True whether no drift detected else False
    """


def compare_feature_relations(imp_feature: pandas.Series, target: pandas.Series) -> bool:
    """
    Function compares relations between important feature and target variable 
    to determine whether this feature is still valuable for predictions or not

    Args:
        imp_feature (pandas.Series) old feature distribution
        target_variable (pandas.Series) new feature distribution

    Returns:
        bool True whether no drift detected else False
    """