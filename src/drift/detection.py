import pandas 
from scipy import stats
import collections 
import typing
from ...model_requirements import requirements

def compare_datasets(old_data: pandas.DataFrame, new_data: pandas.DataFrame) -> typing.Dict[str, int]:
    """
    Function compares 2 datasets and checks for potential drift 
    using statistical tests 
    """
    if old_data.columns != new_data.columns:
        raise ValueError("Datasets are not compatible")
    try:
        driftmap = collections.defaultdict(str)
        for feature in new_data.columns:

            if new_data[feature].dtype in ('category', 'str'):

                driftmap[feature]['p_value'] = compare_categorical_features(
                    old_f=old_data[feature],
                    new_f=new_data[feature]
                )

            elif new_data[feature].dtype.str.startswith(["int", "float", "uint"]):

                driftmap[feature]['p_value'] = compare_numerical_features(
                    old_f=old_data[feature],
                    new_f=new_data[feature]
                )
            else:
                driftmap[feature]['p_value'] = compare_boolean_features(
                    old_f=old_data[feature],
                    new_f=new_data[feature]
                )
            
            driftmap[feature]['null_exceed'] = check_null_proportion(
                feature=new_data[feature]
            )
        return driftmap
    except Exception as err:
        raise err


def check_null_proportion(feature: pandas.Series) -> bool:
    """
    Checks for proportion of the NaN values inside the feature 

    Args:
        feature (pandas.Series) - feature, represents 

    Returns:
        boolean variable 
        True - exceeds limit
        False - okay
    """
    if len(feature) == 0: return False 
    prop = feature.isna().sum() / len(feature) 
    return prop <= requirements.STATS_THRESHOLD_LIMIT

    
def check_variance_distinction(old_f: pandas.Series, new_f: pandas.Series) -> bool:
    """
    Function checks the variance between old and new feature
    using Levene statistical test
    
    Args:
        old_f: pandas.Series
        new_f: pandas.Series object containg
    Returns:
        boolean variable 
        True - variance is okay
        False - variance is not okay (drift)
    """
    stats, p_value = stats.levene(old_f, new_f)
    return p_value >= requirements.P_VALUE_THRESHOLD

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
    pass


def compare_numerical_features(old_f: pandas.Series, new_f: pandas.Series) -> bool:
    """
    Function compares numerical features from old and new dataframes
    to detect drift using Kolmogorov-Smirnov Test
    
    Args:
        old_f (pandas.Series) old feature distribution
        new_f (pandas.Series) new feature distribution

    Returns:
        bool True whether no drift detected else False
    """
    _, p_value = stats.ks_2samp(old_f, new_f)
    return p_value >= requirements.P_VALUE_THRESHOLD 

def compare_boolean_features(old_f: pandas.Series, new_f: pandas.Series):
    """
    Function compares boolean features from old and new dataframes 
    to detect drif
    
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
    pass 
