import pandas 
from scipy import stats
import collections 
import typing
from baseline_requirements import data
import numpy
import logging 
import definitions
import os

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(filename=os.path.join(definitions.ROOT_DIR, "logs/drift_detection.py"))

def compare_datasets(old_data: pandas.DataFrame, new_data: pandas.DataFrame) -> typing.Dict[str, int]:
    """
    Function compares 2 datasets and checks for potential drift 
    using statistical tests 
    """
    if old_data.columns.to_list().sort() != new_data.columns.to_list().sort():
        raise ValueError("Datasets are not compatible")
    try:
        driftmap = collections.defaultdict(dict)
        for feature in new_data.columns:

            if feature in new_data.select_dtypes(include='category').columns:

                driftmap[feature]['p_value'] = compare_categorical_features(
                    old_f=old_data[feature],
                    new_f=new_data[feature],
                    dataset=new_data,
                )
                
                driftmap[feature]['still_relevant'] = compare_cat_feature_relations(
                    imp_feature=new_data[feature],
                    target=new_data['category'],
                )

            elif feature in new_data.select_dtypes(include='number').columns:

                driftmap[feature]['p_value'] = compare_numerical_features(
                    old_f=old_data[feature],
                    new_f=new_data[feature]
                )
                driftmap[feature]['variance'] = check_variance_distinction(
                    old_f=old_data[feature],
                    new_f=new_data[feature]
                )
                driftmap[feature]['still_relevant'] = compare_num_feature_relations(
                    imp_feature=new_data[feature],
                    target=new_data['category']
                )
                
            elif feature in new_data.select_dtypes(include='boolean').columns:
            
                driftmap[feature]['phi_value'] = compare_boolean_features(
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
    return prop >= data.STATS_THRESHOLD_LIMIT

    
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
    return p_value >= data.P_VALUE_THRESHOLD

def compare_categorical_features(old_f: pandas.Series, new_f: pandas.Series, dataset: pandas.DataFrame) -> bool:
    """
    Function compares categorical features
    and detects drift, once they have certain distinctions
    using chi squared test 

    Args:
        old_f (pandas.Series) old feature distribution
        new_f (pandas.Series) new feature distribution

    Returns:
        bool True whether no drift detected else False
    """
    if old_f.shape[0] != new_f.shape[0]: 
        raise ValueError("features should have the same length")
    try:
        old_contingency_table = dataset[old_f].pivot_table(columns='category', values='date', aggfunc='count').iloc[0]
        new_contigency_table = dataset[new_f].pivot_table(columns='category', values='date', aggfunc='count').iloc[0]

        _, p_val_old, _, _ = stats.chi2_contingency(observed=old_contingency_table)
        _, p_val_new, _, _ = stats.chi2_contingency(observed=new_contigency_table)

        return ((p_val_old >= data.P_VALUE_THRESHOLD) and (p_val_new >= data.P_VALUE_THRESHOLD)
        ) or ((p_val_old <= data.P_VALUE_THRESHOLD) and (p_val_new <= data.P_VALUE_THRESHOLD))

    except Exception as err:
        logger.error(err)
        return False

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
    return p_value >= data.P_VALUE_THRESHOLD 

def compare_boolean_features(old_f: pandas.Series, new_f: pandas.Series):
    """
    Function compares boolean features from old and new dataframes 
    to detect drift using PHI coefficient
    
    Args:
        old_f (pandas.Series) old feature distribution
        new_f (pandas.Series) new feature distribution

    Returns:
        bool True whether no drift detected else False
    """
    contingency_table = numpy.array(
        [[numpy.sum((old_f == 1) & (new_f == 1)), numpy.sum((old_f == 1) & (new_f == 0))],
        [numpy.sum((old_f == 0) & (new_f == 1)), numpy.sum((old_f == 0) & (new_f == 0))]]
    )
    
    # Calculate the chi-squared test and ignore the p-value and degrees of freedom
    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
    
    # Calculate the phi coefficient
    phi = numpy.sqrt(chi2 / numpy.sum(contingency_table))
    
    return phi

def compare_cat_feature_relations(imp_feature: pandas.Series, target: pandas.Series) -> bool:
    """
    Function compares relation between categorical feature and target variable 
    using 
    """
    if imp_feature.shape[0] != target.shape[0]:
        raise ValueError("Feature and target should have the same length")

def compare_boolean_feature_relations(imp_feature: pandas.Series, target: pandas.Series) -> bool:
    """
    Function compares correlation between feature and target variable
    model container reote
    """
    if imp_feature.shape[0] != target.shape[0]:
        raise ValueError("Feature and")

    boolean_feature = numpy.asarray(imp_feature, dtype=numpy.int_)
    target_feature = numpy.asarray(target)

    table = [boolean_feature, target_feature]
    _, p_value, _ = stats.chi2_contingency(observed=table)
    return p_value <= data.STATS_THRESHOLD_LIMIT

def compare_num_feature_relations(imp_feature: pandas.Series, target: pandas.Series) -> bool:
    """
    Function compares relations between important feature and target variable 
    to determine whether this feature is still valuable for predictions or not
    using KL Divergence

    Args:
        imp_feature (pandas.Series) old feature distribution
        target_variable (pandas.Series) new feature distribution

    Returns:
        bool True whether no drift detected else False
    """
    if imp_feature.shape[0] != target.shape[0]:
        raise ValueError("shapes of both distributions must be the same size")

    if (~numpy.isclose(imp_feature, 1)) or (~numpy.isclose(target_distribution, 1)):
        raise ValueError("both distributions should sum up to 1.")

    epsilon = 1e-8
    target_distribution = numpy.clip(target, epsilon, 1)
    feature_distribution = numpy.clip(imp_feature, epsilon, 1)

    kl_div = numpy.sum(
        target_distribution * (
        numpy.log(target_distribution / feature_distribution))
    )
    return kl_div
