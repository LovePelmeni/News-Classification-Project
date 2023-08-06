import pandas
import logging
import numpy

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(filename="../experiment_logs/features.log")
logger.addHandler(file_handler)

def build_features(dataset: pandas.DataFrame):
    """
    Function builds feature for the given dataset 
    
    Args:
        dataset (pandas.DataFrame) - experimental dataset
    """
    if dataset.shape[0] == 0: return dataset 
    try:
        __construct_headline_length(dataset)
        __construct_description_length(dataset)
        dataset = __construct_average_headline_and_description_lengths(dataset)
        dataset = __construct_popular_weekday(dataset)
        return dataset
    except(ValueError, TypeError, AttributeError) as err:
        logger.debug(err)
        return dataset

def __construct_headline_length(news_applications: pandas.DataFrame):
    news_applications['headline_len'] = news_applications['headline'].str.split(" ").apply(
    lambda item: numpy.array(item).shape[0])

def __construct_description_length(news_applications: pandas.DataFrame):
    news_applications['description_len'] = news_applications['short_description'].str.split(" ").apply(
    lambda item: numpy.array(item).shape[0])

def __construct_average_headline_and_description_lengths(news_applications: pandas.DataFrame):
    """
    Method constructs average headline and description lengths 
    for each news article category

    Args:
        news_applications: (pandas.DataFrame) dataset
    """
    __construct_description_length(news_applications)
    __construct_headline_length(news_applications)

    averages = news_applications.groupby('category').agg(
        average_headline_len=("headline_len", "mean"),
        average_description_len=("description_len", "mean")
    ).reset_index()
    return news_applications.merge(
        averages, on='category'
    )

def __construct_popular_weekday(news_applications):
    """
    Finds most popular day to publish article
    based on category
    """
    news_applications['weekday'] = news_applications['date'].apply(
        lambda item: pandas.to_datetime(item).weekday()
    )

    most_freq_weekdays = news_applications.groupby('category').agg(
        popular_weekday=('weekday', pandas.Series.mode)
    ).reset_index()

    return news_applications.merge(most_freq_weekdays, on='category')