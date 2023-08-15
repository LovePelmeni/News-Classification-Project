import pandas
import logging
import numpy

logger = logging.getLogger("feature_builder")
file_handler = logging.FileHandler(filename="../experiment_logs/features.log")
logger.addHandler(file_handler)

def build_features(dataset: pandas.DataFrame):
    """
    Function builds feature for the given dataset 
    
    Args:
        dataset (pandas.DataFrame) - experimental dataset
    """
    if dataset.shape[0] == 0: return dataset 
    __construct_headline_length(dataset)
    __construct_description_length(dataset)
    dataset = __construct_average_headline_and_description_lengths(dataset)
    dataset = __construct_popular_weekday(dataset)
    merge_text_features(dataset)
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

def __construct_popular_weekday(news_applications: pandas.DataFrame):
    """
    Finds most popular day to publish article
    based on category
    """
    news_applications['weekday'] = news_applications['date'].apply(
        lambda item: pandas.to_datetime(item).weekday()
    )

    most_freq_weekdays = news_applications.groupby(
    'category')['weekday'].apply(pandas.Series.mode).to_frame("popular_weekday")
    return news_applications.merge(most_freq_weekdays, on='category')


def merge_text_features(dataset: pandas.DataFrame):
    """
    Function merges headline and description of the article 
    into single text feature usign simple join function
    """
    if 'headline' not in dataset.columns: 
        raise ValueError("Headline is not presented")
    
    if 'short_description' not in dataset.columns:
        raise ValueError("Short Description is not presented")

    dataset['label'] = dataset['headline'].str.cat(
        others=dataset['short_description'], sep=" "
    )
    dataset.drop(columns=['headline', 'short_description'], inplace=True)
