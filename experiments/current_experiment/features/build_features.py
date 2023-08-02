import numpy
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
import typing
import logging

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(filename="features.log")
logger.addHandler(file_handler)


def __construct_headline_length(news_applications: pandas.DataFrame):
    news_applications['headline_len'] = news_applications['headline'].str.split(" ").apply(
        lambda headline: numpy.array(headline).shape[0]
    )

def __construct_description_length(news_applications: pandas.DataFrame):
    news_applications['description_len'] = news_applications['short_description'].str.split(" ").apply(
        lambda description: numpy.array(description).shape[0]
    )

def construct_average_headline_and_description_lengths(news_applications: pandas.DataFrame):
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

def construct_popular_weekday(news_applications):
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


def get_most_frequent_headline_words(categories: typing.List, dataset: pandas.DataFrame):
    """
    Function returns hashmap, containing headline most frequent words per category
    limit set up to 5
    Args:
        categories: typing.List - array, containing unique categories
    """
    if len(categories) == 0: return

    headlines = {}
    for category in categories:
        vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
        data = dataset.loc[dataset['category'] == category, 'headline']
        matrix = vec.fit_transform(data)
        word_frequencies = matrix.sum(axis=0).A1
        feature_names = vec.get_feature_names_out()
        headlines[category] = sorted(dict(zip(feature_names, word_frequencies)), key=lambda item: item[1])[:5]
    return headlines


def get_most_frequent_headline_words(categories: typing.List, dataset: pandas.DataFrame):
    """
    Function returns hashmap, containing most description frequent words per category
    limit set up to 5
    Args:
        categories: typing.List - array, containing unique categories
    """
    if len(categories) == 0: return

    headlines = {}
    for category in categories:
        vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
        data = dataset.loc[dataset['category'] == category, 'short_description']
        matrix = vec.fit_transform(data)
        word_frequencies = matrix.sum(axis=0).A1
        feature_names = vec.get_feature_names_out()
        headlines[category] = sorted(dict(zip(feature_names, word_frequencies)), key=lambda item: item[1])[:5]
    return headlines