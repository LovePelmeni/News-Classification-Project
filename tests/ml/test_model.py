import pytest 
from experiments.current_experiment.features import form
from src.models import predict_model
import numpy

@pytest.fixture(scope='module')
def dataset() -> form.NewsClassificationForm:
    """
    Returns Example of Model Dataset
    """
    return form.NewsClassificationForm()

def test_model(dataset):
    predictor = predict_model.baseline_predictor 
    result = predictor.predict_article_tags(dataset)
    expected_resp = numpy.array([], dtype=numpy.object_)
    assert numpy.array_equal(result, expected_resp)


