import pytest 
from fastapi import testclient
from ...rest import settings
from ...src.feature_form import BaseFeatureForm


@pytest.fixture(scope='module')
def client():
    return testclient.TestClient(
        app=settings.application
    )

def get_feature_form() -> BaseFeatureForm:
    pass

def get_invalid_form() -> BaseFeatureForm:
    pass

feature_form = get_feature_form()
invalid_form = get_invalid_form()

def test_prediction_controller(client):
    response = client.post(
        "/predict/news/tags/", 
        content=feature_form
    )
    assert response.status_code == 201


def test_fail_prediction_controller(client):
    response = client.post(
        "/predict/news/tags/", 
        content=invalid_form
    )
    assert response.status_code == 400
    