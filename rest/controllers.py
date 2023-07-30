import logging 
from src.forms import feature_form 
from src.interfaces import model
from fastapi.responses import JSONResponse
from fastapi.requests import Request

rest_logger = logging.getLogger(__name__)

def predict_news_classification_tags(application_data: feature_form.NewsFeatureForm):
    """
    Function reprensents main REST-Endpoint for handling prediction requests
    via HTTP Web Protocol, transfers input data to the model and returns predicted outcome

    Args:
        application_data: feature_form.NewsFeatureForm - customer's form for prediction
    """
    predicted_tags = model.prediction_model.predict_article_tags(application_data)
    return JSONResponse(
        content=predicted_tags,
        status_code=201,
    )

def healthcheck(request: Request):
    """
    Standard Heatlhcheck REST-Endpoint
    """
    return JSONResponse(status_code=200)