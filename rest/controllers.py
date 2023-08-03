import logging 
from fastapi.responses import JSONResponse
from fastapi.requests import Request
import definitions

rest_logger = logging.getLogger(__name__)

file_handler = logging.FileHandler(
    filename=definitions.ROOT_DIR + "/logs/controllers.log"
)

from src.models import predict_model
from src.feature_form import feature_form

def predict_news_classification_tags(application_data: feature_form.BaseFeatureForm):
    """
    Function reprensents main REST-Endpoint for handling prediction requests
    via HTTP Web Protocol, transfers input data to the model and returns predicted outcome

    Args:
        application_data: feature_form.NewsFeatureForm - customer's form for prediction
    """
    predicted_tags = predict_model.predictor.predict_article_tags(application_data)
    return JSONResponse(
        content=predicted_tags,
        status_code=201,
    )

def healthcheck(request: Request):
    """
    Standard Heatlhcheck REST-Endpoint
    """
    return JSONResponse(status_code=200)