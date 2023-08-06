import logging 
from fastapi.responses import JSONResponse
from fastapi.requests import Request

import definitions 
from rest import system_metrics
from system_metrics import check_resource_exceed

rest_logger = logging.getLogger(__name__)

file_handler = logging.FileHandler(
    filename=definitions.ROOT_DIR + "/logs/controllers.log"
)

from src.models import predict_model
from src.feature_form import feature_form

def predict_news_classification_tags(application_data: feature_form.BaseFeatureForm):
    """
    Function represents main REST-Endpoint for handling prediction requests via HTTP Web Protocol.
    Expects Model Feature's Form as an input and then performs prediction.

    Args:
        application_data: feature_form.NewsFeatureForm - customer's form for prediction
    """
    predicted_tags = predict_model.baseline_predictor.predict_article_tags(application_data)
    return JSONResponse(
        content=predicted_tags,
        status_code=201,
    )

def healthcheck(request: Request):
    """
    Standard Heatlhcheck REST-Endpoint
    """
    return JSONResponse(status_code=200)

def check_resource_usage(request: Request):
    """
    Rest Endpoint for Monitoring application's resource usage, including: 
        1. CPU Usage 
        2. Memory Usage 
        3. Disk Space Usage
    """
    output = {}
    resource_info = system_metrics.get_system_resource_metrics()
    for resource_group in resource_info.keys():
        output[resource_group] = {
            "exceeds": check_resource_exceed(
                metric_type=resource_group,
                metrics=resource_info[resource_group]
            ),
            "metric_type": resource_group,
            "usage (%)": resource_info[resource_group]['Usage']
        }
    return resource_group
