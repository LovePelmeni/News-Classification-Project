import logging
import definitions

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(
    filename=definitions.ROOT_DIR + "/logs/settings.log")
logger.addHandler(file_handler)


try:
    open("../experiments/current_experiment/models/baseline_classifier.pkl", mode='rb')
except(FileNotFoundError) as model_file_err:
    raise SystemExit(
        "ML Model File does not exist"
    )

try:
    import fastapi
    import os
    from fastapi.middleware import cors

    from rest import controllers
    from src import exceptions
    from rest import exc_handlers
    from rest import system_metrics

except (ImportError, ModuleNotFoundError) as err:
    logger.error({'msg': err})
    raise SystemExit(
        "Some of the modules failed to be loaded, check logs for more information"
    )

# Environment Variables

DEBUG_MODE = os.environ.get("DEBUG_MODE", False)
VERSION = os.environ.get("VERSION", "1.0.0")

# CORS configuration
ALLOWED_HOSTS = os.environ.get("ALLOWED_HOSTS", "*")
ALLOWED_HEADERS = os.environ.get("ALLOWED_HEADERS", "*")

application = fastapi.FastAPI(
    debug=DEBUG_MODE,
    version=VERSION
)

try:
    # Adding Middlewares
    application.add_middleware(
        middleware_class=cors.CORSMiddleware,
        allowed_hosts=[
            host for host in ALLOWED_HOSTS] if ALLOWED_HOSTS else ["*"],
        allowed_headers=[
            header for header in ALLOWED_HEADERS] if ALLOWED_HEADERS else ["*"],
        allowed_methods=["POST", "OPTIONS", "GET"]
    )

    # Adding Rest Endpoints

    application.add_api_route(
        path="/predict/news/tags/",
        methods=["POST"],
        endpoint=controllers.predict_news_classification_tags,
        description="API Endpoint for classifying news"
    )
    application.add_api_route(
        path="/healthcheck/",
        methods=["GET"],
        endpoint=controllers.healthcheck,
        description="Startard Healthcheck API Endpoint"
    )

    # resource metrics usage endpoints
    application.add_api_route(
        path="/container/",
        methods=["GET"],
        endpoint=system_metrics.get_system_resource_metrics,
    )

    application.add_api_route(
        path="/get/system/resource/consume/",
        methods=['GET'],
        endpoint=controllers.check_resource_exceed
    )

    # Adding Exception Handlers
    application.add_exception_handler(
        exc_class_or_status_code=exceptions.PredictionFailed,
        handler=exc_handlers.handle_invalid_form
    )

except (fastapi.exceptions.FastAPIError, AttributeError, IndexError) as err:
    logger.critical(err)
    raise SystemExit(
        "Failed to start application, check logs"
    )
