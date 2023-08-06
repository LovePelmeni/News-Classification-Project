import pydantic 
import logging

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(filename="../experiment_logs/form.log")
logger.addHandler(file_handler)


class NewsClassificationForm(pydantic.BaseModel):
    pass