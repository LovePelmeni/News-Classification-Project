from fastapi import Request
from fastapi import responses 

def handle_invalid_form(request: Request, exc):
    """
    Basic exception endpoint for handling 
    feature form invalidation
    Args:
        request: incoming http request 
        exc: raised exception
    Returns:
        JSONResponse, containing explanations for the given error
    """
    return responses.JSONResponse(
        status_code=400, 
        content={
            'error': exc
        }
    )