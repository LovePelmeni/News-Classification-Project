#!/bin/bash 

echo "Running deployment pipeline for the News Classification Project"

autopep8 --recursive --in-place 

flake8 . 

if [ $? -ne 0 ]; then 
    echo "Code does not have certain format"
    exit 1;

echo "Running unittests..."

python -m pytest 

if [ $? -ne 0 ]; then 
    echo "unittests failed"
    exit 1;

echo "Running ASGI Server..."

uvicorn rest.settings:application --host 0.0.0.0 --port 8080

if [ $? -ne 0 ]; then 
    echo "Failed to start ASGI Server"
    exit 1;