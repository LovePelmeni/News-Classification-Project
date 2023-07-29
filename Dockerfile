FROM --platform=arm64 python3.9-bullseye 
LABEL author=kirklimushin@gmail.com 
RUN echo "Building project... Relax and get some 🍺"

# Root user credentials 
ARG ROOT_USER 

# Creating custom user
RUN useradd --create-home ${ROOT_USER}

# Initializing working directory 
WORKDIR /project/dir/${ROOT_USER}

COPY ./deployment ./deployment
COPY ./src ./src
COPY ./proj_requirements/prod_requirements.txt ./
COPY ./poetry.lock ./
COPY ./pyproject.toml ./

RUN pip install --upgrade pip 

RUN poetry install && poetry export --format=requirements.txt \\
--output=prod_requirements.txt --without-hashes

RUN pip install -r prod_requirements.txt && pip install 'fastapi[all]' --upgrade

RUN chmod +x entrypoint.sh
ENTRYPOINT ["sh", "entrypoint.sh"]