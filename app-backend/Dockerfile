FROM python:3.11.5

WORKDIR /app/src

RUN apt-get update && apt-get upgrade -y
RUN pip install --no-cache -U pip
RUN pip install --no-cache poetry

RUN poetry config virtualenvs.create false

COPY ./app-backend/pyproject.toml /app/src
COPY ./app-backend/poetry.lock /app/src

RUN poetry install --no-interaction --no-root -vvv

ADD ./app-backend /app/src

RUN chmod +x run.sh

CMD ["bash", "./run.sh"]

