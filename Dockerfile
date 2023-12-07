FROM python:3.11-slim-buster

RUN pip install poetry

WORKDIR /app

COPY . .

RUN poetry install

CMD ["/bin/bash"]