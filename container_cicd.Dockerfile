FROM python:3.10-buster

RUN python3 -m pip install pipx && python3 -m pipx ensurepath
RUN pipx install poetry
RUN pipx ensurepath
RUN . ~/.bashrc

COPY . /app
WORKDIR /app

RUN poetry install

# CMD ["poetry", "run"]
