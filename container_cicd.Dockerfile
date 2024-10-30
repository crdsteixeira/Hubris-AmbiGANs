FROM python:3.10-buster

RUN python3 -m pip install --user pipx && python3 -m pipx ensurepath
RUN source ~/.bashrc
RUN pipx install poetry

COPY . /app
WORKDIR /app

RUN poetry install

# CMD ["poetry", "run"]
