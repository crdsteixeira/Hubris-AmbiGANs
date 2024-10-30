FROM python:3.10-buster

# Install pipx and Poetry
RUN pip install pipx && \
    pipx install poetry && \
    pipx ensurepath

# Add /root/.local/bin to PATH directly in Docker
ENV PATH="/root/.local/bin:$PATH"

COPY . /app
WORKDIR /app

RUN poetry install

# CMD ["poetry", "run"]
