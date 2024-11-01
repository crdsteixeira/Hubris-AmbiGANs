FROM python:3.10-slim

# Install pipx and Poetry
RUN pip install pipx && \
    pipx install poetry && \
    pipx ensurepath

# Install Git and any other dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Add /root/.local/bin to PATH directly in Docker for Poetry
ENV PATH="/root/.local/bin:$PATH"

# Configure Poetry
RUN poetry config virtualenvs.in-project false && \
    poetry config virtualenvs.create true

# Add Poetry’s virtual environment location to PATH directly in Docker
ENV PATH="/root/.cache/pypoetry/virtualenvs/*/bin:$PATH"

COPY pyproject.toml /app/pyproject.toml
WORKDIR /app

RUN poetry install --no-root --no-cache

# CMD ["poetry", "run"]
