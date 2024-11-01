FROM python:3.10-slim

# Install pipx and Poetry
RUN pip install pipx && \
    pipx install poetry && \
    pipx ensurepath

# Install Git and any other dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Add /root/.local/bin to PATH directly in Docker for Poetry
ENV PATH="/root/.local/bin:$PATH"

# Disable Poetry's automatic virtual environment creation
RUN poetry config virtualenvs.create false

# Install libraries
COPY pyproject.toml /app/pyproject.toml
WORKDIR /app
RUN poetry install --no-root --no-cache
