FROM python:3.11-slim

# Install pipx and Poetry
RUN pip install pipx && \
    pipx install poetry && \
    pipx ensurepath

# Install Git and any other dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Add /root/.local/bin to PATH directly in Docker for Poetry
ENV PATH="/root/.local/bin:$PATH"
ENV FILESDIR="/app"

# Disable Poetry's automatic virtual environment creation
RUN poetry config virtualenvs.create false

# Copy entire project
COPY . /app
WORKDIR /app

# Install project with dependencies
RUN poetry install --no-cache --with dev
