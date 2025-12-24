FROM python:3.10-slim

# Prevents Python from writing pyc files to disk and enables unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==2.2.1

# Copy dependency files first (better caching)
COPY pyproject.toml poetry.lock ./
COPY README.md ./


# Install deps (no virtualenv inside container)
RUN poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi --no-root

# Copy source code
COPY src/ ./src

COPY artifacts/ ./artifacts

# Expose API port
EXPOSE 8000

# Default command
CMD ["uvicorn", "mon_mlops_project.serving.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
