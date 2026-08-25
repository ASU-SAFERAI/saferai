FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for psycopg2 (if db extra is needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY pyproject.toml ./
COPY src/ src/

# Install the package with all optional dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir ".[all]"

# Download spacy/presidio models if PII is needed
RUN python -m spacy download en_core_web_lg || true

# Default entrypoint: run the post-deploy CLI
ENTRYPOINT ["post-deploy"]
CMD ["run", "--preset", "safer", "--help"]
