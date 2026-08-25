FROM python:3.10-slim

WORKDIR /app

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
