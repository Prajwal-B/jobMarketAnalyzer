# file: Dockerfile
# Containerized environment for Job Market Analyzer

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data exports/tableau_ready exports/by_region logs

# Set permissions
RUN chmod +x run_all.sh

# Default command: run full pipeline with sample data
CMD ["bash", "run_all.sh"]

# Alternative commands (uncomment as needed):
# CMD ["python", "scripts/data_collection.py", "--use-sample", "--output", "data/raw_jobs.csv"]
# CMD ["python", "scripts/skill_extractor.py", "--input", "data/cleaned_jobs.csv", "--output", "data/jobs_with_skills.csv"]
# CMD ["pytest", "tests/", "-v"]

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Labels
LABEL maintainer="Job Market Analyzer"
LABEL version="1.0.0"
LABEL description="Complete pipeline for analyzing India job market data"

# Expose port for future web interface (optional)
# EXPOSE 8000

# Volume mounts for data persistence
VOLUME ["/app/data", "/app/exports", "/app/logs"]