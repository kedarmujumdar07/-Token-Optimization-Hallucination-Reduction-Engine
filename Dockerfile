FROM python:3.11-slim

WORKDIR /app

# System dependencies (for spaCy, torch CPU, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# spaCy model
RUN python -m spacy download en_core_web_sm

COPY . .

EXPOSE 8000 8501

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
